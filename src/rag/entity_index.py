"""ChromaDB index over the feature tables: one short text chunk per player, team or pair.

This complements the match search in ``retriever.py`` (whole matches).  Here every row of the
batsman, bowler, team, partnership, fielding and matchup feature tables becomes a chunk, so a
question such as "how does Kohli bat in the death overs?" retrieves Kohli's record.

  * ``build_index()``  chunks the Parquet tables and stores them in a persistent Chroma
                       collection under ``data/embeddings/``
  * ``retrieve()``     if the question names a player or team, filter by that name first, then
                       rank semantically among the survivors; otherwise plain semantic search

Every chunk carries metadata ``dataset`` ("cricsheet_main"), ``entity_type``, ``entity_name`` and
``league``.  Only the team table is split by league; the player tables aggregate all leagues, so
their league is "all".

Content rules (from the backtests): the retired Phase-1 scores (legacy_score, chemistry,
vulnerability, fielding score) are never written into a chunk, fielding chunks say they measure
dismissal involvement and not skill, and matchup chunks say they are raw small-sample records.
"""

import re
from collections.abc import Callable
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import EMBEDDINGS_DIR, PROCESSED_DIR
from src.names import DOMINANCE, TEAM_ABBREVIATIONS, _norm, score_name
from src.rag.retriever import default_embed_fn

DATASET = "cricsheet_main"
COLLECTION = "cricketmind_entities"
ENTITY_TYPES = ("batsman", "bowler", "team", "partnership", "fielding", "matchup")
PLAYER_TYPES = ("batsman", "bowler", "fielding")  # chunks whose entity_name is one player
MIN_MATCHUP_BALLS = 30  # most pairs have 6-11 balls: pure noise, and 82k chunks would take ~30 min
MAX_NAME_CANDIDATES = 6
INTENT_BOOST = 0.05  # added to the similarity of chunk types the question's wording points at;
#                      fixed in advance (not tuned): the near-ties among one player's chunks are
#                      typically 0.01-0.03 apart, so this only reorders those ties

LEAGUE_LABEL = {"IPL": "the IPL", "BBL": "the BBL", "CPL": "the CPL", "PSL": "the PSL",
                "T20I": "T20 internationals"}


# ── chunk text ──────────────────────────────────────────────────────────
def _f(x, nd=1):
    """Format a number; None for missing values so the phrase can be dropped."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    x = float(x)
    return f"{x:,.0f}" if nd == 0 else f"{x:.{nd}f}"  # counts use nd=0; rates keep their decimals


def _join(parts: list[str | None]) -> str:
    return " ".join(p for p in parts if p)


def _phase(label: str, value, nd=1) -> str | None:
    v = _f(value, nd)
    return f"{label} {v}" if v else None


def batsman_chunk(r) -> str:
    phases = ", ".join(p for p in (_phase("powerplay", r.sr_powerplay), _phase("middle overs", r.sr_middle),
                                   _phase("death overs", r.sr_death)) if p)
    recent = (f"Last {int(r.recent_innings)} innings: average {_f(r.recent_avg, 2)}, "
              f"strike rate {_f(r.recent_sr, 1)}." if _f(r.recent_avg) and _f(r.recent_sr) else None)
    return _join([
        f"{r.batter} - batting record across the IPL, BBL, CPL, PSL and T20 internationals combined.",
        f"{_f(r.innings, 0)} innings, {_f(r.total_runs, 0)} runs off {_f(r.total_balls, 0)} balls,",
        f"average {_f(r.average, 2)} ({_f(r.times_out, 0)} dismissals)," if _f(r.average) else
        f"never dismissed ({_f(r.times_out, 0)} dismissals),",
        f"strike rate {_f(r.strike_rate, 2)}, highest score {_f(r.highest_score, 0)}.",
        f"Boundaries on {_f(r.boundary_pct)}% of balls, dot balls {_f(r.dot_pct)}%.",
        f"Strike rate by phase: {phases}." if phases else None,
        recent,
    ])


def bowler_chunk(r) -> str:
    phases = ", ".join(p for p in (_phase("powerplay", r.econ_powerplay, 2),
                                   _phase("middle overs", r.econ_middle, 2),
                                   _phase("death overs", r.econ_death, 2)) if p)
    overs = f"{int(r.total_balls) // 6}.{int(r.total_balls) % 6}"
    recent = (f"Last {int(r.recent_matches)} matches: economy {_f(r.recent_econ, 2)}, "
              f"{_f(r.recent_wickets, 0)} wickets." if _f(r.recent_econ) else None)
    return _join([
        f"{r.bowler} - bowling record across the IPL, BBL, CPL, PSL and T20 internationals combined.",
        f"{_f(r.matches, 0)} matches, {overs} overs, {_f(r.total_wickets, 0)} wickets,",
        f"economy {_f(r.economy, 2)},",
        f"bowling average {_f(r.bowling_avg, 2)}, strike rate {_f(r.bowling_sr, 1)} balls per wicket,"
        if _f(r.bowling_avg) else None,
        f"dot balls {_f(r.dot_pct)}%, boundaries conceded on {_f(r.boundary_pct_conceded)}% of balls.",
        f"Economy by phase: {phases}." if phases else None,
        recent,
    ])


def team_chunk(r) -> str:
    where = LEAGUE_LABEL.get(r.league, r.league)
    return _join([
        f"{r.team} in {where}: {_f(r.matches, 0)} matches, {_f(r.wins, 0)} wins ({_f(r.win_pct)}% win rate).",
        f"Average score {_f(r.avg_score)}, run rate {_f(r.avg_run_rate, 2)} per over,",
        f"boundaries on {_f(r.boundary_pct)}% of balls.",
        f"Conceded {_f(r.avg_bowling_econ, 2)} runs per over and took {_f(r.avg_wickets_taken, 1)} "
        "wickets per match.",
        f"Matches from {pd.Timestamp(r.first_match).date()} to {pd.Timestamp(r.last_match).date()}."
        if pd.notna(r.first_match) and pd.notna(r.last_match) else None,
    ])


def partnership_chunk(r) -> str:
    return _join([
        f"{r.player_a} and {r.player_b} batting together: {_f(r.times_paired, 0)} partnerships,",
        f"{_f(r.total_runs, 0)} runs from {_f(r.total_balls, 0)} balls",
        f"(run rate {_f(r.run_rate, 2)} per over, average stand {_f(r.avg_partnership_runs)}).",
        f"Boundaries on {_f(r.boundary_pct)}% of balls, strike rotated on {_f(r.rotation_pct)}% of balls.",
        f"Best stand {_f(r.best_partnership, 0)}; {_f(r.fifty_plus_stands, 0)} stands of 50+ "
        f"and {_f(r.century_stands, 0)} of 100+.",
    ])


def fielding_chunk(r) -> str:
    keeper = ("Has stumpings, so these counts reflect the wicketkeeper role." if bool(r.is_keeper)
              else None)
    return _join([
        f"{r.fielder} - fielding: {_f(r.total_dismissals, 0)} dismissals in {_f(r.matches_played, 0)} "
        f"matches ({_f(r.dismissals_per_match, 2)} per match):",
        f"{_f(r.catches, 0)} catches, {_f(r.caught_and_bowled, 0)} caught-and-bowled, "
        f"{_f(r.run_outs, 0)} run-outs, {_f(r.stumpings, 0)} stumpings.",
        f"By phase: powerplay {_f(r.dismissals_powerplay, 0)}, middle overs {_f(r.dismissals_middle, 0)}, "
        f"death overs {_f(r.dismissals_death, 0)}.",
        keeper,
        "This counts involvement in dismissals, not fielding quality (no positions or drops in the data).",
    ])


def matchup_chunk(r) -> str:
    return _join([
        f"{r.batter} against {r.bowler} (raw head-to-head record): {_f(r.balls_faced, 0)} balls, "
        f"{_f(r.runs_scored, 0)} runs, strike rate {_f(r.strike_rate, 2)},",
        f"out {_f(r.times_dismissed, 0)} times, {_f(r.dots, 0)} dot balls ({_f(r.dot_pct)}%),",
        f"{_f(r.boundaries, 0)} boundaries, {_f(r.encounters, 0)} meetings.",
        "A small-sample fact, not a forecast.",
    ])


def _records(df: pd.DataFrame, entity_type: str, name_fn, chunk_fn, league_fn, weight_fn):
    out = []
    for r in df.itertuples(index=False):
        name, league = name_fn(r), league_fn(r)
        out.append({
            "id": f"{entity_type}:{name}:{league}",
            "text": chunk_fn(r),
            "metadata": {"dataset": DATASET, "entity_type": entity_type, "entity_name": name,
                         "league": league, "weight": float(weight_fn(r))},
        })
    return out


def build_chunks(processed_dir: Path | str | None = None,
                 min_matchup_balls: int | None = MIN_MATCHUP_BALLS) -> list[dict]:
    """Chunk every feature table.  Tables that are missing are skipped with a warning."""
    d = Path(processed_dir or PROCESSED_DIR)
    all_league = lambda r: "all"  # noqa: E731
    specs = {
        "batsman": (lambda r: r.batter, batsman_chunk, all_league, lambda r: r.total_balls),
        "bowler": (lambda r: r.bowler, bowler_chunk, all_league, lambda r: r.total_balls),
        "team": (lambda r: r.team, team_chunk, lambda r: r.league, lambda r: r.matches),
        "partnership": (lambda r: f"{r.player_a} & {r.player_b}", partnership_chunk, all_league,
                        lambda r: r.total_balls),
        "fielding": (lambda r: r.fielder, fielding_chunk, all_league, lambda r: r.matches_played),
        "matchup": (lambda r: f"{r.batter} vs {r.bowler}", matchup_chunk, all_league,
                    lambda r: r.balls_faced),
    }
    chunks = []
    for etype, (name_fn, chunk_fn, league_fn, weight_fn) in specs.items():
        path = d / f"{etype}_features.parquet"
        if not path.exists():
            logger.warning(f"Skipping {etype}: {path} not found")
            continue
        df = pd.read_parquet(path)
        if etype == "matchup" and min_matchup_balls:
            before = len(df)
            df = df[df["balls_faced"] >= min_matchup_balls]
            logger.info(f"matchup: {len(df):,} of {before:,} pairs have >= {min_matchup_balls} balls")
        chunks += _records(df, etype, name_fn, chunk_fn, league_fn, weight_fn)
    return chunks


# ── the Chroma collection ───────────────────────────────────────────────
def _client(persist_dir):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(processed_dir: Path | str | None = None, persist_dir: Path | str | None = None,
                embed_fn: Callable[[list[str]], np.ndarray] | None = None, reset: bool = True,
                min_matchup_balls: int | None = MIN_MATCHUP_BALLS, batch_size: int = 256) -> dict:
    """Chunk the feature tables, embed them and store them in ChromaDB.

    Returns the number of chunks per entity type.  ``reset`` rebuilds from scratch (no duplicates).
    Embedding ~23,000 chunks takes about 10 minutes with the default model.
    """
    persist_dir = persist_dir or EMBEDDINGS_DIR
    embed_fn = embed_fn or default_embed_fn()
    if embed_fn is None:
        raise RuntimeError("No embedding model available (pip install fastembed and check the "
                           "network for the first download)")
    chunks = build_chunks(processed_dir, min_matchup_balls)
    if not chunks:
        raise RuntimeError("No feature tables found; run `python -m src.pipeline` first")
    client = _client(persist_dir)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # first build: nothing to delete
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                          embedding_function=None)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                   metadatas=[c["metadata"] for c in batch],
                   embeddings=np.asarray(embed_fn([c["text"] for c in batch]), dtype=np.float32).tolist())
        if (i // batch_size) % 10 == 0:
            logger.info(f"Indexed {min(i + batch_size, len(chunks)):,} / {len(chunks):,} chunks")
    _NAME_CACHE.clear()
    counts = pd.Series([c["metadata"]["entity_type"] for c in chunks]).value_counts().to_dict()
    logger.success(f"Entity index ready at {persist_dir}: {counts}")
    return counts


def _collection(persist_dir):
    try:
        col = _client(persist_dir).get_collection(COLLECTION, embedding_function=None)
    except Exception as e:
        raise RuntimeError("The entity index has not been built: run "
                           "`python -m src.rag.entity_index`") from e
    if col.count() == 0:
        raise RuntimeError("The entity index is empty: run `python -m src.rag.entity_index`")
    return col


# ── spotting names in a question ────────────────────────────────────────
DOMAIN_WORDS = {
    "runs", "run", "wicket", "wickets", "over", "overs", "ball", "balls", "batting", "bowling",
    "fielding", "batter", "batsman", "bowler", "fielder", "strike", "rate", "economy", "average",
    "best", "worst", "most", "recent", "form", "death", "powerplay", "middle", "phase", "team",
    "player", "match", "matches", "innings", "record", "stats", "against", "versus", "partnership",
    "pair", "catches", "catch", "score", "scored", "league", "season", "career", "tell", "show",
}
# surnames that are also everyday words: accepted only when capitalised (or given a first name)
WORDLIKE_SURNAMES = {"head", "short", "hope", "green", "wood", "black", "white", "brown", "little",
                     "king", "young", "cook", "bell", "fox", "hunt", "rose", "field", "power", "bird",
                     "stone", "cross", "hall", "gold", "law", "pace", "spin"}
_NOT_SURNAMES = set(ENGLISH_STOP_WORDS) | DOMAIN_WORDS
_NAME_CACHE: dict = {}


def _index_names(col, key) -> dict:
    """Player names (with a popularity weight), team names, and a surname lookup, cached."""
    cached = _NAME_CACHE.get(key)
    if cached is not None and cached["count"] == col.count():
        return cached
    got = col.get(include=["metadatas"])
    players: dict[str, float] = {}
    teams: dict[str, set] = {}
    for m in got["metadatas"]:
        if m["entity_type"] in PLAYER_TYPES:
            players[m["entity_name"]] = max(players.get(m["entity_name"], 0.0), m["weight"])
        elif m["entity_type"] == "team":
            teams.setdefault(m["entity_name"], set()).add(m["league"])
    by_surname: dict[str, list[str]] = {}
    for name in players:
        parts = _norm(name).split()
        if parts:
            by_surname.setdefault(parts[-1], []).append(name)
    out = {"count": col.count(), "players": players, "teams": teams, "by_surname": by_surname}
    _NAME_CACHE[key] = out
    return out


def spot_names(question: str, players: dict[str, float], teams, by_surname=None) -> dict:
    """Players and teams a question names.

    Teams: the full name, or an upper-case abbreviation such as "CSK".  Players: a surname (or
    first-name / initials plus surname, which picks out one player).  A surname shared by several
    players returns all of them, most-played first, up to six.  "head to head" is ignored so it does
    not summon Travis Head.
    """
    q = re.sub(r"head[\s-]+to[\s-]+head", " ", question, flags=re.I)
    q = re.sub(r"['\u2019]s\b", "", q)  # possessives: "Kohli's strike rate" names Kohli
    found_teams = []
    qn = f" {_norm(q)} "
    for t in teams:
        if f" {_norm(t)} " in qn:
            found_teams.append(t)
    canon = {_norm(v): v for v in teams}
    for tok in re.findall(r"\b[A-Z]{2,5}\b", q):  # abbreviations must be written in capitals
        full = TEAM_ABBREVIATIONS.get(tok.lower())
        if full and full in teams and full not in found_teams:
            found_teams.append(full)
        elif full and _norm(full) in canon and canon[_norm(full)] not in found_teams:
            found_teams.append(canon[_norm(full)])

    if by_surname is None:
        by_surname = {}
        for name in players:
            parts = _norm(name).split()
            if parts:
                by_surname.setdefault(parts[-1], []).append(name)
    tokens = re.findall(r"[A-Za-z][A-Za-z'.\-]*", q)
    lows = [re.sub(r"[^a-z]", "", t.lower()) for t in tokens]

    def cands_for(i):
        low = lows[i]
        if len(low) < 3 or low in _NOT_SURNAMES or low not in by_surname:
            return None
        return by_surname[low]

    def given_evidence(i, cand):
        """Is the word before token i plausibly this candidate's first name or initials?"""
        prev = tokens[i - 1]
        initials_style = prev.isalpha() and prev.isupper() and len(prev) <= 3  # "AS", "MS", "RG"
        if not initials_style and (lows[i - 1] in _NOT_SURNAMES or not lows[i - 1]):
            return False  # "about Sharma": 'about' is not a first name (but "AS Khan" is)
        given = "".join(_norm(cand).split()[:-1])
        if given and given == lows[i - 1]:
            return True  # the exact first name, in any case ("rashid khan")
        # initials in the data ("V Kohli", "RG Sharma") vs a first name in the question: only
        # when it is written like a name, so a filler word cannot pick players by its first letter
        return prev[0].isupper() and score_name(f"{prev} {tokens[i]}", cand) >= 0.95

    found_players: list[str] = []
    ambiguous: list[str] = []  # surnames that still matched several players
    handled: set[int] = set()  # token positions already explained (surnames and their first names)

    def add(names):
        for n in names:
            if n not in found_players:
                found_players.append(n)

    # pass 1: "first name / initials + surname" pins down one player, and its first-name word is
    # then not mistaken for somebody else's surname ("Babar Azam" must not summon "Zulfiqar Babar")
    for i in range(1, len(tokens)):
        cands = cands_for(i)
        if not cands:
            continue
        with_given = [c for c in cands if given_evidence(i, c)]
        if with_given:
            add(with_given)
            handled |= {i, i - 1}

    # pass 2: a surname on its own
    for i in range(len(tokens)):
        cands = cands_for(i)
        if not cands or i in handled:
            continue
        if lows[i] in WORDLIKE_SURNAMES and not tokens[i][0].isupper():
            continue  # "head", "short"... in lower case is an ordinary word
        chosen = sorted(cands, key=lambda n: -players[n])[:MAX_NAME_CANDIDATES]
        # an obscure namesake must not hijack the obvious player ("Kohli" is V Kohli, not the
        # 19-ball T Kohli): keep only the leader if it has DOMINANCE x the next one's data
        if len(chosen) > 1 and players[chosen[0]] >= DOMINANCE * players[chosen[1]]:
            chosen = chosen[:1]
        if len(chosen) > 1:
            ambiguous.append(tokens[i])
        add(chosen)
    return {"players": found_players, "teams": found_teams, "ambiguous": ambiguous}


def _name_in(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text) is not None


# ── what kind of chunk is the question after? ───────────────────────────
_INTENT_PATTERNS = {
    "batsman": r"\b(bat|bats|batting|batter|batsman|strike rate|innings|highest score|"
               r"how many runs|runs scored|average)\b",
    "bowler": r"\b(bowl|bowls|bowling|bowler|economy|wickets?|dot balls?|conceded|overs bowled)\b",
    "fielding": r"\b(catch|catches|caught|fielding|fielder|stumping|stumpings|stumped|run[- ]?outs?|"
                r"wicket[- ]?keeper|keeper|dismissals)\b",
    "partnership": r"\b(partnership|partnerships|together|pair|pairs|opening stand|stand)\b",
}
_SUPERLATIVE = re.compile(r"\b(best|worst|most|least|highest|lowest|fastest|slowest|top|greatest|"
                          r"leading|biggest|smallest)\b", re.I)


def detect_intent(question: str, n_players: int = 0, n_teams: int = 0) -> set[str]:
    """Chunk types the wording points at ("economy" -> bowler, "catches" -> fielding, ...).

    A matchup is only suggested when two or more players are named, a team only when a team is.
    """
    q = question.lower()
    out = {t for t, pat in _INTENT_PATTERNS.items() if re.search(pat, q)}
    if n_players >= 2 or re.search(r"head[\s-]+to[\s-]+head", q):
        if re.search(r"\b(against|vs\.?|versus|facing|faced|head[\s-]+to[\s-]+head)\b", q):
            out.add("matchup")
    if n_teams or re.search(r"\b(team|side|franchise)\b", q):
        out.add("team")
    return out


# ── retrieval ───────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = 5, entity_type: str | None = None,
             persist_dir: Path | str | None = None,
             embed_fn: Callable[[list[str]], np.ndarray] | None = None,
             intent_boost: float = INTENT_BOOST) -> dict:
    """Chunks that answer a question.

    If the question names a player or team, chunks mentioning that name are selected first and
    ranked semantically among themselves; if nothing matches (or no name is found) it falls back to
    plain semantic search and says so.  ``entity_type`` restricts the chunk kind.  Chunk types the
    wording points at ("economy", "catches", ...) get ``intent_boost`` added to their similarity
    (set 0 to disable), and ``score`` is similarity plus that boost.
    """
    if entity_type is not None and entity_type not in ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
    top_k = max(1, int(top_k))
    persist_dir = persist_dir or EMBEDDINGS_DIR
    col = _collection(persist_dir)
    embed_fn = embed_fn or default_embed_fn()
    if embed_fn is None:
        raise RuntimeError("No embedding model available for the question")
    qv = np.asarray(embed_fn([question]), dtype=np.float32)[0].tolist()

    idx = _index_names(col, str(persist_dir))
    names = spot_names(question, idx["players"], idx["teams"], idx["by_surname"])
    wanted = names["players"] + names["teams"]

    intent = detect_intent(question, len(names["players"]), len(names["teams"]))

    def where_for(etype):
        if etype is None:
            return {"dataset": DATASET}
        return {"$and": [{"dataset": DATASET}, {"entity_type": etype}]}

    def run(where_document, n, etype):
        r = col.query(query_embeddings=[qv], n_results=n, where=where_for(etype),
                      where_document=where_document, include=["documents", "metadatas", "distances"])
        return [(i, d, m, dist) for i, d, m, dist in
                zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True)]

    # candidate pool: the plain top results, plus the top results *of each chunk type the wording
    # points at* -- otherwise a player's short numeric matchup / partnership chunks can crowd the
    # right chunk out of the pool before the intent hint gets a chance to promote it
    fetch_types = [entity_type] if entity_type else [None, *sorted(intent & set(ENTITY_TYPES))]

    def pooled(where_document, n, verify):
        seen, rows = set(), []
        for et in fetch_types:
            for row in run(where_document, n, et):
                if row[0] in seen or (verify and not any(_name_in(row[1], nm) for nm in wanted)):
                    continue
                seen.add(row[0])
                rows.append(row)
        return rows

    rows, filtered = [], False
    if wanted:
        doc_filter = ({"$contains": wanted[0]} if len(wanted) == 1
                      else {"$or": [{"$contains": n} for n in wanted]})
        # over-fetch, then keep only chunks that contain a name as a whole word ("S Khan" must
        # not match "AS Khan"); Chroma's filter is a plain substring test
        rows = pooled(doc_filter, top_k * 6, verify=True)
        filtered = bool(rows)
    if not rows:
        rows = pooled(None, top_k, verify=False)
    scored = [(1 - dist + (intent_boost if m["entity_type"] in intent else 0.0), 1 - dist, d, m)
              for _, d, m, dist in rows]
    scored.sort(key=lambda x: -x[0])
    hits = [{"text": d, "entity_type": m["entity_type"], "entity_name": m["entity_name"],
             "league": m["league"], "score": round(score, 4), "similarity": round(sim, 4),
             "metadata": m} for score, sim, d, m in scored[:top_k]]
    warning = None
    if not wanted and _SUPERLATIVE.search(question):
        warning = ("Similarity search finds text that resembles the question; it cannot rank by a "
                   "statistic. For 'best', 'most' or 'highest' questions use the ranking and "
                   "leaderboard tools: these hits are not ordered by the statistic.")
    return {"question": question, "entity_type": entity_type, "filtered_by": wanted,
            "named": {"players": names["players"], "teams": names["teams"]},
            "ambiguous_surnames": names["ambiguous"],
            "name_filtered": filtered, "intent": sorted(intent), "hits": hits,
            **({"warning": warning} if warning else {}),
            **({"note": "No chunk mentions the name(s) found in the question; showing plain "
                        "semantic matches."} if wanted and not filtered else {})}


if __name__ == "__main__":
    print(build_index())
    for q in ("How does Kohli bat in the death overs?", "Which team scores fastest in the IPL?"):
        r = retrieve(q, top_k=2)
        print(f"\n{q}  [filtered_by={r['filtered_by']}]")
        for h in r["hits"]:
            print(f"  ({h['score']:.2f}) {h['text'][:130]}...")
