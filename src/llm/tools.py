"""Read-only tools an LLM can call to answer cricket questions with real numbers.

Every number the assistant quotes should come from one of these functions, which wrap the
backtested models.  Each result is a JSON-serialisable dict with the data plus ``caveats``
(plain-language limits the assistant should pass on), so honesty travels with the numbers.

The tool layer is model-agnostic: ``TOOL_SPECS`` uses the common name / description /
input_schema shape, and ``Toolbox.call`` never raises (errors come back as ``{"error": ...}``)
so a model can recover from a bad argument.
"""

import difflib
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.config import DB_PATH, TEAM_ALIASES
from src.features.bowling_matchup import head_to_head as _head_to_head
from src.features.fielding import compute_fielding_features
from src.models.player_ratings import PHASES, RatingBook, load_balls
from src.models.team_dataset import LEAGUES
from src.models.team_strength import MODEL_DIR, predict_match as _predict_match

POOLS = ("club", "intl", "assoc")
POOL_HELP = (
    "club = IPL/BBL/CPL/PSL combined; intl = T20Is between two ICC Full Members; "
    "assoc = T20Is involving an associate nation (ratings are NOT comparable with intl)"
)
SMALL_SAMPLE_BALLS = 300
DOMINANCE = 4  # a name match with 4x the data beats look-alikes

TEAM_ABBREVIATIONS = {
    "csk": "Chennai Super Kings", "mi": "Mumbai Indians", "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders", "srh": "Sunrisers Hyderabad", "dc": "Delhi Capitals",
    "pbks": "Punjab Kings", "rr": "Rajasthan Royals", "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans", "aus": "Australia", "ind": "India", "eng": "England",
    "sa": "South Africa", "nz": "New Zealand", "pak": "Pakistan", "sl": "Sri Lanka",
    "ban": "Bangladesh", "wi": "West Indies", "zim": "Zimbabwe", "ire": "Ireland",
}


# ── name resolution (pure functions) ────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace(".", " ").replace("-", " ")).strip()


def score_name(query: str, candidate: str) -> float:
    """How well a user-typed name matches a Cricsheet name such as 'V Kohli' (0-1).

    Cricsheet abbreviates given names ('V Kohli', 'MS Dhoni'), so a surname match plus a
    compatible initial is a strong match, e.g. 'Virat Kohli' -> 'V Kohli'.
    """
    q, c = _norm(query).split(), _norm(candidate).split()
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q[-1] == c[-1]:  # same surname
        q_given, c_given = q[:-1], "".join(c[:-1])
        if not q_given:
            return 0.9  # surname only
        initials = "".join(t[0] for t in q_given)
        if c_given and (c_given == initials or initials.startswith(c_given)
                        or c_given.startswith(initials)):
            return 0.97
        if c_given and c_given[0] == initials[0]:
            return 0.85
        return 0.55  # same surname, different person
    return round(0.8 * difflib.SequenceMatcher(None, " ".join(q), " ".join(c)).ratio(), 3)


def resolve(query: str, candidates: dict[str, int], limit: int = 5) -> dict:
    """Rank candidate names for a query.  ``candidates`` maps name -> sample size (balls)."""
    scored = sorted(((score_name(query, n), b, n) for n, b in candidates.items()), reverse=True)
    top = [{"name": n, "match": round(s, 2), "balls": int(b)} for s, b, n in scored[:limit]
           if s >= 0.6]
    if not top:
        return {"query": query, "candidates": [], "ambiguous": False, "resolved": None}
    # several people matching about equally well (e.g. plain "Sharma") are ambiguous, unless one
    # has far more data than the rest (an obscure 19-ball "T Kohli" must not block "Kohli")
    tied = [c for c in top if top[0]["match"] - c["match"] < 0.05]
    leader = max(tied, key=lambda c: c["balls"])
    others = [c for c in tied if c is not leader]
    out = {"query": query, "candidates": top, "ambiguous": False, "resolved": leader["name"]}
    if others:
        if leader["balls"] >= DOMINANCE * max(c["balls"] for c in others):
            out["note"] = (f"Assumed {leader['name']}; also matches "
                           f"{', '.join(c['name'] for c in others)} (far less data).")
        else:
            out.update(ambiguous=True, resolved=None)
    return out


# ── the toolbox ─────────────────────────────────────────────────────────
class Toolbox:
    """Lazily loads ratings / models and exposes the tools."""

    def __init__(self, db_path: str | None = None, model_dir=None, book: RatingBook | None = None):
        self.db_path = db_path or str(DB_PATH)
        self.model_dir = model_dir or MODEL_DIR
        self._book = book
        self._fielding: pd.DataFrame | None = None
        self._teams: dict[str, set[str]] | None = None

    # -- lazy loaders
    @property
    def book(self) -> RatingBook:
        if self._book is None:
            self._book = RatingBook(load_balls(self.db_path))
        return self._book

    @property
    def fielding(self) -> pd.DataFrame:
        if self._fielding is None:
            self._fielding = compute_fielding_features(self.db_path)
        return self._fielding

    @property
    def teams(self) -> dict[str, set[str]]:
        if self._teams is None:
            con = duckdb.connect(self.db_path, read_only=True)
            rows = con.execute("""
                SELECT league, team FROM (
                    SELECT league, team1 AS team FROM match_info
                    UNION SELECT league, team2 FROM match_info)""").fetchall()
            con.close()
            self._teams = {}
            for league, team in rows:
                self._teams.setdefault(league, set()).add(team)
        return self._teams

    def _player_names(self) -> dict[str, int]:
        names: dict[str, int] = {}
        for idx, col in ((self.book.batters, "batter"), (self.book.bowlers, "bowler")):
            for (_, name), balls in idx["balls"].items():
                names[name] = max(names.get(name, 0), int(balls))
        return names

    # -- caveats
    @staticmethod
    def _sample_caveat(balls: int, what: str) -> list[str]:
        if balls < SMALL_SAMPLE_BALLS:
            return [f"Small sample: only {balls} {what}; the rating is shrunk toward average "
                    "and may be unreliable."]
        return []

    @staticmethod
    def _pool_caveats(pool: str) -> list[str]:
        if pool == "assoc":
            return ["Associate-nation pool: players face weaker opposition, so these ratings "
                    "are not comparable with club or full-member international ratings."]
        if pool == "club":
            return ["The club pool combines IPL, BBL, CPL and PSL; it cannot be split by league."]
        return []

    def _check_pool(self, pool: str) -> str:
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS} ({POOL_HELP})")
        return pool

    def _need_player(self, name: str, pool: str, kind: str) -> str:
        idx = self.book.batters if kind == "batter" else self.book.bowlers
        if (pool, name) in idx.index:
            return name
        r = resolve(name, self._player_names())
        if r["resolved"]:
            return r["resolved"]
        raise ValueError(
            f"Could not uniquely identify {kind} {name!r}. Candidates: "
            f"{[c['name'] for c in r['candidates']] or 'none'} - call find_player first.")

    # ── tools ───────────────────────────────────────────────────────
    def find_player(self, query: str) -> dict:
        """Resolve a name a user typed to the exact names used in the data."""
        return resolve(query, self._player_names())

    def find_team(self, query: str, league: str | None = None) -> dict:
        """Resolve a team name / abbreviation (e.g. 'RCB') to the canonical team name."""
        q = _norm(query)
        if q in TEAM_ABBREVIATIONS:
            q = _norm(TEAM_ABBREVIATIONS[q])
        old_to_new = {_norm(o): n for names in TEAM_ALIASES.values() for o, n in names.items()}
        q = _norm(old_to_new.get(q, q))
        pool = {(lg, t) for lg, ts in self.teams.items() for t in ts if league in (None, lg)}
        scored = sorted(((difflib.SequenceMatcher(None, q, _norm(t)).ratio()
                          + (0.3 if q and q in _norm(t) else 0), lg, t) for lg, t in pool),
                        reverse=True)[:5]
        cands = [{"team": t, "league": lg, "match": round(min(s, 1.0), 2)} for s, lg, t in scored
                 if s >= 0.6]
        return {"query": query, "candidates": cands,
                "resolved": cands[0] if cands and (len(cands) == 1 or
                                                   cands[0]["match"] - cands[1]["match"] >= 0.1)
                else None}

    def player_profile(self, name: str, pool: str) -> dict:
        """Batting and bowling ratings for one player in one pool, with rank and caveats."""
        self._check_pool(pool)
        out: dict[str, Any] = {"pool": pool, "caveats": self._pool_caveats(pool)}
        found, errors = False, []
        for kind, idx, rank_fn in (
            ("batter", self.book.batters, self.book.rank_batters),
            ("bowler", self.book.bowlers, self.book.rank_bowlers),
        ):
            try:
                n = self._need_player(name, pool, kind)
            except ValueError as e:
                errors.append(str(e))
                continue
            if (pool, n) not in idx.index:
                continue
            found = True
            row = idx.loc[(pool, n)]
            out["name"] = n
            ranked = rank_fn(pool, min_balls=SMALL_SAMPLE_BALLS)
            rank = int(ranked.loc[n, "rank"]) if n in ranked.index else None
            if kind == "batter":
                out["batting"] = {
                    "balls_faced": int(row["balls"]), "innings": int(row["innings"]),
                    "strike_rate": round(row["runs"] * 100 / row["balls"], 1),
                    "runs_above_par_per_100_balls": round(row["runs_per_ball"] * 100, 1),
                    "impact_runs_above_par_per_innings": round(row["impact"], 2),
                    "rank_in_pool": rank, "ranked_among": len(ranked),
                    "top_percent_of_pool": _top_percent(rank, len(ranked)),
                }
                out["caveats"] += self._sample_caveat(int(row["balls"]), "balls faced")
            else:
                out["bowling"] = {
                    "balls_bowled": int(row["balls"]), "matches": int(row["matches"]),
                    "economy": round(row["economy"], 2),
                    "runs_saved_per_over": round(row["saved_per_ball"] * 6, 2),
                    "rank_in_pool": rank, "ranked_among": len(ranked),
                    "top_percent_of_pool": _top_percent(rank, len(ranked)),
                }
                out["caveats"] += self._sample_caveat(int(row["balls"]), "balls bowled")
                out["caveats"].append("Bowlers are ranked on runs saved; wicket counts barely "
                                      "persist from year to year, so they are not used.")
        if not found:
            ambiguous = [e for e in errors if "uniquely" in e]
            raise ValueError(ambiguous[0] if ambiguous else
                             f"No batting or bowling record for {name!r} in pool {pool!r}")
        return out

    def rank_batters(self, pool: str, top: int = 10) -> dict:
        self._check_pool(pool)
        df = self.book.rank_batters(pool, top=min(int(top), 50))
        return {"pool": pool, "as_of": str(self.book.as_of.date()),
                "ranking_basis": "runs above par per innings (era- and phase-adjusted, "
                                 "shrunk, 2-year recency weighting)",
                "rows": df.reset_index().to_dict("records"),
                "caveats": self._pool_caveats(pool) + [
                    "Ratings predict next-year performance only moderately (rank correlation "
                    "about 0.3 club); the adjusted rating is no better than raw strike rate "
                    "for full-member internationals."]}

    def rank_bowlers(self, pool: str, top: int = 10) -> dict:
        self._check_pool(pool)
        df = self.book.rank_bowlers(pool, top=min(int(top), 50))
        return {"pool": pool, "as_of": str(self.book.as_of.date()),
                "ranking_basis": "runs saved per match (wickets deliberately excluded)",
                "rows": df.reset_index().to_dict("records"),
                "caveats": self._pool_caveats(pool) + [
                    "Wicket-taking barely persists year to year, so this ranks economy, not "
                    "wickets. Raw economy predicts about as well as this rating."]}

    def matchup(self, batter: str, bowler: str, pool: str, phase: str | None = None) -> dict:
        """Expected runs per over and dismissal chance for one batter against one bowler."""
        self._check_pool(pool)
        b = self._need_player(batter, pool, "batter")
        w = self._need_player(bowler, pool, "bowler")
        e = self.book.expected(b, w, pool, phase)
        e["caveats"] = self._pool_caveats(pool) + [
            "Built from each player's general rating plus the phase par. Their own head-to-head "
            "history is not used: backtests showed it adds no predictive value.",
            "Individual balls are very random; treat this as a tendency, not a prediction."]
        for who, balls, kind in ((b, e["batter_balls"], "batter"), (w, e["bowler_balls"], "bowler")):
            e["caveats"] += self._sample_caveat(balls, f"balls for {who}") if balls else [
                f"No {kind} history for {who} in this pool; assumed average."]
        return e

    def recommend_bowlers(self, batter: str, bowlers: list[str], pool: str,
                          phase: str | None = None) -> dict:
        self._check_pool(pool)
        b = self._need_player(batter, pool, "batter")
        ws = [self._need_player(w, pool, "bowler") for w in bowlers]
        df = self.book.recommend_bowlers(b, ws, pool, phase)
        return {"batter": b, "pool": pool, "phase": phase or "all",
                "ranking": df.to_dict("records"),
                "caveats": self._pool_caveats(pool) + [
                    "Ranked by expected runs per over (fewest first) from individual ratings; "
                    "head-to-head history is not used. Differences of a few tenths of a run "
                    "per over are within noise."]}

    def best_pairs(self, batters: list[str], pool: str, phase: str | None = None,
                   top: int = 5) -> dict:
        self._check_pool(pool)
        names = [self._need_player(n, pool, "batter") for n in batters]
        df = self.book.best_pairs(names, pool, phase, top=min(int(top), 20))
        return {"pool": pool, "phase": phase or "all", "pairs": df.to_dict("records"),
                "caveats": self._pool_caveats(pool) + [
                    "Pair strength is the average of the two batters' individual ratings. A "
                    "pair's own record showed no 'chemistry' effect in backtests."]}

    def head_to_head(self, batter: str, bowler: str) -> dict:
        """The raw historical record between two players (a fact, not a forecast)."""
        names = self._player_names()
        b = batter if batter in names else resolve(batter, names)["resolved"]
        w = bowler if bowler in names else resolve(bowler, names)["resolved"]
        unresolved = [q for q, n in ((batter, b), (bowler, w)) if not n]
        if unresolved:
            raise ValueError(f"Could not uniquely identify {unresolved}; call find_player first.")
        rec = _head_to_head(b, w, self.db_path)
        if rec is None:
            return {"found": False, "batter": b, "bowler": w,
                    "caveats": ["These two have not met for at least 6 balls in the data."]}
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
        return {"found": True, "record": rec, "caveats": [
            "Historical fact only: head-to-head records are small samples and did NOT predict "
            "future outcomes better than league averages."]}

    def fielding_profile(self, name: str) -> dict:
        df = self.fielding
        hit = df[df["fielder"] == name]
        if hit.empty:
            r = resolve(name, df.set_index("fielder")["matches_played"].to_dict())
            if not r["resolved"]:
                raise ValueError(f"No fielding record for {name!r}. Candidates: "
                                 f"{[c['name'] for c in r['candidates']]}")
            hit = df[df["fielder"] == r["resolved"]]
        row = hit.iloc[0].to_dict()
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        return {"record": row, "caveats": [
            "This counts involvement in dismissals per match played. It reflects role (keepers, "
            "slip fielders, bowlers taking return catches) and opportunity at least as much as "
            "fielding quality; the data has no positions, chances or drops."]}

    def predict_match(self, team_a: str, team_b: str, league: str,
                      venue: str | None = None) -> dict:
        """Probability that team_a beats team_b, with an honest read on reliability."""
        if league not in LEAGUES:
            raise ValueError(f"league must be one of {LEAGUES}")
        ta, tb = (self.find_team(t, league) for t in (team_a, team_b))
        for q, r in ((team_a, ta), (team_b, tb)):
            if not r["resolved"]:
                raise ValueError(f"Could not identify team {q!r} in {league}: {r['candidates']}")
        if not all((Path(self.model_dir) / f).exists()
                   for f in ("team_strength.json", "team_state.pkl")):
            raise ValueError("The team-strength model has not been trained: run "
                             "`python -m src.models.team_strength`.")
        p = _predict_match(ta["resolved"]["team"], tb["resolved"]["team"], league, venue,
                           model_dir=self.model_dir)
        p.pop("xi_a", None)
        p.pop("xi_b", None)
        caveats = ["Uses each team's latest known XI; injuries, toss and conditions are not "
                   "modelled."]
        if league == "T20I":
            caveats.append("Backtested on 2021-2026 internationals: about 70% of winners "
                           "called correctly, largely from established nations beating weaker "
                           "sides.")
        else:
            caveats.append("Club-league matches are close to coin flips: in backtests the "
                           "model called about 50-53% of winners, no better than chance. The "
                           "probability shown is deliberately pulled toward 50%.")
        p["caveats"] = caveats
        p["reliability"] = "moderate" if league == "T20I" else "very low (near coin flip)"
        return p

    # ── dispatch ────────────────────────────────────────────────────
    def call(self, name: str, arguments: dict | str | None = None) -> dict:
        """Run a tool by name.  Never raises: failures come back as {'error': ...}."""
        fn = getattr(self, name, None) if name in TOOL_NAMES else None
        if fn is None:
            return {"error": f"Unknown tool {name!r}. Available: {sorted(TOOL_NAMES)}"}
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            result = fn(**(arguments or {}))
            return json.loads(json.dumps(result, default=_json_default))
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except (ValueError, KeyError) as e:
            return {"error": str(e).strip("'\"")}
        except Exception as e:  # keep the assistant alive whatever happens
            return {"error": f"{type(e).__name__}: {e}"}


def _top_percent(rank: int | None, total: int) -> float | None:
    """Rank as 'top X%' of the ranked pool, so the assistant can quote it instead of computing."""
    return round(rank * 100 / total, 1) if rank and total else None


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return str(o)


_PHASE = {"type": "string", "enum": list(PHASES.values()),
          "description": "Innings phase; omit for the over-weighted average of all phases."}
_POOL = {"type": "string", "enum": list(POOLS), "description": POOL_HELP}


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required}}


TOOL_SPECS = [
    _spec("find_player",
          "Resolve a player name as a user typed it (e.g. 'Virat Kohli', 'Bumrah') to the exact "
          "name in the data. If 'ambiguous' is true, ask the user which player they mean.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("find_team",
          "Resolve a team name or abbreviation (e.g. 'RCB', 'Aus') to the canonical team name.",
          {"query": {"type": "string"},
           "league": {"type": "string", "enum": list(LEAGUES)}}, ["query"]),
    _spec("player_profile",
          "Batting and/or bowling rating, rank and sample size for one player in one pool.",
          {"name": {"type": "string"}, "pool": _POOL}, ["name", "pool"]),
    _spec("rank_batters", "Top batters in a pool by era- and phase-adjusted runs above par.",
          {"pool": _POOL, "top": {"type": "integer", "minimum": 1, "maximum": 50}}, ["pool"]),
    _spec("rank_bowlers",
          "Top bowlers in a pool by runs saved (wickets are deliberately not used).",
          {"pool": _POOL, "top": {"type": "integer", "minimum": 1, "maximum": 50}}, ["pool"]),
    _spec("matchup",
          "Expected runs per over and dismissal chance for one batter against one bowler.",
          {"batter": {"type": "string"}, "bowler": {"type": "string"}, "pool": _POOL,
           "phase": _PHASE}, ["batter", "bowler", "pool"]),
    _spec("recommend_bowlers",
          "Rank a list of available bowlers against a batter (fewest expected runs first).",
          {"batter": {"type": "string"},
           "bowlers": {"type": "array", "items": {"type": "string"}, "minItems": 2},
           "pool": _POOL, "phase": _PHASE}, ["batter", "bowlers", "pool"]),
    _spec("best_pairs", "Rank every pair from a list of batters by expected runs per over.",
          {"batters": {"type": "array", "items": {"type": "string"}, "minItems": 2},
           "pool": _POOL, "phase": _PHASE, "top": {"type": "integer", "minimum": 1}},
          ["batters", "pool"]),
    _spec("head_to_head",
          "The raw historical record between a batter and a bowler. A fact, not a forecast.",
          {"batter": {"type": "string"}, "bowler": {"type": "string"}}, ["batter", "bowler"]),
    _spec("fielding_profile",
          "Dismissals (catches, run-outs, stumpings) per match played for one fielder.",
          {"name": {"type": "string"}}, ["name"]),
    _spec("predict_match",
          "Probability that team_a beats team_b in a league (IPL/BBL/CPL/PSL/T20I). Club "
          "leagues are near coin flips; say so.",
          {"team_a": {"type": "string"}, "team_b": {"type": "string"},
           "league": {"type": "string", "enum": list(LEAGUES)}, "venue": {"type": "string"}},
          ["team_a", "team_b", "league"]),
]
TOOL_NAMES = {s["name"] for s in TOOL_SPECS}
