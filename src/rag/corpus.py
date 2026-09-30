"""Match summaries for retrieval, generated deterministically from the ball-by-ball data.

The data has no narrative text, so each document is built from structured facts (result, margin,
innings totals, top performers, centuries, five-fors, ...) plus derived *tags* such as
"last-ball finish" or "collapse".  Nothing is invented: every sentence comes from a computed
field, and the fields are stored beside the text so answers can cite them.

Tag rules (T20, 20-over matches only where balls matter):
  last-ball finish   chasing side won on the final legal ball
  last-over finish   chasing side won with 6 or fewer balls left
  close finish       won by <= 5 runs, or by <= 2 wickets, or a last-ball finish
  thumping win       won by >= 60 runs, or by >= 8 wickets with >= 36 balls left
  big chase          chasing side won a target of 190+
  high-scoring       an innings of 200+;   low-scoring: both innings <= 130
  collapse           a side bowled out for 100 or fewer
  century            a batter scored 100+;   five-wicket haul: a bowler took 5+
  tie / super over / no result / rain-affected (D/L) / awarded (no margin is derived)
"""

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR

CORPUS_PATH = PROCESSED_DIR / "rag" / "match_corpus.parquet"

# words a user might type for each tag; they are indexed in the text and used to expand queries
TAG_WORDS = {
    "last-ball finish": "last-ball finish last ball final ball thriller",
    "last-over finish": "last-over finish went to the last over",
    "close finish": "close finish nail-biter thriller tight game narrow win",
    "thumping win": "thumping win one-sided crushing comfortable easy win",
    "big chase": "big chase huge chase successful chase",
    "high-scoring": "high-scoring run-fest big total batting fireworks",
    "low-scoring": "low-scoring bowlers dominated defensive total",
    "collapse": "collapse batting collapse all out cheaply bowled out",
    "century": "century hundred ton",
    "five-wicket haul": "five-wicket haul five-for fifer",
    "tie": "tie tied match",
    "super over": "super over tie-breaker",
    "no result": "no result abandoned washed out",
    "rain-affected": "rain-affected rain Duckworth-Lewis D/L reduced overs",
    "awarded": "awarded result awarded match awarded forfeit not played to a finish",
}
TAGS = tuple(TAG_WORDS)

# the exact rule behind each tag, returned with search results so answers can state it
TAG_DEFINITIONS = {
    "last-ball finish": "the chasing side won on the final legal ball of a 20-over innings",
    "last-over finish": "the chasing side won with 6 or fewer balls remaining",
    "close finish": "won by 5 runs or fewer, or by 2 wickets or fewer, or on the last ball",
    "thumping win": "won by 60+ runs, or by 8+ wickets with 36+ balls to spare",
    "big chase": "the chasing side won a target of 190 or more",
    "high-scoring": "at least one innings of 200 or more",
    "low-scoring": "both innings totalled 130 or fewer (full 20-over matches)",
    "collapse": "a side was bowled out for 100 or fewer (full 20-over matches)",
    "century": "a batter scored 100 or more",
    "five-wicket haul": "a bowler took 5 or more wickets (run-outs excluded)",
    "tie": "the scores finished level",
    "super over": "a tied match with a recorded winner or eliminator",
    "no result": "no result was declared",
    "rain-affected": "the result was set by Duckworth-Lewis(-Stern)",
    "awarded": "the result was awarded without the match being played to a finish",
}


def _load(db_path: str | None):
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)
    mi = con.execute("SELECT * FROM match_info").df()
    bb = con.execute("""
        SELECT match_id, innings, batting_team, bowling_team, batter, bowler, runs_batter,
               runs_extras, bowler_runs, is_legal, is_faced, is_wicket, is_bowler_wicket,
               player_out
        FROM ball_by_ball""").df()
    con.close()
    return mi, bb


def _clean(x):
    """Nulls arrive from pandas as NaN, which is truthy in Python: turn them into None."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA else x


def _innings_table(bb: pd.DataFrame) -> pd.DataFrame:
    g = bb.groupby(["match_id", "innings", "batting_team"], sort=False)
    inn = g.agg(runs=("runs_batter", "sum"), extras=("runs_extras", "sum"),
                wkts=("is_wicket", "sum"), balls=("is_legal", "sum")).reset_index()
    inn["score"] = inn["runs"] + inn["extras"]
    return inn


def _top_batters(bb: pd.DataFrame) -> pd.DataFrame:
    faced = bb[bb["is_faced"]].groupby(["match_id", "innings", "batter"], sort=False).agg(
        runs=("runs_batter", "sum"), balls=("is_faced", "sum")).reset_index()
    outs = bb[bb["is_wicket"] & bb["player_out"].notna()][["match_id", "innings", "player_out"]]
    outs = outs.drop_duplicates().rename(columns={"player_out": "batter"}).assign(out=True)
    faced = faced.merge(outs, on=["match_id", "innings", "batter"], how="left")
    faced["out"] = faced["out"].fillna(False).astype(bool)
    return faced


def _bowlers(bb: pd.DataFrame) -> pd.DataFrame:
    b = bb[bb["is_legal"] | (bb["bowler_runs"] > 0)]
    return b.groupby(["match_id", "innings", "bowler"], sort=False).agg(
        wkts=("is_bowler_wicket", "sum"), runs=("bowler_runs", "sum"),
        balls=("is_legal", "sum")).reset_index()


def _fmt_bat(r) -> str:
    return f"{r['batter']} {int(r['runs'])}{'' if r['out'] else '*'} ({int(r['balls'])})"


def _fmt_bowl(r) -> str:
    return f"{r['bowler']} {int(r['wkts'])}/{int(r['runs'])}"


def build_corpus(db_path: str | None = None) -> pd.DataFrame:
    """One row per match: structured facts, tags and searchable text."""
    mi, bb = _load(db_path)
    inn = _innings_table(bb)
    bat = _top_batters(bb)
    bowl = _bowlers(bb)
    bat_by = {k: g for k, g in bat.groupby(["match_id", "innings"], sort=False)}
    bowl_by = {k: g for k, g in bowl.groupby(["match_id", "innings"], sort=False)}
    inn_by = {k: g for k, g in inn.groupby("match_id", sort=False)}

    rows = []
    for m in mi.itertuples():
        team1, team2, winner = m.team1, m.team2, _clean(m.winner)
        matchinn = inn_by.get(m.match_id)
        if matchinn is None:
            continue
        first = matchinn[matchinn["innings"] == 1]
        second = matchinn[matchinn["innings"] == 2]
        f = first.iloc[0] if len(first) else None
        s = second.iloc[0] if len(second) else None
        overs = getattr(m, "overs", 20) or 20
        full = overs == 20 and getattr(m, "method", None) not in ("D/L", "Awarded")
        outcome = _clean(getattr(m, "outcome", None))
        method = _clean(getattr(m, "method", None))
        elim = _clean(getattr(m, "eliminator", None))
        tie = outcome == "tie"

        # margin: the official figure when the source has one, else derived from the innings
        wr, ww = getattr(m, "winner_runs", None), getattr(m, "winner_wickets", None)
        awarded = method == "Awarded"  # not played to a finish: partial innings, no margin
        margin_type = margin = None
        if awarded:
            pass
        elif pd.notna(wr) and wr:
            margin_type, margin = "runs", int(wr)
        elif pd.notna(ww) and ww:
            margin_type, margin = "wickets", int(ww)
        elif winner and f is not None and s is not None and not tie:
            if winner == f["batting_team"] and f["score"] > s["score"]:
                margin_type, margin = "runs", int(f["score"] - s["score"])
            elif winner == s["batting_team"] and s["wkts"] < 10:
                margin_type, margin = "wickets", int(10 - s["wkts"])

        chase_win = bool(winner and s is not None and winner == s["batting_team"] and not tie
                         and not awarded)
        balls_left = int(overs * 6 - s["balls"]) if (chase_win and full and s is not None) else None
        target = int(f["score"]) + 1 if (chase_win and full and f is not None) else None

        # performers
        cent, five, fifties = [], [], 0
        perf = []
        for i in (1, 2):
            bi, bo = bat_by.get((m.match_id, i)), bowl_by.get((m.match_id, i))
            if bi is not None:
                cent += list(bi.loc[bi["runs"] >= 100, "batter"])
                fifties += int(((bi["runs"] >= 50) & (bi["runs"] < 100)).sum())
                perf += list(bi.loc[bi["runs"] >= 30, "batter"])
            if bo is not None:
                five += list(bo.loc[bo["wkts"] >= 5, "bowler"])
                perf += list(bo.loc[bo["wkts"] >= 2, "bowler"])
        potm = _clean(getattr(m, "player_of_match1", None))
        if isinstance(potm, str):
            perf.append(potm)

        top = {}
        for i in (1, 2):
            bi, bo = bat_by.get((m.match_id, i)), bowl_by.get((m.match_id, i))
            top[i] = (
                _fmt_bat(bi.sort_values(["runs", "balls"], ascending=[False, True]).iloc[0])
                if bi is not None and len(bi) else None,
                _fmt_bowl(bo.sort_values(["wkts", "runs"], ascending=[False, True]).iloc[0])
                if bo is not None and len(bo) else None,
            )

        # tags
        tags = []
        if chase_win and full and balls_left == 0:
            tags += ["last-ball finish"]
        if chase_win and full and balls_left is not None and balls_left <= 6:
            tags += ["last-over finish"]
        if (margin_type == "runs" and margin <= 5) or (margin_type == "wickets" and margin <= 2) \
                or "last-ball finish" in tags:
            tags += ["close finish"]
        if (margin_type == "runs" and margin >= 60) or (
                margin_type == "wickets" and margin >= 8 and balls_left is not None
                and balls_left >= 36):
            tags += ["thumping win"]
        if chase_win and target is not None and target >= 190:
            tags += ["big chase"]
        scores = [int(x["score"]) for x in (f, s) if x is not None]
        if scores and max(scores) >= 200:
            tags += ["high-scoring"]
        if len(scores) == 2 and max(scores) <= 130 and full:
            tags += ["low-scoring"]
        if full and any(x is not None and x["wkts"] >= 10 and x["score"] <= 100 for x in (f, s)):
            tags += ["collapse"]
        if cent:
            tags += ["century"]
        if five:
            tags += ["five-wicket haul"]
        if tie:
            tags += ["tie"]
            if winner or elim:
                tags += ["super over"]
        if outcome == "no result":
            tags += ["no result"]
        if awarded:
            tags += ["awarded"]
        if method == "D/L":
            tags += ["rain-affected"]

        # text
        loser = team2 if winner == team1 else team1
        if tie and winner:
            result = f"{team1} and {team2} tied; {winner} won the super over"
        elif tie:
            result = f"{team1} and {team2} tied"
        elif awarded and winner:
            result = f"{winner} were awarded the match against {loser}"
        elif winner and margin_type:
            result = f"{winner} beat {loser} by {margin} {margin_type[:-1] if margin == 1 else margin_type}"
        elif winner:
            result = f"{winner} beat {loser}"
        else:
            result = f"{team1} v {team2}: {outcome or 'no result'}"
        city = m.city if isinstance(getattr(m, "city", None), str) else None
        where = m.venue if (not city or city.lower() in m.venue.lower()) else f"{m.venue}, {city}"
        parts = [f"{m.league} {m.season}: {result} at {where} on {pd.Timestamp(m.date).date()}."]
        for label, x, i in (("batting first", f, 1), ("chasing", s, 2)):
            if x is not None:
                bt, bw = top[i]
                parts.append(f"{x['batting_team']} ({label}) {int(x['score'])}/{int(x['wkts'])}"
                             + (f" in {x['balls'] // 6}.{x['balls'] % 6} overs"
                                if x["balls"] < overs * 6 else "")
                             + (f", top scorer {bt}" if bt else "")
                             + (f", best bowler for the opposition {bw}" if bw else "") + ".")
        if target and chase_win:
            parts.append(f"The chase of {target} was completed with {balls_left} "
                         f"{'ball' if balls_left == 1 else 'balls'} to spare.")
        if isinstance(potm, str):
            parts.append(f"Player of the match: {potm}.")
        if cent:
            parts.append("Century: " + ", ".join(cent) + ".")
        if five:
            parts.append("Five-wicket haul: " + ", ".join(five) + ".")
        parts.append("Tags: " + "; ".join(TAG_WORDS[t] for t in tags) + "." if tags else "")
        text = " ".join(p for p in parts if p)

        rows.append({
            "match_id": int(m.match_id), "date": pd.Timestamp(m.date), "league": m.league,
            "season": m.season, "venue": m.venue,
            "city": m.city if isinstance(getattr(m, "city", None), str) else None,
            "team1": team1, "team2": team2, "winner": winner,
            "result": result, "margin_type": margin_type, "margin": margin,
            "first_team": f["batting_team"] if f is not None else None,
            "first_score": int(f["score"]) if f is not None else None,
            "first_wkts": int(f["wkts"]) if f is not None else None,
            "second_team": s["batting_team"] if s is not None else None,
            "second_score": int(s["score"]) if s is not None else None,
            "second_wkts": int(s["wkts"]) if s is not None else None,
            "second_balls": int(s["balls"]) if s is not None else None,
            "target": target, "balls_left": balls_left,
            "player_of_match": potm if isinstance(potm, str) else None,
            "centuries": cent, "five_wicket_hauls": five, "fifties": fifties,
            "notable_players": sorted(set(perf)), "tags": tags,
            "top_first": " | ".join(x for x in top[1] if x) or None,
            "top_second": " | ".join(x for x in top[2] if x) or None,
            "text": text,
        })
    df = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    logger.info(f"Match corpus: {len(df):,} matches, "
                f"{int((df['tags'].str.len() > 0).sum()):,} with at least one tag")
    return df


def save_corpus(df: pd.DataFrame, path=CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved match corpus to {path}")


if __name__ == "__main__":
    c = build_corpus()
    save_corpus(c)
    print(c["text"].iloc[-1])
    print(c["tags"].explode().value_counts().to_string())
