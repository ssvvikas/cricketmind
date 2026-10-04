"""Era- and phase-adjusted player ratings, built to be backtested.

Every ball is compared with the *par* for its (pool, year, phase): a powerplay ball in 2025
is not worth the same as a death-over ball in 2012.  A player's rating is the shrunk,
optionally recency-weighted average of their runs / outs (batters) or runs / wickets
(bowlers) above that par.

All functions take an ``as_of`` date and use only balls strictly before it, so the same code
serves live rankings and leak-free backtests.  Ratings are kept in three pools - club leagues,
internationals between ICC Full Members ("intl") and internationals involving an associate
("assoc") - because players look elite against weak opposition.
"""

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from src.config import DB_PATH, FULL_MEMBERS

PHASES = {0: "powerplay", 1: "middle", 2: "death"}
PHASE_OVERS = {0: 6.0, 1: 9.0, 2: 5.0}
BAT_K = 150.0  # shrinkage, in balls faced
BOWL_K = 150.0  # shrinkage, in legal balls bowled
DEFAULT_HALFLIFE_YEARS = None  # None = no recency weighting


def load_balls(db_path: str | None = None) -> pd.DataFrame:
    """Ball-level table with pool, year and phase for every delivery."""
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT match_id, date, league, batting_team, bowling_team, innings, over_number,
               batter, non_striker, bowler, runs_batter, runs_extras, bowler_runs,
               is_faced, is_legal, is_bowler_wicket,
               (is_wicket AND player_out = batter) AS batter_out
        FROM ball_by_ball
    """).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    both_full = df["batting_team"].isin(FULL_MEMBERS) & df["bowling_team"].isin(FULL_MEMBERS)
    df["pool"] = np.where(df["league"] != "T20I", "club", np.where(both_full, "intl", "assoc"))
    df["phase"] = np.select(
        [df["over_number"] <= 5, df["over_number"] <= 14], [0, 1], default=2
    ).astype("int8")
    df["days"] = (df["date"] - pd.Timestamp("2000-01-01")).dt.days.astype("int32")
    # a bowler is only credited a wicket the batter actually lost to them
    df["bowler_wkt"] = df["is_bowler_wicket"] & df["batter_out"]
    for c in ("is_faced", "is_legal", "bowler_wkt", "batter_out"):
        df[c] = df[c].astype(bool)
    logger.info(f"Loaded {len(df):,} balls")
    return df


def compute_par(balls: pd.DataFrame) -> pd.DataFrame:
    """Par rates per (pool, year, phase) from the balls given."""
    faced = balls[balls["is_faced"]].groupby(["pool", "year", "phase"]).agg(
        bat_rpb=("runs_batter", "mean"), out_rate=("batter_out", "mean"))
    legal = balls[balls["is_legal"]].groupby(["pool", "year", "phase"]).agg(
        bowl_rpb=("bowler_runs", "mean"), wkt_rate=("bowler_wkt", "mean"))
    return faced.join(legal)


def _with_par(balls: pd.DataFrame, par: pd.DataFrame) -> pd.DataFrame:
    """Attach par columns; years beyond the par table fall back to the latest year per pool."""
    keys = ["pool", "year", "phase"]
    out = balls.merge(par.reset_index(), on=keys, how="left")
    if out["bat_rpb"].isna().any():
        latest = par.reset_index().sort_values("year").groupby(["pool", "phase"]).tail(1)
        latest = latest.drop(columns="year")
        missing = out["bat_rpb"].isna()
        fill = out.loc[missing, ["pool", "phase"]].merge(latest, on=["pool", "phase"], how="left")
        for c in ("bat_rpb", "out_rate", "bowl_rpb", "wkt_rate"):
            out.loc[missing, c] = fill[c].to_numpy()
    return out


def _weights(days: pd.Series, as_of: pd.Timestamp, halflife_years: float | None) -> np.ndarray:
    if halflife_years is None:
        return np.ones(len(days))
    age = (as_of - pd.Timestamp("2000-01-01")).days - days.to_numpy()
    return 0.5 ** (age / (365.25 * halflife_years))


def batter_ratings(
    balls: pd.DataFrame, as_of: str | pd.Timestamp, halflife_years: float | None = None,
    k: float = BAT_K, pool: str | None = None,
) -> pd.DataFrame:
    """Shrunk above-par batting rate per batter using balls before ``as_of``.

    Columns: runs_per_ball / out_per_ball above par (shrunk), balls, innings, and
    ``impact`` = expected runs above par per innings (rate x typical balls per innings).
    """
    as_of = pd.Timestamp(as_of)
    hist = balls[(balls["date"] < as_of) & balls["is_faced"]]
    if pool:
        hist = hist[hist["pool"] == pool]
    par = compute_par(balls[balls["date"] < as_of])
    h = _with_par(hist, par)
    w = _weights(h["days"], as_of, halflife_years)
    h = h.assign(
        w=w,
        wruns=w * (h["runs_batter"] - h["bat_rpb"]),
        wout=w * (h["batter_out"].astype(float) - h["out_rate"]),
        wballs=w,
    )
    g = h.groupby(["pool", "batter"]).agg(
        wruns=("wruns", "sum"), wout=("wout", "sum"), wballs=("wballs", "sum"),
        balls=("w", "size"), innings=("match_id", "nunique"),
        runs=("runs_batter", "sum"), outs=("batter_out", "sum"),
    )
    g["runs_per_ball"] = g["wruns"] / (g["wballs"] + k)  # shrunk toward par (0)
    g["out_per_ball"] = g["wout"] / (g["wballs"] + k)
    g["balls_per_innings"] = g["balls"] / g["innings"]
    g["impact"] = g["runs_per_ball"] * g["balls_per_innings"]
    return g.reset_index().drop(columns=["wruns", "wout", "wballs"])


def bowler_ratings(
    balls: pd.DataFrame, as_of: str | pd.Timestamp, halflife_years: float | None = None,
    k: float = BOWL_K, pool: str | None = None,
) -> pd.DataFrame:
    """Shrunk above-par bowling rate per bowler (positive = runs saved / extra wickets)."""
    as_of = pd.Timestamp(as_of)
    hist = balls[(balls["date"] < as_of) & balls["is_legal"]]
    if pool:
        hist = hist[hist["pool"] == pool]
    par = compute_par(balls[balls["date"] < as_of])
    h = _with_par(hist, par)
    w = _weights(h["days"], as_of, halflife_years)
    h = h.assign(
        w=w,
        wsaved=w * (h["bowl_rpb"] - h["bowler_runs"]),
        wwkt=w * (h["bowler_wkt"].astype(float) - h["wkt_rate"]),
    )
    g = h.groupby(["pool", "bowler"]).agg(
        wsaved=("wsaved", "sum"), wwkt=("wwkt", "sum"), wballs=("w", "sum"),
        balls=("w", "size"), matches=("match_id", "nunique"),
        runs=("bowler_runs", "sum"), wkts=("bowler_wkt", "sum"),
    )
    g["saved_per_ball"] = g["wsaved"] / (g["wballs"] + k)
    g["wkt_per_ball"] = g["wwkt"] / (g["wballs"] + k)
    g["balls_per_match"] = g["balls"] / g["matches"]
    g["impact"] = g["saved_per_ball"] * g["balls_per_match"]
    g["economy"] = g["runs"] * 6.0 / g["balls"]
    return g.reset_index().drop(columns=["wsaved", "wwkt", "wballs"])


# ── rating book: rankings + matchup expectations ─────────────────────────
HALFLIFE_YEARS = 2.0  # backtested: 2-3 years beat no recency weighting (see README)
MIN_RANK_BALLS = 300


class RatingBook:
    """All ratings as of a date, plus rankings and matchup expectations.

    The matchup model is deliberately **additive** (par + batter rating - bowler rating,
    by phase).  Backtests showed a pair's own head-to-head record adds nothing beyond
    that (small samples are noise, and raw head-to-head does worse than par), so it is
    not used.
    """

    def __init__(self, balls: pd.DataFrame, as_of: str | pd.Timestamp | None = None,
                 halflife_years: float | None = HALFLIFE_YEARS) -> None:
        self.as_of = (
            pd.Timestamp(as_of) if as_of is not None
            else balls["date"].max() + pd.Timedelta(days=1)
        )
        hist = balls[balls["date"] < self.as_of]
        self.par = compute_par(hist)
        self.batters = batter_ratings(balls, self.as_of, halflife_years).set_index(["pool", "batter"])
        self.bowlers = bowler_ratings(balls, self.as_of, halflife_years).set_index(["pool", "bowler"])

    # ── rankings ───────────────────────────────────────────────────
    def rank_batters(self, pool: str, min_balls: int = MIN_RANK_BALLS, top: int | None = None):
        """Batters by runs above par per innings (phase- and era-adjusted, shrunk)."""
        d = self.batters.xs(pool, level="pool")
        d = d[d["balls"] >= min_balls].copy()
        d["runs_above_par_per_100_balls"] = d["runs_per_ball"] * 100
        d["strike_rate"] = d["runs"] * 100 / d["balls"]
        d["average"] = d["runs"] / d["outs"].replace(0, np.nan)
        d = d.sort_values("impact", ascending=False)
        d.insert(0, "rank", range(1, len(d) + 1))
        cols = ["rank", "balls", "innings", "strike_rate", "average",
                "runs_above_par_per_100_balls", "impact"]
        d = d[cols].round(3)
        return d.head(top) if top else d

    def rank_bowlers(self, pool: str, min_balls: int = MIN_RANK_BALLS, top: int | None = None):
        """Bowlers by runs saved per match.  Wicket-taking is deliberately not in the rank:
        it barely persists from one year to the next (Spearman ~0.05-0.10)."""
        d = self.bowlers.xs(pool, level="pool")
        d = d[d["balls"] >= min_balls].copy()
        d["runs_saved_per_over"] = d["saved_per_ball"] * 6
        d = d.sort_values("impact", ascending=False)
        d.insert(0, "rank", range(1, len(d) + 1))
        d = d[["rank", "balls", "matches", "economy", "runs_saved_per_over", "impact"]].round(3)
        return d.head(top) if top else d

    # ── matchups ───────────────────────────────────────────────────
    def _par_row(self, pool: str, phase: int) -> pd.Series | None:
        try:
            rows = self.par.xs((pool, phase), level=("pool", "phase")).sort_index()
        except KeyError:
            return None
        return rows.iloc[-1]  # latest year

    def expected(self, batter: str, bowler: str, pool: str, phase: str | int | None = None) -> dict:
        """Expected outcome per ball for this batter against this bowler.

        ``phase`` is 'powerplay' / 'middle' / 'death' (or 0/1/2); None averages the three.
        """
        names = {v: k for k, v in PHASES.items()}
        # "all phases" averages the phases weighted by their overs (6 / 9 / 5)
        phases = {names.get(phase, phase): 1.0} if phase is not None else PHASE_OVERS
        b = self.batters.loc[(pool, batter)] if (pool, batter) in self.batters.index else None
        w = self.bowlers.loc[(pool, bowler)] if (pool, bowler) in self.bowlers.index else None
        runs, out, weights = [], [], []
        for ph, weight in phases.items():
            par = self._par_row(pool, ph)
            if par is None:
                continue
            runs.append(par["bat_rpb"] + (b["runs_per_ball"] if b is not None else 0)
                        - (w["saved_per_ball"] if w is not None else 0))
            out.append(par["out_rate"] + (b["out_per_ball"] if b is not None else 0)
                       + (w["wkt_per_ball"] if w is not None else 0))
            weights.append(weight)
        if not weights:
            raise KeyError(f"no par data for pool={pool!r}, phase={phase!r}")
        e_runs = float(np.average(runs, weights=weights))
        e_out = float(np.clip(np.average(out, weights=weights), 0.0, 1.0))
        return {
            "batter": batter, "bowler": bowler, "pool": pool,
            "phase": phase or "all",
            "expected_runs_per_over": round(e_runs * 6, 2),
            "dismissal_prob_per_ball": round(e_out, 4),
            "dismissal_prob_per_over": round(1 - (1 - e_out) ** 6, 4),
            "batter_balls": int(b["balls"]) if b is not None else 0,
            "bowler_balls": int(w["balls"]) if w is not None else 0,
        }

    def recommend_bowlers(self, batter: str, bowlers: list[str], pool: str,
                          phase: str | int | None = None) -> pd.DataFrame:
        """Rank available bowlers against a batter: fewest expected runs per over first."""
        rows = [self.expected(batter, bw, pool, phase) for bw in bowlers]
        df = pd.DataFrame(rows).sort_values(
            ["expected_runs_per_over", "dismissal_prob_per_ball"], ascending=[True, False])
        df = df.reset_index(drop=True)
        # the gap is computed here so the assistant can quote it rather than subtract itself
        df["extra_runs_per_over_vs_best"] = (
            df["expected_runs_per_over"] - df["expected_runs_per_over"].iloc[0]).round(2)
        return df

    def pair_expectation(self, a: str, b: str, pool: str, phase: str | int | None = None) -> dict:
        """Expected batting output while two batters are together.

        Deliberately the average of the two individual ratings on top of par: backtests found
        a pair's own history adds nothing (and hurts), so there is no 'chemistry' term.
        """
        ea = self.expected(a, "__none__", pool, phase)
        eb = self.expected(b, "__none__", pool, phase)
        return {
            "a": a, "b": b, "pool": pool, "phase": phase or "all",
            "expected_runs_per_over": round((ea["expected_runs_per_over"]
                                             + eb["expected_runs_per_over"]) / 2, 2),
            "a_balls": ea["batter_balls"], "b_balls": eb["batter_balls"],
        }

    def best_pairs(self, batters: list[str], pool: str, phase: str | int | None = None,
                   top: int | None = None) -> pd.DataFrame:
        """Rank every pair from a squad by expected runs per over together."""
        from itertools import combinations

        rows = [self.pair_expectation(a, b, pool, phase) for a, b in combinations(batters, 2)]
        df = pd.DataFrame(rows).sort_values("expected_runs_per_over", ascending=False)
        df = df.reset_index(drop=True)
        df["runs_per_over_below_best_pair"] = (
            df["expected_runs_per_over"].iloc[0] - df["expected_runs_per_over"]).round(2)
        return df.head(top) if top else df

    def save(self, out_dir=None) -> None:
        from src.config import PROCESSED_DIR

        out = out_dir or PROCESSED_DIR
        self.batters.reset_index().to_parquet(f"{out}/player_ratings_batters.parquet", index=False)
        self.bowlers.reset_index().to_parquet(f"{out}/player_ratings_bowlers.parquet", index=False)
        logger.info(f"Saved player ratings (as of {self.as_of.date()}) to {out}")


if __name__ == "__main__":
    book = RatingBook(load_balls())
    book.save()
    for pool in ("club", "intl"):
        print(f"\nTOP BATTERS ({pool})\n", book.rank_batters(pool, top=10).to_string())
        print(f"\nTOP BOWLERS ({pool})\n", book.rank_bowlers(pool, top=10).to_string())
