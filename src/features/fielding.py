"""Fielding features: dismissal involvement per match played (no position data exists)."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_fielding_features(db_path: str | None = None) -> pd.DataFrame:
    """Per-fielder dismissal involvement, normalised by matches actually played.

    Cricsheet records who took a catch / stumping / run-out, but not field position, chances
    offered or drops.  So this measures *involvement in dismissals*, which reflects role
    (wicketkeepers, slip fielders, bowlers taking return catches) and opportunity at least as
    much as fielding quality.  Backtests found it persists year to year (Spearman ~0.55 for
    dismissals per match) but that a keeper flag explains much of that; there is no data to
    separate skill from role, so no "fielding score" is produced.

    Matches played come from the team sheets.  (An earlier version divided by matches in which
    the fielder had a dismissal, which overstated the rate about 3x.)
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    logger.info("Computing fielding features...")

    df = con.execute("""
        WITH credits AS (
            -- one row per fielder credited; run-outs can credit two fielders
            SELECT match_id, over_number, wicket_type, fielder FROM ball_by_ball
            WHERE fielder IS NOT NULL
              AND wicket_type IN ('caught', 'caught and bowled', 'run out', 'stumped')
            UNION ALL
            SELECT match_id, over_number, wicket_type, fielder_2 FROM ball_by_ball
            WHERE fielder_2 IS NOT NULL AND wicket_type = 'run out'
        ),
        played AS (
            SELECT person_name AS fielder, COUNT(DISTINCT match_id) AS matches_played
            FROM match_players WHERE role = 'player' GROUP BY person_name
        ),
        agg AS (
            SELECT
                fielder,
                COUNT(*) AS total_dismissals,
                SUM(CASE WHEN wicket_type = 'caught' THEN 1 ELSE 0 END) AS catches,
                SUM(CASE WHEN wicket_type = 'caught and bowled' THEN 1 ELSE 0 END)
                    AS caught_and_bowled,
                SUM(CASE WHEN wicket_type = 'run out' THEN 1 ELSE 0 END) AS run_outs,
                SUM(CASE WHEN wicket_type = 'stumped' THEN 1 ELSE 0 END) AS stumpings,
                SUM(CASE WHEN over_number BETWEEN 0 AND 5 THEN 1 ELSE 0 END)
                    AS dismissals_powerplay,
                SUM(CASE WHEN over_number BETWEEN 6 AND 14 THEN 1 ELSE 0 END)
                    AS dismissals_middle,
                SUM(CASE WHEN over_number >= 15 THEN 1 ELSE 0 END) AS dismissals_death,
                COUNT(DISTINCT match_id) AS matches_with_dismissal
            FROM credits
            GROUP BY fielder
            HAVING COUNT(*) >= 3
        )
        SELECT
            agg.*,
            -- never fewer than the matches we can see a dismissal in
            GREATEST(COALESCE(played.matches_played, 0), agg.matches_with_dismissal)
                AS matches_played
        FROM agg LEFT JOIN played USING (fielder)
        ORDER BY total_dismissals DESC
    """).df()

    con.close()

    if not df.empty:
        n = df["matches_played"]
        df["dismissals_per_match"] = (df["total_dismissals"] / n).round(3)
        df["catches_per_match"] = ((df["catches"] + df["caught_and_bowled"]) / n).round(3)
        df["run_outs_per_match"] = (df["run_outs"] / n).round(3)
        df["is_keeper"] = df["stumpings"] > 0  # anyone credited with a stumping
        df = df.drop(columns="matches_with_dismissal")
        df = df.sort_values("dismissals_per_match", ascending=False).reset_index(drop=True)

    logger.info(f"Fielding features computed for {len(df)} fielders")
    return df


def save_fielding_features(df: pd.DataFrame) -> None:
    dest = PROCESSED_DIR / "fielding_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved fielding features to {dest}")


if __name__ == "__main__":
    features = compute_fielding_features()
    save_fielding_features(features)
    print(features.head(20).to_string())
