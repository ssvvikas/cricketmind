"""Fielding feature engineering — catches, run-outs, efficiency by zone."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_fielding_features(db_path: str | None = None) -> pd.DataFrame:
    """Compute per-fielder features from dismissal data.

    Note: Cricsheet ball-by-ball data has fielder names on dismissals
    (caught, stumped, run out) but not field position.  We derive
    position proxies from dismissal type + bowling phase.
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    logger.info("Computing fielding features...")

    df = con.execute("""
        WITH dismissals AS (
            SELECT
                b.match_id,
                m.date,
                m.event_name AS league,
                b.fielder,
                b.wicket_type,
                b.bowler,
                b.batter,
                b.over_number,
                CASE
                    WHEN b.over_number BETWEEN 0 AND 5 THEN 'powerplay'
                    WHEN b.over_number BETWEEN 6 AND 14 THEN 'middle'
                    ELSE 'death'
                END AS phase,
                -- Proxy for fielding zone based on dismissal type
                CASE
                    WHEN b.wicket_type = 'caught' AND b.fielder = b.bowler THEN 'caught_and_bowled'
                    WHEN b.wicket_type = 'caught' AND b.over_number <= 5 THEN 'close_catching'
                    WHEN b.wicket_type = 'caught' AND b.over_number >= 16 THEN 'boundary_catching'
                    WHEN b.wicket_type = 'caught' THEN 'general_catching'
                    WHEN b.wicket_type = 'run out' THEN 'run_out'
                    WHEN b.wicket_type = 'stumped' THEN 'stumping'
                    ELSE 'other'
                END AS fielding_zone
            FROM ball_by_ball b
            JOIN match_info m ON b.match_id = m.match_id
            WHERE b.fielder IS NOT NULL
              AND b.wicket_type IN ('caught', 'run out', 'stumped')
        )
        SELECT
            fielder,
            COUNT(*) AS total_dismissals,
            SUM(CASE WHEN wicket_type = 'caught' THEN 1 ELSE 0 END) AS catches,
            SUM(CASE WHEN wicket_type = 'run out' THEN 1 ELSE 0 END) AS run_outs,
            SUM(CASE WHEN wicket_type = 'stumped' THEN 1 ELSE 0 END) AS stumpings,
            -- Zone breakdown
            SUM(CASE WHEN fielding_zone = 'close_catching' THEN 1 ELSE 0 END) AS close_catches,
            SUM(CASE WHEN fielding_zone = 'boundary_catching' THEN 1 ELSE 0 END) AS boundary_catches,
            SUM(CASE WHEN fielding_zone = 'general_catching' THEN 1 ELSE 0 END) AS general_catches,
            SUM(CASE WHEN fielding_zone = 'caught_and_bowled' THEN 1 ELSE 0 END) AS caught_and_bowled,
            -- Phase breakdown
            SUM(CASE WHEN phase = 'powerplay' THEN 1 ELSE 0 END) AS dismissals_powerplay,
            SUM(CASE WHEN phase = 'middle' THEN 1 ELSE 0 END) AS dismissals_middle,
            SUM(CASE WHEN phase = 'death' THEN 1 ELSE 0 END) AS dismissals_death,
            COUNT(DISTINCT match_id) AS matches_fielded
        FROM dismissals
        GROUP BY fielder
        HAVING COUNT(*) >= 3
        ORDER BY total_dismissals DESC
    """).df()

    con.close()

    if not df.empty:
        # Dismissals per match (fielding efficiency proxy)
        df["dismissals_per_match"] = (df["total_dismissals"] / df["matches_fielded"]).round(3)

        # Fielding score — weighted combination
        df["fielding_score"] = (
            0.30 * df["catches"].rank(pct=True, na_option="bottom")
            + 0.25 * df["run_outs"].rank(pct=True, na_option="bottom")
            + 0.20 * df["dismissals_per_match"].rank(pct=True, na_option="bottom")
            + 0.15 * df["boundary_catches"].rank(pct=True, na_option="bottom")
            + 0.10 * df["close_catches"].rank(pct=True, na_option="bottom")
        ).round(4)

        df = df.sort_values("fielding_score", ascending=False).reset_index(drop=True)

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
