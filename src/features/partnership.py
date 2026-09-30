"""Partnership feature engineering — batting pair analysis."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_partnership_features(db_path: str | None = None) -> pd.DataFrame:
    """Descriptive per-partnership records: runs together, run rate, best stand.

    There is deliberately no "chemistry score": backtests showed a pair's own history predicts
    its future scoring far worse than the two batters' individual ratings (Spearman 0.03 vs 0.22
    in club cricket), and adding pair history to those ratings made predictions worse.  For a
    forecast use ``RatingBook.pair_expectation``.
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    logger.info("Computing partnership features...")

    df = con.execute("""
        WITH partnerships AS (
            -- Each ball has a batter and a non-striker (the current pair)
            SELECT
                match_id,
                date,
                league,
                LEAST(batter, non_striker) AS player_a,
                GREATEST(batter, non_striker) AS player_b,
                runs_batter,
                runs_extras,
                innings,
                is_legal,
                CASE WHEN is_wicket THEN 1 ELSE 0 END AS wicket_fell
            FROM ball_by_ball
        ),
        pair_innings AS (
            -- Aggregate per partnership per innings
            SELECT
                player_a,
                player_b,
                match_id,
                innings,
                date,
                league,
                SUM(runs_batter + runs_extras) AS partnership_runs,
                SUM(is_legal::INT) AS balls_together,
                SUM(CASE WHEN runs_batter >= 4 THEN 1 ELSE 0 END) AS boundaries,
                SUM(CASE WHEN runs_batter IN (1, 3) THEN 1 ELSE 0 END) AS rotations,
                MAX(wicket_fell) AS broken
            FROM partnerships
            GROUP BY player_a, player_b, match_id, innings, date, league
        )
        SELECT
            player_a,
            player_b,
            COUNT(*) AS times_paired,
            SUM(partnership_runs) AS total_runs,
            SUM(balls_together) AS total_balls,
            ROUND(AVG(partnership_runs), 2) AS avg_partnership_runs,
            ROUND(SUM(partnership_runs) * 6.0 / NULLIF(SUM(balls_together), 0), 2) AS run_rate,
            ROUND(SUM(boundaries) * 100.0 / NULLIF(SUM(balls_together), 0), 2) AS boundary_pct,
            ROUND(SUM(rotations) * 100.0 / NULLIF(SUM(balls_together), 0), 2) AS rotation_pct,
            MAX(partnership_runs) AS best_partnership,
            SUM(CASE WHEN partnership_runs >= 50 THEN 1 ELSE 0 END) AS fifty_plus_stands,
            SUM(CASE WHEN partnership_runs >= 100 THEN 1 ELSE 0 END) AS century_stands
        FROM pair_innings
        GROUP BY player_a, player_b
        HAVING COUNT(*) >= 3
        ORDER BY avg_partnership_runs DESC
    """).df()

    con.close()

    logger.info(f"Partnership features computed for {len(df)} pairs")
    return df


def save_partnership_features(df: pd.DataFrame) -> None:
    dest = PROCESSED_DIR / "partnership_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved partnership features to {dest}")


if __name__ == "__main__":
    features = compute_partnership_features()
    save_partnership_features(features)
    print(features.head(20).to_string())
