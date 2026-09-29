"""Batsman feature engineering from ball-by-ball data."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR, settings


def compute_batsman_features(
    db_path: str | None = None,
    recent_n: int | None = None,
) -> pd.DataFrame:
    """Compute per-batsman features across all filtered matches.

    Returns a DataFrame with one row per batsman containing:
    - career stats (average, strike rate, boundary %, dot %)
    - recent form (last N innings)
    - phase splits (powerplay, middle, death)
    - vs pace / vs spin splits
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)
    n = recent_n or settings.recent_form_innings
    min_inn = settings.min_innings_threshold

    logger.info("Computing batsman features...")

    # ── career stats ───────────────────────────────────────────────
    career = con.execute("""
        WITH innings AS (
            SELECT
                b.batter,
                b.match_id,
                m.date,
                m.event_name AS league,
                SUM(b.runs_batter) AS runs,
                COUNT(*) AS balls,
                SUM(CASE WHEN b.runs_batter = 0 THEN 1 ELSE 0 END) AS dots,
                SUM(CASE WHEN b.runs_batter = 4 THEN 1 ELSE 0 END) AS fours,
                SUM(CASE WHEN b.runs_batter = 6 THEN 1 ELSE 0 END) AS sixes,
                MAX(CASE WHEN b.wicket_type IS NOT NULL
                         AND b.player_out = b.batter THEN 1 ELSE 0 END) AS dismissed
            FROM ball_by_ball b
            JOIN match_info m ON b.match_id = m.match_id
            GROUP BY b.batter, b.match_id, m.date, m.event_name
        )
        SELECT
            batter,
            COUNT(*) AS innings,
            SUM(runs) AS total_runs,
            SUM(balls) AS total_balls,
            SUM(dismissed) AS times_out,
            ROUND(SUM(runs) * 1.0 / NULLIF(SUM(dismissed), 0), 2) AS average,
            ROUND(SUM(runs) * 100.0 / NULLIF(SUM(balls), 0), 2) AS strike_rate,
            ROUND(SUM(dots) * 100.0 / NULLIF(SUM(balls), 0), 2) AS dot_pct,
            ROUND((SUM(fours) + SUM(sixes)) * 100.0 / NULLIF(SUM(balls), 0), 2) AS boundary_pct,
            ROUND(STDDEV_SAMP(runs), 2) AS consistency_std,
            MAX(runs) AS highest_score
        FROM innings
        GROUP BY batter
        HAVING COUNT(*) >= ?
    """, [min_inn]).df()

    # ── phase splits (powerplay 1-6, middle 7-15, death 16-20) ────
    phase = con.execute("""
        SELECT
            batter,
            CASE
                WHEN over_number BETWEEN 0 AND 5 THEN 'powerplay'
                WHEN over_number BETWEEN 6 AND 14 THEN 'middle'
                ELSE 'death'
            END AS phase,
            SUM(runs_batter) AS runs,
            COUNT(*) AS balls
        FROM ball_by_ball
        GROUP BY batter, phase
    """).df()

    phase_pivot = phase.pivot_table(
        index="batter",
        columns="phase",
        values=["runs", "balls"],
        fill_value=0,
    )
    phase_pivot.columns = [f"{col[0]}_{col[1]}" for col in phase_pivot.columns]
    phase_pivot = phase_pivot.reset_index()

    # Add strike rates per phase
    for p in ["powerplay", "middle", "death"]:
        runs_col = f"runs_{p}"
        balls_col = f"balls_{p}"
        if runs_col in phase_pivot.columns and balls_col in phase_pivot.columns:
            phase_pivot[f"sr_{p}"] = (
                phase_pivot[runs_col] * 100 / phase_pivot[balls_col].replace(0, pd.NA)
            ).round(2)

    # ── recent form (last N innings) ──────────────────────────────
    recent = con.execute("""
        WITH innings_ranked AS (
            SELECT
                b.batter,
                b.match_id,
                SUM(b.runs_batter) AS runs,
                COUNT(*) AS balls,
                m.date,
                ROW_NUMBER() OVER (
                    PARTITION BY b.batter ORDER BY m.date DESC, b.match_id DESC
                ) AS rn
            FROM ball_by_ball b
            JOIN match_info m ON b.match_id = m.match_id
            GROUP BY b.batter, b.match_id, m.date
        )
        SELECT
            batter,
            ROUND(AVG(runs), 2) AS recent_avg,
            ROUND(SUM(runs) * 100.0 / NULLIF(SUM(balls), 0), 2) AS recent_sr,
            COUNT(*) AS recent_innings
        FROM innings_ranked
        WHERE rn <= ?
        GROUP BY batter
    """, [n]).df()

    con.close()

    # ── merge everything ──────────────────────────────────────────
    df = career.merge(phase_pivot, on="batter", how="left")
    df = df.merge(recent, on="batter", how="left")

    # ── composite ranking score ───────────────────────────────────
    # Weighted combination — weights are tunable
    df["rank_score"] = (
        0.25 * df["strike_rate"].rank(pct=True, na_option="bottom")
        + 0.20 * df["average"].rank(pct=True, na_option="bottom")
        + 0.15 * (1 - df["consistency_std"].rank(pct=True, na_option="top"))
        + 0.15 * df["boundary_pct"].rank(pct=True, na_option="bottom")
        + 0.15 * df["recent_sr"].rank(pct=True, na_option="bottom")
        + 0.10 * df["recent_avg"].rank(pct=True, na_option="bottom")
    ).round(4)

    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df.index.name = "rank"
    df.index += 1

    logger.info(f"Batsman features computed for {len(df)} players")
    return df


def save_batsman_features(df: pd.DataFrame) -> None:
    """Write batsman features to Parquet."""
    dest = PROCESSED_DIR / "batsman_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=True)
    logger.info(f"Saved batsman features to {dest}")


if __name__ == "__main__":
    features = compute_batsman_features()
    save_batsman_features(features)
    print(features.head(20).to_string())
