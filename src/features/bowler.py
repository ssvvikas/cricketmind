"""Bowler feature engineering from ball-by-ball data."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR, settings


def compute_bowler_features(db_path: str | None = None) -> pd.DataFrame:
    """Compute per-bowler features: economy, strike rate, phase splits, etc."""
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)
    min_inn = settings.min_innings_threshold

    logger.info("Computing bowler features...")

    career = con.execute("""
        WITH spells AS (
            SELECT
                bowler,
                match_id,
                SUM(is_legal::INT) AS balls,
                SUM(bowler_runs) AS runs_conceded,
                SUM(is_bowler_wicket::INT) AS wickets,
                SUM(CASE WHEN runs_batter = 0 AND runs_extras = 0 THEN 1 ELSE 0 END) AS dots,
                SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END) AS fours_conceded,
                SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END) AS sixes_conceded
            FROM ball_by_ball
            GROUP BY bowler, match_id
        )
        SELECT
            bowler,
            COUNT(*) AS matches,
            SUM(balls) AS total_balls,
            SUM(runs_conceded) AS total_runs_conceded,
            SUM(wickets) AS total_wickets,
            ROUND(SUM(runs_conceded) * 6.0 / NULLIF(SUM(balls), 0), 2) AS economy,
            ROUND(SUM(balls) * 1.0 / NULLIF(SUM(wickets), 0), 2) AS bowling_sr,
            ROUND(SUM(runs_conceded) * 1.0 / NULLIF(SUM(wickets), 0), 2) AS bowling_avg,
            ROUND(SUM(dots) * 100.0 / NULLIF(SUM(balls), 0), 2) AS dot_pct,
            ROUND((SUM(fours_conceded) + SUM(sixes_conceded)) * 100.0
                  / NULLIF(SUM(balls), 0), 2) AS boundary_pct_conceded
        FROM spells
        GROUP BY bowler
        HAVING COUNT(*) >= ?
    """, [min_inn]).df()

    # ── phase splits ──────────────────────────────────────────────
    phase = con.execute("""
        SELECT
            bowler,
            CASE
                WHEN over_number BETWEEN 0 AND 5 THEN 'powerplay'
                WHEN over_number BETWEEN 6 AND 14 THEN 'middle'
                ELSE 'death'
            END AS phase,
            SUM(is_legal::INT) AS balls,
            SUM(bowler_runs) AS runs_conceded,
            SUM(is_bowler_wicket::INT) AS wickets
        FROM ball_by_ball
        GROUP BY bowler, phase
    """).df()

    phase_pivot = phase.pivot_table(
        index="bowler",
        columns="phase",
        values=["balls", "runs_conceded", "wickets"],
        fill_value=0,
    )
    # always emit every phase/metric column, even if a phase has no data
    phase_pivot = phase_pivot.reindex(
        columns=pd.MultiIndex.from_product(
            [["balls", "runs_conceded", "wickets"], ["powerplay", "middle", "death"]]
        ),
        fill_value=0,
    )
    phase_pivot.columns = [f"{col[0]}_{col[1]}" for col in phase_pivot.columns]
    phase_pivot = phase_pivot.reset_index()

    for p in ["powerplay", "middle", "death"]:
        rc = f"runs_conceded_{p}"
        bl = f"balls_{p}"
        if rc in phase_pivot.columns and bl in phase_pivot.columns:
            phase_pivot[f"econ_{p}"] = (
                phase_pivot[rc] * 6 / phase_pivot[bl].replace(0, float("nan"))
            ).round(2)

    # ── recent form ───────────────────────────────────────────────
    n = settings.recent_form_innings
    recent = con.execute("""
        WITH spells_ranked AS (
            SELECT
                b.bowler, b.match_id,
                SUM(b.is_legal::INT) AS balls,
                SUM(b.bowler_runs) AS runs_conceded,
                SUM(b.is_bowler_wicket::INT) AS wickets,
                m.date,
                ROW_NUMBER() OVER (PARTITION BY b.bowler ORDER BY m.date DESC, b.match_id DESC) AS rn
            FROM ball_by_ball b
            JOIN match_info m ON b.match_id = m.match_id
            GROUP BY b.bowler, b.match_id, m.date
        )
        SELECT
            bowler,
            ROUND(SUM(runs_conceded) * 6.0 / NULLIF(SUM(balls), 0), 2) AS recent_econ,
            SUM(wickets) AS recent_wickets,
            COUNT(*) AS recent_matches
        FROM spells_ranked
        WHERE rn <= ?
        GROUP BY bowler
    """, [n]).df()

    con.close()

    # ── merge ─────────────────────────────────────────────────────
    df = career.merge(phase_pivot, on="bowler", how="left")
    df = df.merge(recent, on="bowler", how="left")

    df = df.sort_values("total_wickets", ascending=False).reset_index(drop=True)
    logger.info(f"Bowler features computed for {len(df)} bowlers")
    return df


def save_bowler_features(df: pd.DataFrame) -> None:
    """Write bowler features to Parquet."""
    dest = PROCESSED_DIR / "bowler_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved bowler features to {dest}")


if __name__ == "__main__":
    features = compute_bowler_features()
    save_bowler_features(features)
    print(features.head(20).to_string())
