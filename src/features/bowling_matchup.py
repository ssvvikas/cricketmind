"""Bowler-vs-batsman matchup analysis for bowling strategy."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_matchup_features(db_path: str | None = None) -> pd.DataFrame:
    """Compute bowler-vs-batsman matchup stats.

    For every bowler-batsman pair that has faced at least 6 balls,
    compute runs, strike rate, dismissals, and a vulnerability score.
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    logger.info("Computing bowler-batsman matchup features...")

    df = con.execute("""
        SELECT
            b.bowler,
            b.batter,
            COUNT(*) AS balls_faced,
            SUM(b.runs_batter) AS runs_scored,
            ROUND(SUM(b.runs_batter) * 100.0 / COUNT(*), 2) AS strike_rate,
            SUM(CASE WHEN b.runs_batter = 0 THEN 1 ELSE 0 END) AS dots,
            ROUND(SUM(CASE WHEN b.runs_batter = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                AS dot_pct,
            SUM(CASE WHEN b.runs_batter >= 4 THEN 1 ELSE 0 END) AS boundaries,
            SUM(CASE WHEN b.is_bowler_wicket AND b.player_out = b.batter
                 THEN 1 ELSE 0 END) AS times_dismissed,
            COUNT(DISTINCT b.match_id) AS encounters
        FROM ball_by_ball b
        WHERE b.is_faced
        GROUP BY b.bowler, b.batter
        HAVING COUNT(*) >= 6
        ORDER BY times_dismissed DESC, dot_pct DESC
    """).df()

    con.close()

    if not df.empty:
        # Vulnerability score: how much trouble this bowler causes this batsman
        # Higher = batsman struggles more against this bowler
        df["vulnerability_score"] = (
            0.30 * df["dot_pct"].rank(pct=True, na_option="bottom")
            + 0.30 * (1 - df["strike_rate"].rank(pct=True, na_option="top"))
            + 0.25 * df["times_dismissed"].rank(pct=True, na_option="bottom")
            + 0.15 * (1 - df["boundaries"].rank(pct=True, na_option="top"))
        ).round(4)

    logger.info(f"Matchup features computed for {len(df)} bowler-batsman pairs")
    return df


def get_batsman_weaknesses(
    batter: str, db_path: str | None = None, top_n: int = 5
) -> pd.DataFrame:
    """Find the bowlers who are most dangerous to a specific batsman."""
    matchups = compute_matchup_features(db_path)
    return (
        matchups[matchups["batter"] == batter]
        .sort_values("vulnerability_score", ascending=False)
        .head(top_n)
    )


def recommend_bowler(
    target_batter: str,
    available_bowlers: list[str],
    db_path: str | None = None,
) -> pd.DataFrame:
    """Given a target batsman and a list of available bowlers,
    rank bowlers by how effective they are against that batsman.
    """
    matchups = compute_matchup_features(db_path)
    relevant = matchups[
        (matchups["batter"] == target_batter)
        & (matchups["bowler"].isin(available_bowlers))
    ].sort_values("vulnerability_score", ascending=False)
    return relevant


def save_matchup_features(df: pd.DataFrame) -> None:
    dest = PROCESSED_DIR / "matchup_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved matchup features to {dest}")


if __name__ == "__main__":
    features = compute_matchup_features()
    save_matchup_features(features)
    print(features.head(20).to_string())
