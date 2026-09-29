"""Bowler-vs-batsman matchup analysis for bowling strategy."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_matchup_features(db_path: str | None = None) -> pd.DataFrame:
    """Compute bowler-vs-batsman matchup stats.

    For every bowler-batsman pair that has faced at least 6 balls,
    compute runs, strike rate and dismissals (descriptive only - see ``head_to_head``).
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

    logger.info(f"Matchup features computed for {len(df)} bowler-batsman pairs")
    return df


def head_to_head(batter: str, bowler: str, db_path: str | None = None) -> dict | None:
    """The raw head-to-head record between one batter and one bowler (a fact, not a forecast).

    Samples are small, and backtests showed a pair's own history predicts future balls *worse*
    than league par.  For predictions use ``RatingBook.expected`` in
    ``src.models.player_ratings``.
    """
    matchups = compute_matchup_features(db_path)
    row = matchups[(matchups["batter"] == batter) & (matchups["bowler"] == bowler)]
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    rec["small_sample"] = bool(rec["balls_faced"] < 30)
    return rec


def recommend_bowler(target_batter: str, available_bowlers: list[str], pool: str = "club",
                     phase: str | None = None, book=None):
    """Rank bowlers against a batter with the additive rating model (phase-aware)."""
    from src.models.player_ratings import RatingBook, load_balls

    book = book or RatingBook(load_balls())
    return book.recommend_bowlers(target_batter, available_bowlers, pool, phase)


def save_matchup_features(df: pd.DataFrame) -> None:
    dest = PROCESSED_DIR / "matchup_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved matchup features to {dest}")


if __name__ == "__main__":
    features = compute_matchup_features()
    save_matchup_features(features)
    print(features.head(20).to_string())
