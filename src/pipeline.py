"""End-to-end Phase 1 pipeline: download → load → features."""

from loguru import logger

from src.features.batsman import compute_batsman_features, save_batsman_features
from src.features.bowler import compute_bowler_features, save_bowler_features
from src.features.bowling_matchup import compute_matchup_features, save_matchup_features
from src.features.fielding import compute_fielding_features, save_fielding_features
from src.features.partnership import compute_partnership_features, save_partnership_features
from src.features.team import compute_team_features, save_team_features
from src.ingestion.download import download_all
from src.ingestion.load_db import create_database, print_summary


def run_phase1(skip_download: bool = False) -> None:
    """Run the full Phase 1 pipeline."""
    logger.info("=== CricketMind Phase 1 Pipeline ===")

    # Step 1: Download raw data
    if not skip_download:
        logger.info("Step 1: Downloading Cricsheet data...")
        download_all()
    else:
        logger.info("Step 1: Skipping download (--skip-download)")

    # Step 2: Load into DuckDB
    logger.info("Step 2: Loading data into DuckDB...")
    create_database()
    print_summary()

    # Step 3: Compute features
    logger.info("Step 3: Computing features...")

    batsman_df = compute_batsman_features()
    save_batsman_features(batsman_df)

    bowler_df = compute_bowler_features()
    save_bowler_features(bowler_df)

    team_df = compute_team_features()
    save_team_features(team_df)

    partnership_df = compute_partnership_features()
    save_partnership_features(partnership_df)

    fielding_df = compute_fielding_features()
    save_fielding_features(fielding_df)

    matchup_df = compute_matchup_features()
    save_matchup_features(matchup_df)

    logger.success("=== Phase 1 complete! ===")
    logger.info(f"  Batsmen:      {len(batsman_df):,}")
    logger.info(f"  Bowlers:      {len(bowler_df):,}")
    logger.info(f"  Teams:        {len(team_df):,}")
    logger.info(f"  Partnerships: {len(partnership_df):,}")
    logger.info(f"  Fielders:     {len(fielding_df):,}")
    logger.info(f"  Matchups:     {len(matchup_df):,}")


if __name__ == "__main__":
    import sys

    skip = "--skip-download" in sys.argv
    run_phase1(skip_download=skip)
