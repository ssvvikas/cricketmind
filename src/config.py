"""Project-wide configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


# ── paths ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DB_PATH = PROCESSED_DIR / "cricketmind.duckdb"


# ── Cricsheet download URLs ───────────────────────────────────────
# TIGZIG publishes Cricsheet data as Parquet/CSV/DuckDB.
# We download individual Parquet tables for flexibility.
CRICSHEET_BASE_URL = "https://db-mcp.tigzig.com/downloads"

CRICSHEET_TABLES = {
    "ball_by_ball": f"{CRICSHEET_BASE_URL}/ball_by_ball.parquet",
    "match_info": f"{CRICSHEET_BASE_URL}/match_info.parquet",
    "match_players": f"{CRICSHEET_BASE_URL}/match_players.parquet",
    "people": f"{CRICSHEET_BASE_URL}/people.parquet",
}

# Leagues we care about (used for filtering)
TARGET_LEAGUES = [
    "Indian Premier League",
    "Big Bash League",
    "Caribbean Premier League",
    "Pakistan Super League",
    "T20I",  # International T20s
]


class Settings(BaseSettings):
    """Runtime settings — override with env vars or a .env file."""

    duckdb_path: str = str(DB_PATH)
    raw_dir: str = str(RAW_DIR)
    processed_dir: str = str(PROCESSED_DIR)
    log_level: str = "INFO"

    # Feature engineering windows
    recent_form_innings: int = 10  # last N innings for "form" features
    min_innings_threshold: int = 5  # minimum innings to include a player

    model_config = {"env_prefix": "CRICKETMIND_"}


settings = Settings()
