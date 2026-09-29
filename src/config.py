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

# ── Direct Cricsheet downloads (JSON zips) ────────────────────────
# TIGZIG's export has no BBL/CPL/PSL, so those leagues come straight from
# cricsheet.org and are converted to the same raw layout (see cricsheet_json.py).
CRICSHEET_ZIP_URL = "https://cricsheet.org/downloads/{key}_json.zip"
CRICSHEET_LEAGUES = ["bbl", "cpl", "psl"]

# Scope of the analysis (men's T20 club leagues + T20 internationals).
TARGET_MATCH_TYPES = ["T20", "IPL"]  # drops ODI / Test
TARGET_GENDER = "male"

# Cricsheet event name -> short league code used throughout the features.
LEAGUE_BY_EVENT = {
    "Indian Premier League": "IPL",
    "Big Bash League": "BBL",
    "Caribbean Premier League": "CPL",
    "Pakistan Super League": "PSL",
}


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
