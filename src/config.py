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

# ICC Full Members.  Player ratings split internationals by opposition level: matches between
# two Full Members form the "intl" pool, anything involving an associate the "assoc" pool
# (associate-nation batters/bowlers look elite against weak opposition).  Afghanistan is a
# Full Member but has no matches in Cricsheet at all.
FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka",
    "Bangladesh", "West Indies", "Afghanistan", "Ireland", "Zimbabwe",
}

# Same franchise under an older name -> canonical (current) name, keyed by league so
# a name can never be renamed across leagues.  Deliberately NOT merged: Deccan Chargers
# (not Sunrisers), Pune Warriors, and CPL Antigua Hawksbills vs Antigua and Barbuda
# Falcons (different franchises).  Unverified: Jamaica Tallawahs -> Jamaica Kingsmen.
TEAM_ALIASES = {
    "IPL": {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Kings XI Punjab": "Punjab Kings",
        "Delhi Daredevils": "Delhi Capitals",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
    },
    "CPL": {
        "St Lucia Zouks": "St Lucia Kings",
        "St Lucia Stars": "St Lucia Kings",
        "Barbados Tridents": "Barbados Royals",
        "Trinidad & Tobago Red Steel": "Trinbago Knight Riders",
    },
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
