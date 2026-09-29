"""Load raw Parquet files into DuckDB and apply league filters."""

import duckdb
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR, RAW_DIR, TARGET_GENDER, TARGET_MATCH_TYPES


def _match_type_filter_sql() -> str:
    """Build a SQL IN clause for target match types."""
    return "(" + ", ".join(f"'{t}'" for t in TARGET_MATCH_TYPES) + ")"


def create_database(db_path: str | None = None, raw_dir: str | None = None) -> None:
    """Read raw Parquet, filter to target leagues, and write to DuckDB."""
    db = db_path or str(DB_PATH)
    raw = raw_dir or str(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)

    # ── match_info (T20 + IPL, men's) ──────────────────────────────
    # Adds `date` and `league` so downstream SQL does not depend on raw naming.
    logger.info("Loading match_info...")
    con.execute(f"""
        CREATE OR REPLACE TABLE match_info AS
        SELECT
            *,
            start_date AS date,
            CASE WHEN match_type = 'IPL' THEN 'IPL' ELSE 'T20I' END AS league
        FROM read_parquet('{raw}/match_info.parquet')
        WHERE match_type IN {_match_type_filter_sql()}
          AND gender = '{TARGET_GENDER}'
    """)
    match_count = con.execute("SELECT count(*) FROM match_info").fetchone()[0]
    logger.info(f"  match_info: {match_count:,} matches")

    # ── ball_by_ball (normalised) ──────────────────────────────────
    # wides/noballs are NULL when absent.  Legal ball = no wide, no no-ball;
    # a batter faces every ball except wides; bowlers concede bat runs + wides
    # + no-balls (not byes / leg-byes).
    logger.info("Loading ball_by_ball...")
    con.execute(f"""
        CREATE OR REPLACE TABLE ball_by_ball AS
        SELECT
            b.match_id,
            m.date,
            m.league,
            b.innings,
            b.over_no AS over_number,
            b.batting_team,
            b.bowling_team,
            b.striker AS batter,
            b.non_striker,
            b.bowler,
            b.runs_off_bat AS runs_batter,
            b.extras AS runs_extras,
            COALESCE(b.wides, 0) AS wides,
            COALESCE(b.noballs, 0) AS noballs,
            (COALESCE(b.wides, 0) = 0 AND COALESCE(b.noballs, 0) = 0) AS is_legal,
            (COALESCE(b.wides, 0) = 0) AS is_faced,
            b.runs_off_bat + COALESCE(b.wides, 0) + COALESCE(b.noballs, 0) AS bowler_runs,
            b.wicket_type,
            b.player_dismissed AS player_out,
            (b.wicket_type IS NOT NULL
                AND b.wicket_type NOT IN ('retired hurt', 'retired not out')) AS is_wicket,
            (b.wicket_type IN ('bowled', 'caught', 'caught and bowled', 'lbw',
                               'stumped', 'hit wicket')) AS is_bowler_wicket,
            CASE WHEN b.wicket_type = 'caught and bowled' THEN b.bowler
                 ELSE b.fielder_1 END AS fielder,
            b.fielder_2
        FROM read_parquet('{raw}/ball_by_ball.parquet') b
        INNER JOIN match_info m ON b.match_id = m.match_id
    """)
    ball_count = con.execute("SELECT count(*) FROM ball_by_ball").fetchone()[0]
    logger.info(f"  ball_by_ball: {ball_count:,} deliveries")

    # ── match_players ──────────────────────────────────────────────
    logger.info("Loading match_players...")
    con.execute(f"""
        CREATE OR REPLACE TABLE match_players AS
        SELECT mp.*
        FROM read_parquet('{raw}/match_players.parquet') mp
        INNER JOIN match_info m ON mp.match_id = m.match_id
    """)
    player_count = con.execute("SELECT count(*) FROM match_players").fetchone()[0]
    logger.info(f"  match_players: {player_count:,} entries")

    # ── people (full table — lightweight) ──────────────────────────
    logger.info("Loading people...")
    con.execute(f"""
        CREATE OR REPLACE TABLE people AS
        SELECT * FROM read_parquet('{raw}/people.parquet')
    """)
    people_count = con.execute("SELECT count(*) FROM people").fetchone()[0]
    logger.info(f"  people: {people_count:,} players")

    con.close()
    logger.success(f"Database ready at {db}")


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the analytics database."""
    return duckdb.connect(db_path or str(DB_PATH), read_only=True)


def print_summary(db_path: str | None = None) -> None:
    """Print a quick summary of what is in the database."""
    con = get_connection(db_path)
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"\nDatabase: {db_path or DB_PATH}")
    print(f"Tables: {[t[0] for t in tables]}\n")
    for (table_name,) in tables:
        row_count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {row_count:,} rows")
    con.close()


if __name__ == "__main__":
    create_database()
    print_summary()
