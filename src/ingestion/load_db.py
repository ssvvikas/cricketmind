"""Load raw Parquet files into DuckDB and apply league filters."""

import duckdb
from loguru import logger

from pathlib import Path

from src.config import (
    DB_PATH,
    LEAGUE_BY_EVENT,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_GENDER,
    TARGET_MATCH_TYPES,
    TEAM_ALIASES,
)


def _match_type_filter_sql() -> str:
    """Build a SQL IN clause for target match types."""
    return "(" + ", ".join(f"'{t}'" for t in TARGET_MATCH_TYPES) + ")"


def _league_case_sql() -> str:
    """SQL CASE mapping event names to league codes (IPL, BBL, CPL, PSL, T20I)."""
    whens = " ".join(f"WHEN event = '{e}' THEN '{c}'" for e, c in LEAGUE_BY_EVENT.items())
    return (
        f"CASE {whens} WHEN match_type = 'IPL' THEN 'IPL' "
        "WHEN team_type = 'international' THEN 'T20I' "
        "ELSE COALESCE(event, 'Other T20') END"
    )


def _canonical_team_sql(col: str, league_col: str = "league") -> str:
    """SQL expression mapping renamed franchises in `col` to their canonical name."""
    whens = " ".join(
        f"WHEN {league_col} = '{lg}' AND {col} = '{old.replace(chr(39), chr(39) * 2)}' "
        f"THEN '{new}'"
        for lg, names in TEAM_ALIASES.items()
        for old, new in names.items()
    )
    return f"CASE {whens} ELSE {col} END"


def _source_sql(raw: str, table: str) -> str:
    """Read a raw table from TIGZIG, plus the direct-Cricsheet copy when present.

    Direct-Cricsheet rows whose match_id already exists in the TIGZIG export are
    skipped so no match is counted twice.
    """
    tigzig = f"SELECT * FROM read_parquet('{raw}/{table}.parquet')"
    extra = Path(raw) / f"cricsheet_{table}.parquet"
    if not extra.exists():
        return tigzig
    return (
        f"{tigzig} UNION ALL BY NAME "
        f"SELECT * FROM read_parquet('{extra}') WHERE match_id NOT IN "
        f"(SELECT match_id FROM read_parquet('{raw}/match_info.parquet'))"
    )


def create_database(db_path: str | None = None, raw_dir: str | None = None) -> None:
    """Read raw Parquet, filter to target leagues, and write to DuckDB."""
    db = db_path or str(DB_PATH)
    raw = raw_dir or str(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)

    # ── match_info (men's T20 leagues + T20I) ──────────────────────
    # Adds `date` and `league` so downstream SQL does not depend on raw naming.
    logger.info("Loading match_info...")
    con.execute(f"""
        CREATE OR REPLACE TABLE match_info AS
        SELECT
            * REPLACE (
                {_canonical_team_sql("team1")} AS team1,
                {_canonical_team_sql("team2")} AS team2,
                {_canonical_team_sql("toss_winner")} AS toss_winner,
                {_canonical_team_sql("winner")} AS winner
            )
        FROM (
            SELECT
                *,
                start_date AS date,
                {_league_case_sql()} AS league
            FROM ({_source_sql(raw, "match_info")})
            WHERE match_type IN {_match_type_filter_sql()}
              AND gender = '{TARGET_GENDER}'
        )
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
            {_canonical_team_sql("b.batting_team", "m.league")} AS batting_team,
            {_canonical_team_sql("b.bowling_team", "m.league")} AS bowling_team,
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
            COALESCE(b.wicket_type IN ('bowled', 'caught', 'caught and bowled', 'lbw',
                                       'stumped', 'hit wicket'), FALSE) AS is_bowler_wicket,
            CASE WHEN b.wicket_type = 'caught and bowled' THEN b.bowler
                 ELSE b.fielder_1 END AS fielder,
            b.fielder_2
        FROM ({_source_sql(raw, "ball_by_ball")}) b
        INNER JOIN match_info m ON b.match_id = m.match_id
        WHERE b.innings <= 2  -- innings 3+ are super overs
    """)
    ball_count = con.execute("SELECT count(*) FROM ball_by_ball").fetchone()[0]
    logger.info(f"  ball_by_ball: {ball_count:,} deliveries")

    # ── match_players ──────────────────────────────────────────────
    logger.info("Loading match_players...")
    con.execute(f"""
        CREATE OR REPLACE TABLE match_players AS
        SELECT mp.* REPLACE ({_canonical_team_sql("mp.team", "m.league")} AS team)
        FROM ({_source_sql(raw, "match_players")}) mp
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
