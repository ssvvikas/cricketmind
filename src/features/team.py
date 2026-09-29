"""Team-level feature engineering."""

import duckdb
import pandas as pd
from loguru import logger

from src.config import DB_PATH, PROCESSED_DIR


def compute_team_features(db_path: str | None = None) -> pd.DataFrame:
    """Compute per-team features: win rate, NRR, batting/bowling strength."""
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    logger.info("Computing team features...")

    df = con.execute("""
        WITH team_matches AS (
            -- Expand each match into two rows (one per team)
            SELECT
                match_id, date, event_name, venue,
                team1 AS team,
                team2 AS opponent,
                CASE WHEN winner = team1 THEN 1 ELSE 0 END AS won,
                toss_winner = team1 AS won_toss
            FROM match_info
            WHERE winner IS NOT NULL
            UNION ALL
            SELECT
                match_id, date, event_name, venue,
                team2 AS team,
                team1 AS opponent,
                CASE WHEN winner = team2 THEN 1 ELSE 0 END AS won,
                toss_winner = team2 AS won_toss
            FROM match_info
            WHERE winner IS NOT NULL
        ),
        team_batting AS (
            -- Aggregate batting stats per team per match
            SELECT
                b.match_id,
                mp.team,
                SUM(b.runs_batter) AS team_runs,
                COUNT(*) AS team_balls,
                SUM(CASE WHEN b.runs_batter >= 4 THEN 1 ELSE 0 END) AS boundaries,
                SUM(CASE WHEN b.runs_batter = 0 THEN 1 ELSE 0 END) AS dots
            FROM ball_by_ball b
            JOIN match_players mp ON b.match_id = mp.match_id AND b.batter = mp.player_id
            GROUP BY b.match_id, mp.team
        ),
        team_bowling AS (
            SELECT
                b.match_id,
                mp.team AS bowling_team,
                SUM(CASE WHEN b.wicket_type IS NOT NULL
                         AND b.wicket_type NOT IN ('run out', 'retired hurt', 'retired out')
                     THEN 1 ELSE 0 END) AS wickets_taken,
                SUM(b.runs_batter + b.runs_extras) AS runs_conceded,
                COUNT(*) AS balls_bowled
            FROM ball_by_ball b
            JOIN match_players mp ON b.match_id = mp.match_id AND b.bowler = mp.player_id
            GROUP BY b.match_id, mp.team
        )
        SELECT
            tm.team,
            tm.event_name AS league,
            COUNT(*) AS matches,
            SUM(tm.won) AS wins,
            ROUND(SUM(tm.won) * 100.0 / COUNT(*), 2) AS win_pct,
            ROUND(AVG(tb.team_runs), 2) AS avg_score,
            ROUND(AVG(tb.team_runs * 6.0 / NULLIF(tb.team_balls, 0)), 2) AS avg_run_rate,
            ROUND(AVG(tb.boundaries * 100.0 / NULLIF(tb.team_balls, 0)), 2) AS boundary_pct,
            ROUND(AVG(tbo.wickets_taken), 2) AS avg_wickets_taken,
            ROUND(AVG(tbo.runs_conceded * 6.0 / NULLIF(tbo.balls_bowled, 0)), 2) AS avg_bowling_econ,
            MIN(tm.date) AS first_match,
            MAX(tm.date) AS last_match
        FROM team_matches tm
        LEFT JOIN team_batting tb ON tm.match_id = tb.match_id AND tm.team = tb.team
        LEFT JOIN team_bowling tbo ON tm.match_id = tbo.match_id AND tm.team = tbo.bowling_team
        GROUP BY tm.team, tm.event_name
        HAVING COUNT(*) >= 5
        ORDER BY win_pct DESC
    """).df()

    con.close()
    logger.info(f"Team features computed for {len(df)} team-league combos")
    return df


def compute_head_to_head(
    team_a: str, team_b: str, db_path: str | None = None
) -> pd.DataFrame:
    """Head-to-head record between two teams."""
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)

    df = con.execute(f"""
        SELECT
            date, venue, event_name AS league, winner,
            CASE WHEN winner = '{team_a}' THEN '{team_a}'
                 WHEN winner = '{team_b}' THEN '{team_b}'
                 ELSE 'no result' END AS result
        FROM match_info
        WHERE (team1 = '{team_a}' AND team2 = '{team_b}')
           OR (team1 = '{team_b}' AND team2 = '{team_a}')
        ORDER BY date DESC
    """).df()

    con.close()
    return df


def save_team_features(df: pd.DataFrame) -> None:
    dest = PROCESSED_DIR / "team_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(f"Saved team features to {dest}")


if __name__ == "__main__":
    features = compute_team_features()
    save_team_features(features)
    print(features.head(20).to_string())
