"""Unit tests for feature engineering modules.

These tests use a small in-memory DuckDB database with synthetic data
so they run fast without needing the full Cricsheet download.
"""

import duckdb
import pytest

from src.features.batsman import compute_batsman_features
from src.features.bowler import compute_bowler_features
from src.features.bowling_matchup import compute_matchup_features
from src.features.fielding import compute_fielding_features
from src.features.partnership import compute_partnership_features
from src.features.team import compute_team_features


@pytest.fixture()
def test_db(tmp_path):
    """Create a small DuckDB with synthetic cricket data."""
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)

    # match_info
    con.execute("""
        CREATE TABLE match_info AS
        SELECT * FROM (VALUES
            ('m1', '2024-04-01', 'Mumbai', 'Team A', 'Team B', 'Team A', 'Team A',
             'Indian Premier League', 'T20'),
            ('m2', '2024-04-02', 'Delhi', 'Team A', 'Team B', 'Team B', 'Team B',
             'Indian Premier League', 'T20'),
            ('m3', '2024-04-03', 'Chennai', 'Team A', 'Team C', 'Team A', 'Team A',
             'Indian Premier League', 'T20'),
            ('m4', '2024-04-04', 'Kolkata', 'Team B', 'Team C', 'Team B', 'Team C',
             'Indian Premier League', 'T20'),
            ('m5', '2024-04-05', 'Bangalore', 'Team A', 'Team B', 'Team A', 'Team A',
             'Indian Premier League', 'T20'),
            ('m6', '2024-04-06', 'Hyderabad', 'Team A', 'Team C', 'Team A', 'Team B',
             'Indian Premier League', 'T20')
        ) AS t(match_id, date, venue, team1, team2, winner, toss_winner,
               event_name, match_type)
    """)

    # match_players
    con.execute("""
        CREATE TABLE match_players AS
        SELECT * FROM (VALUES
            ('m1', 'p1', 'Team A'), ('m1', 'p2', 'Team A'), ('m1', 'p3', 'Team B'),
            ('m1', 'p4', 'Team B'), ('m2', 'p1', 'Team A'), ('m2', 'p2', 'Team A'),
            ('m2', 'p3', 'Team B'), ('m2', 'p4', 'Team B'), ('m3', 'p1', 'Team A'),
            ('m3', 'p2', 'Team A'), ('m3', 'p5', 'Team C'), ('m4', 'p3', 'Team B'),
            ('m4', 'p4', 'Team B'), ('m4', 'p5', 'Team C'), ('m5', 'p1', 'Team A'),
            ('m5', 'p2', 'Team A'), ('m5', 'p3', 'Team B'), ('m5', 'p4', 'Team B'),
            ('m6', 'p1', 'Team A'), ('m6', 'p2', 'Team A'), ('m6', 'p5', 'Team C')
        ) AS t(match_id, player_id, team)
    """)

    # ball_by_ball — 120 synthetic deliveries spread across matches
    rows = []
    ball_id = 0
    for match in ["m1", "m2", "m3", "m4", "m5", "m6"]:
        for innings in [1, 2]:
            for over_num in range(20):
                for ball_num in range(1, 4):  # 3 balls per over for speed
                    batter = "p1" if innings == 1 else "p3"
                    non_striker = "p2" if innings == 1 else "p4"
                    bowler = "p4" if innings == 1 else "p2"
                    runs = ball_id % 7  # 0-6 runs cycling
                    extras = 0
                    wicket = None
                    player_out = None
                    fielder = None

                    # Add a wicket every 30 balls
                    if ball_id % 30 == 15:
                        wicket = "caught"
                        player_out = batter
                        fielder = "p5"

                    rows.append(
                        f"('{match}', {innings}, {over_num}, {ball_num}, "
                        f"'{batter}', '{non_striker}', '{bowler}', "
                        f"{runs}, {extras}, "
                        f"{'NULL' if not wicket else repr(wicket)}, "
                        f"{'NULL' if not player_out else repr(player_out)}, "
                        f"{'NULL' if not fielder else repr(fielder)})"
                    )
                    ball_id += 1

    values = ",\n".join(rows)
    con.execute(f"""
        CREATE TABLE ball_by_ball AS
        SELECT * FROM (VALUES
            {values}
        ) AS t(match_id, innings, over_number, ball_number,
               batter, non_striker, bowler,
               runs_batter, runs_extras,
               wicket_type, player_out, fielder)
    """)

    # people
    con.execute("""
        CREATE TABLE people AS
        SELECT * FROM (VALUES
            ('p1', 'Player One', 'Country A'),
            ('p2', 'Player Two', 'Country A'),
            ('p3', 'Player Three', 'Country B'),
            ('p4', 'Player Four', 'Country B'),
            ('p5', 'Player Five', 'Country C')
        ) AS t(player_id, name, country)
    """)

    con.close()
    return db_path


class TestBatsmanFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_batsman_features(db_path=test_db, recent_n=5)
        assert len(df) > 0, "Should return at least one batsman"

    def test_has_required_columns(self, test_db):
        df = compute_batsman_features(db_path=test_db, recent_n=5)
        required = ["batter", "innings", "total_runs", "strike_rate", "average", "rank_score"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_rank_score_between_0_and_1(self, test_db):
        df = compute_batsman_features(db_path=test_db, recent_n=5)
        assert df["rank_score"].min() >= 0
        assert df["rank_score"].max() <= 1


class TestBowlerFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_bowler_features(db_path=test_db)
        assert len(df) > 0

    def test_has_economy(self, test_db):
        df = compute_bowler_features(db_path=test_db)
        assert "economy" in df.columns
        assert df["economy"].notna().all()


class TestTeamFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_team_features(db_path=test_db)
        assert len(df) > 0

    def test_win_pct_range(self, test_db):
        df = compute_team_features(db_path=test_db)
        assert df["win_pct"].min() >= 0
        assert df["win_pct"].max() <= 100


class TestPartnershipFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_partnership_features(db_path=test_db)
        assert len(df) > 0

    def test_has_chemistry_score(self, test_db):
        df = compute_partnership_features(db_path=test_db)
        if len(df) > 0:
            assert "chemistry_score" in df.columns


class TestFieldingFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_fielding_features(db_path=test_db)
        assert len(df) > 0

    def test_has_fielding_score(self, test_db):
        df = compute_fielding_features(db_path=test_db)
        if len(df) > 0:
            assert "fielding_score" in df.columns


class TestMatchupFeatures:
    def test_returns_dataframe(self, test_db):
        df = compute_matchup_features(db_path=test_db)
        assert len(df) > 0

    def test_has_vulnerability_score(self, test_db):
        df = compute_matchup_features(db_path=test_db)
        if len(df) > 0:
            assert "vulnerability_score" in df.columns
