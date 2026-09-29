"""Tests for the loader and feature engineering modules.

The fixture writes small synthetic Parquet files in the *raw* Cricsheet layout
and pushes them through ``create_database``, so the loader's normalisation and
the feature SQL are tested together against the real column names.

Per match (m1-m6), innings 1 is team1 batting, built so expected values can be
worked out by hand:

* 24 legal balls, runs cycling 0-6 (i % 7)
* + 1 wide (1 run) and 1 no-ball (bat 4 + 1 extra)
* p1 is on strike for legal balls 0-21, p2 for balls 22-23
* ball 5 : run out - non-striker p2 dismissed, fielders p5 + p6
* ball 10: caught and bowled by p4 (p1 out, fielder column empty)
* ball 15: caught by p5 (p1 out)
"""

import duckdb
import pandas as pd
import pytest

from src.features.batsman import compute_batsman_features
from src.features.bowler import compute_bowler_features
from src.features.bowling_matchup import compute_matchup_features
from src.features.fielding import compute_fielding_features
from src.features.partnership import compute_partnership_features
from src.features.team import compute_head_to_head, compute_team_features
from src.ingestion.load_db import create_database

N_MATCHES = 6

# (match_id, team1, team2, winner)
MATCHES = [
    (1, "Team A", "Team B", "Team A"),
    (2, "Team A", "Team B", "Team B"),
    (3, "Team A", "Team C", "Team A"),
    (4, "Team B", "Team C", "Team C"),
    (5, "Team A", "Team B", "Team A"),
    (6, "Team A", "Team C", "Team C"),
]


def _ball(match_id, innings, batting, bowling, i, batter, non_striker, bowler, **kw):
    row = {
        "match_id": match_id, "season": "2024",
        "start_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=int(match_id)),
        "venue": "V", "innings": innings, "ball": float(i), "batting_team": batting,
        "bowling_team": bowling, "striker": batter, "non_striker": non_striker,
        "bowler": bowler, "runs_off_bat": 0, "extras": 0, "wides": None, "noballs": None,
        "byes": None, "legbyes": None, "penalty": None, "wicket_type": None,
        "player_dismissed": None, "other_wicket_type": None, "other_player_dismissed": None,
        "match_type": "IPL", "over_no": i // 6, "delivery_in_over": i % 6 + 1,
        "actual_delivery": float(i), "non_boundary": False, "fielder_1": None,
        "fielder_2": None, "fielder_3": None, "gender": "male", "team_type": "club",
    }
    row.update(kw)
    return row


def _innings_rows(match_id, innings, batting, bowling, striker, other, bowler, specials):
    rows = []
    for i in range(24):
        batter, non_striker = (striker, other) if i < 22 else (other, striker)
        kw = {"runs_off_bat": i % 7}
        if specials and i == 5:
            kw.update(wicket_type="run out", player_dismissed=other,
                      fielder_1="p5", fielder_2="p6")
        if specials and i == 10:
            kw.update(wicket_type="caught and bowled", player_dismissed=batter)
        if specials and i == 15:
            kw.update(wicket_type="caught", player_dismissed=batter, fielder_1="p5")
        rows.append(_ball(match_id, innings, batting, bowling, i, batter, non_striker,
                          bowler, **kw))
    if specials:
        rows.append(_ball(match_id, innings, batting, bowling, 24, striker, other, bowler,
                          extras=1, wides=1))
        rows.append(_ball(match_id, innings, batting, bowling, 25, striker, other, bowler,
                          runs_off_bat=4, extras=1, noballs=1))
    return rows


@pytest.fixture(scope="module")
def raw_and_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raw")
    info, balls = [], []
    for mid, t1, t2, winner in MATCHES:
        info.append({
            "match_id": mid, "match_type": "IPL", "team1": t1, "team2": t2,
            "gender": "male", "season": "2024",
            "start_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=mid),
            "venue": "V", "city": "C", "event": "Indian Premier League",
            "toss_winner": t1, "winner": winner, "team_type": "club",
        })
        balls += _innings_rows(mid, 1, t1, t2, "p1", "p2", "p4", specials=True)
        balls += _innings_rows(mid, 2, t2, t1, "p3", "p4", "p2", specials=False)

    # out-of-scope matches: a men's ODI and a women's T20 - must be filtered out
    for mid, mtype, gender in [(7, "ODI", "male"), (8, "T20", "female")]:
        info.append({
            "match_id": mid, "match_type": mtype, "team1": "Team A", "team2": "Team B",
            "gender": gender, "season": "2024", "start_date": pd.Timestamp("2024-05-01"),
            "venue": "V", "city": "C", "event": None, "toss_winner": "Team A",
            "winner": "Team A", "team_type": "international",
        })
        balls += _innings_rows(mid, 1, "Team A", "Team B", "p1", "p2", "p4", specials=True)

    pd.DataFrame(info).to_parquet(tmp / "match_info.parquet")
    pd.DataFrame(balls).to_parquet(tmp / "ball_by_ball.parquet")
    # rosters: team1 = p1, p2 ; team2 = p3, p4 (matches the ball-by-ball above)
    pd.DataFrame([
        {"match_id": mid, "person_name": name, "identifier": name, "team": team, "role": "player"}
        for mid, t1, t2, _ in MATCHES
        for team, names in ((t1, ["p1", "p2"]), (t2, ["p3", "p4"]))
        for name in names
    ]).to_parquet(tmp / "match_players.parquet")
    pd.DataFrame({"identifier": ["x"], "name": ["p1"]}).to_parquet(tmp / "people.parquet")

    db = str(tmp / "test.duckdb")
    create_database(db_path=db, raw_dir=str(tmp))
    return db


@pytest.fixture(scope="module")
def db(raw_and_db):
    return raw_and_db


class TestLoader:
    def test_filters_out_odi_and_womens(self, db):
        con = duckdb.connect(db, read_only=True)
        assert con.execute("SELECT count(*) FROM match_info").fetchone()[0] == N_MATCHES
        ids = {r[0] for r in con.execute("SELECT DISTINCT match_id FROM ball_by_ball").fetchall()}
        assert ids == {1, 2, 3, 4, 5, 6}

    def test_null_wides_become_flags(self, db):
        con = duckdb.connect(db, read_only=True)
        legal, faced, total = con.execute(
            "SELECT sum(is_legal::INT), sum(is_faced::INT), count(*) FROM ball_by_ball "
            "WHERE match_id = 1 AND innings = 1"
        ).fetchone()
        assert (legal, faced, total) == (24, 25, 26)  # wide not legal/faced; no-ball faced

    def test_boolean_flags_are_never_null(self, db):
        # NULL flags would silently break `NOT flag` filters (NULL IN (...) is NULL)
        con = duckdb.connect(db, read_only=True)
        for col in ("is_legal", "is_faced", "is_wicket", "is_bowler_wicket"):
            nulls = con.execute(f"SELECT count(*) FROM ball_by_ball WHERE {col} IS NULL").fetchone()[0]
            assert nulls == 0, col

    def test_league_column(self, db):
        con = duckdb.connect(db, read_only=True)
        assert con.execute("SELECT DISTINCT league FROM match_info").fetchall() == [("IPL",)]


class TestBatsmanFeatures:
    def test_required_columns(self, db):
        df = compute_batsman_features(db_path=db, recent_n=5)
        for col in ["batter", "innings", "total_runs", "strike_rate", "average", "rank_score"]:
            assert col in df.columns

    def test_wide_not_counted_as_ball_faced(self, db):
        df = compute_batsman_features(db_path=db, recent_n=5).set_index("batter")
        # p1: 22 legal balls + no-ball = 23 faced per match, wide excluded
        assert df.loc["p1", "total_balls"] == 23 * N_MATCHES

    def test_no_ball_runs_credited_to_batter(self, db):
        df = compute_batsman_features(db_path=db, recent_n=5).set_index("batter")
        # p1 per match: sum(i % 7 for i in 0..21) + 4 (no-ball)
        expected = sum(i % 7 for i in range(22)) + 4
        assert df.loc["p1", "total_runs"] == expected * N_MATCHES

    def test_non_striker_run_out_credited_to_dismissed_player(self, db):
        df = compute_batsman_features(db_path=db, recent_n=5).set_index("batter")
        assert df.loc["p2", "times_out"] == N_MATCHES  # out at the non-striker end
        assert df.loc["p1", "times_out"] == N_MATCHES

    def test_rank_score_between_0_and_1(self, db):
        df = compute_batsman_features(db_path=db, recent_n=5)
        assert df["rank_score"].between(0, 1).all()


class TestBowlerFeatures:
    def test_economy_uses_legal_balls_and_wides(self, db):
        df = compute_bowler_features(db_path=db).set_index("bowler")
        assert df.loc["p4", "total_balls"] == 24 * N_MATCHES  # wide + no-ball not legal
        bat_runs = sum(i % 7 for i in range(24)) + 4  # + no-ball off the bat
        conceded = bat_runs + 1 + 1  # + wide + no-ball penalty run
        assert df.loc["p4", "total_runs_conceded"] == conceded * N_MATCHES

    def test_bowler_wickets_exclude_run_outs(self, db):
        df = compute_bowler_features(db_path=db).set_index("bowler")
        # caught and bowled + caught per match; the run out is not the bowler's
        assert df.loc["p4", "total_wickets"] == 2 * N_MATCHES

    def test_phase_econ_present(self, db):
        df = compute_bowler_features(db_path=db)
        assert {"econ_powerplay", "econ_middle", "econ_death"} <= set(df.columns)


class TestTeamFeatures:
    def test_win_pct(self, db):
        df = compute_team_features(db_path=db).set_index("team")
        # Team A: 5 matches, 3 wins; B and C are below the 5-match threshold
        assert list(df.index) == ["Team A"]
        assert df.loc["Team A", "win_pct"] == 60.0

    def test_head_to_head(self, db):
        h2h = compute_head_to_head("Team A", "Team B", db_path=db)
        assert len(h2h) == 3
        assert h2h["result"].value_counts().to_dict() == {"Team A": 2, "Team B": 1}

    def test_head_to_head_handles_quotes(self, db):
        assert compute_head_to_head("O'Brien XI", "Team B", db_path=db).empty


class TestPartnershipFeatures:
    def test_pair_is_order_independent(self, db):
        df = compute_partnership_features(db_path=db)
        pair = df[(df.player_a == "p1") & (df.player_b == "p2")]
        assert len(pair) == 1
        assert pair.iloc[0]["times_paired"] == N_MATCHES
        assert "chemistry_score" in df.columns


class TestFieldingFeatures:
    def test_run_out_credits_both_fielders(self, db):
        df = compute_fielding_features(db_path=db).set_index("fielder")
        assert df.loc["p5", "run_outs"] == N_MATCHES
        assert df.loc["p6", "run_outs"] == N_MATCHES

    def test_caught_and_bowled_credited_to_bowler(self, db):
        df = compute_fielding_features(db_path=db).set_index("fielder")
        assert df.loc["p4", "caught_and_bowled"] == N_MATCHES
        assert df.loc["p5", "catches"] == N_MATCHES

    def test_has_fielding_score(self, db):
        assert "fielding_score" in compute_fielding_features(db_path=db).columns


class TestMatchupFeatures:
    def test_wides_not_counted_as_balls_faced(self, db):
        df = compute_matchup_features(db_path=db)
        row = df[(df.bowler == "p4") & (df.batter == "p1")].iloc[0]
        assert row["balls_faced"] == 22 * N_MATCHES + N_MATCHES  # 22 legal + no-ball

    def test_dismissals_are_bowler_credited_only(self, db):
        df = compute_matchup_features(db_path=db)
        row = df[(df.bowler == "p4") & (df.batter == "p1")].iloc[0]
        assert row["times_dismissed"] == 2 * N_MATCHES  # c&b + caught
        p2 = df[(df.bowler == "p4") & (df.batter == "p2")]
        assert p2.empty or p2.iloc[0]["times_dismissed"] == 0  # run out is not the bowler's

    def test_has_vulnerability_score(self, db):
        assert "vulnerability_score" in compute_matchup_features(db_path=db).columns
