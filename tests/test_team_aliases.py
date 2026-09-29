"""Franchise renames must collapse to one team, without leaking across leagues."""

import duckdb
import pandas as pd
import pytest

from src.ingestion.load_db import create_database
from tests.test_features import _innings_rows


def _info(match_id, event, t1, t2, winner, date):
    return {
        "match_id": match_id, "match_type": "T20", "team1": t1, "team2": t2,
        "gender": "male", "season": "x", "start_date": pd.Timestamp(date), "venue": "V",
        "city": "C", "event": event, "toss_winner": t1, "winner": winner,
        "team_type": "club",
    }


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alias")
    old, new = "Royal Challengers Bangalore", "Royal Challengers Bengaluru"
    info = [
        _info(1, "Indian Premier League", old, "Team B", old, "2015-04-01"),
        _info(2, "Indian Premier League", new, "Team B", "Team B", "2025-04-01"),
        # same *old* name in a different league must be left alone
        _info(3, "Pakistan Super League", old, "Team B", old, "2020-04-01"),
        _info(4, "Caribbean Premier League", "St Lucia Zouks", "Barbados Tridents",
              "St Lucia Zouks", "2014-08-01"),
    ]
    balls = []
    for mid, m in enumerate(info, start=1):
        balls += _innings_rows(mid, 1, m["team1"], m["team2"], "p1", "p2", "p4", False)
    pd.DataFrame(info).to_parquet(tmp / "match_info.parquet")
    pd.DataFrame(balls).to_parquet(tmp / "ball_by_ball.parquet")
    pd.DataFrame({"match_id": [1, 3], "person_name": ["p1", "p1"], "identifier": ["x", "x"],
                  "team": [old, old], "role": ["player", "player"]}
                 ).to_parquet(tmp / "match_players.parquet")
    pd.DataFrame({"identifier": ["x"], "name": ["p1"]}).to_parquet(tmp / "people.parquet")
    db = str(tmp / "a.duckdb")
    create_database(db_path=db, raw_dir=str(tmp))
    return duckdb.connect(db, read_only=True)


def test_ipl_rename_collapsed_everywhere(con):
    teams = {r[0] for r in con.execute(
        "SELECT team1 FROM match_info WHERE league='IPL' UNION SELECT winner FROM match_info "
        "WHERE league='IPL' AND winner LIKE 'Royal%'").fetchall()}
    assert teams == {"Royal Challengers Bengaluru"}
    bat = {r[0] for r in con.execute(
        "SELECT DISTINCT batting_team FROM ball_by_ball WHERE league='IPL'").fetchall()}
    assert "Royal Challengers Bangalore" not in bat
    assert con.execute("SELECT DISTINCT team FROM match_players WHERE match_id=1"
                       ).fetchall() == [("Royal Challengers Bengaluru",)]


def test_alias_does_not_leak_across_leagues(con):
    assert con.execute("SELECT team1 FROM match_info WHERE league='PSL'"
                       ).fetchone() == ("Royal Challengers Bangalore",)
    assert con.execute("SELECT DISTINCT team FROM match_players WHERE match_id=3"
                       ).fetchall() == [("Royal Challengers Bangalore",)]


def test_cpl_renames(con):
    row = con.execute("SELECT team1, team2, winner FROM match_info WHERE league='CPL'").fetchone()
    assert row == ("St Lucia Kings", "Barbados Royals", "St Lucia Kings")
