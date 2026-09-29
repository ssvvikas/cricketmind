"""Tests for the direct Cricsheet JSON ingest and its integration in the loader."""

import json
import zipfile

import duckdb
import pandas as pd
import pytest

from src.ingestion.cricsheet_json import parse_zip
from src.ingestion.load_db import create_database
from tests.test_features import _innings_rows


def _delivery(batter, bowler, non_striker, bat=0, extras=None, wickets=None, actual=None):
    dl = {
        "batter": batter, "bowler": bowler, "non_striker": non_striker,
        "runs": {"batter": bat, "extras": sum((extras or {}).values()),
                 "total": bat + sum((extras or {}).values())},
    }
    if actual:
        dl["actual_delivery"] = actual
    if extras:
        dl["extras"] = extras
    if wickets:
        dl["wickets"] = wickets
    return dl


def _match_json(event="Pakistan Super League", super_over=False):
    over0 = [
        _delivery("A", "X", "B", bat=1),
        _delivery("A", "X", "B", extras={"wides": 1}),
        _delivery("B", "X", "A", bat=0, extras={"byes": 1, "noballs": 1}),
        _delivery("B", "X", "A", wickets=[
            {"player_out": "A", "kind": "run out",
             "fielders": [{"name": "F1"}, {"name": "F2"}, {"name": "F3"}, {"name": "F4"}]},
            {"player_out": "B", "kind": "retired hurt"},
        ]),
        _delivery("C", "X", "B", wickets=[{"player_out": "C", "kind": "caught and bowled"}]),
    ]
    innings = [{"team": "Team A", "overs": [{"over": 0, "deliveries": over0}]},
               {"team": "Team B", "overs": [{"over": 0, "deliveries": [
                   _delivery("D", "Y", "E", bat=6)]}]}]
    if super_over:
        innings.append({"team": "Team A", "super_over": True, "overs": [
            {"over": 0, "deliveries": [_delivery("A", "Y", "B", bat=4)]}]})
    return {
        "meta": {"data_version": "1.2.0"},
        "info": {
            "balls_per_over": 6, "city": "Lahore", "dates": ["2024-03-01"],
            "event": {"name": event}, "gender": "male", "match_type": "T20",
            "outcome": {"winner": "Team B", "by": {"wickets": 5}}, "overs": 20,
            "player_of_match": ["D"], "season": "2024", "team_type": "club",
            "teams": ["Team A", "Team B"], "toss": {"decision": "bat", "winner": "Team A"},
            "venue": "Gaddafi Stadium",
            "players": {"Team A": ["A", "B"], "Team B": ["D"]},
            "registry": {"people": {"A": "id-a", "B": "id-b", "D": "id-d"}},
        },
        "innings": innings,
    }


@pytest.fixture()
def parsed(tmp_path):
    zpath = tmp_path / "psl_json.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("1001.json", json.dumps(_match_json(super_over=True)))
        z.writestr("README.txt", "not a match")
    return parse_zip(zpath)


class TestParser:
    def test_skips_non_json_and_super_over(self, parsed):
        balls = parsed["ball_by_ball"]
        assert len(parsed["match_info"]) == 1
        assert set(balls["innings"]) == {1, 2}  # super over dropped
        assert len(balls) == 6

    def test_extras_split_into_columns(self, parsed):
        b = parsed["ball_by_ball"].reset_index(drop=True)
        assert b.loc[1, "wides"] == 1 and pd.isna(b.loc[1, "noballs"])
        assert b.loc[2, "byes"] == 1 and b.loc[2, "noballs"] == 1
        assert pd.isna(b.loc[0, "wides"])

    def test_run_out_fielders_and_second_wicket(self, parsed):
        b = parsed["ball_by_ball"].reset_index(drop=True)
        row = b.loc[3]
        assert (row["wicket_type"], row["player_dismissed"]) == ("run out", "A")
        assert (row["fielder_1"], row["fielder_2"], row["fielder_3"]) == ("F1", "F2", "F3")
        assert (row["other_wicket_type"], row["other_player_dismissed"]) == ("retired hurt", "B")

    def test_caught_and_bowled_has_no_fielder(self, parsed):
        row = parsed["ball_by_ball"].reset_index(drop=True).loc[4]
        assert row["wicket_type"] == "caught and bowled" and pd.isna(row["fielder_1"])

    def test_teams_and_missing_actual_delivery(self, parsed):
        b = parsed["ball_by_ball"].reset_index(drop=True)
        assert (b.loc[0, "batting_team"], b.loc[0, "bowling_team"]) == ("Team A", "Team B")
        assert (b.loc[5, "batting_team"], b.loc[5, "bowling_team"]) == ("Team B", "Team A")
        assert b.loc[0, "actual_delivery"] == pytest.approx(0.1)  # fallback: over + j/10

    def test_match_info_and_players(self, parsed):
        m = parsed["match_info"].iloc[0]
        assert (m["match_id"], m["event"], m["winner"], m["winner_wickets"]) == (
            1001, "Pakistan Super League", "Team B", 5)
        p = parsed["match_players"].set_index("person_name")
        assert p.loc["A", "identifier"] == "id-a" and len(p) == 3


class TestLoaderIntegration:
    @pytest.fixture()
    def db(self, tmp_path):
        """TIGZIG-style raw files (IPL + a match with super overs) + Cricsheet extras."""
        info = pd.DataFrame([{
            "match_id": 1, "match_type": "IPL", "team1": "Team A", "team2": "Team B",
            "gender": "male", "season": "2024", "start_date": pd.Timestamp("2024-04-01"),
            "venue": "V", "city": "C", "event": "Indian Premier League",
            "toss_winner": "Team A", "winner": "Team A", "team_type": "club",
        }])
        balls = pd.DataFrame(_innings_rows(1, 1, "Team A", "Team B", "p1", "p2", "p4", False)
                             + _innings_rows(1, 3, "Team A", "Team B", "p1", "p2", "p4", False))
        info.to_parquet(tmp_path / "match_info.parquet")
        balls.to_parquet(tmp_path / "ball_by_ball.parquet")
        pd.DataFrame({"match_id": [1], "person_name": ["p1"], "identifier": ["x"],
                      "team": ["Team A"], "role": ["player"]}
                     ).to_parquet(tmp_path / "match_players.parquet")
        pd.DataFrame({"identifier": ["x"], "name": ["p1"]}).to_parquet(tmp_path / "people.parquet")

        # Cricsheet copy: PSL match 1001, plus a duplicate of TIGZIG match 1
        zpath = tmp_path / "z.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("1001.json", json.dumps(_match_json()))
            z.writestr("1.json", json.dumps(_match_json(event="Indian Premier League")))
        for table, df in parse_zip(zpath).items():
            df.to_parquet(tmp_path / f"cricsheet_{table}.parquet")

        db = str(tmp_path / "t.duckdb")
        create_database(db_path=db, raw_dir=str(tmp_path))
        return duckdb.connect(db, read_only=True)

    def test_leagues_mapped_and_deduplicated(self, db):
        rows = dict(db.execute("SELECT match_id, league FROM match_info").fetchall())
        assert rows == {1: "IPL", 1001: "PSL"}

    def test_duplicate_match_uses_tigzig_rows_only(self, db):
        n = db.execute("SELECT count(*) FROM ball_by_ball WHERE match_id = 1").fetchone()[0]
        assert n == 24  # not doubled by the Cricsheet copy

    def test_super_over_innings_dropped(self, db):
        assert db.execute("SELECT max(innings) FROM ball_by_ball").fetchone()[0] == 2

    def test_cricsheet_rows_flow_through_flags(self, db):
        legal, faced = db.execute(
            "SELECT sum(is_legal::INT), sum(is_faced::INT) FROM ball_by_ball WHERE match_id = 1001"
        ).fetchone()
        # 6 deliveries: the wide is neither legal nor faced; the no-ball is faced but not legal
        assert (legal, faced) == (4, 5)
