"""Counting stats: cricket definitions checked against hand-computed fixture values.

Fixture facts (6 IPL matches, season 2024; see tests/test_features.py):
  innings 1 (team1 bats): 24 legal balls with runs i % 7, p1 on strike for balls 0-21 and p2 for
  22-23, plus a wide (1 extra) and a no-ball (4 off the bat + 1 extra), all faced by p1.
  Ball 5 run out (non-striker p2), ball 10 caught & bowled (p1), ball 15 caught (p1).  Bowler p4.
  innings 2 (team2 bats): p3 on strike 0-21, p4 22-23, bowler p2, no extras or wickets.
"""

import json
from datetime import date

import duckdb
import pytest

from src.llm import stats as S
from src.llm.tools import Toolbox
from src.models.player_ratings import RatingBook, load_balls
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)


@pytest.fixture(scope="module")
def con(db):  # noqa: F811
    return duckdb.connect(db, read_only=True)


class TestBatting:
    def test_p1_line_matches_hand_computation(self, con):
        b = S.batting_stats(con, "p1")
        # per match: sum(i % 7, i=0..21) = 63, plus a no-ball four = 67 runs off 23 balls
        assert (b["innings"], b["runs"], b["balls_faced"]) == (6, 402, 138)
        assert (b["dismissals"], b["not_outs"], b["average"]) == (6, 0, 67.0)
        assert b["strike_rate"] == round(402 * 100 / 138, 2)
        assert (b["fours"], b["sixes"]) == (24, 18) and b["ducks"] == 0
        assert (b["fifties"], b["hundreds"], b["highest_score"]) == (6, 0, "67")

    def test_wide_is_not_a_ball_faced_but_the_no_ball_is(self, con):
        assert S.batting_stats(con, "p1")["balls_faced"] == 6 * (22 + 1)

    def test_run_out_at_the_non_strikers_end_counts_as_a_dismissal(self, con):
        b = S.batting_stats(con, "p2")
        assert (b["innings"], b["runs"], b["balls_faced"], b["dismissals"]) == (6, 18, 12, 6)

    def test_never_dismissed_gives_no_average_and_a_star_on_highest_score(self, con):
        b = S.batting_stats(con, "p3")  # p3 is never out in the fixture
        assert b["dismissals"] == 0 and b["average"] is None and b["highest_score"].endswith("*")

    def test_breakdown_and_filters(self, con):
        b = S.batting_stats(con, "p1", league="IPL", season="2024", breakdown="season")
        assert b["runs"] == 402 and b["breakdown"][0]["season"] == "2024"
        assert S.batting_stats(con, "p1", breakdown="league")["breakdown"][0]["league"] == "IPL"

    def test_no_data_messages_are_actionable(self, con):
        with pytest.raises(S.NoData, match="Available seasons"):
            S.batting_stats(con, "p1", season="1999")
        with pytest.raises(S.NoData, match="no batting innings"):
            S.batting_stats(con, "p1", league="BBL")
        with pytest.raises(ValueError, match="league must be"):
            S.batting_stats(con, "p1", league="XYZ")


class TestBowling:
    def test_p4_line_matches_hand_computation(self, con):
        b = S.bowling_stats(con, "p4")
        # per match: 24 legal balls; bat runs 66 + no-ball 4, plus 1 for the wide and 1 for the
        # no-ball = 72 conceded; wickets = c&b + caught (the run-out is not the bowler's)
        assert (b["balls"], b["overs"], b["runs_conceded"], b["wickets"]) == (144, "24.0", 432, 12)
        assert b["economy"] == 18.0 and b["average"] == 36.0 and b["strike_rate"] == 12.0
        assert b["best_figures"] == "2/72" and b["five_wicket_hauls"] == 0
        assert b["dot_ball_percent"] == round(24 * 100 / 144, 1)

    def test_wicketless_bowler_has_no_average(self, con):
        b = S.bowling_stats(con, "p2")
        assert b["wickets"] == 0 and b["average"] is None and b["economy"] == 16.5

    def test_overs_notation(self):
        assert S._overs(24) == "4.0" and S._overs(25) == "4.1" and S._overs(5) == "0.5"


class TestLeaderboards:
    def test_orange_cap_order(self, con):
        rows = S.leaderboard(con, "runs", "IPL", "2024", top=3)["rows"]
        assert [(r["player"], r["value"]) for r in rows[:2]] == [("p1", 402.0), ("p3", 378.0)]
        assert [r["rank"] for r in rows] == [1, 2, 3]

    def test_purple_cap_and_economy_qualification(self, con):
        assert S.leaderboard(con, "wickets", top=1)["rows"][0]["player"] == "p4"
        eco = S.leaderboard(con, "economy", min_balls=100)["rows"]
        assert [r["player"] for r in eco] == ["p2", "p4"]  # 16.5 then 18.0, lower is better
        with pytest.raises(S.NoData):
            S.leaderboard(con, "economy", min_balls=100_000)

    def test_rate_stats_say_who_qualifies(self, con):
        r = S.leaderboard(con, "strike_rate", min_balls=100)
        assert any("at least 100 balls" in c for c in r["caveats"])

    def test_highest_score_and_bad_stat(self, con):
        assert S.leaderboard(con, "highest_score", top=1)["rows"][0]["value"] == 67.0
        with pytest.raises(ValueError, match="stat must be"):
            S.leaderboard(con, "catches")


class TestTeams:
    def test_team_a_record(self, con):
        t = S.team_record(con, "Team A", league="IPL")
        assert (t["played"], t["won"], t["lost"], t["win_percent"]) == (5, 3, 2, 60.0)
        assert t["avg_first_innings_score_when_batting_first"] == 72.0
        assert (t["matches_batting_first"], t["won_batting_first"], t["toss_wins"]) == (5, 3, 5)

    def test_head_to_head(self, con):
        h = S.team_record(con, "Team A", opponent="Team B")
        assert (h["played"], h["won"], h["lost"]) == (3, 2, 1) and h["opponent"] == "Team B"

    def test_standings_order_by_wins(self, con):
        rows = S.standings(con, "IPL", "2024")["rows"]
        assert [(r["team"], r["won"], r["lost"]) for r in rows] == [
            ("Team A", 3, 2), ("Team C", 2, 1), ("Team B", 1, 3)]
        with pytest.raises(ValueError):
            S.standings(con, "IPL", None)

    def test_unknown_team(self, con):
        with pytest.raises(S.NoData):
            S.team_record(con, "Nobody FC")


class TestVenues:
    def test_venue_summary(self, con):
        v = S.venue_stats(con, "V")
        assert (v["matches"], v["avg_first_innings_score"], v["avg_second_innings_score"]) == (
            6, 72.0, 66.0)
        assert v["highest_total"] == 72 and v["decided_matches"] == 6
        assert v["batting_first_win_percent"] == 50.0  # 3 of 6 won by the side batting first
        assert v["lowest_total_full_innings"] is None  # fixture innings are far shorter

    def test_venue_search_is_a_contains_match_and_reports_what_it_matched(self, con):
        assert S.venue_stats(con, "v")["matched_venues"] == ["V"]
        with pytest.raises(S.NoData, match="No venue matching"):
            S.venue_stats(con, "Lord's")


class TestSeasonResolution:
    @pytest.fixture()
    def labels(self):
        c = duckdb.connect()
        c.execute("CREATE TABLE match_info (league VARCHAR, season VARCHAR, date DATE)")
        rows = [("IPL", "2007/08", date(2008, 6, 1)), ("IPL", "2009", date(2009, 5, 24)),
                ("IPL", "2009/10", date(2010, 4, 25)), ("IPL", "2020/21", date(2020, 11, 10)),
                ("IPL", "2024", date(2024, 5, 26)), ("BBL", "2023/24", date(2024, 2, 4)),
                ("T20I", "2010", date(2010, 12, 1))]
        c.executemany("INSERT INTO match_info VALUES (?, ?, ?)", rows)
        return c

    @pytest.mark.parametrize("league,query,expected", [
        ("IPL", "2008", ["2007/08"]),      # IPL 2008 is stored as 2007/08
        ("IPL", "2009", ["2009"]),         # ...but IPL 2009 is plain 2009 (the 2009/10 edition
        ("IPL", "2010", ["2009/10"]),      #    concluded in 2010)
        ("IPL", "2020", ["2020/21"]),
        ("BBL", "2024", ["2023/24"]), ("BBL", "2023/24", ["2023/24"]),
        (None, "2010", ["2009/10", "2010"]),  # every league's edition concluding in 2010
    ])
    def test_year_resolves_to_the_edition_that_concluded_then(self, labels, league, query, expected):
        assert S.resolve_seasons(labels, league, query) == expected

    def test_no_season_and_unknown_season(self, labels):
        assert S.resolve_seasons(labels, "IPL", None) is None
        with pytest.raises(S.NoData, match="Available seasons"):
            S.resolve_seasons(labels, "IPL", "1999")


class TestToolboxIntegration:
    @pytest.fixture()
    def tb(self, db):  # noqa: F811
        box = Toolbox(db_path=db, model_dir="/nonexistent")
        box._book = RatingBook(load_balls(db), as_of="2030-01-01", halflife_years=None)
        box._teams = {"IPL": {"Team A", "Team B", "Team C"}}
        return box

    def test_every_stat_tool_returns_json_with_caveats(self, tb):
        calls = [("batting_stats", {"name": "p1"}), ("bowling_stats", {"name": "p4"}),
                 ("leaderboard", {"stat": "runs", "league": "IPL", "season": "2024"}),
                 ("team_record", {"team": "Team A", "league": "IPL"}),
                 ("standings", {"league": "IPL", "season": "2024"}),
                 ("venue_stats", {"venue": "V"})]
        for name, args in calls:
            r = tb.call(name, args)
            assert "error" not in r, (name, r)
            json.dumps(r)
            assert r["caveats"] and any("Covers only" in c for c in r["caveats"])

    def test_errors_come_back_as_data(self, tb):
        assert "Available seasons" in tb.call("batting_stats", {"name": "p1", "season": "1999"})["error"]
        assert "find_player" in tb.call("batting_stats", {"name": "Nobody Atall"})["error"]
        assert "Could not identify team" in tb.call("team_record", {"team": "Zzz"})["error"]
        assert "stat must be" in tb.call("leaderboard", {"stat": "catches"})["error"]

    def test_head_to_head_between_teams_resolves_names(self, tb):
        r = tb.call("team_record", {"team": "team a", "opponent": "Team B", "league": "IPL"})
        assert (r["played"], r["won"]) == (3, 2)

    def test_t20i_caveat_mentions_associates(self, tb):
        assert any("associate" in c for c in S._caveats("T20I"))
        assert any("associate" in c for c in S._caveats(None))
        assert not any("associate" in c for c in S._caveats("IPL"))
