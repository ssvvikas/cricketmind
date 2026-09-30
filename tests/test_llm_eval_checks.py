"""The eval's own checks must be right, or the eval proves nothing."""

import json
from pathlib import Path

import pytest

from src.llm.eval_checks import Case, check, load_cases, ungrounded_numbers

R = [{"expected_runs_per_over": 8.97, "dismissal_prob_per_over": 0.2945, "p_team_a": 0.4983,
      "balls": 6926, "note": "strike rate 134.8"}]


class TestGrounding:
    def test_numbers_present_in_tool_results_pass(self):
        assert ungrounded_numbers("He concedes 8.97 runs an over; a wicket chance of 29.5%.", R) == []

    def test_percentages_match_fractions_and_rounding(self):
        assert ungrounded_numbers("About 49.8% for Mumbai, roughly 50%.", R) == []
        assert ungrounded_numbers("A strike rate of 134.8.", R) == []

    def test_a_rounded_claim_may_quote_a_more_precise_source(self):
        src = [{"top_percent_of_pool": 6.1, "rank": 25, "ranked_among": 407}]
        assert ungrounded_numbers("He is in roughly the top 6%.", src) == []     # 6% ~ 6.1
        assert ungrounded_numbers("He is in the top 6.1%.", src) == []
        assert ungrounded_numbers("He is in the top 6.4%.", src) == ["6.4%"]     # too precise
        assert ungrounded_numbers("He is in the top 9%.", src) == ["9%"]

    def test_invented_percentage_near_a_real_probability_is_still_caught(self):
        src = [{"p_team_a": 0.4983}]
        assert ungrounded_numbers("Mumbai win 50% of the time.", src) == []
        assert ungrounded_numbers("Mumbai win 62% of the time.", src) == ["62%"]

    def test_invented_statistics_are_caught(self):
        bad = ungrounded_numbers("His strike rate is 148.3 and he wins 62% of duels.", R)
        assert set(bad) == {"148.3", "62%"}

    def test_plain_integers_are_ignored(self):
        assert ungrounded_numbers("The top 3 of 407 batters, in 2025.", R) == []

    def test_a_negative_figure_matches_its_magnitude(self):
        src = [{"runs_saved_per_over": -0.01}]
        # the answer's digits carry no sign, and Claude writes "-0.01" or "\u22120.01" (a real minus)
        assert ungrounded_numbers("about average (-0.01 runs saved per over)", src) == []
        assert ungrounded_numbers("about average (\u22120.01 runs saved per over)", src) == []
        assert ungrounded_numbers("about average (0.09 runs saved per over)", src) == ["0.09"]

    def test_no_tool_results_means_any_decimal_is_ungrounded(self):
        assert ungrounded_numbers("He averages 42.5.", []) == ["42.5"]


class TestChecks:
    def call(self, name, result=None):
        return {"name": name, "arguments": {}, "result": result or {}}

    def test_missing_expected_tool_fails(self):
        c = Case("x", "q", expect_tools_any=["predict_match"])
        assert check(c, "answer", [self.call("find_player")])
        assert not check(c, "answer", [self.call("predict_match")])

    def test_required_honesty_phrase(self):
        c = Case("x", "q", must_mention_any=["coin flip"], require_grounding=False)
        assert check(c, "Mumbai are slight favourites.", [])
        assert not check(c, "It is close to a COIN FLIP.", [])

    def test_forbidden_phrase(self):
        c = Case("x", "q", must_not_mention=["guaranteed"], require_grounding=False)
        assert check(c, "A guaranteed win.", [])

    def test_a_successful_tool_that_should_not_have_run_fails(self):
        c = Case("x", "q", forbid_successful_tools=["matchup"], require_grounding=False)
        assert check(c, "ok", [self.call("matchup", {"expected_runs_per_over": 8.0})])
        assert not check(c, "ok", [self.call("matchup", {"error": "ambiguous"})])

    def test_ungrounded_number_fails_the_case(self):
        c = Case("x", "q")
        assert any("not found" in f for f in check(c, "Kohli averages 51.2.", []))
        assert not check(c, "Kohli faced 6926 balls.", [self.call("player_profile", R[0])])


class TestCaseFile:
    def test_cases_load_with_unique_ids_and_valid_tools(self):
        from src.llm.tools import TOOL_NAMES

        cases = load_cases(Path(__file__).resolve().parent.parent / "eval" / "cases.json")
        assert len(cases) >= 10 and len({c.id for c in cases}) == len(cases)
        for c in cases:
            assert c.question.strip()
            assert set(c.expect_tools_any) <= TOOL_NAMES, c.id
            assert set(c.forbid_successful_tools) <= TOOL_NAMES, c.id
            assert c.expect_tools_any or c.must_mention_any or c.forbid_successful_tools, c.id


class TestFinalWinnerCase:
    """The 'final' question has no answer in the data; how the assistant may respond."""

    @pytest.fixture()
    def case(self):
        cases = {c.id: c for c in load_cases(
            Path(__file__).resolve().parent.parent / "eval" / "cases.json")}
        return cases["final-winner-unknown"]

    def test_an_answer_that_states_its_assumption_passes(self, case):
        honest = ("Sunrisers won: they beat RCB by 8 runs on 29 May 2016. My data doesn't mark "
                  "which match was the final; I'm treating this one as the final because it was "
                  "the last match of the season.")
        assert check(case, honest, []) == []

    def test_a_flat_claim_with_no_caveat_fails(self, case):
        assert check(case, "Sunrisers Hyderabad won the IPL 2016 final against RCB.", [])

    def test_the_plain_no_answer_still_passes(self, case):
        assert check(case, "I can't tell you: the data has no playoff information.", []) == []
