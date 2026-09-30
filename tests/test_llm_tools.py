"""LLM tool layer: name resolution, schema/implementation parity, honesty, error handling."""

import inspect
import json

import pytest

from src.llm import tools as T
from src.llm.tools import TOOL_NAMES, TOOL_SPECS, Toolbox, resolve, score_name
from src.models.player_ratings import RatingBook, load_balls
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)


class TestNameScoring:
    @pytest.mark.parametrize("query,cand,low", [
        ("Virat Kohli", "V Kohli", 0.95), ("V Kohli", "V Kohli", 1.0), ("kohli", "V Kohli", 0.9),
        ("Mahendra Singh Dhoni", "MS Dhoni", 0.95), ("M.S. Dhoni", "MS Dhoni", 0.95),
        ("Jasprit Bumrah", "JJ Bumrah", 0.85),
    ])
    def test_matches(self, query, cand, low):
        assert score_name(query, cand) >= low

    def test_same_surname_different_person_scores_low(self):
        assert score_name("Rohit Kumar", "V Kumar") < 0.6

    def test_unrelated_names_score_low(self):
        assert score_name("Kohli", "Bumrah") < 0.6


class TestResolve:
    def test_unique_match_resolves(self):
        r = resolve("Virat Kohli", {"V Kohli": 6000, "RG Sharma": 5000})
        assert r["resolved"] == "V Kohli" and not r["ambiguous"]

    def test_genuinely_ambiguous_surname_is_not_guessed(self):
        r = resolve("Sharma", {"RG Sharma": 5500, "Sandeep Sharma": 3100, "I Sharma": 2400})
        assert r["ambiguous"] and r["resolved"] is None and len(r["candidates"]) == 3

    def test_dominant_player_beats_obscure_lookalike_with_a_note(self):
        r = resolve("Kohli", {"V Kohli": 6926, "T Kohli": 19})
        assert r["resolved"] == "V Kohli" and not r["ambiguous"] and "T Kohli" in r["note"]

    def test_comparable_lookalikes_stay_ambiguous(self):
        assert resolve("Kohli", {"V Kohli": 6000, "T Kohli": 3000})["ambiguous"]

    def test_no_match_returns_empty(self):
        r = resolve("Zzzzz", {"V Kohli": 100})
        assert r["candidates"] == [] and r["resolved"] is None


class TestSchemaParity:
    def test_every_spec_maps_to_a_method_with_matching_arguments(self):
        for spec in TOOL_SPECS:
            fn = getattr(Toolbox, spec["name"])
            params = set(inspect.signature(fn).parameters) - {"self"}
            props = set(spec["input_schema"]["properties"])
            required = set(spec["input_schema"]["required"])
            assert props <= params, f"{spec['name']}: schema has unknown args {props - params}"
            assert required <= props
            assert {p for p, v in inspect.signature(fn).parameters.items()
                    if p != "self" and v.default is inspect.Parameter.empty} <= required

    def test_names_are_unique_and_specs_are_json(self):
        assert len({s["name"] for s in TOOL_SPECS}) == len(TOOL_SPECS) == len(TOOL_NAMES)
        json.dumps(TOOL_SPECS)
        assert all(len(s["description"]) > 20 for s in TOOL_SPECS)


@pytest.fixture(scope="module")
def tb(db):  # noqa: F811
    box = Toolbox(db_path=db, model_dir="/nonexistent")
    box._book = RatingBook(load_balls(db), as_of="2030-01-01", halflife_years=None)
    box._teams = {"IPL": {"Royal Challengers Bengaluru", "Mumbai Indians", "Chennai Super Kings"},
                  "T20I": {"India", "Australia"}}
    return box


class TestErrorHandling:
    def test_call_never_raises_and_reports_problems(self, tb):
        assert "Unknown tool" in tb.call("does_not_exist")["error"]
        assert "Bad arguments" in tb.call("rank_batters", {"nope": 1})["error"]
        assert "pool must be" in tb.call("rank_batters", {"pool": "moon"})["error"]
        assert "league must be" in tb.call("predict_match", {
            "team_a": "a", "team_b": "b", "league": "XYZ"})["error"]

    def test_json_string_arguments_are_accepted(self, tb):
        assert "rows" in tb.call("rank_batters", json.dumps({"pool": "club", "top": 2}))

    def test_unknown_or_ambiguous_players_give_actionable_errors(self, tb):
        assert "find_player" in tb.call("matchup", {"batter": "Nobody", "bowler": "p4",
                                                    "pool": "club"})["error"]

    def test_model_not_trained_message(self, tb):
        r = tb.call("predict_match", {"team_a": "MI", "team_b": "CSK", "league": "IPL"})
        assert "not been trained" in r["error"]


class TestHonesty:
    def test_results_are_json_serialisable_and_carry_caveats(self, tb):
        calls = [
            ("rank_batters", {"pool": "club"}), ("rank_bowlers", {"pool": "club"}),
            ("matchup", {"batter": "p1", "bowler": "p4", "pool": "club"}),
            ("recommend_bowlers", {"batter": "p1", "bowlers": ["p4", "p2"], "pool": "club"}),
            ("best_pairs", {"batters": ["p1", "p2", "p3"], "pool": "club"}),
            ("player_profile", {"name": "p1", "pool": "club"}),
            ("fielding_profile", {"name": "p5"}),
        ]
        for name, args in calls:
            r = tb.call(name, args)
            assert "error" not in r, (name, r)
            json.dumps(r)
            assert r["caveats"], name

    def test_profile_reports_top_percent_so_the_model_need_not_compute_it(self, tb):
        b = tb.call("player_profile", {"name": "p1", "pool": "club"})["batting"]
        assert b["rank_in_pool"] is None and b["top_percent_of_pool"] is None  # too few balls
        assert T._top_percent(25, 407) == 6.1 and T._top_percent(1, 100) == 1.0
        assert T._top_percent(None, 10) is None and T._top_percent(5, 0) is None

    def test_small_samples_are_flagged(self, tb):
        r = tb.call("player_profile", {"name": "p3", "pool": "club"})
        assert any("Small sample" in c for c in r["caveats"])

    def test_matchup_says_head_to_head_is_not_used(self, tb):
        r = tb.call("matchup", {"batter": "p1", "bowler": "p4", "pool": "club"})
        assert any("head-to-head" in c for c in r["caveats"])

    @pytest.fixture()
    def fake_model_dir(self, tb, tmp_path, monkeypatch):
        for f in ("team_strength.json", "team_state.pkl"):
            (tmp_path / f).write_text("x")
        monkeypatch.setattr(tb, "model_dir", tmp_path)

    def test_club_prediction_is_labelled_near_coin_flip(self, tb, monkeypatch, fake_model_dir):
        monkeypatch.setattr(T, "_predict_match", lambda *a, **k: {
            "team_a": "Mumbai Indians", "team_b": "Chennai Super Kings", "league": "IPL",
            "p_team_a": 0.51, "p_team_b": 0.49, "xi_a": [], "xi_b": []})
        r = tb.call("predict_match", {"team_a": "MI", "team_b": "CSK", "league": "IPL"})
        assert r["reliability"].startswith("very low") and "coin flip" in " ".join(r["caveats"])
        assert r["team_a"] == "Mumbai Indians" and "xi_a" not in r

    def test_international_prediction_is_labelled_moderate(self, tb, monkeypatch, fake_model_dir):
        monkeypatch.setattr(T, "_predict_match", lambda *a, **k: {
            "team_a": "India", "team_b": "Australia", "league": "T20I", "p_team_a": 0.68,
            "p_team_b": 0.32, "xi_a": [], "xi_b": []})
        r = tb.call("predict_match", {"team_a": "India", "team_b": "Australia", "league": "T20I"})
        assert r["reliability"] == "moderate"

    def test_team_lookup_handles_abbreviations_and_old_names(self, tb):
        assert tb.find_team("RCB")["resolved"]["team"] == "Royal Challengers Bengaluru"
        assert tb.find_team("Royal Challengers Bangalore")["resolved"]["team"] == \
            "Royal Challengers Bengaluru"
        assert tb.find_team("mumbai")["resolved"]["team"] == "Mumbai Indians"
        assert tb.find_team("Nonsense FC")["resolved"] is None

    def test_associate_pool_carries_a_comparability_warning(self, tb):
        assert any("associate" in c.lower() for c in tb._pool_caveats("assoc"))
