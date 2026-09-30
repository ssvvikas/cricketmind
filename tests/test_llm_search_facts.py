"""The search_facts tool: stored records as text, with the caveats that go with them."""

import json
import sys

import pytest

from src.llm.tools import TOOL_SPECS, Toolbox
from src.rag import entity_index as EI
from tests.test_entity_index import fake_embed, synthetic_tables


@pytest.fixture(scope="module")
def persist(tmp_path_factory):
    processed = synthetic_tables(tmp_path_factory.mktemp("processed"))
    d = tmp_path_factory.mktemp("chroma")
    EI.build_index(processed, d, embed_fn=fake_embed)
    return d


@pytest.fixture()
def tb(persist):
    return Toolbox(entity_dir=persist, entity_embed_fn=fake_embed)


def call(tb, **args):
    return tb.call("search_facts", args)


class TestResult:
    def test_returns_compact_hits_for_a_named_player(self, tb):
        r = call(tb, question="How does Kohli bat in the death overs?", top_k=3)
        assert "error" not in r
        assert r["named"] == {"players": ["V Kohli"], "teams": []} and r["name_filtered"] is True
        assert len(r["hits"]) == 3 and all("V Kohli" in h["text"] for h in r["hits"])
        assert set(r["hits"][0]) == {"entity_type", "entity_name", "league", "score", "text"}
        assert r["intent"] == ["batsman"]

    def test_the_result_is_plain_json_and_carries_caveats(self, tb):
        r = call(tb, question="Kohli against Bumrah")
        json.loads(json.dumps(r))
        assert r["caveats"] and any("Covers only" in c for c in r["caveats"])
        assert any("does not rank players" in c for c in r["caveats"])

    def test_team_hits_carry_their_league(self, tb):
        r = call(tb, question="How have Chennai Super Kings done in the IPL?", top_k=1)
        assert r["hits"][0]["entity_type"] == "team" and r["hits"][0]["league"] == "IPL"

    def test_entity_type_and_top_k(self, tb):
        r = call(tb, question="Kohli", entity_type="fielding")
        assert [h["entity_type"] for h in r["hits"]] == ["fielding"]
        assert len(call(tb, question="batting", top_k=100)["hits"]) == 10  # capped at 10


class TestCaveatsFollowTheContent:
    def test_type_specific_caveats_appear_only_when_that_type_is_returned(self, tb):
        joined = lambda r: " ".join(r["caveats"])  # noqa: E731
        m = call(tb, question="Kohli against Bumrah", entity_type="matchup")
        assert "small samples" in joined(m) and "involvement" not in joined(m)
        f = call(tb, question="Dhoni catches", entity_type="fielding")
        assert "involvement in dismissals" in joined(f) and "small samples" not in joined(f)
        p = call(tb, question="Kohli and Sharma together", entity_type="partnership")
        assert "raw records" in joined(p)

    def test_an_ambiguous_surname_tells_the_assistant_to_ask(self, tb):
        r = call(tb, question="Sharma record")
        note = next(c for c in r["caveats"] if "matched several players" in c)
        assert "RG Sharma" in note and "Sandeep Sharma" in note and "ask the user" in note

    def test_a_resolved_name_raises_no_ambiguity_caveat(self, tb):
        r = call(tb, question="What about Rohit Sharma?")
        assert not any("matched several" in c for c in r["caveats"])

    def test_a_ranking_question_carries_the_no_ranking_warning(self, tb):
        r = call(tb, question="Who scored the most runs?")
        assert any("cannot rank by a statistic" in c for c in r["caveats"])

    def test_a_named_entity_with_no_matching_chunk_says_so(self, tb):
        r = call(tb, question="Kohli", entity_type="team")
        assert r["name_filtered"] is False and "plain semantic" in r["note"]


class TestErrors:
    def test_an_invalid_entity_type_is_an_actionable_error(self, tb):
        assert "entity_type must be" in call(tb, question="Kohli", entity_type="umpire")["error"]

    def test_an_unbuilt_index_says_how_to_build_it(self, tmp_path):
        box = Toolbox(entity_dir=tmp_path / "nothing", entity_embed_fn=fake_embed)
        err = call(box, question="Kohli")["error"]
        assert "not been built" in err and "src.rag.entity_index" in err

    def test_missing_dependencies_say_what_to_install(self, tb, monkeypatch):
        monkeypatch.setitem(sys.modules, "src.rag.entity_index", None)
        err = call(tb, question="Kohli")["error"]
        assert "pip install" in err and "rag" in err

    def test_question_is_required(self, tb):
        assert "Bad arguments" in tb.call("search_facts", {})["error"]


class TestSpec:
    def test_the_spec_tells_the_model_when_to_use_it(self):
        spec = next(s for s in TOOL_SPECS if s["name"] == "search_facts")
        d = spec["description"]
        for phrase in ("by phase", "last 10 innings", "partnership", "cannot rank"):
            assert phrase in d
        assert spec["input_schema"]["required"] == ["question"]
        assert spec["input_schema"]["properties"]["top_k"]["maximum"] == 10

    def test_the_system_prompt_mentions_it(self):
        from src.llm import agent

        assert "search_facts" in agent.SYSTEM_PROMPT and "cannot rank" in agent.SYSTEM_PROMPT
