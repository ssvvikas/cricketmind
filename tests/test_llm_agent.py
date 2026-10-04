"""The agent loop, tested with a scripted fake client (no network, no API key)."""

from types import SimpleNamespace as NS

import pytest

from src.llm import agent as A
from src.llm.agent import CricketAgent
from src.llm.tools import TOOL_SPECS


def text(t):
    return NS(type="text", text=t)


def thinking(t="hmm"):
    return NS(type="thinking", thinking=t, signature="sig")


def tool_use(id_, name, **args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop="end_turn", **extra):
    return NS(content=list(blocks), stop_reason=stop,
              usage=NS(input_tokens=100, output_tokens=20), **extra)


class FakeClient:
    """Serves scripted responses and records every request it receives."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create_beta))

    def _record(self, kw, beta):
        # snapshot: the agent mutates its message list after the call
        self.requests.append({**kw, "messages": list(kw["messages"]), "_beta": beta})
        return self.responses.pop(0)

    def _create(self, **kw):
        return self._record(kw, False)

    def _create_beta(self, **kw):
        return self._record(kw, True)


class StubToolbox:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def call(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self.results.get(name, {"data": 1, "caveats": ["c"]})


def make(*responses, toolbox=None, **kw):
    client = FakeClient(*responses)
    return CricketAgent(toolbox=toolbox or StubToolbox(), client=client, **kw), client


class TestLoop:
    def test_answers_directly_without_tools(self):
        agent, client = make(reply(text("Hello")))
        r = agent.ask("hi")
        assert r.text == "Hello" and r.tool_calls == [] and r.steps == 1

    def test_tool_round_trip(self):
        tb = StubToolbox({"rank_batters": {"rows": [{"batter": "X"}], "caveats": []}})
        agent, client = make(
            reply(thinking(), tool_use("t1", "rank_batters", pool="club", top=3), stop="tool_use"),
            reply(text("X is best")), toolbox=tb)
        r = agent.ask("who is the best batter?")
        assert r.text == "X is best" and r.steps == 2
        assert tb.calls == [("rank_batters", {"pool": "club", "top": 3})]
        second = client.requests[1]["messages"]
        assert second[-1]["role"] == "user"
        res = second[-1]["content"][0]
        assert res["type"] == "tool_result" and res["tool_use_id"] == "t1"
        assert "X" in res["content"] and "is_error" not in res

    def test_parallel_tool_calls_return_in_one_user_message(self):
        agent, client = make(
            reply(tool_use("a", "find_player", query="Kohli"),
                  tool_use("b", "find_player", query="Bumrah"), stop="tool_use"),
            reply(text("done")))
        agent.ask("compare")
        last = client.requests[1]["messages"][-1]
        assert last["role"] == "user" and [c["tool_use_id"] for c in last["content"]] == ["a", "b"]

    def test_tool_errors_are_flagged_so_the_model_can_recover(self):
        tb = StubToolbox({"matchup": {"error": "Could not uniquely identify batter"}})
        agent, client = make(
            reply(tool_use("t1", "matchup", batter="Sharma", bowler="x", pool="club"),
                  stop="tool_use"),
            reply(text("Which Sharma do you mean?")), toolbox=tb)
        r = agent.ask("Sharma vs x")
        assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
        assert r.text == "Which Sharma do you mean?"

    def test_assistant_turns_are_kept_exactly_as_returned(self):
        blocks = [thinking("secret"), tool_use("t1", "find_player", query="k")]
        agent, client = make(reply(*blocks, stop="tool_use"), reply(text("ok")))
        agent.ask("q")
        assistant = client.requests[1]["messages"][1]
        assert assistant["role"] == "assistant" and assistant["content"] == blocks  # unchanged

    def test_conversation_history_carries_across_questions(self):
        agent, client = make(reply(text("one")), reply(text("two")))
        agent.ask("first")
        agent.ask("second")
        roles = [m["role"] for m in client.requests[1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_step_limit_stops_a_runaway_loop(self):
        loop = [reply(tool_use(f"t{i}", "find_player", query="x"), stop="tool_use")
                for i in range(3)]
        agent, _ = make(*loop, max_steps=3)
        r = agent.ask("loop forever")
        assert r.stop_reason == "max_steps" and "ran out of steps" in r.text and r.steps == 3

    def test_pause_turn_continues(self):
        agent, _ = make(reply(text("partial"), stop="pause_turn"), reply(text("final")))
        assert agent.ask("q").text == "final"

    def test_truncated_answers_are_labelled(self):
        agent, _ = make(reply(text("half an ans"), stop="max_tokens"))
        assert "cut off" in agent.ask("q").text

    def test_token_usage_is_summed(self):
        agent, _ = make(reply(tool_use("t", "find_player", query="k"), stop="tool_use"),
                        reply(text("ok")))
        r = agent.ask("q")
        assert (r.input_tokens, r.output_tokens) == (200, 40)


class TestRefusals:
    def test_refusal_returns_a_message_and_rolls_history_back(self):
        agent, _ = make(
            reply(tool_use("t1", "find_player", query="k"), stop="tool_use"),
            reply(stop="refusal", stop_details=NS(category="cyber")),
            reply(text("fine")))
        r = agent.ask("something declined")
        assert "can't help" in r.text and "cyber" in r.text
        assert agent.messages == []  # the whole refused exchange is gone
        assert agent.ask("next").text == "fine"


class TestRequestShape:
    def test_defaults_match_the_skills_guidance(self):
        agent, client = make(reply(text("ok")))
        agent.ask("q")
        req = client.requests[0]
        assert req["model"] == "claude-opus-5-5"
        assert req["tools"] == TOOL_SPECS and req["system"] == A.SYSTEM_PROMPT
        assert req["output_config"] == {"effort": "medium"}
        assert req["max_tokens"] == 16000
        # not allowed on Opus 5.5 (400): forced tool use, thinking config, sampling, prefill
        for bad in ("tool_choice", "thinking", "temperature", "top_p", "top_k"):
            assert bad not in req
        assert req["messages"][-1]["role"] == "user"  # no assistant prefill

    def test_refusal_fallbacks_are_on_by_default(self):
        agent, client = make(reply(text("ok")))
        agent.ask("q")
        req = client.requests[0]
        assert req["_beta"] and req["betas"] == [A.FALLBACK_BETA] and req["fallbacks"] == "default"

    def test_fallbacks_can_be_switched_off(self):
        agent, client = make(reply(text("ok")), refusal_fallbacks=False)
        agent.ask("q")
        req = client.requests[0]
        assert not req["_beta"] and "fallbacks" not in req and "betas" not in req

    def test_system_prompt_states_the_honesty_rules(self):
        s = A.SYSTEM_PROMPT
        for phrase in ("tool result", "coin flip", "ambiguous", "caveats", "pool"):
            assert phrase in s


class TestProgressEvents:
    def test_events_arrive_in_order_around_each_tool(self):
        events = []
        tb = StubToolbox({"matchup": {"error": "ambiguous"}})
        agent, _ = make(
            reply(tool_use("a", "find_player", query="Kohli"),
                  tool_use("b", "matchup", batter="x", bowler="y", pool="club"), stop="tool_use"),
            reply(text("done")), toolbox=tb)
        agent.ask("q", on_event=events.append)
        assert [(e["type"], e.get("name")) for e in events] == [
            ("step", None), ("tool_call", "find_player"), ("tool_result", "find_player"),
            ("tool_call", "matchup"), ("tool_result", "matchup"), ("step", None)]
        assert events[1]["arguments"] == {"query": "Kohli"}
        assert events[2]["ok"] is True and events[4]["ok"] is False  # the error result is flagged
        assert [e["n"] for e in events if e["type"] == "step"] == [1, 2]

    def test_a_failing_listener_never_breaks_the_answer(self):
        def boom(_):
            raise RuntimeError("listener bug")

        agent, _ = make(reply(tool_use("a", "find_player", query="k"), stop="tool_use"),
                        reply(text("still answered")))
        assert agent.ask("q", on_event=boom).text == "still answered"

    def test_no_listener_is_fine(self):
        agent, _ = make(reply(text("hi")))
        assert agent.ask("q").text == "hi"
