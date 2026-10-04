"""The HTTP API, driven with a scripted fake model (no network, no API key)."""

import json
import threading
from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app, describe_error
from src.api.settings import ApiSettings
from src.llm.agent import CricketAgent
from src.llm.tools import TOOL_SPECS, Toolbox
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)
from tests.test_llm_agent import FakeClient, StubToolbox, reply, text, tool_use


def settings(**kw):
    base = dict(warmup_on_start=False, rate_limit_per_minute=1000, access_token=None)
    return ApiSettings(**{**base, **kw})


def agent_factory(*replies, toolbox=None):
    """A factory whose agents each get their own copy of the scripted replies."""
    def make():
        return CricketAgent(toolbox=toolbox or StubToolbox(), client=FakeClient(*list(replies)))
    return make


def client_for(*replies, toolbox=None, **kw):
    app = create_app(settings(**kw), toolbox=toolbox or StubToolbox(),
                     agent_factory=agent_factory(*replies, toolbox=toolbox))
    return TestClient(app)


HELLO = [reply(text("Hello from the assistant"))] * 40


def chat(c, message="hi", **body):
    return c.post("/api/chat", json={"message": message, **body})


# ── read-only endpoints ─────────────────────────────────────────────────
class TestInfo:
    def test_info_lists_tools_examples_and_limits(self):
        r = client_for(*HELLO).get("/api/info").json()
        assert [t["name"] for t in r["tools"]] == [s["name"] for s in TOOL_SPECS]
        assert len(r["examples"]) >= 6 and r["auth_required"] is False
        assert r["max_message_chars"] == 1000 and r["model"] == "claude-opus-5-5"
        assert all(t["description"] for t in r["tools"])

    def test_health(self):
        r = client_for(*HELLO).get("/api/health").json()
        assert r["status"] == "ok" and r["warmup"] == "skipped" and r["conversations"] == 0
        assert isinstance(r["credentials_in_environment"], bool)

    @pytest.mark.parametrize("warmup,ready", [
        ("skipped", True),                                      # lazy loading: usable at once
        ("running", False),                                     # still loading the data
        ({"ratings": True, "entity_search": False}, True),      # an optional resource is missing
        ({"ratings": "FileNotFoundError: no data"}, False),     # the core data failed to load
    ])
    def test_ready_reflects_whether_the_data_is_loaded(self, warmup, ready):
        c = client_for(*HELLO)
        c.app.state.warmup = warmup
        body = c.get("/api/health").json()
        assert body["ready"] is ready and body["status"] == "ok"

    def test_info_and_health_are_open_even_when_a_token_is_required(self):
        c = client_for(*HELLO, access_token="s3cret")
        assert c.get("/api/info").json()["auth_required"] is True
        assert c.get("/api/health").status_code == 200


# ── chat ────────────────────────────────────────────────────────────────
class TestChat:
    def test_a_question_gets_an_answer_and_a_session(self):
        r = chat(client_for(*HELLO))
        body = r.json()
        assert r.status_code == 200 and body["answer"] == "Hello from the assistant"
        assert len(body["session_id"]) == 32 and body["new_session"] is True and body["turns"] == 1
        assert body["tool_calls"] == []
        # 100 input + 20 output tokens at the model's list price ($4 / $20 per million)
        assert body["usage"] == {"input_tokens": 100, "output_tokens": 20,
                                 "estimated_cost_usd": 0.0008}

    def test_the_tools_used_are_reported_with_their_outcome(self):
        tb = StubToolbox({"matchup": {"error": "ambiguous"}})
        c = client_for(reply(tool_use("a", "find_player", query="Kohli"),
                             tool_use("b", "matchup", batter="x", bowler="y", pool="club"),
                             stop="tool_use"),
                       reply(text("done")), toolbox=tb)
        body = chat(c).json()
        assert body["tool_calls"] == [
            {"name": "find_player", "arguments": {"query": "Kohli"}, "ok": True},
            {"name": "matchup", "arguments": {"batter": "x", "bowler": "y", "pool": "club"}, "ok": False}]
        assert body["steps"] == 2

    def test_a_conversation_continues_with_its_history(self):
        c = client_for(*HELLO)
        first = chat(c, "first").json()
        second = chat(c, "second", session_id=first["session_id"]).json()
        assert second["session_id"] == first["session_id"] and second["new_session"] is False
        assert second["turns"] == 2
        agent = c.app.state.store.get_or_create(first["session_id"])[0].agent
        roles = [m["role"] for m in agent._client.requests[-1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_conversations_are_isolated_from_each_other(self):
        c = client_for(*HELLO)
        a, b = chat(c, "from a").json(), chat(c, "from b").json()
        assert a["session_id"] != b["session_id"]
        store = c.app.state.store
        msgs = lambda sid: [m["content"] for m in store.get_or_create(sid)[0].agent.messages  # noqa: E731
                            if m["role"] == "user"]
        assert msgs(a["session_id"]) == ["from a"] and msgs(b["session_id"]) == ["from b"]

    def test_an_unknown_session_id_starts_a_new_conversation(self):
        body = chat(client_for(*HELLO), session_id="0" * 32).json()
        assert body["new_session"] is True and body["session_id"] != "0" * 32

    def test_whitespace_is_trimmed(self):
        c = client_for(*HELLO)
        sid = chat(c, "   padded   ").json()["session_id"]
        agent = c.app.state.store.get_or_create(sid)[0].agent
        assert agent.messages[0]["content"] == "padded"


class TestValidation:
    @pytest.mark.parametrize("message", ["", "   ", "\n\t"])
    def test_empty_messages_are_rejected(self, message):
        r = chat(client_for(*HELLO), message)
        assert r.status_code == 422 and "empty" in r.json()["detail"]

    def test_over_long_messages_are_rejected_with_the_limit(self):
        r = chat(client_for(*HELLO, max_message_chars=50), "x" * 51)
        assert r.status_code == 422 and "50 characters" in r.json()["detail"]
        assert chat(client_for(*HELLO, max_message_chars=50), "x" * 50).status_code == 200

    def test_a_missing_or_wrongly_typed_message_is_a_422(self):
        c = client_for(*HELLO)
        assert c.post("/api/chat", json={}).status_code == 422
        assert c.post("/api/chat", json={"message": 123}).status_code == 422
        assert c.post("/api/chat", content="not json").status_code == 422

    def test_rejected_messages_do_not_create_conversations(self):
        c = client_for(*HELLO)
        chat(c, "   ")
        assert len(c.app.state.store) == 0


class TestLimits:
    def test_a_conversation_has_a_turn_limit(self):
        c = client_for(*HELLO, max_turns_per_session=2)
        sid = chat(c).json()["session_id"]
        assert chat(c, session_id=sid).status_code == 200
        r = chat(c, session_id=sid)
        assert r.status_code == 409 and r.json()["detail"]["code"] == "session_full"
        assert chat(c).status_code == 200  # a fresh conversation is fine

    def test_one_message_at_a_time_per_conversation(self):
        gate, started = threading.Event(), threading.Event()

        class Blocking:
            messages: list = []

            def ask(self, message, on_event=None):
                started.set()
                gate.wait(5)
                return CricketAgent(toolbox=StubToolbox(), client=FakeClient(*HELLO)).ask(message)

        app = create_app(settings(), toolbox=StubToolbox(), agent_factory=Blocking)
        c = TestClient(app)
        sid = {}

        def first():
            sid["r"] = chat(c, "slow")

        t = threading.Thread(target=first)
        t.start()
        assert started.wait(5)
        session_id = next(iter(app.state.store._sessions))
        r = chat(c, "impatient", session_id=session_id)
        assert r.status_code == 409 and r.json()["detail"]["code"] == "busy"
        gate.set()
        t.join(5)
        assert sid["r"].status_code == 200
        assert chat(c, "now fine", session_id=session_id).status_code in (200, 500)  # lock released

    def test_the_rate_limit_returns_429_with_retry_after(self):
        c = client_for(*HELLO, rate_limit_per_minute=2)
        assert [chat(c).status_code for _ in range(2)] == [200, 200]
        r = chat(c)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/api/health").status_code == 200  # only chat is limited

    def test_clients_are_limited_separately_when_a_proxy_is_trusted(self):
        c = client_for(*HELLO, rate_limit_per_minute=1, trust_proxy_headers=True)
        h = lambda ip: {"X-Forwarded-For": f"{ip}, 10.0.0.1"}  # noqa: E731
        assert c.post("/api/chat", json={"message": "a"}, headers=h("1.1.1.1")).status_code == 200
        assert c.post("/api/chat", json={"message": "a"}, headers=h("2.2.2.2")).status_code == 200
        assert c.post("/api/chat", json={"message": "a"}, headers=h("1.1.1.1")).status_code == 429

    def test_forwarded_headers_are_ignored_unless_trusted(self):
        c = client_for(*HELLO, rate_limit_per_minute=1)
        for ip in ("1.1.1.1", "2.2.2.2"):  # spoofing must not dodge the limit
            r = c.post("/api/chat", json={"message": "a"}, headers={"X-Forwarded-For": ip})
        assert r.status_code == 429

    def test_the_store_refuses_new_conversations_when_every_slot_is_busy(self):
        gate = threading.Event()

        class Blocking:
            def ask(self, message, on_event=None):
                gate.wait(5)
                raise RuntimeError("stop")

        app = create_app(settings(max_sessions=1), toolbox=StubToolbox(), agent_factory=Blocking)
        c = TestClient(app)
        t = threading.Thread(target=lambda: chat(c, "occupy"))
        t.start()
        for _ in range(100):
            if len(app.state.store):
                break
            threading.Event().wait(0.02)
        r = chat(c, "second")
        gate.set()
        t.join(5)
        assert r.status_code == 503 and "busy" in r.json()["detail"]


# ── access token ────────────────────────────────────────────────────────
class TestAccessToken:
    def test_chat_needs_the_token_when_one_is_set(self):
        c = client_for(*HELLO, access_token="s3cret")
        r = chat(c)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert chat(c, headers=None).status_code == 401
        assert c.post("/api/chat", json={"message": "x"},
                      headers={"Authorization": "Bearer wrong"}).status_code == 401

    def test_bearer_and_header_forms_both_work(self):
        c = client_for(*HELLO, access_token="s3cret")
        assert c.post("/api/chat", json={"message": "x"},
                      headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.post("/api/chat", json={"message": "x"},
                      headers={"X-Access-Token": "s3cret"}).status_code == 200

    def test_streaming_and_reset_are_guarded_too(self):
        c = client_for(*HELLO, access_token="s3cret")
        assert c.post("/api/chat/stream", json={"message": "x"}).status_code == 401
        assert c.delete("/api/chat/" + "0" * 32).status_code == 401

    def test_guessing_the_token_is_rate_limited(self):
        c = client_for(*HELLO, access_token="s3cret", rate_limit_per_minute=3)
        codes = [c.post("/api/chat", json={"message": "x"},
                        headers={"Authorization": f"Bearer guess{i}"}).status_code for i in range(5)]
        assert codes == [401, 401, 401, 429, 429]

    def test_the_page_stays_open_so_a_token_can_be_entered(self):
        assert client_for(*HELLO, access_token="s3cret").get("/").status_code == 200

    def test_the_token_is_never_echoed(self):
        c = client_for(*HELLO, access_token="s3cret-value")
        for r in (c.get("/api/info"), c.get("/api/health"), chat(c)):
            assert "s3cret-value" not in r.text


# ── errors ──────────────────────────────────────────────────────────────
def api_error(cls, status, message):
    req = httpx.Request("POST", "https://api.example/v1/messages")
    return cls(message, response=httpx.Response(status, request=req), body=None)


class Boom:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def ask(self, message, on_event=None):
        self.calls += 1
        if self.calls == 1:
            raise self.exc
        return CricketAgent(toolbox=StubToolbox(), client=FakeClient(*HELLO)).ask(message)


class TestErrors:
    CASES = [
        (api_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the API"),
         503, "out of credit"),
        (api_error(anthropic.AuthenticationError, 401, "invalid x-api-key sk-ant-SECRET"), 503, "credentials"),
        (api_error(anthropic.RateLimitError, 429, "slow down"), 503, "rate limited"),
        (api_error(anthropic.InternalServerError, 500, "upstream exploded"), 502, "provider returned an error"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), 502, "Could not reach"),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), 504, "too long"),
        (RuntimeError("secret internal detail /Users/someone/file.py"), 500, "Something went wrong"),
    ]

    @pytest.mark.parametrize("exc,status,fragment", CASES)
    def test_failures_become_clear_messages(self, exc, status, fragment):
        assert describe_error(exc)[0] == status and fragment in describe_error(exc)[1]

    @pytest.mark.parametrize("exc,status,fragment", CASES)
    def test_over_http_and_without_leaking_details(self, exc, status, fragment):
        app = create_app(settings(), toolbox=StubToolbox(), agent_factory=lambda: Boom(exc))
        r = chat(TestClient(app))
        assert r.status_code == status and fragment in r.json()["detail"]
        assert "SECRET" not in r.text and "/Users/" not in r.text and "upstream exploded" not in r.text

    def test_a_failed_turn_releases_the_conversation(self):
        app = create_app(settings(), toolbox=StubToolbox(),
                         agent_factory=lambda: Boom(RuntimeError("x")))
        c = TestClient(app)
        assert chat(c).status_code == 500
        sid = next(iter(app.state.store._sessions))
        assert chat(c, session_id=sid).status_code == 200  # the lock was released, and it recovers
        assert app.state.store.get_or_create(sid)[0].turns == 1  # the failed turn did not count

    def test_missing_credentials_are_reported_as_a_configuration_problem(self):
        def no_key():
            raise anthropic.AnthropicError("The api_key client option must be set")

        c = TestClient(create_app(settings(), toolbox=StubToolbox(), agent_factory=no_key))
        r = chat(c)
        assert r.status_code == 503 and "no API credentials" in r.json()["detail"]

    def test_the_api_key_in_the_environment_never_appears_in_any_response(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-SUPERSECRET")
        c = client_for(*HELLO)
        outputs = [c.get("/api/info").text, c.get("/api/health").text, chat(c).text,
                   c.get("/").text, c.get("/docs").text, c.get("/openapi.json").text]
        assert not any("SUPERSECRET" in o for o in outputs)
        assert c.get("/api/health").json()["credentials_in_environment"] is True


# ── streaming ───────────────────────────────────────────────────────────
def sse_events(response):
    events, event, data = [], None, []
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and event:
            events.append((event, json.loads("".join(data))))
            event, data = None, []
    return events


class TestStreaming:
    def stream(self, c, message="hi", **body):
        with c.stream("POST", "/api/chat/stream", json={"message": message, **body}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
            return sse_events(r)

    def test_events_arrive_in_order_ending_with_the_answer(self):
        c = client_for(reply(tool_use("a", "find_player", query="Kohli"), stop="tool_use"),
                       reply(text("final answer")))
        ev = self.stream(c)
        assert [e[0] for e in ev] == ["session", "step", "tool_call", "tool_result", "step",
                                      "answer", "done"]
        assert ev[0][1]["new_session"] is True and len(ev[0][1]["session_id"]) == 32
        assert ev[2][1]["name"] == "find_player" and ev[2][1]["arguments"] == {"query": "Kohli"}
        assert ev[3][1] == {"type": "tool_result", "name": "find_player", "ok": True}
        answer = ev[5][1]
        assert answer["answer"] == "final answer" and answer["turns"] == 1
        assert answer["tool_calls"][0]["name"] == "find_player"

    def test_a_streamed_turn_continues_the_same_conversation(self):
        c = client_for(*HELLO)
        sid = self.stream(c)[0][1]["session_id"]
        ev = self.stream(c, session_id=sid)
        assert ev[0][1] == {"session_id": sid, "new_session": False}
        assert ev[-2][1]["turns"] == 2

    def test_a_failure_is_an_error_event_then_done(self):
        app = create_app(settings(), toolbox=StubToolbox(),
                         agent_factory=lambda: Boom(api_error(anthropic.BadRequestError, 400,
                                                              "credit balance is too low")))
        ev = self.stream(TestClient(app))
        kinds = [e[0] for e in ev]
        assert kinds[-2:] == ["error", "done"] and "answer" not in kinds
        err = dict(ev)["error"]
        assert err["status"] == 503 and "out of credit" in err["message"]

    def test_problems_found_before_streaming_are_ordinary_http_errors(self):
        c = client_for(*HELLO, max_message_chars=10, access_token="t")
        h = {"Authorization": "Bearer t"}
        assert c.post("/api/chat/stream", json={"message": "x" * 11}, headers=h).status_code == 422
        assert c.post("/api/chat/stream", json={"message": "x"}).status_code == 401

    def test_a_broken_listener_is_not_possible_but_a_slow_turn_still_finishes(self):
        c = client_for(reply(tool_use("a", "find_player", query="k"), stop="tool_use"),
                       reply(text("ok")))
        assert self.stream(c)[-2][0] == "answer"


# ── reset ───────────────────────────────────────────────────────────────
class TestReset:
    def test_deleting_forgets_the_conversation(self):
        c = client_for(*HELLO)
        sid = chat(c).json()["session_id"]
        assert c.delete(f"/api/chat/{sid}").json() == {"deleted": True}
        assert c.delete(f"/api/chat/{sid}").json() == {"deleted": False}
        assert chat(c, session_id=sid).json()["new_session"] is True

    def test_a_malformed_id_is_a_404(self):
        assert client_for(*HELLO).delete("/api/chat/not-an-id").status_code == 404


# ── the page and headers ────────────────────────────────────────────────
class TestPage:
    def test_the_chat_page_is_served_with_a_strict_content_security_policy(self):
        r = client_for(*HELLO).get("/")
        assert r.status_code == 200 and "CricketMind" in r.text
        csp = r.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "script-src 'self'" in csp
        assert "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
        assert r.headers["cache-control"] == "no-cache"

    def test_the_page_uses_no_inline_script_style_or_handlers(self):
        html = client_for(*HELLO).get("/").text
        assert "<style" not in html and not any(
            f" {h}=" in html for h in ("onclick", "onload", "onerror", "onsubmit", "style"))
        import re
        assert all(re.search(r"src=", tag) for tag in re.findall(r"<script[^>]*>", html))

    def test_static_files_and_favicon(self):
        c = client_for(*HELLO)
        for path in ("/static/app.js", "/static/render.js", "/static/style.css"):
            assert c.get(path).status_code == 200, path
        r = c.get("/favicon.ico")
        assert r.status_code == 204 and r.content == b""

    def test_security_headers_on_api_responses(self):
        r = client_for(*HELLO).get("/api/info")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["cache-control"] == "no-store"

    def test_no_cors_headers_are_sent(self):
        r = client_for(*HELLO).get("/api/info", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers


# ── real tools through the HTTP layer ───────────────────────────────────
class TestRealToolsOverHttp:
    def test_a_scripted_model_calling_real_stat_tools_through_the_api(self, db):  # noqa: F811
        tb = Toolbox(db_path=db, model_dir="/nonexistent")
        c = client_for(
            # built by hand: the helper's own `name` parameter would clash with the tool's
            reply(NS(type="tool_use", id="a", name="batting_stats", input={"name": "p1"}),
                  tool_use("b", "leaderboard", stat="runs", top=2), stop="tool_use"),
            reply(text("p1 leads")), toolbox=tb)
        body = chat(c).json()
        assert body["answer"] == "p1 leads"
        assert [(t["name"], t["ok"]) for t in body["tool_calls"]] == [
            ("batting_stats", True), ("leaderboard", True)]
        agent = c.app.state.store.get_or_create(body["session_id"])[0].agent
        results = agent._client.requests[1]["messages"][-1]["content"]
        assert '"runs": 402' in results[0]["content"]  # the real tool result reached the model
