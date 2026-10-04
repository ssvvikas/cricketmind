# Web app

The API, the chat page, the spend guards and what was verified. Deployment is in [DEPLOYMENT.md](DEPLOYMENT.md). Back to the [README](../README.md).

## Web app

One FastAPI process serves the assistant over HTTP and a chat page (`frontend/`, plain HTML/CSS/JS,
no build step).

```bash
pip install -e ".[serve]"
export ANTHROPIC_API_KEY=...               # stays on the server; never sent to the browser
python -m src.api                          # http://127.0.0.1:8000  (data loads in ~10 s)
```

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | one question, one JSON answer (with the tools it used and their outcome) |
| `POST /api/chat/stream` | the same as server-sent events: `session`, `step`, `tool_call`, `tool_result`, `answer` / `error`, `done` |
| `DELETE /api/chat/{id}` | forget a conversation |
| `GET /api/info`, `/api/health` | examples, limits and tool list; what is loaded or missing (no secrets) |

The page streams progress while the assistant works ("Building a leaderboard", with a tick when
each tool finishes), renders answers with a small markdown renderer that escapes everything first
and can only emit a fixed set of tags (no links, images or attributes), and shows a "How I got
this" list of the exact tool calls behind every answer.

**Every message spends API credit, so the chat endpoints are guarded.** Settings come from
environment variables (prefix `CRICKETMIND_API_`):

| Setting | Default | Meaning |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | loopback only until you decide otherwise |
| `ACCESS_TOKEN` | unset | when set, chat needs `Authorization: Bearer ...` (the page prompts for it). **Set this before exposing the server** |
| `RATE_LIMIT_PER_MINUTE` | 10 | per client address; checked first, so guessing the token is throttled too |
| `MAX_MESSAGE_CHARS` | 1000 | per message |
| `MAX_TURNS_PER_SESSION` | 20 | bounds context size and spend per conversation |
| `MAX_SESSIONS` / `SESSION_TTL_SECONDS` | 200 / 1800 | idle conversations are forgotten; busy ones are never evicted |
| `TRUST_PROXY_HEADERS` | false | honour `X-Forwarded-For` (only behind your own proxy; otherwise it can be spoofed) |

One message at a time per conversation (a second gets `409`); provider failures become clear
messages (`503` "out of credit", "rate limited", ...) with no internals leaked; the page ships with
a strict Content-Security-Policy and no inline script or style.

**Concurrency.** One `Toolbox` serves every request. Making that safe turned up real bugs, each now
covered by a test that fails without the fix: DuckDB connections are not thread-safe (stat tools
returned wrong results under load, so every thread gets its own cursor), lazy loads raced (now
locked), `MatchIndex` kept per-query state on the shared object (searches could swap results),
and tied leaderboard rows came back in a different order from run to run (ties now break by
name). A real `uvicorn` process on the real data answered 18 concurrent requests with answers
identical to the sequential ones.

**What was verified, and what was not.** 400+ new tests: the API with a scripted fake model,
the page's JavaScript under Node (an XSS payload battery, a ReDoS check, the stream parser at
every split point of a real server stream), and a real browser session driving the page: example
clicks, live progress, the expandable tool trace, "New chat", the access-token form, the rate-limit
message, a 390-pixel phone layout, and WCAG AA contrast in both themes. **Not yet run: a real
model behind the API** (the stand-in picked the tool calls), and typing from a keyboard in a
browser (the test browser could not deliver key presses; the Enter path was verified by script).
Conversations live in server memory, so restart loses them and you should run a single worker.
