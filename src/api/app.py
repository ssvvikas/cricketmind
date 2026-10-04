"""FastAPI app: the cricket assistant over HTTP, plus the static chat page.

    POST   /api/chat            one question, one JSON answer (with the tools it used)
    POST   /api/chat/stream     the same, as server-sent events (progress while it works)
    DELETE /api/chat/{id}       forget a conversation
    GET    /api/info            examples, limits, tool list, whether a token is needed
    GET    /api/health          what is loaded / missing (no secrets)
    GET    /                    the chat page (frontend/)

Every chat message spends API credit, so the chat endpoints are guarded: a per-client rate limit
(checked first, so guessing the access token is throttled too), an optional access token, a
message-length cap, a cap on turns per conversation and on simultaneous conversations.  The model's
API key stays on the server and is never sent to the browser.
"""

import json
import math
import os
import queue
import secrets
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.api.ratelimit import RateLimiter
from src.api.sessions import SESSION_ID, SessionStore, StoreFull
from src.api.settings import EXAMPLES, ApiSettings
from src.llm.agent import CricketAgent
from src.llm.tools import TOOL_SPECS, Toolbox

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
KEEPALIVE_SECONDS = 15

# the page ships no inline script or style, so a strict policy costs nothing and limits the damage
# of any rendering bug
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def describe_error(exc: Exception) -> tuple[int, str]:
    """Map a failure to (HTTP status, message that is safe to show a user)."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the SDK is a dependency of the agent
        anthropic = None
    text = str(exc).lower()
    if anthropic is not None:
        if isinstance(exc, anthropic.AuthenticationError):
            return 503, "The assistant's API credentials were rejected. The server needs attention."
        if isinstance(exc, anthropic.RateLimitError):
            return 503, "The assistant is being rate limited by its provider. Try again shortly."
        if isinstance(exc, anthropic.BadRequestError) and "credit balance" in text:
            return 503, "The assistant's API account is out of credit. The server owner needs to top it up."
        if isinstance(exc, anthropic.APITimeoutError):
            return 504, "The assistant took too long to respond. Try again."
        if isinstance(exc, anthropic.APIConnectionError):
            return 502, "Could not reach the assistant's provider. Try again shortly."
        if isinstance(exc, anthropic.APIStatusError):
            return 502, "The assistant's provider returned an error. Try again shortly."
    if "api_key" in text or "api key" in text or "credentials" in text or "auth_token" in text:
        return 503, "The assistant is not configured: no API credentials were found on the server."
    logger.exception("Unhandled error while answering")
    return 500, "Something went wrong while answering. Please try again."


def create_app(settings: ApiSettings | None = None, toolbox: Toolbox | None = None,
               agent_factory: Callable[[], object] | None = None,
               clock: Callable[[], float] = time.monotonic) -> FastAPI:
    settings = settings or ApiSettings()
    toolbox = toolbox or Toolbox()
    limiter = RateLimiter(settings.rate_limit_per_minute, 60.0, clock)

    shared_client: dict = {}
    client_lock = threading.Lock()

    def default_factory():
        with client_lock:  # one provider client shared by every conversation
            if "c" not in shared_client:
                import anthropic

                shared_client["c"] = anthropic.Anthropic()
        return CricketAgent(toolbox=toolbox, client=shared_client["c"], model=settings.model,
                            refusal_fallbacks=settings.refusal_fallbacks)

    store = SessionStore(agent_factory or default_factory, settings.max_sessions,
                         settings.session_ttl_seconds, clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.warmup = "skipped"
        if settings.warmup_on_start:
            app.state.warmup = "running"

            def run():
                app.state.warmup = toolbox.warmup()
                logger.info(f"Warmup finished: {app.state.warmup}")

            threading.Thread(target=run, daemon=True, name="warmup").start()
        yield

    app = FastAPI(title="CricketMind", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.toolbox = settings, store, toolbox
    app.state.warmup = "skipped"

    # ── guards ──────────────────────────────────────────────────────
    def client_key(request: Request) -> str:
        if settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def presented_token(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return request.headers.get("x-access-token", "")

    def guard(request: Request) -> None:
        ok, retry = limiter.check(client_key(request))  # first: this also throttles token guessing
        if not ok:
            raise HTTPException(429, "Too many requests. Please wait a moment.",
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
        if settings.access_token and not secrets.compare_digest(
                presented_token(request).encode(), settings.access_token.encode()):
            raise HTTPException(401, "An access token is required.",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ── conversation turns ──────────────────────────────────────────
    def begin_turn(body: ChatRequest):
        message = body.message.strip()
        if not message:
            raise HTTPException(422, "The message must not be empty.")
        if len(message) > settings.max_message_chars:
            raise HTTPException(422, f"The message is too long (limit {settings.max_message_chars} "
                                     "characters).")
        try:
            session, created = store.get_or_create(body.session_id)
        except StoreFull:
            raise HTTPException(503, "The assistant is busy. Try again in a moment.") from None
        except Exception as exc:  # e.g. no API credentials when the agent is first built
            status, msg = describe_error(exc)
            raise HTTPException(status, msg) from None
        if session.turns >= settings.max_turns_per_session:
            raise HTTPException(409, detail={
                "code": "session_full",
                "message": "This conversation has reached its length limit. Start a new chat."})
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, detail={
                "code": "busy", "message": "This conversation is still answering your last message."})
        return session, created, message

    def run_turn(session, message: str, on_event=None):
        """Answer one message.  Releases the session lock when done, whatever happens."""
        started = time.monotonic()
        try:
            result = session.agent.ask(message, on_event=on_event)
            session.turns += 1
            store.touch(session)
            logger.info(f"chat session={session.id[:8]} turn={session.turns} "
                        f"tools={[c.name for c in result.tool_calls]} steps={result.steps} "
                        f"seconds={time.monotonic() - started:.1f}")
            return result
        finally:
            session.lock.release()

    def payload(session, created: bool, result) -> dict:
        return {
            "session_id": session.id, "new_session": created, "turns": session.turns,
            "answer": result.text, "stop_reason": result.stop_reason, "steps": result.steps,
            "tool_calls": [{"name": c.name, "arguments": c.arguments, "ok": "error" not in c.result}
                           for c in result.tool_calls],
            "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        }

    # ── endpoints ───────────────────────────────────────────────────
    @app.post("/api/chat", dependencies=[Depends(guard)])
    def chat(body: ChatRequest):
        session, created, message = begin_turn(body)
        try:
            result = run_turn(session, message)
        except Exception as exc:
            status, msg = describe_error(exc)
            raise HTTPException(status, msg) from None
        return payload(session, created, result)

    @app.post("/api/chat/stream", dependencies=[Depends(guard)])
    def chat_stream(body: ChatRequest):
        session, created, message = begin_turn(body)  # validation / 409 happen before streaming
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                result = run_turn(session, message, on_event=events.put)
                events.put({"type": "answer", **payload(session, created, result)})
            except Exception as exc:
                status, msg = describe_error(exc)
                events.put({"type": "error", "status": status, "message": msg})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True, name="chat-turn").start()

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def stream():
            yield sse("session", {"session_id": session.id, "new_session": created})
            while True:
                try:
                    item = events.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield sse(item["type"], item)
            yield sse("done", {})

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/api/chat/{session_id}", dependencies=[Depends(guard)])
    def reset(session_id: str):
        if not SESSION_ID.match(session_id):
            raise HTTPException(404, "No such conversation.")
        return {"deleted": store.delete(session_id)}

    @app.get("/api/info")
    def info():
        return {
            "model": settings.model, "auth_required": bool(settings.access_token),
            "max_message_chars": settings.max_message_chars,
            "max_turns_per_session": settings.max_turns_per_session,
            "examples": EXAMPLES,
            "tools": [{"name": s["name"], "description": s["description"].split(". ")[0].rstrip(".")}
                      for s in TOOL_SPECS],
        }

    @app.get("/api/health")
    def health():
        return {
            "status": "ok", "warmup": app.state.warmup, "conversations": len(store),
            # best effort: the SDK can also authenticate through a login profile
            "credentials_in_environment": bool(os.environ.get("ANTHROPIC_API_KEY")
                                               or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        }

    # ── the page ────────────────────────────────────────────────────
    index = FRONTEND_DIR / "index.html"

    @app.get("/", include_in_schema=False)
    def page():
        if not index.exists():
            return JSONResponse({"message": "The chat page is missing; use the /api endpoints."})
        return FileResponse(index, media_type="text/html",
                            headers={"Content-Security-Policy": PAGE_CSP, "Cache-Control": "no-cache"})

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)  # no body: a 204 must be empty

    return app
