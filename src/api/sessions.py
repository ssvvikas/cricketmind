"""In-memory conversations: one ``CricketAgent`` per session, with expiry and a hard cap.

A session holds the whole message history, so it is the unit of memory *and* of cost: it expires
when idle, and the store never holds more than ``max_sessions`` (least recently used idle ones are
dropped first; if every session is busy, new ones are refused).
"""

import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


class StoreFull(RuntimeError):
    """Every slot is occupied by a conversation that is answering right now."""


@dataclass
class Session:
    id: str
    agent: object
    lock: threading.Lock = field(default_factory=threading.Lock)  # held while answering
    turns: int = 0
    last_used: float = 0.0


class SessionStore:
    def __init__(self, factory: Callable[[], object], max_sessions: int, ttl_seconds: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory, self._max, self._ttl, self._clock = factory, max_sessions, ttl_seconds, clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _expired(self, s: Session, now: float) -> bool:
        return now - s.last_used > self._ttl and not s.lock.locked()

    def sweep(self) -> int:
        """Drop idle, expired sessions; returns how many were removed."""
        now = self._clock()
        with self._lock:
            dead = [k for k, s in self._sessions.items() if self._expired(s, now)]
            for k in dead:
                del self._sessions[k]
            return len(dead)

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        """(session, created).  An unknown, malformed or expired id yields a fresh session."""
        now = self._clock()
        with self._lock:
            for k in [k for k, s in self._sessions.items() if self._expired(s, now)]:
                del self._sessions[k]
            if session_id and SESSION_ID.match(session_id) and session_id in self._sessions:
                s = self._sessions[session_id]
                s.last_used = now
                self._sessions.move_to_end(session_id)
                return s, False
            while len(self._sessions) >= self._max:  # make room: oldest idle first
                victim = next((k for k, s in self._sessions.items() if not s.lock.locked()), None)
                if victim is None:
                    raise StoreFull("too many conversations are answering right now")
                del self._sessions[victim]
            s = Session(id=uuid.uuid4().hex, agent=self._factory(), last_used=now)
            self._sessions[s.id] = s
            return s, True

    def touch(self, session: Session) -> None:
        session.last_used = self._clock()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
