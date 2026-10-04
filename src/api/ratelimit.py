"""A small sliding-window rate limiter (per key, in memory)."""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    """Allow ``limit`` events per ``window`` seconds for each key (e.g. a client address)."""

    MAX_KEYS = 10_000  # bound memory if someone cycles through addresses

    def __init__(self, limit: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Record an attempt.  Returns (allowed, seconds_until_the_next_slot_if_refused)."""
        if self.limit <= 0:  # 0 disables limiting
            return True, 0.0
        now = self._clock()
        with self._lock:
            if len(self._hits) > self.MAX_KEYS:
                self._prune(now)
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(0.0, self.window - (now - q[0]))
            q.append(now)
            return True, 0.0

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]
