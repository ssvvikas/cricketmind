"""Session store, rate limiter and settings (the API's supporting pieces)."""

import threading

import pytest

from src.api.ratelimit import RateLimiter
from src.api.sessions import SESSION_ID, SessionStore, StoreFull
from src.api.settings import ApiSettings


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestSessionStore:
    def make(self, max_sessions=3, ttl=60.0):
        clock = Clock()
        made = []

        def factory():
            made.append(object())
            return made[-1]

        return SessionStore(factory, max_sessions, ttl, clock), clock, made

    def test_new_session_has_a_valid_random_id_and_its_own_agent(self):
        store, _, made = self.make()
        a, created_a = store.get_or_create(None)
        b, _ = store.get_or_create(None)
        assert created_a and SESSION_ID.match(a.id) and a.id != b.id and len(made) == 2
        assert a.agent is not b.agent

    def test_same_id_returns_the_same_session(self):
        store, _, made = self.make()
        a, _ = store.get_or_create(None)
        again, created = store.get_or_create(a.id)
        assert again is a and not created and len(made) == 1

    @pytest.mark.parametrize("bad", ["", "nope", "../../etc", "A" * 32, "g" * 32, "0" * 31, None])
    def test_unknown_or_malformed_ids_start_a_fresh_session(self, bad):
        store, _, _ = self.make()
        s, created = store.get_or_create(bad)
        assert created and SESSION_ID.match(s.id)

    def test_idle_sessions_expire(self):
        store, clock, _ = self.make(ttl=60)
        a, _ = store.get_or_create(None)
        clock.t += 61
        again, created = store.get_or_create(a.id)
        assert created and again.id != a.id

    def test_using_a_session_keeps_it_alive(self):
        store, clock, _ = self.make(ttl=60)
        a, _ = store.get_or_create(None)
        for _ in range(5):
            clock.t += 40
            s, created = store.get_or_create(a.id)
            assert s is a and not created

    def test_a_busy_session_never_expires_mid_answer(self):
        store, clock, _ = self.make(ttl=60)
        a, _ = store.get_or_create(None)
        a.lock.acquire()
        clock.t += 600
        assert store.sweep() == 0 and store.get_or_create(a.id)[0] is a
        a.lock.release()
        clock.t += 600
        assert store.sweep() == 1

    def test_the_cap_evicts_the_least_recently_used_idle_session(self):
        store, clock, _ = self.make(max_sessions=2)
        a, _ = store.get_or_create(None)
        clock.t += 1
        b, _ = store.get_or_create(None)
        clock.t += 1
        store.get_or_create(a.id)  # a is now more recent than b
        c, _ = store.get_or_create(None)
        assert len(store) == 2
        # a (recently used) and c (new) survived; check them first, because looking up the
        # evicted b creates a replacement and would itself evict something
        assert store.get_or_create(a.id) == (a, False)
        assert store.get_or_create(c.id) == (c, False)
        assert store.get_or_create(b.id)[1] is True  # b was the one evicted

    def test_when_every_session_is_busy_new_ones_are_refused(self):
        store, _, _ = self.make(max_sessions=2)
        for _ in range(2):
            store.get_or_create(None)[0].lock.acquire()
        with pytest.raises(StoreFull):
            store.get_or_create(None)

    def test_delete(self):
        store, _, _ = self.make()
        a, _ = store.get_or_create(None)
        assert store.delete(a.id) is True and store.delete(a.id) is False and len(store) == 0

    def test_concurrent_creation_respects_the_cap(self):
        store = SessionStore(object, max_sessions=5, ttl_seconds=60)
        barrier = threading.Barrier(16)

        def go():
            barrier.wait()
            store.get_or_create(None)

        threads = [threading.Thread(target=go) for _ in range(16)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(store) == 5


class TestRateLimiter:
    def test_allows_up_to_the_limit_then_refuses_with_a_retry_time(self):
        clock = Clock()
        rl = RateLimiter(3, 60, clock)
        assert [rl.check("a")[0] for _ in range(3)] == [True, True, True]
        ok, retry = rl.check("a")
        assert ok is False and 0 < retry <= 60

    def test_keys_are_independent(self):
        rl = RateLimiter(1, 60, Clock())
        assert rl.check("a")[0] and rl.check("b")[0] and not rl.check("a")[0]

    def test_slots_free_up_as_the_window_slides(self):
        clock = Clock()
        rl = RateLimiter(2, 60, clock)
        rl.check("a")
        clock.t += 30
        rl.check("a")
        assert not rl.check("a")[0]
        clock.t += 31  # the first hit is now 61s old
        assert rl.check("a")[0] and not rl.check("a")[0]

    def test_retry_after_counts_down_to_the_oldest_hit(self):
        clock = Clock()
        rl = RateLimiter(1, 60, clock)
        rl.check("a")
        clock.t += 45
        assert rl.check("a")[1] == pytest.approx(15)

    def test_zero_disables_limiting(self):
        rl = RateLimiter(0, 60, Clock())
        assert all(rl.check("a")[0] for _ in range(1000))

    def test_memory_is_bounded(self):
        clock = Clock()
        rl = RateLimiter(5, 60, clock)
        rl.MAX_KEYS = 100
        for i in range(150):
            rl.check(f"ip{i}")
        clock.t += 120
        rl.check("fresh")
        assert len(rl._hits) <= 101


class TestSettings:
    def test_safe_defaults(self):
        s = ApiSettings()
        assert s.host == "127.0.0.1" and s.access_token is None and s.trust_proxy_headers is False
        assert s.max_message_chars == 1000 and s.max_turns_per_session == 20

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("CRICKETMIND_API_ACCESS_TOKEN", "s3cret")
        monkeypatch.setenv("CRICKETMIND_API_PORT", "9001")
        monkeypatch.setenv("CRICKETMIND_API_RATE_LIMIT_PER_MINUTE", "3")
        s = ApiSettings()
        assert (s.access_token, s.port, s.rate_limit_per_minute) == ("s3cret", 9001, 3)
