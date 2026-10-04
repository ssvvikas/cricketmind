"""One Toolbox / one search index serves many simultaneous web requests."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src.llm.tools import Toolbox
from src.models.player_ratings import RatingBook, load_balls
from src.rag import corpus as C
from src.rag.retriever import MatchIndex
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)
from tests.test_rag import TestHybridRanking, _mk

THREADS, ROUNDS = 8, 40


def hammer(fn, n_threads=THREADS):
    """Run fn(thread_no) in n_threads threads released together; return all results."""
    barrier = threading.Barrier(n_threads)

    def run(i):
        barrier.wait()
        return fn(i)

    with ThreadPoolExecutor(n_threads) as pool:
        return list(pool.map(run, range(n_threads)))


class TestToolboxThreads:
    @pytest.fixture()
    def tb(self, db):  # noqa: F811
        box = Toolbox(db_path=db, model_dir="/nonexistent")
        box._book = RatingBook(load_balls(db), as_of="2030-01-01", halflife_years=None)
        return box

    def test_stat_tools_give_the_same_answers_under_concurrent_calls(self, tb):
        want_bat = tb.call("batting_stats", {"name": "p1"})
        want_board = tb.call("leaderboard", {"stat": "runs", "top": 3})
        want_team = tb.call("team_record", {"team": "Team A", "league": "IPL"})

        def work(_):
            out = []
            for _ in range(ROUNDS):
                out.append((tb.call("batting_stats", {"name": "p1"}) == want_bat,
                            tb.call("leaderboard", {"stat": "runs", "top": 3}) == want_board,
                            tb.call("team_record", {"team": "Team A", "league": "IPL"}) == want_team))
            return out

        results = [x for r in hammer(work) for x in r]
        assert len(results) == THREADS * ROUNDS and all(all(t) for t in results)

    def test_each_thread_gets_its_own_cursor_and_the_same_thread_reuses_it(self, tb):
        cursors = hammer(lambda _: (tb.con, tb.con))
        assert all(a is b for a, b in cursors)
        assert len({id(a) for a, _ in cursors}) == THREADS

    def test_a_lazy_resource_is_loaded_once_even_when_requested_concurrently(self, db, monkeypatch):  # noqa: F811
        calls = []

        class Slow(RatingBook):
            def __init__(self, *a, **k):
                calls.append(1)
                time.sleep(0.2)  # long enough for every thread to arrive
                super().__init__(*a, **k)

        monkeypatch.setattr("src.llm.tools.RatingBook", Slow)
        box = Toolbox(db_path=db, model_dir="/nonexistent")
        books = hammer(lambda _: box.book)
        assert len(calls) == 1 and all(b is books[0] for b in books)

    def test_player_names_are_computed_once(self, tb):
        names = hammer(lambda _: tb._player_names())
        assert all(n is names[0] for n in names)


class TestWarmup:
    def test_reports_each_resource_and_never_raises(self, db, tmp_path):  # noqa: F811
        box = Toolbox(db_path=db, model_dir=tmp_path / "no_model", entity_dir=tmp_path / "no_index",
                      entity_embed_fn=TestHybridranking_fake)
        box._book = RatingBook(load_balls(db), as_of="2030-01-01", halflife_years=None)
        status = box.warmup()
        assert set(status) == {"ratings", "fielding", "teams", "match_search", "entity_search",
                               "match_model"}
        assert status["ratings"] is True and status["teams"] is True and status["fielding"] is True
        assert status["match_search"] is True  # built in memory from the fixture database
        assert status["match_model"] is False
        assert "not been built" in status["entity_search"]  # reported as text, not raised


TestHybridranking_fake = TestHybridRanking.fake_embed


class TestSearchIndexThreads:
    @pytest.fixture()
    def index(self):
        emb = TestHybridRanking.fake_embed
        corpus = _mk([dict(text="close finish nail-biter thriller tight game", result="A"),
                      dict(text="thumping win one-sided crushing comfortable", result="B"),
                      dict(text="high-scoring run-fest big total fireworks", result="C"),
                      dict(text="low-scoring bowlers dominated defensive", result="D"),
                      dict(text="rain-affected duckworth-lewis reduced overs", result="E")])
        return MatchIndex(corpus, emb(list(corpus["text"])), emb)

    def test_a_search_leaves_no_per_query_state_on_the_shared_index(self, index):
        before = set(vars(index))
        index.search("nail-biter thriller")
        index.search(sort="newest")
        assert set(vars(index)) == before  # nothing like last_ranking / _last_scores

    def test_concurrent_searches_never_swap_each_others_results(self, index):
        queries = ["nail-biter thriller", "crushing one-sided win", "run-fest fireworks",
                   "defensive low scoring", "duckworth lewis rain", "tight game"]
        expected = {q: [(h["match_id"], h["relevance"]) for h in index.search(q, top=5)["hits"]]
                    for q in queries}
        expected_rank = {q: index.search(q)["ranking"] for q in queries}

        def work(i):
            bad = 0
            for k in range(ROUNDS):
                q = queries[(i + k) % len(queries)]
                r = index.search(q, top=5)
                bad += [(h["match_id"], h["relevance"]) for h in r["hits"]] != expected[q]
                bad += r["ranking"] != expected_rank[q]
            return bad

        assert sum(hammer(work)) == 0
