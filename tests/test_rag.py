"""Match summaries and hybrid retrieval.

Fixture facts (6 IPL matches, season 2024; see tests/test_features.py): innings 1 is 72/3 in 24
legal balls (70 off the bat + 2 extras; p1 67 off 23 balls, out), innings 2 is 66/0 in 24 balls
(p3 63* off 22).  The fixture's winners are synthetic, so margins are *derived*: team1 wins by
72 - 66 = 6 runs (m1, m3, m5); when the chasing side is the winner it is "by 10 wickets" with 96
balls left (m2, m4, m6).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.llm.tools import Toolbox
from src.models.player_ratings import RatingBook, load_balls
from src.rag import corpus as C
from src.rag.retriever import MatchIndex, NoMatches
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)


@pytest.fixture(scope="module")
def corpus(db):  # noqa: F811
    return C.build_corpus(db)


@pytest.fixture(scope="module")
def index(corpus):
    return MatchIndex(corpus)


class TestCorpus:
    def test_one_row_per_match_with_derived_facts(self, corpus):
        assert len(corpus) == 6 and corpus["match_id"].is_unique
        m1 = corpus[corpus.match_id == 1].iloc[0]
        assert (m1.first_score, m1.first_wkts, m1.second_score, m1.second_wkts) == (72, 3, 66, 0)
        assert (m1.winner, m1.margin_type, m1.margin) == ("Team A", "runs", 6)
        assert m1.result == "Team A beat Team B by 6 runs"
        assert m1.top_first == "p1 67 (23) | p4 2/72" and m1.top_second.startswith("p3 63* (22)")

    def test_chasing_win_is_by_wickets_with_balls_left(self, corpus):
        m2 = corpus[corpus.match_id == 2].iloc[0]  # Team B (chasing) is the winner
        assert (m2.margin_type, m2.margin, m2.balls_left, m2.target) == ("wickets", 10, 96, 73)

    def test_tags_follow_the_documented_rules(self, corpus):
        tags = {int(r.match_id): set(r.tags) for r in corpus.itertuples()}
        assert tags[1] == {"low-scoring"}                   # 6-run win: not <= 5, not thumping
        assert tags[2] == {"thumping win", "low-scoring"}   # 10 wkts with 96 balls to spare
        assert not any("close finish" in t or "century" in t for t in tags.values())

    def test_text_is_clean(self, corpus):
        for t in corpus["text"]:
            assert not any(bad in t for bad in (" nan", "None", "NaN"))
        assert "Tags: " in corpus["text"].iloc[1]

    def test_every_tag_has_words_and_a_definition(self):
        assert set(C.TAG_WORDS) == set(C.TAG_DEFINITIONS) == set(C.TAGS)

    def test_null_winner_is_not_treated_as_a_winner(self, db, tmp_path):
        import shutil

        import duckdb

        copy = str(tmp_path / "nowinner.duckdb")
        shutil.copy(db, copy)
        con = duckdb.connect(copy)
        con.execute("UPDATE match_info SET winner = NULL WHERE match_id = 1")
        con.close()
        m1 = C.build_corpus(copy).query("match_id == 1").iloc[0]
        assert pd.isna(m1.winner) and "nan" not in m1.text.lower()
        assert m1.result.startswith("Team A v Team B")


def _mk(rows):
    """A tiny hand-made corpus with every column the index needs."""
    base = dict(date=pd.Timestamp("2024-01-01"), league="IPL", season="2024", venue="V",
                city=None, team1="A", team2="B", winner="A", result="A beat B", margin_type="runs",
                margin=10, first_team="A", first_score=150, first_wkts=5, second_team="B",
                second_score=140, second_wkts=8, second_balls=120, target=None, balls_left=None,
                player_of_match=None, centuries=[], five_wicket_hauls=[], fifties=0,
                notable_players=[], tags=[], top_first=None, top_second=None, text="plain")
    return pd.DataFrame([{**base, "match_id": i, **r} for i, r in enumerate(rows, start=1)])


class TestFilters:
    def test_league_team_opponent_venue_season(self, index):
        assert index.search(league="IPL", top=25)["matched_total"] == 6
        assert index.search(team="Team A", top=25)["matched_total"] == 5
        assert index.search(team="team a", opponent="Team C", top=25)["matched_total"] == 2
        assert index.search(venue="v", top=25)["matched_total"] == 6
        assert index.search(season="2024", top=25)["matched_total"] == 6

    def test_matched_total_counts_everything_even_when_top_is_small(self, index):
        r = index.search(tags=["thumping win"], top=1)
        assert r["matched_total"] == 3 and r["returned"] == 1

    def test_tags_are_and_filters_and_validated(self, index):
        assert index.search(tags=["low-scoring", "thumping win"])["matched_total"] == 3
        with pytest.raises(ValueError, match="unknown tag"):
            index.search(tags=["boring"])
        with pytest.raises(NoMatches):
            index.search(tags=["century"])

    def test_bad_season_and_sort(self, index):
        with pytest.raises(NoMatches, match="Available"):
            index.search(season="1999")
        with pytest.raises(ValueError, match="sort must be"):
            index.search(sort="loudest")

    def test_player_means_notable_except_with_century_or_five_for_tags(self):
        ix = MatchIndex(_mk([
            dict(centuries=["Other"], notable_players=["Kohli", "Other"], tags=["century"]),
            dict(centuries=["Kohli"], notable_players=["Kohli"], tags=["century"]),
            dict(five_wicket_hauls=["Bumrah"], notable_players=["Bumrah"], tags=["five-wicket haul"]),
            dict(notable_players=["Kohli"]),
        ]))
        assert ix.search(player="Kohli", top=25)["matched_total"] == 3           # notable in
        hits = ix.search(player="Kohli", tags=["century"], top=25)
        assert hits["matched_total"] == 1 and hits["hits"][0]["match_id"] == 2  # scored it
        assert ix.search(player="Bumrah", tags=["five-wicket haul"])["matched_total"] == 1
        with pytest.raises(NoMatches):
            ix.search(player="Kohli", tags=["five-wicket haul"])

    def test_edition_year_resolution(self):
        ix = MatchIndex(_mk([
            dict(season="2009/10", date=pd.Timestamp("2010-04-25")),
            dict(season="2009", date=pd.Timestamp("2009-05-24")),
            dict(season="2020/21", date=pd.Timestamp("2020-11-10"))]))
        got = lambda y: {h["season"] for h in ix.search(season=y, top=25)["hits"]}  # noqa: E731
        assert got("2010") == {"2009/10"} and got("2009") == {"2009"} and got("2020") == {"2020/21"}
        assert got("2009/10") == {"2009/10"}


class TestOrdering:
    def test_newest_and_oldest(self, index):
        assert index.search(sort="oldest", top=1)["hits"][0]["match_id"] == 1
        assert index.search(sort="newest", top=1)["hits"][0]["match_id"] == 6

    def test_closest_puts_runs_margins_before_a_10_wicket_win(self, index):
        order = [h["match_id"] for h in index.search(sort="closest", top=6)["hits"]]
        assert set(order[:3]) == {1, 3, 5} and set(order[3:]) == {2, 4, 6}

    def test_biggest_chase_only_lists_chases(self, index):
        r = index.search(sort="biggest_chase", top=25)
        assert r["matched_total"] == 3 and all(h["target"] for h in r["hits"])

    def test_biggest_win_only_lists_run_wins_largest_first(self):
        ix = MatchIndex(_mk([dict(margin=5), dict(margin=80), dict(margin=30),
                             dict(margin_type="wickets", margin=9)]))
        assert [h["margin"] for h in ix.search(sort="biggest_win")["hits"]] == [80, 30, 5]

    def test_highest_total(self):
        ix = MatchIndex(_mk([dict(first_score=150), dict(first_score=250), dict(second_score=210)]))
        assert [h["match_id"] for h in ix.search(sort="highest_total")["hits"]][:2] == [2, 3]


class TestTextRanking:
    def test_synonyms_in_the_text_make_paraphrases_findable(self):
        ix = MatchIndex(_mk([
            dict(text="IPL 2024: A beat B by 30 runs. Tags: thumping win one-sided crushing."),
            dict(text="IPL 2024: A beat B by 2 runs. Tags: close finish nail-biter thriller tight game."),
            dict(text="IPL 2024: A beat B by 12 runs. Tags: high-scoring run-fest big total."),
            dict(text="IPL 2024: A beat B by 9 runs."), dict(text="IPL 2024: A beat B by 8 runs.")]))
        r = ix.search("a real nail-biter thriller")
        assert r["hits"][0]["match_id"] == 2 and r["hits"][0]["relevance"] > 0
        assert ix.search("crushing one-sided win")["hits"][0]["match_id"] == 1

    def test_filters_apply_before_text_ranking(self):
        ix = MatchIndex(_mk([dict(league="BBL", text="thriller thriller thriller"),
                             dict(league="IPL", text="quiet game")]))
        assert [h["match_id"] for h in ix.search("thriller", league="IPL")["hits"]] == [2]


class TestPersistenceAndTool:
    def test_save_and_load_roundtrip(self, index, tmp_path):
        path = tmp_path / "idx.pkl"
        index.save(path)
        again = MatchIndex.load(path)
        a = index.search("low scoring", top=3)["hits"]
        b = again.search("low scoring", top=3)["hits"]
        assert [h["match_id"] for h in a] == [h["match_id"] for h in b]

    @pytest.fixture()
    def tb(self, db, index):  # noqa: F811
        box = Toolbox(db_path=db, model_dir="/nonexistent")
        box._book = RatingBook(load_balls(db), as_of="2030-01-01", halflife_years=None)
        box._teams = {"IPL": {"Team A", "Team B", "Team C"}}
        box._index = index
        return box

    def test_search_matches_tool(self, tb):
        r = tb.call("search_matches", {"team": "team a", "tags": ["thumping win"],
                                       "sort": "oldest", "top": 2})
        assert "error" not in r
        json.dumps(r)  # valid JSON: no numpy types, no NaN
        assert r["matched_total"] == 2 and r["returned"] == 2
        assert r["tag_definitions"]["thumping win"].startswith("won by 60+")
        assert any("Summaries are generated" in c for c in r["caveats"])
        assert all(h["first_score"] == 72 and h["target"] is None or h["target"] == 73
                   for h in r["hits"])  # NaN became None (valid JSON), numbers are plain ints

    def test_tool_errors_are_actionable(self, tb):
        assert "unknown tag" in tb.call("search_matches", {"tags": ["boring"]})["error"]
        assert "Could not identify team" in tb.call("search_matches", {"team": "Zzz"})["error"]
        assert "league must be" in tb.call("search_matches", {"league": "XYZ"})["error"]
        assert "Available" in tb.call("search_matches", {"season": "1999"})["error"]


class TestHybridRanking:
    """A fake embedder stands in for the model: text mentioning close/wire -> [1, 0], else [0, 1]."""

    @staticmethod
    def fake_embed(texts):
        return np.array([[1.0, 0.0] if any(w in t.lower() for w in ("close", "wire", "tight"))
                         else [0.0, 1.0] for t in texts], dtype=np.float32)

    @pytest.fixture()
    def corpus(self):
        return _mk([dict(text="IPL 2024: A beat B by 2 runs. Tags: close finish nail-biter"),
                    dict(text="IPL 2024: A beat B by 90 runs. Tags: thumping win crushing"),
                    dict(text="IPL 2024: A beat B by 40 runs. Tags: high-scoring run-fest"),
                    dict(text="IPL 2024: A beat B by 9 runs."), dict(text="IPL 2024: B beat A.")])

    def index(self, corpus, with_embeddings=True):
        emb = self.fake_embed(list(corpus["text"])) if with_embeddings else None
        return MatchIndex(corpus, emb, self.fake_embed if with_embeddings else None)

    def test_embeddings_rescue_a_paraphrase_the_keywords_cannot_see(self, corpus):
        ix = self.index(corpus)
        # no word of the query occurs in any document, so TF-IDF alone has nothing to go on
        kw = ix.search("went down to the wire", ranking="keyword")
        hy = ix.search("went down to the wire")
        assert kw["ranking"] == "keyword" and set(kw["hits"][0]) >= {"relevance"}
        assert hy["ranking"] == "hybrid" and hy["hits"][0]["match_id"] == 1

    def test_keywords_still_matter_in_the_blend(self, corpus):
        ix = self.index(corpus)
        r = ix.search("crushing win")  # embedding says nothing here; the words decide
        assert r["hits"][0]["match_id"] == 2

    def test_without_embeddings_it_says_keyword(self, corpus):
        r = self.index(corpus, with_embeddings=False).search("close finish")
        assert r["ranking"] == "keyword" and r["hits"][0]["match_id"] == 1

    def test_no_query_means_no_ranking_label(self, corpus):
        assert "ranking" not in self.index(corpus).search(sort="newest")

    def test_scores_do_not_depend_on_the_filters(self, corpus):
        ix = self.index(corpus)
        rel = lambda **f: {h["match_id"]: h["relevance"]  # noqa: E731
                           for h in ix.search("tight game", top=25, **f)["hits"]}
        unfiltered, filtered = rel(), rel(league="IPL", team="A")
        assert filtered and all(unfiltered[i] == filtered[i] for i in filtered)

    def test_pickle_drops_the_function_and_recovers_it_on_load(self, corpus, tmp_path, monkeypatch):
        ix = self.index(corpus)
        path = tmp_path / "i.pkl"
        ix.save(path)
        again = MatchIndex.load(path)
        assert again._embed_fn is None and again.embeddings is not None
        monkeypatch.setattr("src.rag.retriever.default_embed_fn", lambda: self.fake_embed)
        assert again.search("went down to the wire")["ranking"] == "hybrid"

    def test_missing_model_falls_back_to_keywords(self, corpus, monkeypatch):
        ix = MatchIndex(corpus, self.fake_embed(list(corpus["text"])), None)
        monkeypatch.setattr("src.rag.retriever.default_embed_fn", lambda: None)
        assert ix.search("close finish")["ranking"] == "keyword"

    def test_tool_warns_when_only_keywords_were_used(self, db, corpus):  # noqa: F811
        box = Toolbox(db_path=db, model_dir="/nonexistent")
        box._teams = {"IPL": {"A", "B"}}
        box._index = self.index(corpus, with_embeddings=False)
        r = box.call("search_matches", {"query": "close finish"})
        assert r["ranking"] == "keyword" and any("embedding model was unavailable" in c
                                                 for c in r["caveats"])
        box._index = self.index(corpus)
        r = box.call("search_matches", {"query": "close finish"})
        assert r["ranking"] == "hybrid" and not any("unavailable" in c for c in r["caveats"])
