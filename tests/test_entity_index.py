"""ChromaDB entity index: chunks, metadata, name-first filtering and semantic retrieval.

A small synthetic dataset stands in for the six feature tables, and a deterministic bag-of-words
embedder stands in for the real model, so nothing is downloaded and the tests are fast.
"""

import re
import zlib

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.rag import entity_index as EI


def fake_embed(texts):
    """Deterministic bag-of-words embedding: stop-words dropped, crude plural stemming, 512
    dimensions (crc32, so it is stable across processes).  It stands in for the real model."""
    out = []
    for t in texts:
        v = np.zeros(512, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w in ENGLISH_STOP_WORDS:
                continue
            v[zlib.crc32(w.rstrip("s").encode()) % 512] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(out, dtype=np.float32)


# ── synthetic feature tables ────────────────────────────────────────────
def bat(name, innings=100, runs=3000.0, balls=2200.0, out=80.0, avg=37.5, sr=136.0, **kw):
    r = dict(batter=name, innings=innings, total_runs=runs, total_balls=balls, times_out=out,
             average=avg, strike_rate=sr, dot_pct=30.0, boundary_pct=17.0, consistency_std=25.0,
             highest_score=110.0, runs_powerplay=900.0, runs_middle=1200.0, runs_death=900.0,
             balls_powerplay=700.0, balls_middle=1000.0, balls_death=500.0, sr_powerplay=128.6,
             sr_middle=120.0, sr_death=180.0, recent_avg=41.0, recent_sr=150.0, recent_innings=10,
             legacy_score=0.91)  # must never reach a chunk
    r.update(kw)
    return r


def bowl(name, **kw):
    r = dict(bowler=name, matches=80, total_balls=1800.0, total_runs_conceded=2300.0,
             total_wickets=90.0, economy=7.67, bowling_sr=20.0, bowling_avg=25.6, dot_pct=38.0,
             boundary_pct_conceded=15.0, balls_powerplay=500.0, balls_middle=700.0, balls_death=600.0,
             runs_conceded_powerplay=600.0, runs_conceded_middle=800.0, runs_conceded_death=900.0,
             wickets_powerplay=25.0, wickets_middle=35.0, wickets_death=30.0, econ_powerplay=7.2,
             econ_middle=6.9, econ_death=9.0, recent_econ=8.1, recent_wickets=9.0, recent_matches=10)
    r.update(kw)
    return r


def synthetic_tables(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    tables = {
        "batsman": pd.DataFrame([
            bat("V Kohli", 387, 13299.0, 9841.0, 313.0, 42.49, 135.14),
            bat("RG Sharma", 421, 11373.0), bat("TM Head", 107), bat("S Khan"), bat("AS Khan"),
            bat("Never Out", 6, 90.0, 60.0, 0.0, np.nan, 150.0)]),
        "bowler": pd.DataFrame([bowl("JJ Bumrah", matches=248), bowl("Sandeep Sharma"),
                                bowl("AS Khan")]),
        "team": pd.DataFrame([
            dict(team=t, league=lg, matches=m, wins=w, win_pct=round(w * 100 / m, 2), avg_score=165.0,
                 avg_run_rate=8.4, boundary_pct=17.0, avg_wickets_taken=5.5, avg_bowling_econ=8.2,
                 first_match=pd.Timestamp("2008-04-19"), last_match=pd.Timestamp("2026-05-21"))
            for t, lg, m, w in [("Chennai Super Kings", "IPL", 264, 148), ("Mumbai Indians", "IPL", 290, 160),
                                ("Royal Challengers Bengaluru", "IPL", 280, 130), ("India", "T20I", 264, 183)]]),
        "partnership": pd.DataFrame([
            dict(player_a="RG Sharma", player_b="V Kohli", times_paired=20, total_runs=900.0,
                 total_balls=600.0, avg_partnership_runs=45.0, run_rate=9.0, boundary_pct=18.0,
                 rotation_pct=40.0, best_partnership=140.0, fifty_plus_stands=6.0, century_stands=1.0)]),
        "fielding": pd.DataFrame([
            dict(fielder="MS Dhoni", total_dismissals=364.0, catches=210.0, caught_and_bowled=0.0,
                 run_outs=73.0, stumpings=81.0, dismissals_powerplay=80.0, dismissals_middle=161.0,
                 dismissals_death=123.0, matches_played=372, dismissals_per_match=0.978,
                 catches_per_match=0.565, run_outs_per_match=0.196, is_keeper=True),
            dict(fielder="V Kohli", total_dismissals=120.0, catches=110.0, caught_and_bowled=2.0,
                 run_outs=8.0, stumpings=0.0, dismissals_powerplay=20.0, dismissals_middle=60.0,
                 dismissals_death=40.0, matches_played=275, dismissals_per_match=0.44,
                 catches_per_match=0.41, run_outs_per_match=0.03, is_keeper=False)]),
        "matchup": pd.DataFrame([
            dict(bowler="JJ Bumrah", batter="V Kohli", balls_faced=108, runs_scored=159.0,
                 strike_rate=147.22, dots=39.0, dot_pct=36.11, boundaries=22.0, times_dismissed=5.0,
                 encounters=18),
            dict(bowler="Sandeep Sharma", batter="V Kohli", balls_faced=12, runs_scored=14.0,
                 strike_rate=116.67, dots=4.0, dot_pct=33.33, boundaries=1.0, times_dismissed=1.0,
                 encounters=3)]),
    }
    for name, df in tables.items():
        df.to_parquet(tmp_path / f"{name}_features.parquet", index=False)
    return tmp_path


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    return synthetic_tables(tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="module")
def built(processed, tmp_path_factory):
    persist = tmp_path_factory.mktemp("chroma")
    counts = EI.build_index(processed, persist, embed_fn=fake_embed)
    return persist, counts


def retrieve(built, q, **kw):
    return EI.retrieve(q, persist_dir=built[0], embed_fn=fake_embed, **kw)


# ── chunks ──────────────────────────────────────────────────────────────
class TestChunks:
    def test_one_chunk_per_row_and_matchups_below_the_threshold_are_dropped(self, processed):
        chunks = EI.build_chunks(processed)
        by_type = pd.Series([c["metadata"]["entity_type"] for c in chunks]).value_counts().to_dict()
        assert by_type == {"batsman": 6, "bowler": 3, "team": 4, "partnership": 1, "fielding": 2,
                           "matchup": 1}  # the 12-ball pair is under the 30-ball default
        assert sum(c["metadata"]["entity_type"] == "matchup"
                   for c in EI.build_chunks(processed, min_matchup_balls=None)) == 2

    def test_every_chunk_has_the_required_metadata(self, processed):
        for c in EI.build_chunks(processed):
            m = c["metadata"]
            assert m["dataset"] == "cricsheet_main"
            assert m["entity_type"] in EI.ENTITY_TYPES and m["entity_name"] and m["league"]
            assert isinstance(m["weight"], float)

    def test_league_is_real_for_teams_and_all_elsewhere(self, processed):
        chunks = EI.build_chunks(processed)
        assert {c["metadata"]["entity_name"]: c["metadata"]["league"] for c in chunks
                if c["metadata"]["entity_type"] == "team"} == {
            "Chennai Super Kings": "IPL", "Mumbai Indians": "IPL",
            "Royal Challengers Bengaluru": "IPL", "India": "T20I"}
        assert all(c["metadata"]["league"] == "all" for c in chunks
                   if c["metadata"]["entity_type"] != "team")

    def test_ids_are_unique(self, processed):
        ids = [c["id"] for c in EI.build_chunks(processed)]
        assert len(ids) == len(set(ids))

    def test_no_retired_scores_and_no_missing_value_words(self, processed):
        for c in EI.build_chunks(processed, min_matchup_balls=None):
            assert not re.search(r"legacy|chemistry|vulnerab|\bnan\b|\bNone\b|\binf\b", c["text"], re.I)

    def test_chunk_text_carries_the_figures(self, processed):
        text = {c["metadata"]["entity_name"] + c["metadata"]["entity_type"]: c["text"]
                for c in EI.build_chunks(processed)}
        k = text["V Kohlibatsman"]
        for bit in ("387 innings", "13,299 runs", "average 42.49", "strike rate 135.14",
                    "death overs 180.0", "Last 10 innings"):
            assert bit in k
        assert "Chennai Super Kings in the IPL: 264 matches, 148 wins (56.1% win rate)" in \
            text["Chennai Super Kingsteam"]
        assert "India in T20 internationals" in text["Indiateam"]
        assert "248 matches" in text["JJ Bumrahbowler"] and "death overs 9.00" in text["JJ Bumrahbowler"]

    def test_never_dismissed_batter_is_worded_not_nan(self, processed):
        t = next(c["text"] for c in EI.build_chunks(processed) if c["metadata"]["entity_name"] == "Never Out")
        assert "never dismissed" in t and "balls, average" not in t

    def test_fielding_says_it_is_involvement_not_skill_and_flags_keepers(self, processed):
        t = {c["metadata"]["entity_name"]: c["text"] for c in EI.build_chunks(processed)
             if c["metadata"]["entity_type"] == "fielding"}
        assert "wicketkeeper role" in t["MS Dhoni"] and "wicketkeeper" not in t["V Kohli"]
        assert all("not fielding quality" in x for x in t.values())

    def test_matchups_and_partnerships_are_worded_as_records_between_named_players(self, processed):
        by = {c["metadata"]["entity_type"]: c for c in EI.build_chunks(processed)
              if c["metadata"]["entity_type"] in ("matchup", "partnership")}
        assert by["matchup"]["metadata"]["entity_name"] == "V Kohli vs JJ Bumrah"
        assert "small-sample fact, not a forecast" in by["matchup"]["text"]
        assert by["partnership"]["metadata"]["entity_name"] == "RG Sharma & V Kohli"
        assert "RG Sharma and V Kohli batting together" in by["partnership"]["text"]

    def test_missing_tables_are_skipped(self, tmp_path):
        (tmp_path / "x").mkdir()
        pd.DataFrame([bat("V Kohli")]).to_parquet(tmp_path / "batsman_features.parquet")
        assert {c["metadata"]["entity_type"] for c in EI.build_chunks(tmp_path)} == {"batsman"}

    def test_chunks_are_short(self, processed):
        assert max(len(c["text"]) for c in EI.build_chunks(processed)) < 500


# ── building the collection ─────────────────────────────────────────────
class TestBuild:
    def test_build_reports_counts_and_stores_everything(self, built):
        persist, counts = built
        assert counts == {"batsman": 6, "team": 4, "bowler": 3, "fielding": 2, "partnership": 1,
                          "matchup": 1}
        assert EI._collection(persist).count() == 17

    def test_rebuilding_does_not_duplicate(self, processed, tmp_path):
        EI.build_index(processed, tmp_path, embed_fn=fake_embed)
        EI.build_index(processed, tmp_path, embed_fn=fake_embed)
        assert EI._collection(tmp_path).count() == 17

    def test_upsert_without_reset_is_idempotent(self, processed, tmp_path):
        EI.build_index(processed, tmp_path, embed_fn=fake_embed)
        EI.build_index(processed, tmp_path, embed_fn=fake_embed, reset=False)
        assert EI._collection(tmp_path).count() == 17

    def test_stored_metadata_is_queryable(self, built):
        got = EI._collection(built[0]).get(where={"entity_type": "team"}, include=["metadatas"])
        assert {m["league"] for m in got["metadatas"]} == {"IPL", "T20I"}
        assert all(m["dataset"] == "cricsheet_main" for m in got["metadatas"])

    def test_the_matchup_threshold_can_be_changed(self, processed, tmp_path):
        counts = EI.build_index(processed, tmp_path, embed_fn=fake_embed, min_matchup_balls=None)
        assert counts["matchup"] == 2

    def test_no_feature_tables_is_a_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="No feature tables"):
            EI.build_index(tmp_path / "empty", tmp_path / "db", embed_fn=fake_embed)

    def test_retrieving_before_building_is_a_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="not been built"):
            EI.retrieve("Kohli", persist_dir=tmp_path / "nothing", embed_fn=fake_embed)


# ── spotting names ──────────────────────────────────────────────────────
PLAYERS = {"V Kohli": 9841.0, "RG Sharma": 8000.0, "Sandeep Sharma": 3000.0, "TM Head": 2200.0,
           "S Khan": 2200.0, "AS Khan": 1500.0, "JJ Bumrah": 1800.0, "MS Dhoni": 372.0}
TEAMS = {"Chennai Super Kings": {"IPL"}, "India": {"T20I"}, "Royal Challengers Bengaluru": {"IPL"}}


def spot(q):
    return EI.spot_names(q, PLAYERS, TEAMS)


class TestSpotting:
    def test_surname_only(self):
        assert spot("How does Kohli bat in the death overs?")["players"] == ["V Kohli"]
        assert spot("kohli strike rate")["players"] == ["V Kohli"]

    def test_first_name_picks_one_of_several_players_with_the_surname(self):
        assert spot("What about Rohit Sharma?")["players"] == ["RG Sharma"]
        assert spot("and Sandeep Sharma?")["players"] == ["Sandeep Sharma"]

    def test_a_shared_surname_returns_every_candidate_most_played_first(self):
        assert spot("Sharma record")["players"] == ["RG Sharma", "Sandeep Sharma"]

    def test_initial_plus_surname_is_exact(self):
        assert spot("How does S Khan bowl?")["players"] == ["S Khan"]
        assert spot("How does AS Khan bowl?")["players"] == ["AS Khan"]

    def test_two_players_in_one_question(self):
        assert set(spot("Kohli against Bumrah")["players"]) == {"V Kohli", "JJ Bumrah"}

    def test_ordinary_words_are_not_surnames(self):
        for q in ("Who is the best batter?", "show me the most runs", "what is the average"):
            assert spot(q) == {"players": [], "teams": [], "ambiguous": []}

    def test_wordlike_surnames_need_a_capital_and_head_to_head_is_ignored(self):
        assert spot("What is the head to head record?")["players"] == []
        assert spot("head-to-head between them")["players"] == []
        assert spot("how does head bat")["players"] == []
        assert spot("How does Head bat?")["players"] == ["TM Head"]

    def test_teams_by_name_and_by_capitalised_abbreviation(self):
        assert spot("How have Chennai Super Kings done?")["teams"] == ["Chennai Super Kings"]
        assert spot("How is CSK doing?")["teams"] == ["Chennai Super Kings"]
        assert spot("RCB form")["teams"] == ["Royal Challengers Bengaluru"]
        assert spot("csk form")["teams"] == []  # lower case is ambiguous with ordinary words
        assert spot("India batting")["teams"] == ["India"]

    def test_candidates_are_capped(self):
        many = {f"A{c} Patel": float(i) for i, c in enumerate("BCDEFGHIJ")}
        assert len(EI.spot_names("Patel", many, {})["players"]) == EI.MAX_NAME_CANDIDATES


# ── retrieval ───────────────────────────────────────────────────────────
class TestRetrieve:
    def test_a_named_player_filters_first_then_ranks_semantically(self, built, processed):
        q = "How does Kohli bat in the death overs?"
        r = retrieve(built, q, top_k=10)
        assert r["filtered_by"] == ["V Kohli"] and r["name_filtered"] is True
        assert r["hits"] and all("V Kohli" in h["text"] for h in r["hits"])
        # the semantic step orders exactly the chunks that mention Kohli, closest first: recompute
        # that order independently from the same embeddings
        kohli = [c for c in EI.build_chunks(processed) if "V Kohli" in c["text"]]
        qv = fake_embed([q])[0]
        expected = sorted(kohli, key=lambda c: -float(fake_embed([c["text"]])[0] @ qv))
        assert [h["entity_name"] + h["entity_type"] for h in r["hits"]] == [
            c["metadata"]["entity_name"] + c["metadata"]["entity_type"] for c in expected]
        assert len(r["hits"]) == len(kohli) < 17  # everything else was filtered out first

    def test_hits_have_the_documented_fields_and_are_ranked(self, built):
        r = retrieve(built, "Kohli", top_k=4)
        assert len(r["hits"]) == 4
        assert [h["score"] for h in r["hits"]] == sorted((h["score"] for h in r["hits"]), reverse=True)
        h = r["hits"][0]
        assert {"text", "entity_type", "entity_name", "league", "score", "metadata"} <= set(h)
        assert h["metadata"]["dataset"] == "cricsheet_main"

    def test_entity_type_restricts_the_kind_of_chunk(self, built):
        r = retrieve(built, "Kohli", entity_type="fielding")
        assert [h["entity_type"] for h in r["hits"]] == ["fielding"]
        assert r["hits"][0]["entity_name"] == "V Kohli"
        assert {h["entity_type"] for h in retrieve(built, "Kohli", entity_type="partnership")["hits"]} \
            == {"partnership"}

    def test_a_named_team_finds_its_chunk(self, built):
        r = retrieve(built, "How have Chennai Super Kings done in the IPL?")
        assert r["filtered_by"] == ["Chennai Super Kings"]
        assert r["hits"][0]["entity_type"] == "team" and r["hits"][0]["league"] == "IPL"

    def test_abbreviation_and_two_names(self, built):
        assert retrieve(built, "How is CSK doing?")["filtered_by"] == ["Chennai Super Kings"]
        r = retrieve(built, "Kohli against Bumrah", top_k=10)
        assert {h["entity_type"] for h in r["hits"]} >= {"matchup"}
        assert r["hits"][0]["entity_name"] == "V Kohli vs JJ Bumrah"

    def test_a_shared_surname_searches_all_the_candidates(self, built):
        r = retrieve(built, "Sharma", top_k=10)
        assert set(r["filtered_by"]) == {"RG Sharma", "Sandeep Sharma"}
        assert {h["entity_name"] for h in r["hits"]} >= {"RG Sharma", "Sandeep Sharma"}

    def test_a_name_is_matched_as_a_whole_word(self, built):
        """Chroma's document filter is a substring test: 'S Khan' must not return 'AS Khan'."""
        r = retrieve(built, "How does S Khan bat?", top_k=10)
        assert r["filtered_by"] == ["S Khan"]
        assert r["hits"] and all(h["entity_name"] != "AS Khan" for h in r["hits"])
        assert {h["entity_name"] for h in r["hits"]} == {"S Khan"}

    def test_no_name_means_plain_semantic_search(self, built):
        r = retrieve(built, "team win rate in the IPL")
        assert r["filtered_by"] == [] and r["name_filtered"] is False and "note" not in r
        assert r["hits"][0]["entity_type"] == "team"

    def test_falls_back_and_says_so_when_the_named_entity_has_no_such_chunk(self, built):
        r = retrieve(built, "Kohli", entity_type="team")
        assert r["filtered_by"] == ["V Kohli"] and r["name_filtered"] is False
        assert "plain semantic" in r["note"]
        assert {h["entity_type"] for h in r["hits"]} == {"team"}

    def test_top_k_is_respected_and_capped_by_what_exists(self, built):
        assert len(retrieve(built, "batting", top_k=2)["hits"]) == 2
        assert len(retrieve(built, "batting", top_k=100)["hits"]) == 17

    def test_bad_arguments(self, built):
        with pytest.raises(ValueError, match="entity_type must be"):
            retrieve(built, "Kohli", entity_type="umpire")

    def test_the_result_is_plain_json(self, built):
        import json

        json.dumps(retrieve(built, "Kohli against Bumrah"))


# ── ambiguity is reported, not hidden ────────────────────────────────────
class TestAmbiguityReporting:
    def test_a_shared_surname_is_reported_as_ambiguous(self):
        assert spot("Sharma record")["ambiguous"] == ["Sharma"]

    def test_resolved_names_are_not_ambiguous(self):
        assert spot("Rohit Sharma")["ambiguous"] == []       # first name picks one
        assert spot("Kohli")["ambiguous"] == []              # only one Kohli
        assert EI.spot_names("Kohli", {"V Kohli": 9841.0, "T Kohli": 19.0}, {})["ambiguous"] == []

    def test_retrieve_passes_the_information_on(self, built):
        r = retrieve(built, "Sharma")
        assert r["ambiguous_surnames"] == ["Sharma"]
        assert set(r["named"]["players"]) == {"RG Sharma", "Sandeep Sharma"}
        assert retrieve(built, "Kohli")["ambiguous_surnames"] == []


# ── name spotting: dominance and possessives ─────────────────────────────
class TestSpottingRefinements:
    def test_an_obscure_namesake_does_not_hijack_the_obvious_player(self):
        players = {"V Kohli": 9841.0, "T Kohli": 19.0}
        assert EI.spot_names("Kohli", players, {})["players"] == ["V Kohli"]

    def test_comparable_namesakes_are_all_kept(self):
        players = {"V Kohli": 9841.0, "T Kohli": 5000.0}
        assert EI.spot_names("Kohli", players, {})["players"] == ["V Kohli", "T Kohli"]

    def test_possessives_still_name_the_entity(self):
        assert spot("What is Kohli's strike rate?")["players"] == ["V Kohli"]
        assert spot("Kohli’s form")["players"] == ["V Kohli"]  # curly apostrophe
        assert spot("What is CSK's win rate?")["teams"] == ["Chennai Super Kings"]


# ── what kind of chunk is being asked for ────────────────────────────────
class TestIntent:
    @pytest.mark.parametrize("q,expected", [
        ("Bumrah economy in the death overs", {"bowler"}),
        ("How many wickets has Chahal taken?", {"bowler"}),
        ("Kohli strike rate and average", {"batsman"}),
        ("How does Buttler bat in the powerplay?", {"batsman"}),
        ("How many catches has Dhoni taken?", {"fielding"}),
        ("Dhoni stumpings", {"fielding"}),
        ("Buttler and Salt partnership", {"partnership"}),
        ("Tell me about Kohli", set()),
    ])
    def test_wording_points_at_a_chunk_type(self, q, expected):
        assert EI.detect_intent(q) == expected

    def test_a_matchup_needs_two_named_players_and_a_versus_word(self):
        assert "matchup" in EI.detect_intent("Kohli against Bumrah", n_players=2)
        assert "matchup" in EI.detect_intent("Warner facing Bumrah", n_players=2)
        assert "matchup" not in EI.detect_intent("How does Kohli do against pace?", n_players=1)
        assert "matchup" not in EI.detect_intent("Kohli and Bumrah", n_players=2)

    def test_team_intent_from_a_named_team_or_the_word(self):
        assert "team" in EI.detect_intent("CSK record", n_teams=1)
        assert "team" in EI.detect_intent("which team wins most")
        assert "team" not in EI.detect_intent("Kohli record")


# ── the intent hint in retrieval ─────────────────────────────────────────
@pytest.fixture(scope="module")
def crowded(tmp_path_factory):
    """Kohli's batsman chunk buried under 40 short matchup chunks that mention his strike rate."""
    d = tmp_path_factory.mktemp("crowded")
    pd.DataFrame([bat("V Kohli", 387, 13299.0, 9841.0, 313.0, 42.49, 135.14)]).to_parquet(
        d / "batsman_features.parquet")
    pd.DataFrame([dict(bowler=f"Bowler{i}", batter="V Kohli", balls_faced=40 + i, runs_scored=50.0,
                       strike_rate=125.0, dots=15.0, dot_pct=35.0, boundaries=5.0,
                       times_dismissed=1.0, encounters=4) for i in range(40)]).to_parquet(
        d / "matchup_features.parquet")
    persist = tmp_path_factory.mktemp("crowded_db")
    EI.build_index(d, persist, embed_fn=fake_embed)
    return persist


class TestIntentHint:
    Q = "Kohli strike rate"

    def get(self, crowded, **kw):
        return EI.retrieve(self.Q, persist_dir=crowded, embed_fn=fake_embed, **kw)

    def test_without_the_hint_the_short_matchup_chunks_crowd_the_batsman_chunk_out(self, crowded):
        r = self.get(crowded, top_k=1, intent_boost=0.0)
        assert r["hits"][0]["entity_type"] == "matchup"  # the precondition for the next test

    def test_the_hint_fetches_and_promotes_the_right_chunk_type(self, crowded):
        r = self.get(crowded, top_k=1)
        assert r["intent"] == ["batsman"]
        assert r["hits"][0]["entity_type"] == "batsman" and r["hits"][0]["entity_name"] == "V Kohli"

    def test_score_is_similarity_plus_the_boost_and_boost_zero_disables_it(self, crowded):
        boosted = self.get(crowded, top_k=5)["hits"]
        for h in boosted:
            extra = EI.INTENT_BOOST if h["entity_type"] == "batsman" else 0.0
            assert h["score"] == pytest.approx(h["similarity"] + extra, abs=1e-3)
        assert [h["score"] for h in boosted] == sorted((h["score"] for h in boosted), reverse=True)
        plain = self.get(crowded, top_k=5, intent_boost=0.0)["hits"]
        assert all(h["score"] == h["similarity"] for h in plain)

    def test_an_explicit_entity_type_still_wins(self, crowded):
        r = self.get(crowded, top_k=3, entity_type="matchup")
        assert {h["entity_type"] for h in r["hits"]} == {"matchup"}

    def test_no_hint_words_means_no_change(self, built):
        a = EI.retrieve("Tell me about Kohli", persist_dir=built[0], embed_fn=fake_embed, top_k=5)
        b = EI.retrieve("Tell me about Kohli", persist_dir=built[0], embed_fn=fake_embed, top_k=5,
                        intent_boost=0.0)
        assert a["intent"] == [] and [h["entity_name"] for h in a["hits"]] == \
            [h["entity_name"] for h in b["hits"]]


# ── superlatives ────────────────────────────────────────────────────────
class TestSuperlativeWarning:
    def test_a_ranking_question_with_no_name_is_flagged(self, built):
        for q in ("Who scored the most runs?", "best wicketkeeper", "highest strike rate"):
            r = retrieve(built, q)
            assert "cannot rank by a statistic" in r["warning"] and "leaderboard" in r["warning"]

    def test_named_or_ordinary_questions_are_not_flagged(self, built):
        assert "warning" not in retrieve(built, "Kohli's highest score")
        assert "warning" not in retrieve(built, "team win rate in the IPL")


# ── regressions found by running the real index ──────────────────────────
class TestSpottingRegressions:
    PLAYERS = {**PLAYERS, "Zulfiqar Babar": 500.0, "Babar Azam": 6000.0, "Aziz Mohammad": 300.0,
               "Mohammad Rizwan": 5000.0, "A Sharma": 400.0, "AB Sharma": 300.0}

    def spot(self, q):
        return EI.spot_names(q, self.PLAYERS, {})

    def test_first_names_are_not_taken_for_other_players_surnames(self):
        """'Babar Azam' must not summon 'Zulfiqar Babar', nor 'Mohammad Rizwan' 'Aziz Mohammad'."""
        r = self.spot("What is the best stand Babar Azam and Mohammad Rizwan have shared?")
        assert set(r["players"]) == {"Babar Azam", "Mohammad Rizwan"} and r["ambiguous"] == []

    def test_a_filler_word_is_not_a_first_name(self):
        """'about Sharma' used to match only Sharmas whose initial is 'A' (via the word 'about')."""
        r = self.spot("What about Sharma?")
        assert set(r["players"]) == {"RG Sharma", "Sandeep Sharma", "A Sharma", "AB Sharma"}
        assert r["ambiguous"] == ["Sharma"]

    def test_capitalised_initials_that_are_also_english_words_still_count(self):
        assert self.spot("How does AS Khan bowl?")["players"] == ["AS Khan"]  # 'as' is a stop-word
        assert self.spot("How does S Khan bowl?")["players"] == ["S Khan"]

    def test_a_lower_case_full_first_name_still_pins_the_player(self):
        assert self.spot("mohammad rizwan form")["players"] == ["Mohammad Rizwan"]
        # a first name that matches nobody gives no evidence, so it falls back to the surname
        r = self.spot("rashid khan economy")  # no Rashid Khan in this dictionary
        assert set(r["players"]) == {"S Khan", "AS Khan"} and r["ambiguous"] == ["khan"]

    def test_initials_style_first_name_still_works(self):
        assert self.spot("What about Rohit Sharma?")["players"] == ["RG Sharma"]
        assert self.spot("Virat Kohli's form")["players"] == ["V Kohli"]
