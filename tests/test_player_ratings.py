"""Player ratings: par adjustment, shrinkage, recency, leakage, matchups, pools."""

import shutil

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.models.player_ratings import (
    RatingBook,
    batter_ratings,
    bowler_ratings,
    compute_par,
    load_balls,
)
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)

EPOCH = pd.Timestamp("2000-01-01")


def mk(rows):
    """Build a ball table in the shape load_balls() returns.

    Each row: batter, bowler, runs, [out, phase, date, pool, match_id].
    """
    recs = []
    for i, r in enumerate(rows):
        date = pd.Timestamp(r.get("date", "2022-06-01"))
        out = bool(r.get("out", False))
        phase = r.get("phase", 1)
        recs.append({
            "match_id": r.get("match_id", i // 6), "date": date, "league": "IPL",
            "batting_team": "A", "bowling_team": "B",
            "over_number": {0: 2, 1: 10, 2: 18}[phase], "batter": r["batter"],
            "bowler": r["bowler"], "runs_batter": r["runs"], "bowler_runs": r["runs"],
            "is_faced": True, "is_legal": True, "is_bowler_wicket": out, "batter_out": out,
            "year": date.year, "pool": r.get("pool", "club"), "phase": phase,
            "days": (date - EPOCH).days, "bowler_wkt": out,
        })
    return pd.DataFrame(recs)


def balls_for(batter, bowler, n, avg_runs, **kw):
    """n balls scoring avg_runs on average (mixing 0s and 4s / 6s deterministically)."""
    pattern = {0.0: [0], 1.0: [0, 0, 0, 4], 2.0: [0, 4], 3.0: [0, 6, 0, 6, 0, 0]}
    if avg_runs not in pattern:
        raise ValueError(avg_runs)
    p = pattern[avg_runs]
    return [dict(batter=batter, bowler=bowler, runs=p[i % len(p)], **kw) for i in range(n)]


@pytest.fixture()
def league():
    """Par ~1.0 run/ball: hitter (2.0), average (1.0), slow (0.0); leaky (2.0) vs tight (0.0)."""
    rows = (
        balls_for("hitter", "leaky", 300, 2.0) + balls_for("avg", "avg_bowl", 300, 1.0)
        + balls_for("slow", "tight", 300, 0.0)
        + balls_for("avg2", "leaky", 300, 1.0) + balls_for("avg3", "tight", 300, 1.0)
    )
    return mk(rows), pd.Timestamp("2023-01-01")


class TestBatterRatings:
    def test_sign_and_order(self, league):
        balls, t = league
        r = batter_ratings(balls, t).set_index("batter")["runs_per_ball"]
        assert r["hitter"] > 0 > r["slow"]
        assert r["hitter"] > r["avg"] > r["slow"]

    def test_uses_only_balls_before_as_of(self, league):
        balls, t = league
        base = batter_ratings(balls, t).set_index("batter")["runs_per_ball"]
        future = mk(balls_for("hitter", "leaky", 500, 0.0, date="2023-02-01")
                    + balls_for("newbie", "leaky", 100, 3.0, date="2023-03-01"))
        more = pd.concat([balls, future], ignore_index=True)
        after = batter_ratings(more, t).set_index("batter")["runs_per_ball"]
        pd.testing.assert_series_equal(base.sort_index(), after.sort_index())
        assert "newbie" not in after.index

    def test_a_ball_on_the_as_of_date_is_excluded(self):
        b = mk(balls_for("x", "y", 50, 1.0, date="2023-01-01"))
        assert batter_ratings(b, "2023-01-01").empty

    def test_shrinkage_pulls_small_samples_toward_par(self):
        def rating(n):
            # a big filler pool keeps par ~1.0 no matter how many balls the star adds
            rows = balls_for("star", "y", n, 2.0) + balls_for("filler", "y", 60_000, 1.0)
            return batter_ratings(mk(rows), "2023-01-01").set_index("batter").loc["star", "runs_per_ball"]
        assert 0 < rating(30) < rating(300) < rating(3000)

    def test_recency_weighting_favours_current_form(self):
        rows = (balls_for("form", "y", 400, 3.0, date="2019-06-01")
                + balls_for("form", "y", 400, 0.0, date="2022-06-01")
                + balls_for("filler", "y", 800, 1.0, date="2022-06-01"))
        b = mk(rows)
        flat = batter_ratings(b, "2023-01-01").set_index("batter").loc["form", "runs_per_ball"]
        recent = batter_ratings(b, "2023-01-01", halflife_years=1).set_index("batter").loc["form", "runs_per_ball"]
        assert recent < flat

    def test_par_is_phase_specific(self):
        # powerplay par ~2.0, death par ~1.0; scoring 1.5 is below par early, above late
        rows = (balls_for("pp_a", "y", 500, 2.0, phase=0) + balls_for("dh_a", "y", 500, 1.0, phase=2)
                + [dict(batter="probe_pp", bowler="y", runs=r, phase=0) for r in [0, 3] * 100]
                + [dict(batter="probe_dh", bowler="y", runs=r, phase=2) for r in [0, 3] * 100])
        r = batter_ratings(mk(rows), "2023-01-01").set_index("batter")["runs_per_ball"]
        assert r["probe_pp"] < 0 < r["probe_dh"]

    def test_par_rows_exist_per_pool_year_phase(self, league):
        par = compute_par(league[0])
        assert {"bat_rpb", "out_rate", "bowl_rpb", "wkt_rate"} <= set(par.columns)
        assert ("club", 2022, 1) in par.index


class TestBowlerRatings:
    def test_sign_and_order(self, league):
        balls, t = league
        r = bowler_ratings(balls, t).set_index("bowler")["saved_per_ball"]
        assert r["tight"] > 0 > r["leaky"]

    def test_wickets_and_runs_are_separate(self):
        rows = ([dict(batter="b", bowler="wicket_taker", runs=2, out=(i % 5 == 0)) for i in range(400)]
                + [dict(batter="b", bowler="miser", runs=int(i % 4 == 0), out=False) for i in range(400)])
        r = bowler_ratings(mk(rows), "2023-01-01").set_index("bowler")
        assert r.loc["wicket_taker", "wkt_per_ball"] > r.loc["miser", "wkt_per_ball"]
        assert r.loc["miser", "saved_per_ball"] > r.loc["wicket_taker", "saved_per_ball"]


class TestRatingBook:
    @pytest.fixture()
    def book(self, league):
        return RatingBook(league[0], as_of=league[1], halflife_years=None)

    def test_matchup_is_additive_in_the_right_direction(self, book):
        strong_vs_weak = book.expected("hitter", "leaky", "club", "middle")
        strong_vs_strong = book.expected("hitter", "tight", "club", "middle")
        weak_vs_weak = book.expected("slow", "leaky", "club", "middle")
        assert strong_vs_weak["expected_runs_per_over"] > strong_vs_strong["expected_runs_per_over"]
        assert strong_vs_weak["expected_runs_per_over"] > weak_vs_weak["expected_runs_per_over"]

    def test_unknown_players_revert_to_par(self, book):
        e = book.expected("nobody", "nobody else", "club", "middle")
        par = book._par_row("club", 1)
        assert e["expected_runs_per_over"] == pytest.approx(par["bat_rpb"] * 6, abs=0.01)
        assert (e["batter_balls"], e["bowler_balls"]) == (0, 0)

    def test_dismissal_probability_is_a_probability(self, book):
        for bat in ("hitter", "slow", "nobody"):
            e = book.expected(bat, "tight", "club")
            assert 0 <= e["dismissal_prob_per_ball"] <= e["dismissal_prob_per_over"] <= 1

    def test_recommendation_puts_the_best_bowler_first(self, book):
        rec = book.recommend_bowlers("hitter", ["leaky", "tight", "avg_bowl"], "club", "middle")
        assert list(rec["bowler"])[0] == "tight" and list(rec["bowler"])[-1] == "leaky"

    def test_rankings_are_sorted_filtered_and_pool_specific(self, book):
        rb = book.rank_batters("club", min_balls=100)
        assert list(rb.index)[0] == "hitter" and list(rb.index)[-1] == "slow"
        assert list(rb["rank"]) == list(range(1, len(rb) + 1))
        assert book.rank_batters("club", min_balls=10_000).empty
        with pytest.raises(KeyError):
            book.rank_batters("intl")  # nobody has intl history
        assert list(book.rank_bowlers("club", min_balls=100).index)[0] == "tight"


class TestPools:
    def test_pool_assignment(self, db, tmp_path):  # noqa: F811
        copy = str(tmp_path / "pools.duckdb")
        shutil.copy(db, copy)
        con = duckdb.connect(copy)
        for mid, bat, bowl in [(1, "India", "Australia"), (2, "India", "Nepal")]:
            con.execute("UPDATE ball_by_ball SET league='T20I', batting_team=?, bowling_team=? "
                        "WHERE match_id=?", [bat, bowl, mid])
        con.close()
        b = load_balls(copy)
        pools = b.groupby("match_id")["pool"].first().to_dict()
        assert pools[1] == "intl"    # two Full Members
        assert pools[2] == "assoc"   # involves an associate
        assert pools[3] == "club"

    def test_flags_are_bool_and_phase_bounds(self, db):  # noqa: F811
        b = load_balls(db)
        for c in ("is_faced", "is_legal", "bowler_wkt", "batter_out"):
            assert b[c].dtype == bool
        assert set(b["phase"].unique()) <= {0, 1, 2}
        assert (b["phase"][b["over_number"] <= 5] == 0).all()


class TestPhaseWeighting:
    def test_all_phases_is_weighted_by_overs_not_equal(self):
        rows = (balls_for("a", "y", 400, 1.0, phase=0) + balls_for("b", "y", 400, 2.0, phase=1)
                + balls_for("c", "y", 400, 3.0, phase=2))
        book = RatingBook(mk(rows), as_of="2023-01-01", halflife_years=None)
        allp = book.expected("nobody", "nobody", "club")["expected_runs_per_over"]
        by = [book.expected("nobody", "nobody", "club", ph)["expected_runs_per_over"]
              for ph in ("powerplay", "middle", "death")]
        assert allp == pytest.approx((6 * by[0] + 9 * by[1] + 5 * by[2]) / 20, abs=0.02)
        assert allp != pytest.approx(np.mean(by), abs=0.02)

    def test_missing_phase_data_is_skipped_or_raises_clearly(self):
        book = RatingBook(mk(balls_for("a", "y", 400, 1.0, phase=1)), as_of="2023-01-01",
                          halflife_years=None)
        assert book.expected("a", "y", "club")["expected_runs_per_over"] > 0  # only middle exists
        with pytest.raises(KeyError):
            book.expected("a", "y", "club", "death")


class TestPairs:
    @pytest.fixture()
    def book(self, league):
        return RatingBook(league[0], as_of=league[1], halflife_years=None)

    def test_pair_is_order_independent_and_averages_individuals(self, book):
        ab = book.pair_expectation("hitter", "slow", "club", "middle")
        ba = book.pair_expectation("slow", "hitter", "club", "middle")
        assert ab["expected_runs_per_over"] == ba["expected_runs_per_over"]
        h = book.expected("hitter", "__none__", "club", "middle")["expected_runs_per_over"]
        s = book.expected("slow", "__none__", "club", "middle")["expected_runs_per_over"]
        assert ab["expected_runs_per_over"] == pytest.approx((h + s) / 2, abs=0.01)

    def test_best_pairs_ranks_strongest_batters_first(self, book):
        pairs = book.best_pairs(["hitter", "avg", "slow", "avg2"], "club", "middle")
        assert len(pairs) == 6
        assert "hitter" in {pairs.loc[0, "a"], pairs.loc[0, "b"]}
        assert "slow" in {pairs.iloc[-1]["a"], pairs.iloc[-1]["b"]}
        assert pairs["expected_runs_per_over"].is_monotonic_decreasing


class TestDerivedFiguresAreSuppliedByTheTools:
    """The assistant must not do its own arithmetic, so gaps come with the results."""

    @pytest.fixture()
    def book(self, league):
        return RatingBook(league[0], as_of=league[1], halflife_years=None)

    def test_recommendation_gaps_to_the_best_bowler(self, book):
        rec = book.recommend_bowlers("hitter", ["leaky", "tight", "avg_bowl"], "club", "middle")
        gap = rec["extra_runs_per_over_vs_best"]
        assert gap.iloc[0] == 0.0 and gap.is_monotonic_increasing
        assert gap.iloc[-1] == pytest.approx(
            rec["expected_runs_per_over"].iloc[-1] - rec["expected_runs_per_over"].iloc[0], abs=0.01)

    def test_pair_gaps_to_the_best_pair(self, book):
        pairs = book.best_pairs(["hitter", "avg", "slow", "avg2"], "club", "middle")
        gap = pairs["runs_per_over_below_best_pair"]
        assert gap.iloc[0] == 0.0 and gap.is_monotonic_increasing
