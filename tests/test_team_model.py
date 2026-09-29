"""Team-strength model: leakage, symmetry and Elo mechanics."""

import shutil

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.models.team_dataset import FEATURES, ELO_START, TeamState, build_dataset, elo_expected
from src.models.team_strength import split, symmetrise
from tests.test_features import db, raw_and_db  # noqa: F401  (fixtures)

DIFF_FEATURES = ["elo_diff", "form_diff", "bat_rr_diff", "bowl_econ_diff", "venue_rate_diff",
                 "exp_diff"]


@pytest.fixture(scope="module")
def dataset(db):  # noqa: F811
    df, state = build_dataset(db)
    return df, state


def _rebuild_with_winner(db, tmp_path, match_id, winner):
    copy = str(tmp_path / f"flip_{match_id}.duckdb")
    shutil.copy(db, copy)
    con = duckdb.connect(copy)
    con.execute("UPDATE match_info SET winner = ? WHERE match_id = ?", [winner, match_id])
    con.close()
    return build_dataset(copy)[0]


class TestNoLeakage:
    @pytest.mark.parametrize("flip_id", [1, 3, 6])
    def test_changing_a_result_never_changes_earlier_or_own_features(
        self, db, dataset, tmp_path, flip_id  # noqa: F811
    ):
        base, _ = dataset
        m = duckdb.connect(db, read_only=True).execute(
            "SELECT team1, team2, winner FROM match_info WHERE match_id = ?", [flip_id]
        ).fetchone()
        other = m[1] if m[2] == m[0] else m[0]
        flipped = _rebuild_with_winner(db, tmp_path, flip_id, other)

        upto = base["match_id"] <= flip_id  # matches up to and including the flipped one
        pd.testing.assert_frame_equal(
            base.loc[upto, FEATURES].reset_index(drop=True),
            flipped.loc[upto, FEATURES].reset_index(drop=True),
        )
        # ...but the label of the flipped match does change
        assert (base.loc[base.match_id == flip_id, "y"].to_numpy()
                != flipped.loc[flipped.match_id == flip_id, "y"].to_numpy()).all()

    def test_result_does_influence_later_matches(self, db, dataset, tmp_path):  # noqa: F811
        base, _ = dataset
        m = duckdb.connect(db, read_only=True).execute(
            "SELECT team1, team2, winner FROM match_info WHERE match_id = 1").fetchone()
        flipped = _rebuild_with_winner(db, tmp_path, 1, m[1] if m[2] == m[0] else m[0])
        later = base["match_id"] > 1
        assert not np.allclose(base.loc[later, "elo_diff"], flipped.loc[later, "elo_diff"])

    def test_first_match_has_neutral_features(self, dataset):
        df, _ = dataset
        first = df[df.match_id == df.match_id.min()]
        assert (first["elo_diff"] == 0).all() and (first["form_diff"] == 0).all()
        assert (first["h2h_n"] == 0).all() and (first["min_matches"] == 0).all()


class TestSymmetry:
    def test_two_perspective_rows_mirror_each_other(self, dataset):
        df, _ = dataset
        for _, g in df.groupby("match_id"):
            a, b = g.iloc[0], g.iloc[1]
            assert a["y"] + b["y"] == 1
            assert a["team_a"] == b["team_b"] and a["team_b"] == b["team_a"]
            for f in DIFF_FEATURES:
                if pd.notna(a[f]):
                    assert a[f] == pytest.approx(-b[f]), f
            assert a["h2h_rate"] + b["h2h_rate"] == pytest.approx(1.0)
            assert a["toss_won"] + b["toss_won"] == pytest.approx(1.0)

    def test_symmetrise_ignores_team_order(self):
        df = pd.DataFrame({"match_id": [1, 1], "team1_is_a": [True, False], "y": [1, 0]})
        out = symmetrise(df, np.array([0.7, 0.3]))  # both views say team1 wins with 70%
        assert out.loc[1, "p"] == pytest.approx(0.7) and out.loc[1, "y"] == 1


class TestElo:
    def test_expected_is_half_for_equal_and_monotonic(self):
        assert elo_expected(1500, 1500) == 0.5
        assert elo_expected(1600, 1500) > 0.5 > elo_expected(1400, 1500)

    def test_update_is_zero_sum_and_rewards_upsets_more(self):
        s = TeamState()
        s.elo[("L", "strong")], s.elo[("L", "weak")] = 1700.0, 1300.0
        before = s.elo[("L", "strong")] + s.elo[("L", "weak")]
        s.update("L", "strong", "weak", "weak", "v", {})  # upset
        assert s.elo[("L", "strong")] + s.elo[("L", "weak")] == pytest.approx(before)
        upset_gain = s.elo[("L", "weak")] - 1300.0

        t = TeamState()
        t.elo[("L", "strong")], t.elo[("L", "weak")] = 1700.0, 1300.0
        t.update("L", "strong", "weak", "strong", "v", {})  # expected result
        assert upset_gain > t.elo[("L", "strong")] - 1700.0 > 0

    def test_leagues_do_not_share_ratings(self):
        s = TeamState()
        s.update("IPL", "X", "Y", "X", "v", {})
        assert s.elo[("BBL", "X")] == ELO_START


class TestSplit:
    def test_temporal_split_is_disjoint_and_ordered(self):
        dates = pd.to_datetime(["2022-05-01", "2023-12-31", "2024-01-01", "2024-12-31",
                                "2025-01-01"])
        df = pd.DataFrame({"match_id": range(5), "date": dates})
        p = split(df)
        assert list(p["train"].match_id) == [0, 1]
        assert list(p["val"].match_id) == [2, 3]
        assert list(p["test"].match_id) == [4]


# ── lineup features ────────────────────────────────────────────────
from collections import deque  # noqa: E402

from src.models.team_dataset import (  # noqa: E402
    LINEUP_FEATURES,
    MIN_WINDOW_MATCHES,
    PlayerState,
    rating_pool,
)


def _match(bat_runs, bat_balls, wkts=0, bowl_runs=0, bowl_balls=0):
    return [bat_runs, bat_balls, bowl_balls, bowl_runs, wkts]


class TestPlayerRatings:
    def test_unknown_players_contribute_nothing(self):
        s = PlayerState().strength(["nobody", "nobody else"])
        assert (s["bat"], s["bowl"], s["wkt"], s["known"]) == (0.0, 0.0, 0.0, 0)

    def test_strong_batter_and_economical_bowler_rate_positive(self):
        ps = PlayerState()
        for _ in range(MIN_WINDOW_MATCHES + 5):  # establish a par of ~1.2 runs/ball
            ps.add_match({"avg": _match(30, 25, bowl_balls=24, bowl_runs=29),
                          "star": _match(60, 30, bowl_balls=24, bowl_runs=20, wkts=2)})
        assert ps.strength(["star"])["bat"] > 0
        assert ps.strength(["star"])["bowl"] > 0 and ps.strength(["star"])["wkt"] > 0
        assert ps.strength(["star"])["bat"] > ps.strength(["avg"])["bat"]

    def test_shrinkage_keeps_tiny_samples_near_zero(self):
        ps = PlayerState()
        for _ in range(MIN_WINDOW_MATCHES + 5):
            ps.add_match({"avg": _match(30, 25)})
        ps.add_match({"fluke": _match(100, 40)})  # one huge innings
        veteran = PlayerState()
        for _ in range(MIN_WINDOW_MATCHES + 5):
            veteran.add_match({"avg": _match(30, 25)})
        for _ in range(30):
            veteran.add_match({"vet": _match(100, 40)})
        assert 0 < ps.strength(["fluke"])["bat"] < veteran.strength(["vet"])["bat"]

    def test_same_performance_rates_lower_in_a_higher_scoring_era(self):
        low, high = PlayerState(), PlayerState()
        for _ in range(MIN_WINDOW_MATCHES + 5):
            # bowling is included because par needs both bat and bowl totals
            low.add_match({"x": _match(24, 25, bowl_balls=24, bowl_runs=24)})    # par 0.96
            high.add_match({"x": _match(40, 25, bowl_balls=24, bowl_runs=40)})   # par 1.6
        low.add_match({"p": _match(30, 25)})
        high.add_match({"p": _match(30, 25)})  # identical innings, different era par
        assert low.strength(["p"])["bat"] > high.strength(["p"])["bat"]

    def test_par_window_is_bounded(self):
        ps = PlayerState()
        for i in range(ps.window.maxlen + 50):
            ps.add_match({"a": _match(30, 25)})
        assert len(ps.window) == ps.window.maxlen and ps.win[0] == ps.window.maxlen

    def test_club_and_international_pools_are_separate(self):
        assert rating_pool("T20I") == "intl" and rating_pool("IPL") == "club"
        s = TeamState()
        s.update("IPL", "X", "Y", "X", "v", {}, {"p": _match(30, 20)})
        assert "p" in s.players["club"].p and "p" not in s.players["intl"].p


class TestLineupFeatures:
    def test_lineup_diffs_mirror_between_perspective_rows(self, dataset):
        df, _ = dataset
        for _, g in df.groupby("match_id"):
            a, b = g.iloc[0], g.iloc[1]
            for f in ["lineup_bat_diff", "lineup_bowl_diff", "lineup_wkt_diff",
                      "lineup_net_diff"]:
                assert a[f] == pytest.approx(-b[f]), f
            assert a["xi_known_min"] == b["xi_known_min"]

    def test_lineup_columns_present_and_first_match_neutral(self, dataset):
        df, _ = dataset
        assert set(LINEUP_FEATURES) <= set(df.columns)
        first = df[df.match_id == df.match_id.min()]
        assert (first["lineup_net_diff"] == 0).all() and (first["xi_known_min"] == 0).all()

    def test_missing_xi_gives_nan_lineup_features(self):
        f = TeamState().features("IPL", "A", "B", "v", None, None, ["p"])
        assert all(np.isnan(f[k]) for k in LINEUP_FEATURES)

    def test_last_xi_is_remembered(self):
        s = TeamState()
        s.update("IPL", "A", "B", "A", "v", {}, {}, {"A": ["a1", "a2"], "B": ["b1"]})
        assert s.last_xi[("IPL", "A")] == ["a1", "a2"] and s.last_xi[("IPL", "B")] == ["b1"]

    @pytest.mark.parametrize("edit_id", [2, 4])
    def test_changing_a_matchs_ball_data_never_alters_its_or_earlier_features(
        self, db, dataset, tmp_path, edit_id  # noqa: F811
    ):
        base, _ = dataset
        copy = str(tmp_path / f"balls_{edit_id}.duckdb")
        shutil.copy(db, copy)
        con = duckdb.connect(copy)
        con.execute("UPDATE ball_by_ball SET runs_batter = runs_batter + 6 WHERE match_id = ?",
                    [edit_id])
        con.close()
        edited, _ = build_dataset(copy)
        upto = base["match_id"] <= edit_id
        pd.testing.assert_frame_equal(base.loc[upto, FEATURES].reset_index(drop=True),
                                      edited.loc[upto, FEATURES].reset_index(drop=True))
        later = base["match_id"] > edit_id
        assert not np.allclose(base.loc[later, "lineup_bat_diff"].fillna(0),
                               edited.loc[later, "lineup_bat_diff"].fillna(0))


class TestSameDayMatches:
    def test_a_match_never_sees_another_match_from_the_same_date(self, db, dataset, tmp_path):  # noqa: F811
        base, _ = dataset
        # fixture: matches 1 and 2 are Team A v Team B on consecutive days
        assert base.loc[base.match_id == 2, "elo_diff"].abs().max() > 0  # sees match 1
        copy = str(tmp_path / "sameday.duckdb")
        shutil.copy(db, copy)
        con = duckdb.connect(copy)
        con.execute("UPDATE match_info SET date = (SELECT date FROM match_info WHERE match_id = 1)"
                    " WHERE match_id = 2")
        con.close()
        same, _ = build_dataset(copy)
        m2 = same[same.match_id == 2]
        assert (m2["elo_diff"] == 0).all() and (m2["h2h_n"] == 0).all()
        assert (m2["lineup_net_diff"] == 0).all()


class TestClubCalibration:
    def _noise_frame(self, signal: float):
        rng = np.random.default_rng(0)
        n = 1500
        dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 7, n), "D")
        x = rng.normal(size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-signal * x))).astype(int)
        rows = []
        for i in range(n):
            for a_first, sign in ((True, 1), (False, -1)):
                r = dict.fromkeys(FEATURES, 0.0)
                r.update(elo_diff=sign * x[i] * 100, match_id=i, date=dates[i], league="IPL",
                         team1_is_a=a_first, y=y[i] if a_first else 1 - y[i])
                rows.append(r)
        return pd.DataFrame(rows)

    def test_pure_noise_is_shrunk_hard(self):
        from src.models.team_strength import fit_club_shrinkage

        # with ~1,100 matches chance patterns leave some apparent skill (lambda ~0.5),
        # so require clearly-below-signal shrinkage rather than near zero
        cal = fit_club_shrinkage(self._noise_frame(signal=0.0))
        assert cal["lambda"] <= 0.6 and cal["log_loss_shrunk"] <= cal["log_loss_raw"]

    def test_real_signal_is_kept(self):
        from src.models.team_strength import fit_club_shrinkage

        noise = fit_club_shrinkage(self._noise_frame(signal=0.0))["lambda"]
        cal = fit_club_shrinkage(self._noise_frame(signal=2.0))
        assert cal["lambda"] >= 0.8 and cal["lambda"] > noise + 0.25
