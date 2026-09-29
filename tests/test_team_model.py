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
