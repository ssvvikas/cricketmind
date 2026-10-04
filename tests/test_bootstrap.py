"""The one-command setup: which steps run, which are skipped, and how failures are handled."""

import pandas as pd
import pytest

from src import bootstrap as B


@pytest.fixture()
def paths(tmp_path):
    return B.Paths(db=tmp_path / "db" / "x.duckdb", processed=tmp_path / "processed",
                   embeddings=tmp_path / "emb")


def touch_data(p):
    for f in B.status(p)["data"]["missing"]:
        from pathlib import Path

        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).write_text("x")


def touch_model(p):
    for f in B.status(p)["model"]["missing"]:
        from pathlib import Path

        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).write_text("x")


class Recorder:
    def __init__(self, paths, fail=None):
        self.calls, self.paths, self.fail = [], paths, fail
        self.actions = {n: (lambda n=n: self(n)) for n in B.STEPS}

    def __call__(self, name):
        self.calls.append(name)
        if name == self.fail:
            raise RuntimeError("boom")
        {"data": touch_data, "model": touch_model}.get(name, lambda p: None)(self.paths)


class TestStatus:
    def test_an_empty_checkout_is_missing_everything(self, paths):
        st = B.status(paths)
        assert [st[n]["ok"] for n in B.STEPS] == [False, False, False]
        assert any("features.parquet" in m for m in st["data"]["missing"])
        assert any("team_strength.json" in m for m in st["model"]["missing"])

    def test_outputs_flip_a_step_to_ok(self, paths):
        touch_data(paths)
        touch_model(paths)
        st = B.status(paths)
        assert st["data"]["ok"] and st["model"]["ok"] and not st["entities"]["ok"]

    def test_one_missing_file_is_enough_to_rebuild(self, paths):
        touch_data(paths)
        next(paths.processed.glob("bowler_features.parquet")).unlink()
        st = B.status(paths)["data"]
        assert not st["ok"] and len(st["missing"]) == 1 and "bowler_features" in st["missing"][0]


class TestRun:
    def test_builds_every_missing_step_in_dependency_order(self, paths):
        rec = Recorder(paths)
        out = B.run(paths=paths, actions=rec.actions)
        assert rec.calls == ["data", "model", "entities"]
        assert all(v.startswith("built") for v in out.values())

    def test_skips_steps_whose_output_exists(self, paths):
        touch_data(paths)
        rec = Recorder(paths)
        out = B.run(paths=paths, actions=rec.actions)
        assert rec.calls == ["model", "entities"] and out["data"] == "skipped"

    def test_force_rebuilds_everything(self, paths):
        touch_data(paths)
        touch_model(paths)
        rec = Recorder(paths)
        B.run(force=True, paths=paths, actions=rec.actions)
        assert rec.calls == ["data", "model", "entities"]

    def test_only_runs_the_chosen_steps_but_keeps_the_order(self, paths):
        rec = Recorder(paths)
        B.run(steps=("entities", "data"), paths=paths, actions=rec.actions)
        assert rec.calls == ["data", "entities"]

    def test_a_failure_stops_the_run_and_is_reported(self, paths):
        rec = Recorder(paths, fail="model")
        out = B.run(paths=paths, actions=rec.actions)
        assert rec.calls == ["data", "model"]  # entities never starts: it needs the earlier steps
        assert out["data"].startswith("built") and out["model"] == "failed: RuntimeError: boom"
        assert "entities" not in out

    def test_a_second_run_does_nothing(self, paths):
        rec = Recorder(paths)
        B.run(paths=paths, actions=rec.actions)
        # entities is checked through chromadb, which the fake does not create, so only the
        # two file-based steps are guaranteed to be skipped
        first = len(rec.calls)
        out = B.run(paths=paths, actions=rec.actions)
        assert out["data"] == "skipped" and out["model"] == "skipped" and len(rec.calls) <= first + 1


class TestCli:
    def test_check_exits_non_zero_and_lists_what_is_missing(self, paths, monkeypatch, capsys):
        monkeypatch.setattr(B, "Paths", lambda: paths)
        assert B.main(["--check"]) == 1
        out = capsys.readouterr().out
        assert "MISSING  data" in out and "missing:" in out

    def test_check_exits_zero_when_everything_exists(self, paths, monkeypatch):
        touch_data(paths)
        touch_model(paths)
        monkeypatch.setattr(B, "Paths", lambda: paths)
        monkeypatch.setattr(B, "status", lambda p=None: {
            n: {"ok": True, "missing": []} for n in B.STEPS})
        assert B.main(["--check"]) == 0

    def test_exit_code_reflects_failure(self, paths, monkeypatch):
        monkeypatch.setattr(B, "ACTIONS", {n: (lambda: (_ for _ in ()).throw(RuntimeError("x")))
                                           for n in B.STEPS})
        monkeypatch.setattr(B, "Paths", lambda: paths)
        assert B.main(["--only", "data"]) == 1

    def test_unknown_step_is_rejected(self):
        with pytest.raises(SystemExit):
            B.main(["--only", "everything"])

    def test_the_real_entity_check_reports_a_missing_index_clearly(self, tmp_path):
        st = B.status(B.Paths(embeddings=tmp_path / "nothing"))["entities"]
        assert not st["ok"] and "not been built" in st["missing"][0]
