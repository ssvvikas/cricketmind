"""Build everything the app needs from a fresh checkout.

    python -m src.bootstrap              # build whatever is missing
    python -m src.bootstrap --check      # report what exists (exit 1 if anything is missing)
    python -m src.bootstrap --force      # rebuild everything
    python -m src.bootstrap --only model # one step: data | model | entities

Steps, in dependency order (each is skipped when its output already exists):

  data      download Cricsheet (the TIGZIG export plus the BBL/CPL/PSL zips), build the DuckDB
            database, the feature tables, player ratings and the match search index
  model     train the team-strength model
  entities  build the ChromaDB entity index over the feature tables

The first run downloads about 30 MB of data plus a 130 MB embedding model and takes roughly
10-15 minutes, most of it embedding text.  Later runs finish in seconds.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.config import DB_PATH, EMBEDDINGS_DIR, PROCESSED_DIR

STEPS = ("data", "model", "entities")


@dataclass(frozen=True)
class Paths:
    """Where each step leaves its output (overridable so tests can use a temp directory)."""

    db: Path = DB_PATH
    processed: Path = PROCESSED_DIR
    embeddings: Path = EMBEDDINGS_DIR


def status(p: Paths | None = None) -> dict[str, dict]:
    """For each step: whether its output exists, and which files are missing."""
    p = p or Paths()
    wanted = {
        "data": [p.db, *(p.processed / f"{n}_features.parquet"
                         for n in ("batsman", "bowler", "team", "partnership", "fielding", "matchup")),
                 p.processed / "player_ratings_batters.parquet",
                 p.processed / "player_ratings_bowlers.parquet",
                 p.processed / "rag" / "match_corpus.parquet"],
        "model": [p.processed / "models" / n
                  for n in ("team_strength.json", "team_state.pkl", "calibration.json")],
    }
    out = {name: {"missing": [str(f) for f in files if not f.exists()]}
           for name, files in wanted.items()}
    # the entity index lives inside a Chroma folder: ask Chroma whether the collection has chunks
    missing = []
    try:
        from src.rag.entity_index import _collection

        _collection(p.embeddings)
    except Exception as e:  # not built, empty, or chromadb not installed
        missing = [f"{p.embeddings} ({type(e).__name__}: {str(e).splitlines()[0][:80]})"]
    out["entities"] = {"missing": missing}
    for v in out.values():
        v["ok"] = not v["missing"]
    return out


def _data() -> None:
    from src.pipeline import run_phase1

    run_phase1()


def _model() -> None:
    from src.models.team_strength import train

    train()


def _entities() -> None:
    from src.rag.entity_index import build_index

    build_index()


ACTIONS: dict[str, Callable[[], None]] = {"data": _data, "model": _model, "entities": _entities}


def run(steps: tuple[str, ...] = STEPS, force: bool = False, paths: Paths | None = None,
        actions: dict[str, Callable[[], None]] | None = None) -> dict[str, str]:
    """Run the steps that are needed.  Returns {step: 'built' | 'skipped' | 'failed: ...'}.

    Steps run in dependency order; one that fails stops the run, because the later ones need it.
    """
    actions = actions or ACTIONS
    result: dict[str, str] = {}
    for name in (s for s in STEPS if s in steps):
        if not force and status(paths)[name]["ok"]:
            result[name] = "skipped"
            logger.info(f"[{name}] already built - skipping (use --force to rebuild)")
            continue
        logger.info(f"[{name}] building...")
        started = time.monotonic()
        try:
            actions[name]()
        except Exception as e:
            result[name] = f"failed: {type(e).__name__}: {e}"
            logger.error(f"[{name}] {result[name]}")
            break
        result[name] = f"built in {time.monotonic() - started:.0f}s"
        logger.success(f"[{name}] {result[name]}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the data, model and indexes the app needs.")
    ap.add_argument("--check", action="store_true", help="report what exists and exit")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--only", choices=STEPS, action="append", help="run just this step")
    args = ap.parse_args(argv)

    if args.check:
        st = status()
        for name in STEPS:
            print(f"{'ok     ' if st[name]['ok'] else 'MISSING'}  {name}")
            for m in st[name]["missing"][:4]:
                print(f"           missing: {m}")
        return 0 if all(v["ok"] for v in st.values()) else 1

    result = run(tuple(args.only) if args.only else STEPS, force=args.force)
    print("\n".join(f"{k:9s} {v}" for k, v in result.items()))
    return 1 if any(v.startswith("failed") for v in result.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
