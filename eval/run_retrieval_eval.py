"""Retrieval quality: do free-text queries surface the right matches?

Each case is a natural-language query (optionally with a league / venue filter) and a *relevance
rule* over the audited corpus fields.  Precision@10 is compared with the base rate (the share of
relevant matches in the filtered pool, i.e. what picking at random would score).  Tags and
structured filters are exact by construction; this stresses the text ranker alone.

    python eval/run_retrieval_eval.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import MatchIndex  # noqa: E402

# (group, query, filters, relevance rule).  "aligned" queries reuse the words indexed as tag
# synonyms; "paraphrase" queries do not; "held-out" paraphrases were written after the ranking
# method and its 50/50 weighting were fixed, so nothing was tuned on them.
CASES = [
    ("aligned", "a last ball thriller", {}, "tag:last-ball finish"),
    ("aligned", "nail-biting close game", {}, "tag:close finish"),
    ("aligned", "batting collapse all out cheaply", {}, "tag:collapse"),
    ("aligned", "run-fest with huge totals", {}, "tag:high-scoring"),
    ("aligned", "huge successful chase", {}, "tag:big chase"),
    ("aligned", "tied match decided by a super over", {}, "tag:super over"),
    ("aligned", "rain affected Duckworth-Lewis game", {}, "tag:rain-affected"),
    ("aligned", "one-sided thrashing", {}, "tag:thumping win"),
    ("aligned", "Kohli hundred", {"league": "IPL"}, "century_by:V Kohli"),
    ("aligned", "Bumrah five wickets", {}, "five_by:JJ Bumrah"),
    ("aligned", "Chinnaswamy high scoring", {"league": "IPL"}, "tag:high-scoring&venue:Chinnaswamy"),
    ("aligned", "Wankhede last over drama", {"league": "IPL"}, "tag:last-over finish&venue:Wankhede"),
    ("paraphrase", "match that went down to the wire", {}, "tag:close finish"),
    ("paraphrase", "edge of the seat cliffhanger", {}, "tag:close finish"),
    ("paraphrase", "won it off the very last delivery", {}, "tag:last-ball finish"),
    ("paraphrase", "a complete rout, demolition job", {}, "tag:thumping win"),
    ("paraphrase", "batting side crumbled for peanuts", {}, "tag:collapse"),
    ("paraphrase", "monster chase, ran down a huge target", {}, "tag:big chase"),
    ("paraphrase", "bowlers on top, hard to score runs", {}, "tag:low-scoring"),
    ("paraphrase", "boundary bonanza, sky-high scores", {}, "tag:high-scoring"),
    ("paraphrase", "skied it and the game was tied, then a decider", {}, "tag:super over"),
    ("paraphrase", "game interrupted by weather", {}, "tag:rain-affected"),
    ("held-out", "came down to the last over", {}, "tag:last-over finish"),
    ("held-out", "hard-fought and nervy", {}, "tag:close finish"),
    ("held-out", "a walkover", {}, "tag:thumping win"),
    ("held-out", "side folded for a tiny total", {}, "tag:collapse"),
    ("held-out", "a chase that seemed impossible", {}, "tag:big chase"),
    ("held-out", "low-scoring grind", {}, "tag:low-scoring"),
    ("held-out", "towering totals", {}, "tag:high-scoring"),
    ("held-out", "tie and one-over eliminator", {}, "tag:super over"),
    ("held-out", "abandoned without a result", {}, "tag:no result"),
    ("held-out", "rain shortened contest decided by DLS method", {}, "tag:rain-affected"),
]


def relevant(row, rule: str) -> bool:
    ok = True
    for part in rule.split("&"):
        kind, _, val = part.partition(":")
        if kind == "tag":
            ok &= val in row.tags
        elif kind == "venue":
            ok &= val.lower() in (row.venue or "").lower()
        elif kind == "century_by":
            ok &= val in row.centuries
        elif kind == "five_by":
            ok &= val in row.five_wicket_hauls
    return bool(ok)


def main() -> int:
    ix = MatchIndex.load()
    if ix.embeddings is None:
        print("The saved index has no embeddings; rebuild it with `python -m src.rag.retriever`.")
    c = ix.corpus
    by_id = c.set_index("match_id")
    rows = []
    for group, q, filt, rule in CASES:
        pool = c[c.league == filt["league"]] if "league" in filt else c
        base = float(np.mean([relevant(r, rule) for r in pool.itertuples()]))
        row = {"group": group, "query": q, "base_rate": base}
        for ranking in ("keyword", "hybrid"):
            hits = ix.search(q, top=10, ranking=ranking, **filt)["hits"]
            row[ranking] = float(np.mean([relevant(by_id.loc[h["match_id"]], rule) for h in hits]))
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("\nPrecision@10 by query group (higher is better; base rate = random picking):")
    print(df.groupby("group")[["base_rate", "keyword", "hybrid"]].mean().round(2).to_string())
    print(f"\nall {len(df)} queries: keyword {df.keyword.mean():.2f}  hybrid {df.hybrid.mean():.2f}"
          f"  base rate {df.base_rate.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
