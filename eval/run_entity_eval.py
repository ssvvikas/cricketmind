"""Does the entity index put the right kind of chunk first?

Each question has a known "right chunk type" (Bumrah's economy lives in the bowler chunk, Dhoni's
stumpings in the fielding chunk, Kohli against Bumrah in the matchup chunk).  Top-1 and top-3
accuracy are compared with the entity-type hint off and on.  Questions whose right chunk does not
exist in the index (for example a matchup below the 30-ball threshold) are skipped and listed.

    python eval/run_entity_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.entity_index import INTENT_BOOST, retrieve  # noqa: E402

CASES = [
    ("How does Buttler bat in the powerplay?", "batsman"),
    ("What is Kohli's strike rate and average?", "batsman"),
    ("Russell batting record", "batsman"),
    ("de Kock strike rate in the death overs", "batsman"),
    ("How many runs has Yadav scored?", "batsman"),
    ("Warner batting strike rate by phase", "batsman"),
    ("Bumrah economy in the death overs", "bowler"),
    ("How many wickets has Chahal taken?", "bowler"),
    ("Narine bowling average", "bowler"),
    ("Boult bowling record in the powerplay", "bowler"),
    ("Rashid Khan economy", "bowler"),
    ("Bumrah dot ball percentage", "bowler"),
    ("How many catches has Dhoni taken?", "fielding"),
    ("Dhoni stumpings", "fielding"),
    ("de Kock dismissals as wicketkeeper", "fielding"),
    ("Kohli catches and run outs", "fielding"),
    ("How many run outs has Buttler been involved in?", "fielding"),
    ("Kohli against Bumrah", "matchup"),
    ("Russell vs Rashid Khan", "matchup"),
    ("How does Buttler fare against Narine?", "matchup"),
    ("Warner facing Bumrah", "matchup"),
    ("Buttler and Salt partnership", "partnership"),
    ("Kohli and du Plessis opening pair", "partnership"),
    ("How have CSK done in the IPL?", "team"),
    ("Mumbai Indians win rate", "team"),
    ("How is India doing in T20 internationals?", "team"),
]

# Written after the weight sweep on CASES (the development set) and before scoring: nothing was
# tuned on these.
HELD_OUT = [
    ("SA Yadav strike rate", "batsman"),
    ("How does Klaasen hit in the death overs?", "batsman"),
    ("Pooran batting average", "batsman"),
    ("How Head bats in the powerplay", "batsman"),
    ("Maxwell strike rate in the middle overs", "batsman"),
    ("Gill runs and average", "batsman"),
    ("Rabada economy rate", "bowler"),
    ("How many wickets has Shami taken?", "bowler"),
    ("Rashid Khan wickets in the death overs", "bowler"),
    ("Starc bowling economy", "bowler"),
    ("Chahal dot ball percentage", "bowler"),
    ("How many catches did Jadeja take?", "fielding"),
    ("Rizwan wicketkeeper dismissals", "fielding"),
    ("Pant stumpings", "fielding"),
    ("Gill against Rashid Khan", "matchup"),
    ("Head facing Bumrah", "matchup"),
    ("Klaasen vs Chahal", "matchup"),
    ("Warner and Head together", "partnership"),
    ("Rajasthan Royals win rate", "team"),
    ("How are KKR doing?", "team"),
]


def evaluate(cases, boost):
    """(rows, skipped) with top-1 / top-3 hit for one boost weight."""
    usable, skipped = [], []
    for q, want in cases:
        probe = retrieve(q, top_k=1, entity_type=want, intent_boost=0.0)
        # usable if a chunk of the right type mentioning the named entity exists at all,
        # independent of how it ranks (a buried right answer is exactly what we want to score)
        (usable if probe["name_filtered"] or not probe["filtered_by"] else skipped).append((q, want))
    rows = []
    for q, want in usable:
        hits = retrieve(q, top_k=3, intent_boost=boost)["hits"]
        rows.append((q, want, hits[0]["entity_type"], any(h["entity_type"] == want for h in hits)))
    return rows, skipped


def main() -> int:
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    print("Development set (used to choose the weight):")
    for b in grid:
        rows, skipped = evaluate(CASES, b)
        top1 = sum(r[2] == r[1] for r in rows)
        print(f"  boost {b:4.2f}: top-1 {top1}/{len(rows)} ({top1 / len(rows):.0%})   "
              f"top-3 {sum(r[3] for r in rows)}/{len(rows)}")
    print(f"\nHeld-out set ({len(HELD_OUT)} questions; never used for tuning), weight {INTENT_BOOST}"
          " vs off:")
    for b in (0.0, INTENT_BOOST):
        rows, skipped = evaluate(HELD_OUT, b)
        top1 = sum(r[2] == r[1] for r in rows)
        print(f"  boost {b:4.2f}: top-1 {top1}/{len(rows)} ({top1 / len(rows):.0%})   "
              f"top-3 {sum(r[3] for r in rows)}/{len(rows)}   skipped {len(skipped)}")
    print("\nHeld-out misses at the configured weight:")
    rows, _ = evaluate(HELD_OUT, INTENT_BOOST)
    for q, want, got, _ in rows:
        if got != want:
            print(f"  want {want:11s} got {got:11s} {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
