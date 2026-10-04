"""Audit every corpus tag and figure against independent SQL on the real database.

The corpus is built with pandas; this script recomputes the same facts with plain SQL and
compares match by match.  Any disagreement is a bug in one of the two derivations.

    python eval/audit_corpus.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH  # noqa: E402
from src.rag.corpus import CORPUS_PATH  # noqa: E402

con = duckdb.connect(str(DB_PATH), read_only=True)
corpus = pd.read_parquet(CORPUS_PATH)
CREATE = """
CREATE TEMP TABLE inn AS
SELECT match_id, innings, batting_team,
       SUM(runs_batter + runs_extras) AS score, SUM(is_wicket::INT) AS wkts,
       SUM(is_legal::INT) AS balls
FROM ball_by_ball GROUP BY ALL;
CREATE TEMP TABLE bat AS
SELECT match_id, innings, batter, SUM(runs_batter) AS runs FROM ball_by_ball
WHERE is_faced GROUP BY ALL;
CREATE TEMP TABLE bowl AS
SELECT match_id, innings, bowler, SUM(is_bowler_wicket::INT) AS wkts FROM ball_by_ball GROUP BY ALL;
"""
con.execute(CREATE)


def ids(sql: str) -> set[int]:
    return {r[0] for r in con.execute(sql).fetchall()}


FULL = "(m.overs = 20 AND COALESCE(m.method, '') NOT IN ('D/L', 'Awarded'))"
truth = {
    "last-ball finish": ids(f"""SELECT m.match_id FROM match_info m
        JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE {FULL} AND m.winner = s.batting_team AND s.balls = 120
          AND COALESCE(m.outcome, '') <> 'tie'"""),
    "last-over finish": ids(f"""SELECT m.match_id FROM match_info m
        JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE {FULL} AND m.winner = s.batting_team AND s.balls >= 114
          AND COALESCE(m.outcome, '') <> 'tie'"""),
    "big chase": ids(f"""SELECT m.match_id FROM match_info m
        JOIN inn f ON f.match_id = m.match_id AND f.innings = 1
        JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE {FULL} AND m.winner = s.batting_team AND f.score + 1 >= 190
          AND COALESCE(m.outcome, '') <> 'tie'"""),
    "high-scoring": ids("SELECT match_id FROM inn WHERE score >= 200"),
    "century": ids("SELECT match_id FROM bat WHERE runs >= 100"),
    "five-wicket haul": ids("SELECT match_id FROM bowl WHERE wkts >= 5"),
    "collapse": ids(f"""SELECT i.match_id FROM inn i JOIN match_info m USING (match_id)
        WHERE {FULL} AND i.wkts >= 10 AND i.score <= 100"""),
    "tie": ids("SELECT match_id FROM match_info WHERE outcome = 'tie'"),
    "no result": ids("SELECT match_id FROM match_info WHERE outcome = 'no result'"),
    "rain-affected": ids("SELECT match_id FROM match_info WHERE method = 'D/L'"),
    # margin-based tags use the source's official margins (D/L-aware), like the corpus does
    "close finish": ids(f"""SELECT m.match_id FROM match_info m
        LEFT JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE COALESCE(m.method, '') <> 'Awarded'
          AND (m.winner_runs BETWEEN 1 AND 5 OR m.winner_wickets BETWEEN 1 AND 2
               OR ({FULL} AND m.winner = s.batting_team AND s.balls = 120
                   AND COALESCE(m.outcome, '') <> 'tie'))"""),
    "thumping win": ids(f"""SELECT m.match_id FROM match_info m
        LEFT JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE m.winner_runs >= 60
           OR (m.winner_wickets >= 8 AND {FULL} AND m.winner = s.batting_team
               AND (120 - s.balls) >= 36)"""),
    "super over": ids("SELECT match_id FROM match_info WHERE outcome = 'tie' AND "
                      "(winner IS NOT NULL OR eliminator IS NOT NULL)"),
    "awarded": ids("SELECT match_id FROM match_info WHERE method = 'Awarded'"),
    "low-scoring": ids(f"""SELECT m.match_id FROM match_info m
        JOIN inn f ON f.match_id = m.match_id AND f.innings = 1
        JOIN inn s ON s.match_id = m.match_id AND s.innings = 2
        WHERE {FULL} AND f.score <= 130 AND s.score <= 130"""),
}
bad = 0
for tag, expected in truth.items():
    got = {int(r.match_id) for r in corpus.itertuples() if tag in r.tags}
    extra, missing = got - expected, expected - got
    ok = not extra and not missing
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {tag:18s} corpus={len(got):5d} sql={len(expected):5d} "
          f"extra={len(extra)} missing={len(missing)}")
    for label, sample in (("extra", extra), ("missing", missing)):
        if sample:
            print(f"       {label} e.g. {sorted(sample)[:5]}")

# per-match figures: innings totals, and the top scorer's runs
tot = con.execute("SELECT match_id, MAX(CASE WHEN innings = 1 THEN score END) f, "
                  "MAX(CASE WHEN innings = 2 THEN score END) s FROM inn GROUP BY 1").df()
m = corpus.merge(tot, on="match_id")
fig_bad = int(((m.first_score != m.f) | (m.second_score.fillna(-1) != m.s.fillna(-1))).sum())
print(f"{'OK  ' if not fig_bad else 'FAIL'} innings totals   mismatches={fig_bad} of {len(m)}")
bad += bool(fig_bad)
n_c = int(sum(len(x) for x in corpus["centuries"]))
n_sql = con.execute("SELECT COUNT(*) FROM bat WHERE runs >= 100").fetchone()[0]
print(f"{'OK  ' if n_c == n_sql else 'FAIL'} century count    corpus={n_c} sql={n_sql}")
bad += n_c != n_sql
print("\nAUDIT", "PASSED" if not bad else f"FAILED ({bad} problems)")
raise SystemExit(1 if bad else 0)
