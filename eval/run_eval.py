"""Run the honesty / tool-use eval against the real assistant.

    ANTHROPIC_API_KEY=... python eval/run_eval.py            # all cases
    python eval/run_eval.py --case club-prediction-honesty   # one case

This calls the Claude API (about 2-6 requests per case).  Results go to eval/results.json.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.agent import CricketAgent, estimate_cost  # noqa: E402
from src.llm.eval_checks import check, load_cases  # noqa: E402
from src.llm.tools import Toolbox  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", help="run only these case ids")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    cases = [c for c in load_cases(here / "cases.json") if not args.case or c.id in args.case]
    toolbox = Toolbox()  # shared: loading ratings once
    import anthropic

    rows, failed = [], 0
    for case in cases:
        agent = CricketAgent(toolbox=toolbox)
        try:
            result = agent.ask(case.question)
        except anthropic.APIStatusError as e:  # e.g. credit balance too low: keep what we have
            (here / "results.json").write_text(json.dumps(rows, indent=2))
            print(f"\nSTOPPED at {case.id!r}: the API returned {e.status_code}: "
                  f"{getattr(e, 'message', e)}\n{len(rows)} completed cases were saved to "
                  "eval/results.json.")
            return 2
        calls = [asdict(c) for c in result.tool_calls]
        fails = check(case, result.text, calls)
        failed += bool(fails)
        print(f"{'PASS' if not fails else 'FAIL'}  {case.id:26s} tools={[c['name'] for c in calls]}")
        for f in fails:
            print(f"        - {f}")
        cost = estimate_cost(agent.model, result.input_tokens, result.output_tokens)
        rows.append({"id": case.id, "question": case.question, "answer": result.text,
                     "tools": [c["name"] for c in calls], "failures": fails,
                     "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                     "estimated_cost_usd": None if cost is None else round(cost, 4)})
        (here / "results.json").write_text(json.dumps(rows, indent=2))  # saved after every case
    (here / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    costs = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
    if costs:
        print(f"estimated cost at list price: ${sum(costs):.2f} for {len(costs)} cases "
              f"(${sum(costs) / len(costs):.3f} per question on average)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
