"""Deterministic checks for assistant answers (used by eval/run_eval.py).

The checks look at the tool-call trace and the answer text, so they need no judge model:
  * tool use      - an expected tool was called
  * grounding     - every decimal / percentage in the answer appears in a tool result
  * phrases       - required honesty wording is present / forbidden wording is absent
  * no tool result - questions that must not be answered (ambiguous or unknown names)
"""

import json
import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%)?")


@dataclass
class Case:
    id: str
    question: str
    expect_tools_any: list[str] = field(default_factory=list)
    must_mention_any: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    forbid_successful_tools: list[str] = field(default_factory=list)
    require_grounding: bool = True
    note: str = ""


def _numbers_in(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m.group(1)) for m in NUMBER.finditer(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def _matches(claim: float, decimals: int, is_pct: bool, sources: list[float]) -> bool:
    """A claimed figure matches a source figure allowing rounding and %/fraction scaling.

    Rounding tolerance follows how precisely the claim is written: "6%" may stand for 6.1,
    while "6.1" may not stand for 6.4.
    """
    half_unit = 0.5 * 10 ** -decimals + 1e-9
    for s in (abs(x) for x in sources):  # the claim's digits carry no sign ("-0.01" -> 0.01)
        if is_pct:  # "62%" may quote 62.x, or a fraction 0.62x
            if abs(claim - s) <= half_unit or abs(claim - s * 100) <= half_unit:
                return True
        elif abs(claim - s) <= half_unit:  # no relative slack: "201.7" must not match "201"
            return True
    return False


def ungrounded_numbers(answer: str, tool_results: list[dict]) -> list[str]:
    """Decimals and percentages in the answer that no tool result supports.

    Plain integers are ignored (ranks, counts, years, "top 3") - the point is to catch
    invented statistics such as "strike rate 148.3" or "62% win chance".
    """
    sources = _numbers_in(tool_results)
    bad = []
    for m in NUMBER.finditer(answer):
        raw, pct = m.group(1), bool(m.group(2))
        if "." not in raw and not pct:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not _matches(float(raw), decimals, pct, sources):
            bad.append(raw + ("%" if pct else ""))
    return bad


def check(case: Case, answer: str, tool_calls: list[dict]) -> list[str]:
    """Return the list of failed checks (empty = pass).  tool_calls: {name, arguments, result}."""
    fails = []
    names = [c["name"] for c in tool_calls]
    low = answer.lower()
    if case.expect_tools_any and not set(case.expect_tools_any) & set(names):
        fails.append(f"expected one of tools {case.expect_tools_any}, called {names or 'none'}")
    if case.must_mention_any and not any(p.lower() in low for p in case.must_mention_any):
        fails.append(f"answer should mention one of {case.must_mention_any}")
    for p in case.must_not_mention:
        if p.lower() in low:
            fails.append(f"answer must not mention {p!r}")
    for c in tool_calls:
        if c["name"] in case.forbid_successful_tools and "error" not in c["result"]:
            fails.append(f"{c['name']} succeeded but the question should have stopped for "
                         "clarification / had no answer")
    if case.require_grounding:
        bad = ungrounded_numbers(answer, [c["result"] for c in tool_calls])
        if bad:
            fails.append(f"numbers not found in any tool result: {bad}")
    return fails


def load_cases(path) -> list[Case]:
    return [Case(**c) for c in json.loads(open(path).read())]
