"""A cricket Q&A agent: Claude + the tools in ``src.llm.tools``.

Every statistic must come from a tool call, and the tools attach plain-language caveats that
the assistant is instructed to pass on (club-league predictions are near coin flips, small
samples, associate-nation pools, ...).

    python -m src.llm.agent "Who should open the bowling against Kohli in the powerplay?"
    python -m src.llm.agent            # interactive chat

Credentials: the SDK reads ANTHROPIC_API_KEY (or an ``ant auth login`` profile).
"""

import sys
from dataclasses import dataclass, field
from typing import Any

from src.llm.tools import TOOL_SPECS, Toolbox

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
MAX_STEPS = 10  # tool-use rounds per question
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are CricketMind, a T20 cricket analyst. You answer questions about \
players, matchups, bowling plans, batting pairs, fielding and match outcomes using tools that \
wrap backtested models over ball-by-ball data (IPL, BBL, CPL, PSL and men's T20 internationals, \
through 2026).

Rules you must follow:
- Every number you state (ratings, ranks, probabilities, expected runs, records) must come \
from a tool result in this conversation. Never recall statistics from memory and never invent \
one. If no tool can answer, say so plainly.
- Names in the data are abbreviated ("V Kohli", "JJ Bumrah"). Call find_player or find_team \
when a name is uncertain. If a name is ambiguous, ask the user which player they mean instead \
of guessing.
- Tool results carry a "caveats" list. Pass on the ones that matter to the answer, in plain \
language and briefly. In particular: club-league (IPL/BBL/CPL/PSL) match predictions are close \
to coin flips and the tool says so; ratings from small samples are unreliable; associate-nation \
ratings are not comparable with club or full-member ratings; head-to-head records are facts, \
not forecasts.
- Choose the pool deliberately: "club" for IPL/BBL/CPL/PSL (combined; it cannot be split by \
league), "intl" for internationals between Full Members, "assoc" for internationals involving \
an associate nation. If the user does not say, pick the sensible one and state which you used.
- Do not do your own arithmetic on statistics (no new differences, ratios, averages or \
complements): quote the figure a tool returns. The tools supply gaps, averages and percentages \
where they matter; if a comparison is not supplied, describe it in words ("slightly more") \
instead of inventing a number.
- Be direct and concise: lead with the answer, then the few numbers that support it. Give \
probabilities as percentages. Do not add generic disclaimers beyond the tool caveats.
- Stay on cricket analytics. You give analysis, not betting advice."""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CricketAgent:
    """Manual tool-use loop around the Messages API, with multi-turn memory."""

    def __init__(
        self,
        toolbox: Toolbox | None = None,
        client: Any = None,
        model: str = MODEL,
        max_steps: int = MAX_STEPS,
        refusal_fallbacks: bool = True,
        effort: str = "medium",
    ) -> None:
        self.toolbox = toolbox or Toolbox()
        self._client = client
        self.model = model
        self.max_steps = max_steps
        self.refusal_fallbacks = refusal_fallbacks
        self.effort = effort
        self.messages: list[dict] = []

    # ── API plumbing ────────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - environment specific
                raise RuntimeError("Install the SDK first: pip install anthropic") from e
            self._client = anthropic.Anthropic()
        return self._client

    def _create(self):
        """One Messages API call.  Thinking is left at the model default (always on for
        Opus 5.5), and effort is set explicitly because Opus 5.5 defaults to medium."""
        params = dict(
            model=self.model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=TOOL_SPECS, messages=self.messages, output_config={"effort": self.effort},
        )
        if self.refusal_fallbacks:  # server-side retry on another model if a classifier declines
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params)
        return self.client.messages.create(**params)

    # ── the loop ────────────────────────────────────────────────────
    def ask(self, question: str) -> AgentResult:
        """Answer one question, continuing any earlier conversation."""
        start = len(self.messages)  # where this question begins, for rollback on refusal
        self.messages.append({"role": "user", "content": question})
        result = AgentResult(text="")

        for step in range(1, self.max_steps + 1):
            response = self._create()
            result.steps = step
            usage = getattr(response, "usage", None)
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0
            result.stop_reason = response.stop_reason

            if response.stop_reason == "refusal":
                # roll the whole exchange back: a refused turn has nothing safe to continue from
                details = getattr(response, "stop_details", None)
                cat = getattr(details, "category", None)
                result.text = ("I can't help with that request"
                               + (f" (declined: {cat})." if cat else "."))
                del self.messages[start:]
                return result

            # append the assistant turn exactly as returned (keeps thinking blocks intact)
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                result.text = self._text(response)
                if response.stop_reason == "max_tokens":
                    result.text += "\n\n[The answer was cut off at the length limit.]"
                return result

            # run every tool the model asked for; all results go back in ONE user message
            tool_results = []
            for block in (b for b in response.content if b.type == "tool_use"):
                out = self.toolbox.call(block.name, block.input)
                result.tool_calls.append(ToolCall(block.name, dict(block.input), out))
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": _dump(out), **({"is_error": True} if "error" in out else {}),
                })
            self.messages.append({"role": "user", "content": tool_results})

        result.text = ("I ran out of steps before finishing. Try a narrower question."
                       + (f"\n\nPartial: {result.text}" if result.text else ""))
        result.stop_reason = "max_steps"
        return result

    @staticmethod
    def _text(response) -> str:
        return "\n".join(b.text for b in response.content if b.type == "text").strip()

    def reset(self) -> None:
        self.messages = []


def _dump(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def main(argv: list[str]) -> None:
    agent = CricketAgent()
    if argv:
        print(agent.ask(" ".join(argv)).text)
        return
    print("CricketMind - ask about players, matchups, pairs or matches. Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q:
            r = agent.ask(q)
            print(f"\n{r.text}\n\n[{len(r.tool_calls)} tool calls, {r.output_tokens} output tokens]")


if __name__ == "__main__":
    main(sys.argv[1:])
