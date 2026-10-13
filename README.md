# CricketMind 🏏

A T20 cricket analyst. Ask it about players, matchups, matches and teams in plain English and it
answers with numbers that come from backtested models and ball by ball data, and it shows which tools
it used. It also tells you when the honest answer is "this is close to a coin flip".

Covers the IPL, BBL, CPL, PSL and men's T20 internationals (6,262 matches, 1.4 million deliveries,
from [Cricsheet](https://cricsheet.org/)).

## What it found

Every model here was tested on data it had not seen, and several ideas did not survive. The honest
results are the main output (details in [docs/RESULTS.md](docs/RESULTS.md)):

| Question | Result |
|---|---|
| Who wins a T20 international? | **70%** of 2,587 matches called correctly (2021-26), against 63% for plain Elo |
| Who wins an IPL / BBL / CPL / PSL match? | **No better than a coin flip** (50-53%). Squads turn over too much and single T20 matches are mostly luck, so the assistant says so and shrinks its probabilities toward 50% |
| Rate batters and bowlers | Era- and phase-adjusted ratings that predict next-year form (rank correlation 0.30 for club batting), but no better than raw strike rate / economy |
| Do batting pairs have "chemistry"? | **No.** A pair's own history predicts its future scoring *worse* than the two batters' individual ratings |
| Does head to head history predict a matchup? | **No.** Adding it made predictions worse at every setting tried |
| Who is the best fielder? | **Not measurable**: the data has no positions, chances or drops. Dismissal counts mostly reflect role (keepers) |
| Do wickets persist year to year? | Barely (0.06 to 0.11): rank bowlers by runs saved, not wickets |

## How it works

```
User question
     │
     ▼
┌──────────────────┐   tool calls   ┌────────────────────────────────┐
│ Claude assistant │───────────────▶│ 19 read-only tools, each with  │
│ (agent loop)     │◀───────────────│ plain-language caveats         │
└──────────────────┘   results      │                                │
     │                              │  ratings, matchups, pairs      │
     ▼                              │  team-strength model (XGBoost) │
Answer, plus the exact              │  counting stats, leaderboards  │
tool calls behind it                │  match search + entity search  │
                                    └────────────────────────────────┘
                                                    │
                                                    ▼
                                      DuckDB over Cricsheet ball-by-ball
```

- **Every number comes from a tool.** The assistant is told never to state a figure a tool did not
  return, and a deterministic eval checks it (every decimal in an answer must appear in a tool result).
- **Caveats travel with the numbers**: small samples, associate-nation opposition, keeper-driven
  fielding counts, near coin flip club predictions.
- **Search over generated summaries**: one summary per match with derived tags ("last-ball finish",
  "big chase"), and an index over the player / team / pair feature tables, both searched with filters
  plus a TF-IDF and embedding blend.

## Quick start

Needs Python 3.11+ (3.12 recommended). On Linux, xgboost needs `libgomp1`; on macOS, `brew install libomp`.

```bash
git clone https://github.com/ssvvikas/cricketmind.git && cd cricketmind
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]"                  # everything the app needs

python -m src.bootstrap                    # downloads the data, builds the database, model and indexes
export ANTHROPIC_API_KEY=...               # stays on the server; never sent to the browser
python -m src.api                          # open http://127.0.0.1:8000
```

`python -m src.bootstrap` is safe to re-run: it skips what already exists (`--check` reports what is
missing, `--force` rebuilds). The first run downloads the data and an embedding model and took 8 minutes on a
clean clone (the second run takes 3 seconds); see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for timings and for running it as a service.

Without an API key everything except chat still works: the models, the statistics and the search.
Try the tools directly:

```python
from src.llm.tools import Toolbox
Toolbox().call("batting_stats", {"name": "Kohli", "league": "IPL", "season": "2016"})
```

## Documentation

| | |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | The models and backtests, including what did not work and why |
| [docs/ASSISTANT.md](docs/ASSISTANT.md) | The 19 tools, the honesty design, the eval, match search and entity search |
| [docs/WEB_APP.md](docs/WEB_APP.md) | The API, the chat page, spend guards, concurrency fixes, what was verified |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Running it for other people: Docker, a TLS proxy, limits, costs, troubleshooting |
| [docs/DATA.md](docs/DATA.md) | Sources, coverage and known data gaps |

## Development

```bash
pip install -e ".[serve,dev]"
make test        # ~500 tests, hermetic: no data, network or API key needed
make lint        # ruff
make eval        # live honesty / tool-use eval (spends API credit; prints an estimated cost)
```

```
src/
  ingestion/   download (TIGZIG export + Cricsheet JSON), parser, DuckDB loading
  features/    feature tables (batsman, bowler, team, partnership, fielding, matchups)
  models/      team-strength model, player ratings (the backtested parts)
  llm/         tools, agent loop, counting statistics, eval checks
  rag/         match summaries + tags + hybrid search, entity index (ChromaDB)
  api/         FastAPI app, sessions, rate limiter
  bootstrap.py one-command setup
frontend/      the chat page (plain HTML / CSS / JS, no build step)
eval/          live eval, corpus audit against independent SQL, retrieval evals
tests/         pytest suite (includes the page's JavaScript, run under Node)
docs/ docker/  documentation, container entrypoint
```

CI (`.github/workflows/ci.yml`) runs lint and the test suite on every push.

## Status

- Data, models, ratings, statistics, search, assistant, API and page are built and tested.
- **Verified:** the full test suite, also passing in CI on GitHub (lint plus tests); a clean clone
  installing and building from scratch; the real server on real data under concurrent load; the page
  in a real browser (progress, tool trace, access token form, rate limit message, phone layout).
- **Not yet verified:** the assistant against the live model through the API (a scripted stand-in
  chose the tool calls) and the Docker image (not built: no Docker on the development machine). The
  last full live eval run was 27/28; the miss was a wording check, since fixed, and the five
  `search_facts` cases have not been run live.
- Not done: fine-tuning a local model (the eval shows no gap that prompting cannot close).

## License

MIT. Data © Cricsheet (CC BY 4.0 / ODC-By; see their site) and the TIGZIG export of it.
