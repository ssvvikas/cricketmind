# Deployment

How to run CricketMind for other people. Read [WEB_APP.md](WEB_APP.md) first for what the server does and
what its guards are. Measured numbers below are marked with the date and machine; anything not measured
says so.

## What you are running

One Python process (FastAPI + uvicorn) that serves the chat page, answers questions through the Claude
API, and reads a local DuckDB file plus a few parquet files, a model file and a ChromaDB folder. There is
no other service. The only outside call at runtime is to the Anthropic API.

Conversations live in memory: restarting the process forgets them (users just start a new chat).
Run **one worker**. Sessions, the rate limiter and the warm-up state are per process, so several
workers would each keep their own and the limits would multiply.

## Setup (any machine)

```bash
pip install -e ".[serve]"
python -m src.bootstrap          # builds data, model, entity index; skips what exists
python -m src.bootstrap --check  # exit 1 if anything is missing (use in deploy scripts)
python -m src.api
```

Bootstrap steps, each skipped when its output exists (`--force` rebuilds, `--only data|model|entities`
limits it): download Cricsheet data and build the database and feature tables; train the team-strength
model; build the entity index (this downloads the embedding model on first use, set
`FASTEMBED_CACHE_PATH` to keep it somewhere persistent).

Measured on 2026-10-04 on the development Mac, clean clone, home broadband:

| | |
|---|---|
| `pip install -e ".[serve]"` | 29 s (with dev extras) |
| `bootstrap`, data step | 188 s |
| `bootstrap`, model step | 16 s |
| `bootstrap`, entity index step | 274 s (includes the embedding model download) |
| **Total first build** | **480 s (8 minutes)** |
| Second `bootstrap` (all skipped) | 3 s |
| Server start to `ready: true` | 7 s |
| Memory after warm-up (RSS) | about 1.5 GB |
| `data/` on disk | 219 MB |

Plan for **2 GB of RAM** at least; memory under sustained load was not measured. Timings on a
small cloud VM will differ (not measured).

The app finds `frontend/` and `data/` relative to the source tree, so run it from a checkout
(`pip install -e .`), not from an installed wheel.

## Configuration

All settings are environment variables with the prefix `CRICKETMIND_API_`:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | The model credential. Server side only; never sent to the browser |
| `CRICKETMIND_API_HOST` / `_PORT` | `127.0.0.1` / `8000` | Bind address. Loopback until you put a proxy in front |
| `CRICKETMIND_API_ACCESS_TOKEN` | none | When set, chat requests need it (`Authorization: Bearer` or `X-Access-Token`) |
| `CRICKETMIND_API_RATE_LIMIT_PER_MINUTE` | 10 | Chat requests per client address per minute |
| `CRICKETMIND_API_MAX_TURNS_PER_SESSION` | 20 | Bounds context size and spend per conversation |
| `CRICKETMIND_API_MAX_SESSIONS` | 200 | Simultaneous conversations |
| `CRICKETMIND_API_SESSION_TTL_SECONDS` | 1800 | Idle conversations are dropped after this |
| `CRICKETMIND_API_MAX_MESSAGE_CHARS` | 1000 | Longest accepted question |
| `CRICKETMIND_API_MODEL` | `claude-opus-5-5` | Model for answers (see costs) |
| `CRICKETMIND_API_TRUST_PROXY_HEADERS` | false | See "Behind a proxy" |

## Putting it on the internet

Anything public spends your API credit, so do all of these:

1. **Set an access token** (`CRICKETMIND_API_ACCESS_TOKEN`) unless the page is meant to be open to everyone.
2. **Set a spend limit** on the API account in the Anthropic console, and a low alert. The app's limits
   (rate, turns per chat, chats at once) bound the worst case, but only the console limit is a hard stop.
3. **Terminate TLS in a reverse proxy** (Caddy, nginx, a platform load balancer). The app speaks plain HTTP.
4. Keep the app bound to loopback or a private network and expose only the proxy.

### Behind a proxy

- **Turn response buffering off for `/api/chat/stream`**, or progress updates arrive in one lump at the
  end. The app sends `X-Accel-Buffering: no` (nginx honours it); for others disable it explicitly.
  Allow responses to stay open for a few minutes (nginx `proxy_read_timeout 300s`). The app sends a
  keep-alive comment every 15 seconds.
- Without `TRUST_PROXY_HEADERS`, every user appears to come from the proxy's address and **shares one
  rate limit**. Set it to `true` **only** when the proxy overwrites `X-Forwarded-For` (rather than
  appending to what the client sent), otherwise anyone can dodge the limit by forging the header.

Minimal Caddyfile:

```
cricket.example.com {
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1
        header_up X-Forwarded-For {remote_host}
    }
}
```

(with `CRICKETMIND_API_TRUST_PROXY_HEADERS=true`.) This Caddyfile has not been run; check it against
your Caddy version.

## Docker

`Dockerfile`, `docker-compose.yml` and `docker/entrypoint.sh` are provided. **They have not been built
or run**: the development machine has no Docker. Treat them as a starting point and expect to fix small
things on the first build. The design:

- The image holds code and dependencies, not data. On start the entrypoint runs `python -m src.bootstrap`
  (a no-op when the volume already has the data) and then the server, so a fresh volume builds itself.
- `data/` and the embedding-model cache sit on a volume so rebuilds and restarts are fast.
- `--build-arg BAKE_DATA=1` builds the data into the image instead (bigger image, no first-start wait).
- The health check calls `/api/health` and passes only when `ready` is true, with a long start period
  for the first build.

```bash
docker compose up --build     # reads ANTHROPIC_API_KEY (and optional token) from your environment
```

## Cost

The assistant uses a tool loop, so one question costs several model calls. Rough list-price formula,
which the server computes for you:

    cost = input_tokens × input_price + output_tokens × output_price

At `claude-opus-5-5` list prices ($4 / $20 per million input / output tokens) the server logs each
answer and returns it to the client:

    chat session=ab12cd34 turn=2 tools=['batting_stats'] steps=3 tokens=9120+640 cost=$0.049 seconds=14.2

`usage.estimated_cost_usd` is in every chat response too. Prices come from `PRICES` in
`src/llm/agent.py` and are a **list-price estimate**: it ignores caching and any discounts, and
returns nothing for a model it has no price for. Confirm real spend in the Anthropic console.

The eval run prints the same estimate (`make eval`), so you can measure your own per-question average
before opening the page to anyone. To spend less per answer, set `CRICKETMIND_API_MODEL` to a cheaper
model and re-run the eval first: the honesty checks are the part to watch.

## Backups and rebuilding

Everything under `data/` is derived and can be rebuilt, so there is nothing to back up except your
settings. To rebuild after the source data changes or to pick up new matches:

```bash
python -m src.bootstrap --force
```

then restart the server. Rebuilding changes numbers (new matches shift ratings), so expect small
differences against earlier answers.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Chat returns 503 "no API credentials" | `ANTHROPIC_API_KEY` is not set in the server's environment. `/api/health` shows `credentials_in_environment` |
| Chat returns 503 "out of credit" | The API account balance is empty; top it up in the Anthropic console |
| Chat returns 503 "credentials were rejected" | The key is wrong or revoked |
| `/api/health` says `ready: false` | Still warming up (first minute after start), or the core ratings failed to load: check the log and `python -m src.bootstrap --check` |
| Everything works except entity/name search | The entity index is missing: `python -m src.bootstrap --only entities` |
| Import error mentioning `libomp` (macOS) or `libgomp` (Linux) | xgboost needs its OpenMP runtime: `brew install libomp` or `apt-get install libgomp1` |
| Progress shows all at once at the end | The proxy is buffering the stream; see "Behind a proxy" |
| Everyone gets 429 | Behind a proxy without `TRUST_PROXY_HEADERS`: all users share one address |
| Answers feel stale | Data is a snapshot from the last bootstrap; see "Backups and rebuilding" |
| Afghanistan (or another side) has no data | Cricsheet does not cover it; see [DATA.md](DATA.md) |

## What has and has not been verified

- Verified: install and a full from-scratch bootstrap in a clean clone; the second bootstrap skipping
  everything; the server starting from that clone with health reporting ready; chat without credentials
  returning the 503 above; the concurrent-load behaviour described in WEB_APP.md.
- Not verified: Docker build and run, the Caddy snippet, any real hosting platform, long-running memory
  behaviour (hours or days), and the live model through the API under real traffic.
