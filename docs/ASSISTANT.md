# The assistant, match search and entity search

The tool layer behind the chat assistant, the retrieval indexes it uses, and the checks that keep it honest. Back to the [README](../README.md).

## The assistant

The assistant is Claude wired to a set of read-only tools (`src/llm/tools.py`) that wrap the
backtested models, so every number it quotes comes from a tested function. Fine-tuning a local
model is deferred; the tool layer is model-agnostic (`TOOL_SPECS` uses the standard
name / description / input_schema shape).

| Tool | Answers |
|---|---|
| `find_player`, `find_team` | "Virat Kohli" -> `V Kohli`, "RCB" -> Royal Challengers Bengaluru; ambiguous names ("Sharma") are refused with candidates |
| `player_profile`, `rank_batters`, `rank_bowlers` | Ratings, ranks and sample sizes per pool |
| `matchup`, `recommend_bowlers` | Phase-aware expected runs per over and dismissal chance |
| `best_pairs`, `head_to_head`, `fielding_profile` | Pair strength, raw records, dismissals per match |
| `predict_match` | Win probability; club leagues are labelled "very low (near coin flip)" |
| `batting_stats`, `bowling_stats` | Career / season / per-league lines: runs, average, strike rate, highest score, 50s/100s, wickets, economy, best figures |
| `leaderboard` | Orange/Purple Cap style tables (runs, wickets, sixes, fours, strike rate, economy, ...) |
| `team_record`, `standings` | Win/loss records, team head-to-heads, season tables |
| `venue_stats` | Average scores and chasing record at a ground |
| `search_matches` | Find matches by tags / filters / description ("last-ball finishes at Chinnaswamy", "biggest chases") with exact counts |
| `search_facts` | Stored records as text for what no other tool returns: strike rate / economy by phase, last-10-innings form, partnership records, team scoring rates |

The counting stats are plain facts using standard definitions (average = runs / dismissals, run-outs
are not the bowler's, wides are not balls faced) and were checked against well-known figures, e.g.
Kohli's 973 runs at 81.08 in IPL 2016, Gill's 890 and Shami's 28 in 2023. Season labels in the
source are inconsistent (IPL 2010 is stored as `2009/10`), so a plain year resolves to the edition
that concluded that year.

The tools do any arithmetic the assistant might want (gaps to the best bowler or pair, runs per dismissal, chasing win %, top X% of a pool), because the eval's grounding check can only verify figures a tool returned. Letting the check accept derived numbers was measured and rejected: it would have accepted 83% of random invented decimals.

Honesty is built in: each result carries `caveats` (small samples, associate-nation pools,
head-to-head not predictive, keeper-driven fielding numbers) and the system prompt tells the
model to pass them on and to state nothing that a tool did not return.

```bash
pip install -e ".[serve]"                 # everything the app needs (the anthropic SDK included)
export ANTHROPIC_API_KEY=...              # or `ant auth login`
python -m src.bootstrap                   # once: builds the data, the match model and the indexes
python -m src.llm.agent "Who should open the bowling to Kohli in the powerplay?"
python eval/run_eval.py                   # 33 honesty / tool-use checks against the live model
```

The agent loop is a manual tool-use loop on `claude-opus-5-5` (effort `medium`, all parallel tool
results returned in one message, assistant turns kept unchanged). Server-side refusal fallbacks
(`fallbacks: "default"`) are on by default and can be turned off with `refusal_fallbacks=False`.
`eval/` scores answers deterministically: the right tool was called, every decimal or percentage
in the answer appears in a tool result, club predictions are called coin flips, ambiguous names
trigger a question, unknown players get no invented numbers.

**Status:** unit-tested offline (a scripted fake client stands in for the API). The most recent
full live eval run was 27/28; the miss was an over-strict wording check, since fixed. The five
`search_facts` cases added afterwards have not been run live.

## Match search

The data has no commentary or match reports, so there is no text to embed in the usual sense. The
RAG layer instead generates one **summary per match** (6,262 of them) from the ball-by-ball data:
result and margin, both innings, top performers, player of the match, centuries and five-wicket
hauls, plus derived **tags** (last-ball finish, close finish, thumping win, big chase, collapse,
century, super over, rain-affected, awarded, ...). Every sentence comes from a computed field,
and the fields are returned with each hit so answers can cite them.

`search_matches` is a hybrid: exact filters (league, season, team, opponent, player, venue,
tags) and a sort (`biggest_chase`, `closest`, `highest_total`, ...), then free-text ranking that
fuses TF-IDF with dense embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`, ONNX, no PyTorch).
It also returns `matched_total`, so "how many last-ball finishes are there?" is an exact count.

```bash
pip install -e ".[rag]"
python -m src.rag.retriever          # builds summaries + index (~3 min, mostly embedding)
python eval/audit_corpus.py          # every tag and figure checked against independent SQL
python eval/run_retrieval_eval.py    # retrieval quality vs ground truth
```

**Correctness.** `eval/audit_corpus.py` recomputes all 16 tags, every innings total and every
century with plain SQL and compares match by match; it found and fixed three real bugs (a null
winner treated as a winner, margins derived for awarded matches, and D/L / awarded matches
handled inconsistently).

**Retrieval quality** (precision@10 over 32 free-text queries; random picking scores 0.08):

| Query group | Keyword only | Hybrid |
|---|---:|---:|
| Worded like the indexed tag synonyms (12) | 0.92 | 0.92 |
| Paraphrases used while developing (10) | 0.48 | 0.66 |
| Held-out paraphrases, never tuned on (10) | 0.69 | 0.82 |

Embeddings were only added because keyword search measurably failed on paraphrases ("went down to
the wire" found nothing). The match search keeps its embeddings in a NumPy array, because 6,262 documents do not need a vector database (the entity index below does use ChromaDB). The
assistant is told to prefer tags and filters for precise questions and to check each hit's facts,
because free-text ranking is still wrong about one time in five.

**Limits.** These are generated summaries, not commentary: no stories, quotes, crowd or
"final / semi-final" information (the source does not record the stage). Margins are the
official ones where present; awarded and rain-shortened matches are excluded from ball-based tags.

## Entity search

`src/rag/entity_index.py` turns every row of the six feature tables (batsman, bowler, team,
partnership, fielding, matchup) into a short text chunk and stores it in a persistent ChromaDB
collection under `data/embeddings/`. Each chunk carries `dataset` ("cricsheet_main"),
`entity_type`, `entity_name` and `league` metadata (only the team table is split by league; the
player tables aggregate all leagues, so their league is "all").

```python
from src.rag.entity_index import build_index, retrieve
build_index()                                        # ~23,000 chunks, ~10 min (embedding)
retrieve("How does Kohli bat in the death overs?")   # -> Kohli's chunks, closest first
retrieve("Sharma", entity_type="bowler", top_k=3)
```

**Name first, then meaning.** If the question names a player or team ("Kohli", "Rohit Sharma",
"CSK"), chunks mentioning that name are selected first and ranked semantically among themselves;
a surname shared by several players searches all of them (up to six, most-played first). Names are
verified as whole words (Chroma's own filter is a substring test, so "S Khan" would otherwise
return "AS Khan"), "head to head" does not summon Travis Head, and everyday words that are also
surnames (Head, Short, Hope, ...) count only when capitalised. If no chunk mentions the name, it
falls back to plain semantic search and says so.

**Content rules, from the backtests.** The retired Phase 1 scores (`legacy_score`, chemistry,
vulnerability, fielding score) are never written into a chunk; fielding chunks say they count
dismissal involvement, not skill; matchup chunks say they are raw small-sample records. Only pairs
with at least 30 balls are indexed (5,121 of 82,722; most pairs have 6-11 balls, which is noise,
and embedding all of them would take about half an hour): `build_index(min_matchup_balls=None)`
indexes everything.

**Chunk-type hint.** Words like "bat", "economy", "catches" or "partnership" point at a chunk type
(and "against" with two named players at a matchup). Those types are fetched into the candidate
pool explicitly and get a small fixed boost (0.05) on their similarity; `retrieve(...,
intent_boost=0)` turns it off. Without this, a player's short numeric matchup and partnership
chunks crowd out the right chunk: "Kohli's strike rate" returned a Kohli-vs-Arshdeep matchup.

Measured on questions with a known right chunk type (`python eval/run_entity_eval.py`), top-1 is
the right chunk type:

| Question set | Hint off | Hint on |
|---|---:|---:|
| Development set, 25 questions (used to look for weaknesses) | 68% | 100% |
| Held-out set, 20 questions (never tuned on) | 45% | **85%** (top-3: 95%) |

The weight is not sensitive (anything from 0.05 to 1.0 scores the same on the development set), so
the 0.05 chosen in advance was kept. Known misses: "How does Klaasen **hit**..." (no such word in
the hint vocabulary) and two "batting average" questions where a partnership chunk still wins.

**What it cannot do.** Embeddings find text that resembles a question; they cannot rank by a
statistic. "Who scored the most runs?" or "best wicketkeeper" return arbitrary chunks, so a
question with a superlative and no name comes back with a `warning` pointing to the leaderboard and
ranking tools. It also has no team-vs-team chunks (use `team_record`).

The index is exposed to the assistant as the `search_facts` tool. Its result names the players it matched and carries caveats that depend on what came back (a surname shared by several players tells the assistant to ask which one; matchup, fielding and partnership records carry their own limits; a ranking question carries the no-ranking warning).
