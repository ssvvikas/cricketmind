# CricketMind 🏏

Cricket analytics LLM — a hybrid ML + fine tuned model + RAG system for T20 cricket predictions. Covers the IPL, BBL, CPL, PSL and men's T20 internationals.

## What It Does

CricketMind answers five core questions about T20 cricket:

| Module | Question | Method |
|--------|----------|--------|
| **Team Strength** | Which team is stronger head-to-head? | XGBoost classifier with SHAP explainability |
| **Batsman Ranking** | Who's the best batsman in current form? | Composite scoring with weighted metrics |
| **Fielding Analysis** | Who fields best, and how? | Dismissal-based fielding score (catches, run-outs, stumpings by phase). Cricsheet has no fielding positions, so true zone analysis isn't possible |
| **Partnership** | Which batting pair works best together? | Chemistry score from complementary metrics |
| **Bowling Strategy** | Which bowler to start against a batsman? | Vulnerability mapping (bowler strength vs batsman weakness) |

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  RAG Layer  │────▶│  ML Models   │────▶│ Fine-tuned   │
│ (ChromaDB)  │     │ (XGBoost,    │     │ LLM (Qwen/   │
│ Historical  │     │  LightGBM,   │     │ Llama 3B     │
│ stats       │     │  K-Means)    │     │ + LoRA)      │
└─────────────┘     └──────────────┘     └──────────────┘
                                               │
                                               ▼
                                    Natural Language Answer
                                    with supporting numbers
```

- **Classical ML** does the predictions (fast, accurate on tabular data)
- **Fine-tuned LLM** generates insights in natural language
- **RAG pipeline** grounds everything in real Cricsheet data

## Data

Built on [Cricsheet](https://cricsheet.org/) ball-by-ball data, from two sources:

- **TIGZIG Parquet export** of Cricsheet: IPL + men's T20 internationals
- **Direct Cricsheet JSON** (`src/ingestion/cricsheet_json.py`): BBL, CPL, PSL, converted to the same raw layout. The parser was cross-checked against TIGZIG on all 1,243 IPL matches (deliveries, runs, extras, legal balls, wickets, fielders: zero mismatches)

Filtered to men's T20 cricket, innings 1-2 only (super overs are dropped):

| League | Matches | Seasons |
|--------|--------:|---------|
| T20I | 3,558 | 2005-2026 |
| IPL | 1,243 | 2008-2026 |
| BBL | 662 | 2011-2026 |
| CPL | 442 | 2013-2026 |
| PSL | 357 | 2016-2026 |

6,262 matches, ~1.44M deliveries, stored in DuckDB.

**Known limitations**
- Player names are strings, not IDs, so two players sharing a name are merged (Cricsheet's JSON has a name→ID registry that could fix this)
- Renamed franchises appear as separate teams (e.g. Royal Challengers Bangalore/Bengaluru)
- T20I includes associate nations, which skews unadjusted team win rates
- Women's cricket is out of scope

## Team-Strength Model (Phase 2)

`python -m src.models.team_strength` trains an XGBoost classifier for P(team A beats team B)
and writes the model, state, calibration and a metrics/SHAP report to `data/processed/models/`.

**Features** (all computed only from earlier matches; all matches on one date are scored
before any of that date's results are applied):
- Team: Elo per league, recent form and scoring rate, head-to-head, venue record, toss
- **Lineup:** the announced XI is rated from each player's earlier matches: batting runs above
  par, bowling runs saved and wickets above par, weighted by workload and shrunk toward average
  for small samples. Ratings are measured against the par of their era (rolling 400-match
  window) and kept in separate club / international pools, because associate-nation players
  look elite against weak opposition
- Each match is scored from both teams' viewpoints and averaged, so team order can't bias it

**Evaluation:** expanding window, train on everything before year Y and test on Y for
2021-2026, pooled, with bootstrap 95% intervals. Leakage tests flip a result (or edit a match's
ball data) and check no earlier feature changes; they were verified to fail on a deliberate leak.

| Rolling test 2021-26 | Matches | Best model | Accuracy [95% CI] | Log loss (coin flip 0.693) |
|---|---:|---|---|---|
| T20I | 2,587 | XGBoost, team + lineup | 70.1% [68.4, 71.8] | 0.563 (Elo alone: 0.636) |
| Club leagues | 1,105 | any | 50-53% [±3] | 0.69-0.70 (no better than a coin flip) |

**Honest reading**
- **Internationals:** useful. Lineup features improve on team-only features (log loss 0.572 to
  0.563). A plain logistic regression does about as well as XGBoost
- **Club leagues (IPL, BBL, CPL, PSL): no usable win prediction.** The lineup ratings are real:
  a team's rating edge predicts the runs it scores with a slope of about 0.6 runs per point
  (after removing league-year trends). But that edge moves the predicted match margin by only
  ~4 runs against a match-to-match spread of ~43 runs, capping accuracy near 53%. Single T20
  matches between club sides are mostly luck. Reported club probabilities are therefore
  shrunk toward 50% (factor 0.15, fitted out of sample), so they stay honest
- Team history is a poor guide in club cricket because squads turn over (only ~36% of an IPL
  squad returns the next season; season-to-season win% correlation is 0.09)

```python
from src.models.team_strength import predict_match
predict_match("India", "Australia", "T20I")
predict_match("Mumbai Indians", "Chennai Super Kings", "IPL", xi_a=[...])  # optional XI
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/ssvvikas/cricketmind.git
cd cricketmind
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run Phase 1 pipeline (download both sources + compute features)
python -m src.pipeline

# Run tests
pytest tests/ -v
```

## Project Structure

```
cricketmind/
├── data/                    # Data files (gitignored)
│   ├── raw/                 # Cricsheet Parquet downloads
│   ├── processed/           # Feature tables + DuckDB
│   └── embeddings/          # Vector store data
├── src/
│   ├── ingestion/           # TIGZIG download, Cricsheet JSON parser, DB loading
│   ├── features/            # Feature engineering modules
│   │   ├── batsman.py       # Batsman stats + ranking
│   │   ├── bowler.py        # Bowler stats + phase splits
│   │   ├── team.py          # Team strength + head-to-head
│   │   ├── partnership.py   # Partnership chemistry scores
│   │   ├── fielding.py      # Fielding zone analysis
│   │   └── bowling_matchup.py  # Bowler-batsman vulnerability
│   ├── models/              # ML model training (Phase 2)
│   ├── llm/                 # LLM fine-tuning (Phase 3)
│   ├── rag/                 # RAG pipeline (Phase 4)
│   └── api/                 # FastAPI endpoints (Phase 5)
├── tests/                   # pytest test suite
├── pyproject.toml           # Project config + dependencies
└── README.md
```

## Tech Stack

Python 3.11+ · DuckDB · XGBoost · Hugging Face (Transformers + PEFT) · ChromaDB · FastAPI · Docker

## Status

- [x] Phase 1: Data foundation + feature engineering (IPL + T20I)
- [x] Phase 1.5: Direct Cricsheet ingestion (BBL, CPL, PSL)
- [ ] Phase 2: ML models (team-strength done: strong for T20I, ~coin flip for club leagues; batsman, fielding, partnership and bowling-strategy models remaining)
- [ ] Phase 3: LLM fine-tuning
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: API + frontend
- [ ] Phase 6: Polish + deploy

## License

MIT
