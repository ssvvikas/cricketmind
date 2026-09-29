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
- [ ] Phase 2: ML models
- [ ] Phase 3: LLM fine-tuning
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: API + frontend
- [ ] Phase 6: Polish + deploy

## License

MIT
