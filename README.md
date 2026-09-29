# CricketMind 🏏

**Cricket analytics LLM** — a hybrid ML + fine-tuned model + RAG system for T20 cricket predictions across IPL, BBL, CPL, PSL, and international T20s.

## What It Does

CricketMind answers five core questions about T20 cricket:

| Module | Question | Method |
|--------|----------|--------|
| **Team Strength** | Which team is stronger head-to-head? | XGBoost classifier with SHAP explainability |
| **Batsman Ranking** | Who's the best batsman in current form? | Composite scoring with weighted metrics |
| **Fielding Analysis** | Who fields best in which zone? | K-Means clustering over zone-mapped dismissals |
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

Built on [Cricsheet](https://cricsheet.org/) ball-by-ball data:
- 11,000+ matches, 5M+ deliveries from 2001 onward
- Covers IPL, BBL, CPL, PSL, and international T20s
- Downloaded as Parquet files, stored in DuckDB

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/cricketmind.git
cd cricketmind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run Phase 1 pipeline (download data + compute features)
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
│   ├── ingestion/           # Data download + DB loading
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

- [x] Phase 1: Data foundation + feature engineering
- [ ] Phase 2: ML models
- [ ] Phase 3: LLM fine-tuning
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: API + frontend
- [ ] Phase 6: Polish + deploy

## License

MIT
