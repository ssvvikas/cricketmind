"""Hybrid retrieval over match summaries: exact filters + tags + optional sort + TF-IDF text rank.

  * filters   league, season (a plain year finds the edition that concluded then), team,
              opponent, player, venue/city, required tags.  ``player`` means "notable in the
              match" (30+ runs, 2+ wickets or player of the match) - except with the century /
              five-wicket haul tags, where the player must be the one who scored / took them
  * sort      relevance (default with a query), newest, oldest, biggest_chase, highest_total,
              biggest_win, closest
  * text      hybrid ranking: TF-IDF (exact words, names, venues; tag synonyms are indexed) fused
              50/50 with dense embeddings (paraphrases such as "went down to the wire").  On
              paraphrases the index had never seen, precision@10 was 0.74 for the hybrid vs 0.56
              for TF-IDF alone and 0.62 for embeddings alone.  If the embedding model is not
              available, ranking falls back to TF-IDF only and says so.

The number of matches that passed the filters is returned with the hits, so "how many" questions
about tagged matches are exact.
"""

import threading
from collections.abc import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PROCESSED_DIR
from src.rag.corpus import CORPUS_PATH, TAGS, build_corpus, save_corpus

EMBEDDINGS_PATH = PROCESSED_DIR / "rag" / "match_embeddings.npy"
INDEX_PATH = CORPUS_PATH  # the index is rebuilt from this file (+ the embeddings) on load
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HYBRID_WEIGHT = 0.5  # share of the fused score from embeddings (fixed in advance, not tuned)
_embedder = None
_embed_lock = threading.Lock()
SORTS = ("relevance", "newest", "oldest", "biggest_chase", "highest_total", "biggest_win",
         "closest")
FACT_FIELDS = ["match_id", "date", "league", "season", "venue", "team1", "team2", "result",
               "winner", "margin_type", "margin",
               "first_team", "first_score", "first_wkts", "second_team", "second_score",
               "second_wkts", "target", "balls_left", "player_of_match", "centuries",
               "five_wicket_hauls", "tags", "top_first", "top_second"]


class NoMatches(ValueError):
    pass


def _plain(v):
    """numpy -> Python, NaN -> None (so results are valid JSON)."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple, np.ndarray)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def default_embed_fn() -> Callable[[list[str]], np.ndarray] | None:
    """The fastembed ONNX model (downloaded on first use), or None if unavailable."""
    global _embedder
    if _embedder is None:
        with _embed_lock:  # concurrent first requests must not each load the model
            if _embedder is None:
                try:
                    from fastembed import TextEmbedding

                    model = TextEmbedding(EMBED_MODEL)
                    _embedder = lambda texts: np.array(  # noqa: E731
                        list(model.embed(texts, batch_size=64)), dtype=np.float32)
                except Exception as e:  # missing package, offline, corrupt cache...
                    logger.warning(f"Embeddings unavailable ({type(e).__name__}); keyword "
                                   "search only")
                    return None
    return _embedder


class MatchIndex:
    def __init__(self, corpus: pd.DataFrame, embeddings: np.ndarray | None = None,
                 embed_fn: Callable[[list[str]], np.ndarray] | None = None) -> None:
        self.corpus = corpus.reset_index(drop=True)
        self.embeddings = embeddings
        self._embed_fn = embed_fn  # not pickled; re-created on load
        # min_df=1: a name or venue that appears in a single match must stay searchable
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                   lowercase=True)
        self.matrix = self.vec.fit_transform(self.corpus["text"])
        self._season_last = (self.corpus.groupby(["league", "season"])["date"].max()
                             .reset_index())

    # ── filters ─────────────────────────────────────────────────────
    def _season_mask(self, league, season) -> np.ndarray:
        sl = self._season_last if not league else self._season_last[self._season_last.league == league]
        season = str(season).strip()
        picks = set(sl.loc[sl.season == season, "season"])
        if season.isdigit() and len(season) == 4:
            picks |= set(sl.loc[sl.date.dt.year == int(season), "season"])
        if not picks:
            raise NoMatches(f"No matches for season {season!r}"
                            + (f" in {league}" if league else "")
                            + f". Available: {sorted(set(sl.season))}")
        return self.corpus["season"].isin(picks).to_numpy()

    def mask(self, league=None, season=None, team=None, opponent=None, player=None,
             venue=None, tags=None) -> np.ndarray:
        c = self.corpus
        m = np.ones(len(c), dtype=bool)
        if league:
            m &= (c["league"] == league).to_numpy()
        if season:
            m &= self._season_mask(league, season)
        if team:
            t = _norm(team)
            m &= (c["team1"].map(_norm).eq(t) | c["team2"].map(_norm).eq(t)).to_numpy()
        if opponent:
            o = _norm(opponent)
            m &= (c["team1"].map(_norm).eq(o) | c["team2"].map(_norm).eq(o)).to_numpy()
        if player:
            p = _norm(player)
            # with these tags the player must be the one who did it, not merely someone notable
            col = ("centuries" if "century" in (tags or [])
                   else "five_wicket_hauls" if "five-wicket haul" in (tags or [])
                   else "notable_players")
            m &= c[col].map(lambda ps: any(_norm(x) == p for x in ps)).to_numpy()
        if venue:
            v = _norm(venue)
            m &= (c["venue"].map(_norm).str.contains(v, regex=False)
                  | c["city"].fillna("").map(_norm).str.contains(v, regex=False)).to_numpy()
        for t in tags or []:
            if t not in TAGS:
                raise ValueError(f"unknown tag {t!r}; tags are {TAGS}")
            m &= c["tags"].map(lambda ts, t=t: t in ts).to_numpy()
        return m

    # ── scoring ─────────────────────────────────────────────────────
    def _embedder(self):
        if self.embeddings is None:
            return None
        return self._embed_fn or default_embed_fn()

    def scores(self, query: str, ranking: str = "hybrid") -> tuple[np.ndarray, str]:
        """Relevance of every document to the query, and which ranking produced it.

        Scores are standardised over the whole corpus (not the filtered pool), so a query is
        scored the same whatever filters are applied.
        """
        tf = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        embed = self._embedder() if ranking == "hybrid" else None
        if embed is None:
            return tf, "keyword"
        qv = embed([query])[0]
        D = self.embeddings
        em = D @ qv / (np.linalg.norm(D, axis=1) * np.linalg.norm(qv) + 1e-12)
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
        return (1 - HYBRID_WEIGHT) * z(tf) + HYBRID_WEIGHT * z(em), "hybrid"

    # ── ordering ────────────────────────────────────────────────────
    def _order(self, idx: np.ndarray, query, sort, ranking: str = "hybrid"):
        """(ordered row indices, {row: relevance}, ranking label).

        Everything is returned, never stored on ``self``: one index is shared by concurrent
        requests, and per-query state on the instance would let two searches swap results.
        """
        c = self.corpus.iloc[idx]
        if sort in (None, "relevance") and query:
            full, label = self.scores(query, ranking)
            score = full[idx]
            order = np.lexsort((-c["date"].astype("int64").to_numpy(), -score))
            return idx[order], dict(zip(idx, score)), label
        key = {
            "newest": -c["date"].astype("int64").to_numpy().astype(float),
            "oldest": c["date"].astype("int64").to_numpy().astype(float),
            "biggest_chase": -c["target"].fillna(-1).to_numpy(float),
            "highest_total": -c[["first_score", "second_score"]].max(axis=1).fillna(-1).to_numpy(float),
            "biggest_win": -np.where(c["margin_type"] == "runs", c["margin"].fillna(-1), -1
                                     ).astype(float),
        }.get(sort or "newest")
        if sort == "closest":
            tie = c["tags"].map(lambda t: "tie" in t).to_numpy()
            runs = np.where(c["margin_type"] == "runs", c["margin"], np.nan).astype(float)
            wk_balls = np.where(c["margin_type"] == "wickets",
                                c["balls_left"].fillna(c["margin"] * 6), np.nan).astype(float)
            closeness = np.where(tie, -1.0, np.where(np.isnan(runs), wk_balls, runs))
            closeness = np.where(np.isnan(closeness), 1e9, closeness)  # no margin -> last
            key = closeness
        order = np.argsort(key, kind="stable")
        return idx[order], {}, None

    # ── public ──────────────────────────────────────────────────────
    def search(self, query: str | None = None, *, league=None, season=None, team=None,
               opponent=None, player=None, venue=None, tags=None, sort=None,
               top: int = 8, ranking: str = "hybrid") -> dict:
        if sort and sort not in SORTS:
            raise ValueError(f"sort must be one of {SORTS}")
        m = self.mask(league, season, team, opponent, player, venue, tags)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            raise NoMatches("No matches satisfy those filters")
        if sort == "biggest_chase":  # only chases have a target
            idx = idx[self.corpus.iloc[idx]["target"].notna().to_numpy()]
        if sort == "biggest_win":
            idx = idx[(self.corpus.iloc[idx]["margin_type"] == "runs").to_numpy()]
        if len(idx) == 0:
            raise NoMatches("No matches satisfy those filters for that sort")
        ordered, relevance, label = self._order(idx, query, sort, ranking)
        top = max(1, min(int(top), 25))
        hits = []
        for i in ordered[:top]:
            row = self.corpus.iloc[i]
            hit = {k: _plain(row[k]) for k in FACT_FIELDS}
            hit["date"] = str(pd.Timestamp(row["date"]).date())
            hit["tags"], hit["centuries"] = list(row["tags"]), list(row["centuries"])
            hit["five_wicket_hauls"] = list(row["five_wicket_hauls"])
            hit["summary"] = row["text"]
            if query and (sort in (None, "relevance")):
                hit["relevance"] = round(float(relevance.get(i, 0.0)), 3)
            hits.append(hit)
        out = {"matched_total": int(len(idx)), "returned": len(hits), "hits": hits}
        if label:
            out["ranking"] = label
        return out

    # ── persistence ─────────────────────────────────────────────────
    # No pickling of this class: a pickle written by ``python -m src.rag.retriever`` records the
    # class as ``__main__.MatchIndex`` and cannot be loaded from any other entry point.  The index
    # is stored as plain files (Parquet + NumPy) and the fast TF-IDF step is refitted on load.
    def save(self, corpus_path=CORPUS_PATH, embeddings_path=EMBEDDINGS_PATH) -> None:
        save_corpus(self.corpus, corpus_path)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)
        elif embeddings_path.exists():
            embeddings_path.unlink()  # never leave embeddings that do not match the corpus

    @staticmethod
    def load(corpus_path=CORPUS_PATH, embeddings_path=EMBEDDINGS_PATH) -> "MatchIndex":
        corpus = pd.read_parquet(corpus_path)
        emb = np.load(embeddings_path) if embeddings_path.exists() else None
        if emb is not None and len(emb) != len(corpus):
            logger.warning("Embeddings do not match the corpus; ignoring them (rebuild the index)")
            emb = None
        return MatchIndex(corpus, emb)


def build_index(db_path: str | None = None, save: bool = True, embed: bool = True) -> MatchIndex:
    """Build the corpus and index; embeddings take ~2.5 minutes for the full data."""
    corpus = build_corpus(db_path)
    embeddings, fn = None, None
    if embed:
        fn = default_embed_fn()
        if fn is not None:
            logger.info(f"Embedding {len(corpus):,} match summaries with {EMBED_MODEL}...")
            embeddings = fn(list(corpus["text"]))
    idx = MatchIndex(corpus, embeddings, fn)
    if save:
        idx.save()
        logger.info(f"Saved match index to {CORPUS_PATH.parent}")
    return idx


if __name__ == "__main__":
    build_index()
