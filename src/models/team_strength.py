"""Team-strength model: P(team A beats team B), XGBoost with SHAP explanations.

Evaluation is temporal (train <= 2023, validate 2024, test >= 2025) and compared
with two baselines - the pure Elo probability and a logistic regression - so the
gradient-boosted model has to earn its complexity.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR
from src.models.team_dataset import (
    FEATURES,
    LEAGUES,
    TeamState,
    build_dataset,
    elo_expected,
)

MODEL_DIR = PROCESSED_DIR / "models"
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
    "n_jobs": 1,
    "random_state": 42,
}


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Temporal split by match date (both rows of a match land in the same split)."""
    d = pd.to_datetime(df["date"])
    return {
        "train": df[d <= TRAIN_END],
        "val": df[(d > TRAIN_END) & (d <= VAL_END)],
        "test": df[d > VAL_END],
    }


def symmetrise(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Combine the two perspective rows of each match into one team1 probability.

    p1 = mean(P(team1 beats team2), 1 - P(team2 beats team1)).
    """
    tmp = df[["match_id", "team1_is_a", "y"]].copy()
    tmp["p_team1"] = np.where(tmp["team1_is_a"], p, 1 - p)
    tmp["y_team1"] = np.where(tmp["team1_is_a"], tmp["y"], 1 - tmp["y"])
    return tmp.groupby("match_id").agg(p=("p_team1", "mean"), y=("y_team1", "first"))


def metrics(match_df: pd.DataFrame) -> dict:
    p, y = match_df["p"].clip(1e-6, 1 - 1e-6), match_df["y"]
    return {
        "n_matches": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p > 0.5)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
    }


def train(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Train, evaluate against baselines, and (optionally) save artifacts."""
    state = None
    if df is None:
        df, state = build_dataset()
    parts = split(df)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    logger.info(f"Matches - train {tr.match_id.nunique():,}, val {va.match_id.nunique():,}, "
                f"test {te.match_id.nunique():,}")

    # ── XGBoost, early-stopped on the validation season ────────────
    model = xgb.XGBClassifier(n_estimators=1500, early_stopping_rounds=50, **XGB_PARAMS)
    model.fit(tr[FEATURES], tr["y"], eval_set=[(va[FEATURES], va["y"])], verbose=False)
    best_iter = int(model.best_iteration) + 1

    # ── baselines ──────────────────────────────────────────────────
    logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=1000))
    logit.fit(tr[FEATURES], tr["y"])

    predictors = {
        "base_rate": lambda d: np.full(len(d), 0.5),
        "elo_only": lambda d: 1 / (1 + 10 ** (-d["elo_diff"].to_numpy() / 400)),
        "logistic": lambda d: logit.predict_proba(d[FEATURES])[:, 1],
        "xgboost": lambda d: model.predict_proba(d[FEATURES])[:, 1],
    }
    results = {
        name: {s: metrics(symmetrise(d, fn(d))) for s, d in (("val", va), ("test", te))}
        for name, fn in predictors.items()
    }
    # a constant 0.5 has no meaningful accuracy (ties resolve to one side), so drop it
    for s in results["base_rate"].values():
        s["accuracy"] = None

    # T20I is easy (established nations vs associates) and dominates the test set, so
    # club leagues are reported as their own group - that is the hard, interesting one
    groups = {}
    for name, fn in predictors.items():
        groups[name] = {}
        for label, mask in (("club", te["league"] != "T20I"), ("t20i", te["league"] == "T20I")):
            groups[name][label] = metrics(symmetrise(te[mask], fn(te[mask])))
    groups["base_rate"]["club"]["accuracy"] = groups["base_rate"]["t20i"]["accuracy"] = None

    # accuracy by league on test (xgboost) + calibration
    mt = symmetrise(te, predictors["xgboost"](te))
    league_of = te.drop_duplicates("match_id").set_index("match_id")["league"]
    mt["league"] = league_of
    by_league = {
        lg: metrics(g) for lg, g in mt.groupby("league") if len(g) >= 30
    }
    bins = pd.cut(mt["p"], [0, 0.35, 0.45, 0.5, 0.55, 0.65, 1.0])
    calibration = {
        str(k): {"n": int(len(g)), "mean_pred": round(float(g["p"].mean()), 3),
                 "actual_rate": round(float(g["y"].mean()), 3)}
        for k, g in mt.groupby(bins, observed=True)
    }

    # ── SHAP on the test set ───────────────────────────────────────
    import shap

    sv = shap.TreeExplainer(model).shap_values(te[FEATURES])
    importance = dict(sorted(
        zip(FEATURES, np.abs(sv).mean(axis=0).round(4).tolist(), strict=True),
        key=lambda kv: -kv[1],
    ))

    report = {
        "split": {"train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date())},
        "best_iteration": best_iter,
        "results": results,
        "test_by_group": groups,
        "xgboost_test_by_league": by_league,
        "xgboost_test_calibration": calibration,
        "shap_mean_abs": importance,
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # production model: same settings, refit on ALL matches (incl. val + test)
        final = xgb.XGBClassifier(n_estimators=best_iter, **XGB_PARAMS)
        final.fit(df[FEATURES], df["y"])
        final.save_model(MODEL_DIR / "team_strength.json")
        if state is not None:
            with open(MODEL_DIR / "team_state.pkl", "wb") as f:
                pickle.dump(state, f)
        (MODEL_DIR / "team_strength_report.json").write_text(json.dumps(report, indent=2))
        logger.info(f"Saved model, state and report to {MODEL_DIR}")
    return report


def predict_match(
    team_a: str, team_b: str, league: str, venue: str | None = None,
    toss_winner: str | None = None, model_dir: Path | None = None,
) -> dict:
    """P(team_a beats team_b) as of the end of the training data."""
    d = Path(model_dir or MODEL_DIR)
    model = xgb.XGBClassifier()
    model.load_model(d / "team_strength.json")
    with open(d / "team_state.pkl", "rb") as f:
        state: TeamState = pickle.load(f)

    if league not in LEAGUES:
        raise ValueError(f"league must be one of {LEAGUES}")
    for t in (team_a, team_b):
        if state.played[(league, t)] == 0:
            raise ValueError(f"No history for {t!r} in {league}")

    rows = [state.features(league, a, b, venue, toss_winner)
            for a, b in ((team_a, team_b), (team_b, team_a))]
    p = model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    p_a = float((p[0] + 1 - p[1]) / 2)
    return {
        "team_a": team_a, "team_b": team_b, "league": league,
        "p_team_a": round(p_a, 4), "p_team_b": round(1 - p_a, 4),
        "elo_a": round(state.elo[(league, team_a)], 1),
        "elo_b": round(state.elo[(league, team_b)], 1),
        "elo_only_p_a": round(elo_expected(state.elo[(league, team_a)],
                                           state.elo[(league, team_b)]), 4),
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
