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
    FEATURES_TEAM,
    LEAGUES,
    LINEUP_FEATURES,
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


ROLLING_XGB = {**XGB_PARAMS, "n_estimators": 250}


def _fit_predict(kind: str, cols: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    if kind == "xgboost":
        m = xgb.XGBClassifier(**ROLLING_XGB).fit(tr[cols], tr["y"])
        return m.predict_proba(te[cols])[:, 1]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.5, max_iter=1000)).fit(tr[cols], tr["y"])
    return m.predict_proba(te[cols])[:, 1]


def rolling_eval(
    df: pd.DataFrame, leagues: list[str] | None = None, first_test_year: int = 2021,
    n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Expanding-window evaluation: for each year Y train on dates < Y, test on Y.

    Pooling several seasons gives a much tighter estimate than one season.  Models are
    trained and tested only on ``leagues`` (default: the four club leagues).  Bootstrap
    intervals resample whole matches.
    """
    leagues = leagues or [lg for lg in LEAGUES if lg != "T20I"]
    d = df[df["league"].isin(leagues)].copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    sets = {
        "team_only": FEATURES_TEAM,
        "lineup_only": LINEUP_FEATURES + ["league_code"],
        "team+lineup": FEATURES,
    }
    preds: dict[str, list[pd.DataFrame]] = {}
    for year in range(first_test_year, int(d["year"].max()) + 1):
        tr, te = d[d["year"] < year], d[d["year"] == year]
        if te.empty:
            continue
        variants = {"elo_only": 1 / (1 + 10 ** (-te["elo_diff"].to_numpy() / 400))}
        for set_name, cols in sets.items():
            for kind in ("xgboost", "logistic"):
                variants[f"{kind}:{set_name}"] = _fit_predict(kind, cols, tr, te)
        for name, p in variants.items():
            preds.setdefault(name, []).append(symmetrise(te, p))

    rng = np.random.default_rng(seed)
    out = {}
    for name, parts in preds.items():
        m = pd.concat(parts)
        p, y = m["p"].clip(1e-6, 1 - 1e-6).to_numpy(), m["y"].to_numpy()
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        ll = [log_loss(y[i], p[i]) for i in idx]
        auc = [roc_auc_score(y[i], p[i]) for i in idx]
        acc = [accuracy_score(y[i], p[i] > 0.5) for i in idx]
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
        out[name] = {
            "n_matches": int(len(y)),
            "accuracy": round(float(accuracy_score(y, p > 0.5)), 4), "accuracy_ci": ci(acc),
            "log_loss": round(float(log_loss(y, p)), 4), "log_loss_ci": ci(ll),
            "auc": round(float(roc_auc_score(y, p)), 4), "auc_ci": ci(auc),
        }
    return out


def fit_club_shrinkage(df: pd.DataFrame, first_test_year: int = 2021) -> dict:
    """Fit how far club-league probabilities should be pulled toward 0.5.

    Uses out-of-sample expanding-window predictions on club matches and picks the factor
    lam in [0, 1] minimising log loss of  p' = 0.5 + lam * (p - 0.5).  lam = 1 means the
    model is already well calibrated; lam near 0 means it has no club-league skill.
    """
    club = df[df["league"] != "T20I"].copy()
    club["year"] = pd.to_datetime(club["date"]).dt.year
    parts = []
    for year in range(first_test_year, int(club["year"].max()) + 1):
        tr, te = club[club["year"] < year], club[club["year"] == year]
        if not te.empty:
            parts.append(symmetrise(te, _fit_predict("xgboost", FEATURES, tr, te)))
    m = pd.concat(parts)
    grid = np.linspace(0, 1, 101)
    losses = [log_loss(m["y"], (0.5 + lam * (m["p"] - 0.5)).clip(1e-6, 1 - 1e-6)) for lam in grid]
    best = int(np.argmin(losses))
    return {
        "lambda": round(float(grid[best]), 2),
        "log_loss_raw": round(float(losses[-1]), 4),
        "log_loss_shrunk": round(float(losses[best]), 4),
        "n_matches": int(len(m)),
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

    club_cal = fit_club_shrinkage(df)
    logger.info(f"Club shrinkage: {club_cal}")

    report = {
        "club_calibration": club_cal,
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
        (MODEL_DIR / "calibration.json").write_text(json.dumps(club_cal))
        logger.info(f"Saved model, state and report to {MODEL_DIR}")
    return report


def predict_match(
    team_a: str, team_b: str, league: str, venue: str | None = None,
    toss_winner: str | None = None, xi_a: list[str] | None = None,
    xi_b: list[str] | None = None, model_dir: Path | None = None,
) -> dict:
    """P(team_a beats team_b) as of the end of the training data.

    ``xi_a`` / ``xi_b`` are the playing XIs (player names as in Cricsheet); when omitted
    each team's most recent XI is used.
    """
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

    xi_a = xi_a or state.last_xi.get((league, team_a))
    xi_b = xi_b or state.last_xi.get((league, team_b))
    rows = [state.features(league, a, b, venue, toss_winner, xa, xb)
            for a, b, xa, xb in ((team_a, team_b, xi_a, xi_b), (team_b, team_a, xi_b, xi_a))]
    p = model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    p_a = float((p[0] + 1 - p[1]) / 2)
    lam = 1.0
    if league != "T20I":  # club leagues: pull toward 0.5 by the fitted out-of-sample factor
        cal = d / "calibration.json"
        lam = json.loads(cal.read_text())["lambda"] if cal.exists() else 1.0
        p_a = 0.5 + lam * (p_a - 0.5)
    return {
        "team_a": team_a, "team_b": team_b, "league": league,
        "p_team_a": round(p_a, 4), "p_team_b": round(1 - p_a, 4),
        "elo_a": round(state.elo[(league, team_a)], 1),
        "elo_b": round(state.elo[(league, team_b)], 1),
        "elo_only_p_a": round(elo_expected(state.elo[(league, team_a)],
                                           state.elo[(league, team_b)]), 4),
        "xi_a": xi_a, "xi_b": xi_b,
        "shrinkage": lam,  # < 1 means the raw probability was pulled toward 0.5
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
