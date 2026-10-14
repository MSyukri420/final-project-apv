# stage 6: outcome models and how well they predict
#
# - yes/no outcomes use logistic regression with no class weighting, so the
#   probabilities match how often things really happen
# - expected-goals outcomes (xg_for, xg_against) use Poisson regression; they feed APV
# - every model is tested on data it didn't see (unseen matches and unseen
#   competitions) against the average and a model that only knows the defensive shape

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import CHAIN_TARGETS, CLASSIFICATION_TARGETS, MODELLING_TARGETS, OUTCOME_FEATURES, REGRESSION_TARGETS
from src.target_eng import add_chain_targets

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SHAPE_ONLY: Tuple[List[str], List[str]] = ([], ["A_name"])
EPS = 1e-6


# model building

def _task(target: str) -> str:
    return "regression" if target in REGRESSION_TARGETS or target.startswith("xg_") else "classification"


def _base_target(target: str) -> str:
    # "xg_for_h30" -> "xg_for"
    return target.split("_h")[0] if "_h" in target else target


def _feature_lists(target: str) -> Tuple[List[str], List[str]]:
    spec = OUTCOME_FEATURES.get(f"outcome_{_base_target(target)}", {}) or {}
    return list(spec.get("numeric", []) or []), list(spec.get("categorical", []) or [])


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def make_model(task: str, num_cols: List[str], cat_cols: List[str], strength: float) -> Pipeline:
    # strength = C for logistic regression, alpha for Poisson regression
    if task == "classification":
        est: Any = LogisticRegression(C=strength, max_iter=2000)
    else:
        est = PoissonRegressor(alpha=strength, max_iter=2000)
    return Pipeline([("preprocess", _build_preprocessor(num_cols, cat_cols)), ("clf", est)])


def predict(model: Pipeline, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


# cross-validation

def _grouped_splits(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # folds grouped by match, balanced on whether the outcome happened
    strata = (np.asarray(y) > 0).astype(int)
    n_folds = max(2, min(n_folds, int(strata.sum()), int((1 - strata).sum())))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), strata, groups))


def _holdout_splits(holdout_keys: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    # leave out one competition-season at a time
    idx = np.arange(len(holdout_keys))
    return [(idx[holdout_keys != k], idx[holdout_keys == k]) for k in pd.unique(holdout_keys)]


def _out_of_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    fit_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        pred[test_idx] = fit_predict(X.iloc[train_idx], y[train_idx], X.iloc[test_idx])
    return pred


def _model_fit_predict(task: str, cols: Tuple[List[str], List[str]], strength: float):
    def run(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> np.ndarray:
        if task == "classification" and len(np.unique(y_tr)) < 2:
            return np.full(len(X_te), float(np.mean(y_tr)))
        model = make_model(task, cols[0], cols[1], strength).fit(X_tr, y_tr)
        return predict(model, X_te, task)

    return run


def _base_rate_fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> np.ndarray:
    return np.full(len(X_te), float(np.mean(y_tr)))


# metrics

def _calibration_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    # calibration: slope 1 and intercept 0 mean the probabilities are right
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    p = np.clip(p, EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(penalty=None, max_iter=1000).fit(logit, y)
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def compute_metrics(task: str, y: np.ndarray, pred: np.ndarray, base_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    base_pred = np.asarray(base_pred, dtype=float)
    out: Dict[str, float] = {"n": int(len(y)), "observed_mean": float(y.mean()), "predicted_mean": float(pred.mean())}

    if task == "classification":
        yi = y.astype(int)
        both = len(np.unique(yi)) == 2
        brier = float(brier_score_loss(yi, pred))
        brier_base = float(brier_score_loss(yi, base_pred))
        slope, intercept = _calibration_slope(yi, pred)
        out.update(
            {
                "roc_auc": float(roc_auc_score(yi, pred)) if both else float("nan"),
                "avg_precision": float(average_precision_score(yi, pred)) if both else float("nan"),
                "brier": brier,
                "brier_base_rate": brier_base,
                "brier_skill": 1.0 - brier / brier_base if brier_base > 0 else float("nan"),
                "log_loss": float(log_loss(yi, np.clip(pred, EPS, 1 - EPS), labels=[0, 1])),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
    else:
        pos = np.clip(pred, EPS, None)
        dev = float(mean_poisson_deviance(y, pos))
        dev_base = float(mean_poisson_deviance(y, np.clip(base_pred, EPS, None)))
        out.update(
            {
                "poisson_deviance": dev,
                "deviance_base_rate": dev_base,
                "d2_deviance_explained": 1.0 - dev / dev_base if dev_base > 0 else float("nan"),
                "mae": float(mean_absolute_error(y, pred)),
                "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
                "calibration_ratio": float(pred.mean() / y.mean()) if y.mean() > 0 else float("nan"),
            }
        )
    return out


def _score(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    # lower is better; used to pick the regularisation strength
    if task == "classification":
        return float(log_loss(y.astype(int), np.clip(pred, EPS, 1 - EPS), labels=[0, 1]))
    return float(mean_poisson_deviance(y, np.clip(pred, EPS, None)))


# evaluation of one target

def _tune(
    task: str,
    cols: Tuple[List[str], List[str]],
    X: pd.DataFrame,
    y: np.ndarray,
    splits,
    grid: Sequence[float],
) -> Tuple[float, List[Dict[str, float]]]:
    rows = []
    for strength in grid:
        pred = _out_of_fold(X, y, splits, _model_fit_predict(task, cols, strength))
        rows.append({"strength": float(strength), "cv_score": _score(task, y, pred)})
    best = min(rows, key=lambda r: r["cv_score"])["strength"]
    return best, rows


def evaluate_target(
    df: pd.DataFrame,
    target: str,
    grids: Dict[str, Sequence[float]],
    n_folds: int,
    seed: int,
    holdout_keys: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # tune, then test the full model, the shape-only model and the average
    task = _task(target)
    y = df[target].astype(float).to_numpy()
    full_cols = _feature_lists(target)
    X = df[full_cols[0] + full_cols[1] + [c for c in SHAPE_ONLY[1] if c not in full_cols[1]]].copy()
    grid = grids["classification" if task == "classification" else "regression"]

    splits = _grouped_splits(y, df["match_id"].to_numpy(), n_folds, seed)
    strength, tuning = _tune(task, full_cols, X, y, splits, grid)
    shape_strength, _ = _tune(task, SHAPE_ONLY, X, y, splits, grid)

    schemes = {"grouped_cv": splits}
    if holdout_keys is not None and len(pd.unique(holdout_keys)) > 1:
        schemes["leave_one_competition_out"] = _holdout_splits(holdout_keys)

    rows: List[Dict[str, Any]] = []
    oof: Dict[str, np.ndarray] = {}
    for scheme, sp in schemes.items():
        base = _out_of_fold(X, y, sp, _base_rate_fit_predict)
        candidates = {
            "base_rate": base,
            "shape_only": _out_of_fold(X, y, sp, _model_fit_predict(task, SHAPE_ONLY, shape_strength)),
            "full_model": _out_of_fold(X, y, sp, _model_fit_predict(task, full_cols, strength)),
        }
        for name, pred in candidates.items():
            rows.append({"target": target, "scheme": scheme, "model": name, **compute_metrics(task, y, pred, base)})
        oof[scheme] = candidates["full_model"]

    per_holdout: List[Dict[str, Any]] = []
    if "leave_one_competition_out" in schemes and holdout_keys is not None:
        pred = oof["leave_one_competition_out"]
        base = _out_of_fold(X, y, schemes["leave_one_competition_out"], _base_rate_fit_predict)
        for key in pd.unique(holdout_keys):
            m = holdout_keys == key
            per_holdout.append(
                {"target": target, "held_out": key, **compute_metrics(task, y[m], pred[m], base[m])}
            )

    return {
        "task": task,
        "strength": strength,
        "columns": full_cols,
        "rows": rows,
        "tuning": [{"target": target, **r, "chosen": r["strength"] == strength} for r in tuning],
        "per_holdout": per_holdout,
        "oof": oof.get("grouped_cv"),
        "y": y,
    }


# the chance chain, tested on unseen data

def evaluate_chain(
    df: pd.DataFrame,
    strengths: Dict[str, float],
    n_folds: int,
    seed: int,
    holdout_keys: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    # on contested corners: P(win) * P(chance | won) + (1 - P(win)) * P(chance | lost),
    # with all three models fitted on the same folds; compared with one model fitted
    # straight on chance_for, and with the average
    need = ["first_contact_attack", "chance_for", *CHAIN_TARGETS]
    if any(t not in df.columns or t not in strengths for t in need):
        return []
    contested = df["first_contact_attack"].notna().to_numpy()
    d = df[contested].reset_index(drop=True)
    keys = holdout_keys[contested] if holdout_keys is not None else None
    y = d["chance_for"].astype(float).to_numpy()
    won = d["first_contact_attack"].to_numpy() == 1

    def fit_predict(target: str, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        cols = _feature_lists(target)
        tr = d.iloc[train]
        tr = tr[tr[target].notna()]
        return _model_fit_predict("classification", cols, strengths[target])(
            tr[cols[0] + cols[1]], tr[target].astype(float).to_numpy(), d.iloc[test][cols[0] + cols[1]]
        )

    schemes = {"grouped_cv": _grouped_splits(y, d["match_id"].to_numpy(), n_folds, seed)}
    if keys is not None and len(pd.unique(keys)) > 1:
        schemes["leave_one_competition_out"] = _holdout_splits(keys)

    rows: List[Dict[str, Any]] = []
    for scheme, splits in schemes.items():
        base = np.full(len(y), np.nan)
        direct = np.full(len(y), np.nan)
        chain = np.full(len(y), np.nan)
        for train, test in splits:
            base[test] = y[train].mean()
            direct[test] = fit_predict("chance_for", train, test)
            p_win = fit_predict("first_contact_attack", train, test)
            chain[test] = p_win * fit_predict("chance_if_won", train, test) + (1 - p_win) * fit_predict(
                "chance_if_lost", train, test
            )
        for name, pred in (("base_rate", base), ("direct_model", direct), ("chain_model", chain)):
            rows.append(
                {"target": "chance_chain", "scheme": scheme, "model": name, **compute_metrics("classification", y, pred, base)}
            )
    # the best the chain could do: knowing who really won the first ball
    rows.append(
        {
            "target": "chance_chain",
            "scheme": "known_first_contact",
            "model": "rate_if_won_vs_lost",
            **compute_metrics("classification", y, np.where(won, y[won].mean(), y[~won].mean()), np.full(len(y), y.mean())),
        }
    )
    return rows


# figures

def plot_calibration(task: str, y: np.ndarray, pred: np.ndarray, target: str, out_dir: Path) -> None:
    # observed against predicted, in ten equal-sized bins
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y": y, "p": pred}).dropna()
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=10, labels=False)
    agg = df.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    hi = float(max(agg["p"].max(), agg["y"].max()) * 1.1)
    ax.plot([0, hi], [0, hi], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
    ax.plot(agg["p"], agg["y"], marker="o", label="model (out of fold)")
    unit = "probability" if task == "classification" else "xG per corner"
    ax.set_xlabel(f"predicted {unit}")
    ax.set_ylabel(f"observed {unit}")
    ax.set_title(f"Calibration: {target}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{target}.png", dpi=200)
    plt.close(fig)


def _global_shap(model: Pipeline, X: pd.DataFrame, target: str, out_dir: Path, sample_size: int, seed: int) -> None:
    pre = model.named_steps["preprocess"]
    est = model.named_steps["clf"]
    X_bg = X.sample(n=min(sample_size, len(X)), random_state=seed)
    X_proc = pre.transform(X_bg)
    X_proc = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)
    values = np.asarray(shap.LinearExplainer(est, X_proc)(X_proc).values)
    if values.ndim == 3:
        values = values[..., -1]
    names = [n.split("__", 1)[1] if "__" in n else n for n in pre.get_feature_names_out()]
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_dir / f"global_shap_{target}.csv", index=False)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"].astype(str), top["mean_abs_shap"])
    scale = "log-odds" if _task(target) == "classification" else "log expected xG"
    ax.set_xlabel(f"mean |SHAP| ({scale})")
    ax.set_title(f"Global SHAP importance ({target})")
    fig.tight_layout()
    fig.savefig(out_dir / f"global_shap_{target}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# bootstrap models for the uncertainty ranges

def fit_bootstrap_ensemble(
    df: pd.DataFrame,
    specs: Dict[str, Tuple[Tuple[List[str], List[str]], float]],
    n_boot: int,
    seed: int,
) -> List[Dict[str, Pipeline]]:
    # refit the models on matches resampled with replacement; the spread of their
    # predictions is the range shown in the app. All targets share each resample.
    rng = np.random.default_rng(seed)
    matches = df["match_id"].unique()
    rows_by_match = {m: np.flatnonzero(df["match_id"].to_numpy() == m) for m in matches}
    ensemble: List[Dict[str, Pipeline]] = []
    for _ in range(n_boot):
        picked = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([rows_by_match[m] for m in picked])
        sample = df.iloc[idx]
        members: Dict[str, Pipeline] = {}
        for target, (cols, strength) in specs.items():
            rows = sample[sample[target].notna()]
            members[target] = make_model(_task(target), cols[0], cols[1], strength).fit(
                rows[cols[0] + cols[1]], rows[target].astype(float)
            )
        ensemble.append(members)
    return ensemble


# horizon sensitivity

def horizon_sensitivity(
    df: pd.DataFrame,
    horizons: Sequence[float],
    main_horizon: float,
    strengths: Dict[str, float],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    # do the results hold if outcomes are counted over 30, 60 or 90 seconds?
    rows = []
    for h in sorted(set([float(main_horizon), *[float(x) for x in horizons]])):
        for target in ["chance_for", "shot_against_any", "xg_for", "xg_against"]:
            col = target if h == float(main_horizon) else f"{target}_h{int(h)}"
            if col not in df.columns:
                continue
            task = _task(target)
            y = df[col].astype(float).to_numpy()
            cols = _feature_lists(target)
            X = df[cols[0] + cols[1]]
            splits = _grouped_splits(y, df["match_id"].to_numpy(), n_folds, seed)
            base = _out_of_fold(X, y, splits, _base_rate_fit_predict)
            pred = _out_of_fold(X, y, splits, _model_fit_predict(task, cols, strengths[target]))
            m = compute_metrics(task, y, pred, base)
            rows.append(
                {
                    "horizon_secs": int(h),
                    "target": target,
                    "observed_mean": m["observed_mean"],
                    "non_zero_share": float((y > 0).mean()),
                    "roc_auc": m.get("roc_auc", np.nan),
                    "brier_skill": m.get("brier_skill", np.nan),
                    "d2_deviance_explained": m.get("d2_deviance_explained", np.nan),
                    "spearman": m.get("spearman", np.nan),
                }
            )
    return pd.DataFrame(rows)


# stage 6 entry point

def stage6_model_training(
    df_target: pd.DataFrame,
    cfg: dict,
    out_dir: Optional[str] = None,
    holdout_names: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    paths_cfg = cfg.get("paths", {}) or {}
    models_dir = Path(paths_cfg.get("models", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(out_dir) if out_dir is not None else None

    model_cfg = cfg.get("models", {}) or {}
    n_folds = int(model_cfg.get("cv_folds", 5))
    grids = {
        "classification": [float(c) for c in model_cfg.get("logistic_C_grid", [0.01, 0.03, 0.1, 0.3, 1.0])],
        "regression": [float(a) for a in model_cfg.get("poisson_alpha_grid", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0])],
    }
    n_boot = int(model_cfg.get("bootstrap_samples", 100))
    seed = int((cfg.get("project", {}) or {}).get("random_seed", 42))

    explain_cfg = cfg.get("explainability", {}) or {}
    shap_dir = Path(paths_cfg.get("outputs", "outputs")) / "model_importance" / "global"
    do_shap = bool(explain_cfg.get("enable_shap", False)) and HAS_SHAP
    shap_n = int(explain_cfg.get("shap_sample_size", 500))

    df = add_chain_targets(df_target).reset_index(drop=True)
    holdout_keys = None
    if {"competition_id", "season_id"}.issubset(df.columns):
        keys = df["competition_id"].astype(str) + "-" + df["season_id"].astype(str)
        holdout_keys = keys.map(lambda k: (holdout_names or {}).get(k, k)).to_numpy()

    print("------------------------------------------------------")
    print("stage 6: model training and out-of-sample evaluation")
    print("------------------------------------------------------")
    print(f"corners: {len(df)}  matches: {df['match_id'].nunique()}  folds: {n_folds} (grouped by match)")

    eval_rows: List[Dict[str, Any]] = []
    tuning_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    strengths: Dict[str, float] = {}

    for target in MODELLING_TARGETS:
        if target not in df.columns:
            print(f"target '{target}' missing, skipping")
            continue
        # some outcomes are missing for some corners (e.g. no contest for first contact)
        labelled = df[target].notna().to_numpy()
        df_t = df[labelled].reset_index(drop=True)
        keys_t = holdout_keys[labelled] if holdout_keys is not None else None

        res = evaluate_target(df_t, target, grids, n_folds, seed, keys_t)
        strengths[target] = res["strength"]
        eval_rows += res["rows"]
        tuning_rows += res["tuning"]
        holdout_rows += res["per_holdout"]

        num_cols, cat_cols = res["columns"]
        model = make_model(res["task"], num_cols, cat_cols, res["strength"]).fit(
            df_t[num_cols + cat_cols], df_t[target].astype(float)
        )
        joblib.dump(model, models_dir / f"{target}_pooled.joblib")

        cv = next(r for r in res["rows"] if r["scheme"] == "grouped_cv" and r["model"] == "full_model")
        if res["task"] == "classification":
            print(
                f"{target:<18} C={res['strength']:<5} rate={cv['observed_mean']:.3f} "
                f"AUC={cv['roc_auc']:.3f} Brier skill={cv['brier_skill']:+.3f} slope={cv['calibration_slope']:.2f}"
            )
        else:
            print(
                f"{target:<18} alpha={res['strength']:<5} mean={cv['observed_mean']:.4f} "
                f"D2={cv['d2_deviance_explained']:+.3f} spearman={cv['spearman']:.3f}"
            )

        if fig_dir is not None and res["oof"] is not None:
            plot_calibration(res["task"], res["y"], res["oof"], target, fig_dir)
        if do_shap and target in ("first_contact_attack", "chance_for", "shot_against_any", "xg_for", "xg_against"):
            try:
                _global_shap(model, df_t[num_cols + cat_cols], target, shap_dir, shap_n, seed)
            except Exception as exc:
                print(f"global shap failed for '{target}': {exc}")

    chain_rows = evaluate_chain(df, strengths, n_folds, seed, holdout_keys)
    eval_rows += chain_rows
    for r in chain_rows:
        if r["model"] == "chain_model":
            print(f"chance chain ({r['scheme']}): AUC={r['roc_auc']:.3f} Brier skill={r['brier_skill']:+.3f}")

    # bootstrap models for the app: the chance chain and the APV inputs
    specs = {
        t: (_feature_lists(t), strengths[t])
        for t in ["first_contact_attack", *CHAIN_TARGETS, *REGRESSION_TARGETS]
        if t in strengths
    }
    if specs and n_boot > 0:
        ensemble = fit_bootstrap_ensemble(df, specs, n_boot, seed)
        joblib.dump(ensemble, models_dir / "bootstrap_ensemble.joblib")
        print(f"saved {len(ensemble)} bootstrap model pairs to {models_dir / 'bootstrap_ensemble.joblib'}")

    horizons = (cfg.get("targets", {}) or {}).get("sensitivity_horizons", []) or []
    main_h = float((cfg.get("targets", {}) or {}).get("horizon_secs", 60))
    sensitivity = horizon_sensitivity(df, horizons, main_h, strengths, n_folds, seed) if horizons else pd.DataFrame()

    print("stage 6: done")
    return {
        "evaluation": pd.DataFrame(eval_rows),
        "tuning": pd.DataFrame(tuning_rows),
        "per_competition": pd.DataFrame(holdout_rows),
        "horizon_sensitivity": sensitivity,
    }


__all__ = [
    "CLASSIFICATION_TARGETS",
    "REGRESSION_TARGETS",
    "make_model",
    "predict",
    "stage6_model_training",
]
