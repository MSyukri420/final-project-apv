# stage 6 pooled logistic regression models for apv

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import MODELLING_TARGETS, OUTCOME_FEATURES

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _get_outcome_feature_lists(outcome_key: str) -> Tuple[List[str], List[str]]:
    # look up numeric and categorical feature lists for a given outcome
    cfg = OUTCOME_FEATURES.get(outcome_key, {}) or {}
    num_cols = list(cfg.get("numeric", []) or [])
    cat_cols = list(cfg.get("categorical", []) or [])
    return num_cols, cat_cols


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    # build a column transformer for numeric and categorical features
    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_tf, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_tf, cat_cols))

    return ColumnTransformer(transformers=transformers)


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # compute basic binary classification metrics for probabilities
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_samples = int(len(y_true))
    n_pos = int(y_true.sum())
    n_neg = int(n_samples - n_pos)

    metrics: Dict[str, float] = {
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": float(y_true.mean()) if n_samples > 0 else float("nan"),
        "roc_auc": float("nan"),
        "avg_precision": float("nan"),
        "brier": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }

    if n_samples == 0 or n_pos == 0 or n_neg == 0:
        return metrics

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["avg_precision"] = float("nan")

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics["brier"] = float("nan")

    try:
        y_pred = (y_prob >= 0.5).astype(int)
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["precision"] = float("nan")
        metrics["recall"] = float("nan")

    return metrics


def _plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    # plot roc and precision-recall curves for one model
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {model_name}")
    plt.legend(loc="lower right")
    roc_path = out_dir / f"{model_name}_roc.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name} (AP = {ap:.3f})")
    pr_path = out_dir / f"{model_name}_pr.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()


def _make_background(
    df: pd.DataFrame,
    max_samples: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    # pick a background sample for shap
    if df.empty:
        return df
    if len(df) > max_samples:
        return df.sample(n=max_samples, random_state=random_seed)
    return df.copy()


def stage6_model_training(
    df_target: pd.DataFrame,
    cfg: dict,
    out_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    # main entry point for stage 6 pooled logistic regression models
    paths_cfg = cfg.get("paths", {}) or {}
    models_dir = Path(paths_cfg.get("models", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    explain_cfg = cfg.get("explainability", {}) or {}
    enable_shap = bool(explain_cfg.get("enable_shap", False))
    shap_sample_size = int(explain_cfg.get("shap_sample_size", 500))

    outputs_dir = Path(paths_cfg.get("outputs", "outputs"))
    shap_out_dir: Optional[Path] = None
    if enable_shap and HAS_SHAP:
        shap_out_dir = outputs_dir / "model_importance" / "global"
        shap_out_dir.mkdir(parents=True, exist_ok=True)
    elif enable_shap and not HAS_SHAP:
        print("shap is not installed, skipping global importance computation.")

    print("------------------------------------------------------")
    print("stage 6: pooled model training")

    metrics_all: Dict[str, Dict[str, float]] = {}
    out_dir_path = Path(out_dir) if out_dir is not None else None
    random_seed = int(cfg.get("random_seed", 42))

    for target_col in MODELLING_TARGETS:
        model_name = f"{target_col}_pooled"
        outcome_key = f"outcome_{target_col}"

        if target_col not in df_target.columns:
            print(f"target column '{target_col}' not found, skipping.")
            continue

        y = df_target[target_col].astype(int)
        n_samples = len(y)
        n_pos = int(y.sum())
        n_neg = int(n_samples - n_pos)

        if "A_confidence" in df_target.columns:
            sample_weight = (
                df_target["A_confidence"]
                .fillna(df_target["A_confidence"].mean())
                .astype(float)
                .values
            )
        else:
            sample_weight = np.ones(len(y), dtype=float)

        print("")
        print(f"training pooled model for '{target_col}'")
        print(f"samples: {n_samples}  (pos={n_pos}, neg={n_neg})")

        if len(np.unique(y)) < 2:
            print(f"only one class present for '{target_col}', skipping model.")
            metrics_all[model_name] = {
                "n_samples": int(n_samples),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "pos_rate": float(y.mean()) if n_samples > 0 else float("nan"),
                "roc_auc": float("nan"),
                "avg_precision": float("nan"),
                "brier": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "cv_n_splits": 0,
            }
            continue

        num_cols, cat_cols = _get_outcome_feature_lists(outcome_key)
        cols_needed = num_cols + cat_cols

        if not cols_needed:
            print(f"no features defined for '{outcome_key}', skipping model.")
            metrics_all[model_name] = {
                "n_samples": int(n_samples),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "pos_rate": float(y.mean()) if n_samples > 0 else float("nan"),
                "roc_auc": float("nan"),
                "avg_precision": float("nan"),
                "brier": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "cv_n_splits": 0,
            }
            continue

        missing = [c for c in cols_needed if c not in df_target.columns]
        if missing:
            print(
                "missing feature columns for "
                f"'{target_col}': {', '.join(sorted(missing))}"
            )
            metrics_all[model_name] = {
                "n_samples": int(n_samples),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "pos_rate": float(y.mean()) if n_samples > 0 else float("nan"),
                "roc_auc": float("nan"),
                "avg_precision": float("nan"),
                "brier": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "cv_n_splits": 0,
            }
            continue

        X = df_target[cols_needed].copy()

        preprocessor = _build_preprocessor(num_cols, cat_cols)
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None,
            random_state=random_seed,
        )
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

        y_prob_cv: Optional[np.ndarray] = None
        n_splits = 0
        try:
            max_splits = min(5, n_pos, n_neg)
            if max_splits >= 2:
                cv = StratifiedKFold(
                    n_splits=max_splits,
                    shuffle=True,
                    random_state=random_seed,
                )
                y_prob_cv = cross_val_predict(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    method="predict_proba",
                    fit_params={"clf__sample_weight": sample_weight},
                )[:, 1]
                n_splits = max_splits
        except Exception as e:
            print(f"cross validation failed for '{target_col}': {e}")
            y_prob_cv = None
            n_splits = 0

        pipeline.fit(X, y, clf__sample_weight=sample_weight)
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"saved pooled model to {model_path}")

        if y_prob_cv is not None:
            y_prob_used = y_prob_cv
            print(f"using cross validated probabilities for metrics ({n_splits} folds).")
        else:
            y_prob_used = pipeline.predict_proba(X)[:, 1]
            print("using in sample probabilities for metrics (no cv available).")

        metrics = _compute_binary_metrics(y_true=y.values, y_prob=y_prob_used)
        metrics["cv_n_splits"] = int(n_splits)
        metrics_all[model_name] = metrics

        if out_dir_path is not None and target_col in ("chance_for", "shot_against_any"):
            try:
                _plot_roc_pr_curves(
                    y_true=y.values,
                    y_prob=y_prob_used,
                    model_name=model_name,
                    out_dir=out_dir_path,
                )
            except Exception as e:
                print(f"failed to plot roc or pr for '{target_col}': {e}")

        # optional global shap feature importance (chance_for, shot_against_any)
        if shap_out_dir is not None and target_col in ("chance_for", "shot_against_any"):
            try:
                X_bg = _make_background(
                    df=X,
                    max_samples=shap_sample_size,
                    random_seed=random_seed,
                )
                if X_bg.empty:
                    print("shap: no background rows available, skipping.")
                else:
                    bg_pipe = joblib.load(model_path)
                    bg_pre = bg_pipe.named_steps["preprocess"]
                    bg_clf = bg_pipe.named_steps["clf"]

                    # transform background through the same preprocessor
                    X_bg_proc = bg_pre.transform(X_bg)

                    # linear shap for logistic regression
                    explainer = shap.LinearExplainer(
                        bg_clf,
                        X_bg_proc,
                    )
                    shap_values = explainer(X_bg_proc)

                    vals = shap_values.values
                    if vals.ndim == 3:
                        # multi class shape -> take the positive class
                        vals = vals[..., 0]

                    vals = np.asarray(vals)
                    mean_abs = np.abs(vals).mean(axis=0)

                    raw_feature_names = list(bg_pre.get_feature_names_out())

                    def _clean_feature_name(name: str) -> str:
                        # remove 'num__' / 'cat__' prefixes and keep the meaningful part
                        if "__" in name:
                            return name.split("__", 1)[1]
                        return name

                    feature_names = [_clean_feature_name(n) for n in raw_feature_names]

                    df_imp = (
                        pd.DataFrame(
                            {"feature": feature_names, "mean_abs_shap": mean_abs}
                        )
                        .sort_values("mean_abs_shap", ascending=False)
                    )

                    csv_path = shap_out_dir / f"global_shap_{target_col}.csv"
                    df_imp.to_csv(csv_path, index=False)

                    # simple horizontal bar chart
                    top_n = min(20, df_imp.shape[0])
                    df_top = df_imp.head(top_n).iloc[::-1]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(df_top["feature"].astype(str), df_top["mean_abs_shap"])
                    ax.set_xlabel("Mean |SHAP value|")
                    ax.set_title(f"Global SHAP importance ({target_col})")
                    fig.tight_layout()

                    png_path = shap_out_dir / f"global_shap_{target_col}.png"
                    fig.savefig(png_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    print(f"saved global shap importance to {csv_path}")
                    print(f"saved global shap bar chart to {png_path}")
            except Exception as e:
                print(f"failed to compute global importance for '{target_col}': {e}")

    print("")
    print("stage 6: completed pooled model training")
    return metrics_all
