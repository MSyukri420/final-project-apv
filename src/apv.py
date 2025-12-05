from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from src.constants import MODELLING_TARGETS, OUTCOME_FEATURES


# helpers

def _get_outcome_feature_lists(
    outcome_key: str,
) -> Tuple[List[str], List[str]]:
    # read numeric and categorical feature lists for an outcome from outcome_features
    cfg = OUTCOME_FEATURES.get(outcome_key, {}) or {}
    num_cols = list(cfg.get("numeric", []) or [])
    cat_cols = list(cfg.get("categorical", []) or [])
    return num_cols, cat_cols


def _ensure_feature_columns(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_name: str,
) -> None:
    # raise a clear error if any required feature column is missing before prediction
    needed = list(numeric_features) + list(categorical_features)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing feature columns for outcome '{target_name}': {missing}"
        )


def _load_model(models_dir: Path, target_name: str) -> object:
    # load a pooled logistic regression model from joblib
    path = models_dir / f"{target_name}_pooled.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"model file not found for outcome '{target_name}': {path}"
        )
    return joblib.load(path)


def _predict_probs_for_target(
    df: pd.DataFrame,
    models_dir: Path,
    target_col: str,
) -> pd.Series:
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not found in df_target.")

    y = df[target_col].astype(int)
    n_samples = int(len(y))
    n_pos = int(y.sum())
    n_neg = int(n_samples - n_pos)

    proba_name = f"P_{target_col}"

    if n_samples == 0:
        # empty input, return all nan
        return pd.Series(np.nan, index=df.index, name=proba_name)

    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        # only one class present, use a constant baseline probability
        base = float(y.mean()) if n_samples > 0 else 0.0
        print(
            f"target '{target_col}' has only one class "
            f"(pos={n_pos}, neg={n_neg}). using constant p={base:.4f}."
        )
        return pd.Series(base, index=df.index, name=proba_name)

    outcome_key = f"outcome_{target_col}"
    num_cols, cat_cols = _get_outcome_feature_lists(outcome_key)
    if not num_cols and not cat_cols:
        raise ValueError(
            f"no features defined in OUTCOME_FEATURES for '{outcome_key}'."
        )

    _ensure_feature_columns(df, num_cols, cat_cols, target_name=target_col)

    model = _load_model(models_dir, target_col)
    X = df[num_cols + cat_cols].copy()
    proba = model.predict_proba(X)[:, 1]

    return pd.Series(proba, index=df.index, name=proba_name)


def add_apv_columns(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    p_chance_for_col: str = "P_chance_for",
    p_shot_against_col: str = "P_shot_against_any",
    pv_for_col: str = "PV_for",
    pv_against_col: str = "PV_against",
    apv_net_col: str = "APV_net",
) -> pd.DataFrame:
    # add apv columns to a dataframe that already has probability predictions
    required_cols = [p_chance_for_col, p_shot_against_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing probability columns before apv computation: {missing}"
        )

    df_out = df.copy()
    p_cf = df_out[p_chance_for_col].astype(float)
    p_sa = df_out[p_shot_against_col].astype(float)

    df_out[pv_for_col] = alpha * p_cf
    df_out[pv_against_col] = beta * p_sa
    df_out[apv_net_col] = df_out[pv_for_col] - df_out[pv_against_col]

    return df_out


def _compute_apv_from_probs(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    # wrapper used by stage 7 to read apv config and call add_apv_columns
    apv_cfg = cfg.get("apv", {}) or {}
    alpha = float(apv_cfg.get("alpha", 1.0))
    beta = float(apv_cfg.get("beta", 1.0))

    return add_apv_columns(
        df,
        alpha=alpha,
        beta=beta,
        p_chance_for_col="P_chance_for",
        p_shot_against_col="P_shot_against_any",
        pv_for_col="PV_for",
        pv_against_col="PV_against",
        apv_net_col="APV_net",
    )



def stage7_compute_apv(
    df_target: pd.DataFrame,
    cfg: Dict,
    models_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    split: str = "all",
    save: bool = True,
):
    # stage 7 entry point: add probabilities and apv columns, then summarise
    models_dir = Path(models_dir or cfg["paths"]["models"])
    out_dir = Path(out_dir or cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_target is None or df_target.empty:
        print("df_target is empty. stage 7 will return empty outputs.")
        empty_df = pd.DataFrame()
        empty_summary = pd.DataFrame()
        return empty_df, empty_summary

    df = df_target.copy()

    print("------------------------------------------------------")
    print("stage 7: apv probability prediction")
    print("------------------------------------------------------")
    print(f"rows in df_target: {len(df):>8}")
    print(f"models directory:  {models_dir}")
    print("")

    # predict probabilities for each modelling target
    for target_col in MODELLING_TARGETS:
        if target_col not in df.columns:
            print(
                f"target column '{target_col}' not in df_target, skipping."
            )
            continue

        print(f"predicting probabilities for '{target_col}'")
        proba = _predict_probs_for_target(
            df,
            models_dir=models_dir,
            target_col=target_col,
        )
        df[proba.name] = proba

        print(
            f"  added column: {proba.name}  "
            f"min={proba.min():.3f}, max={proba.max():.3f}, mean={proba.mean():.3f}"
        )
        print("")

    df_apv = _compute_apv_from_probs(df, cfg)

    print("------------------------------------------------------")
    print("apv columns added:")
    print("  PV_for, PV_against, APV_net")
    print(df_apv[["PV_for", "PV_against", "APV_net"]].describe())
    print("")

    # summary by archetype for quick diagnostics
    if "A_name" in df_apv.columns:
        group_cols = [
            "P_chance_for",
            "P_chance_against",
            "P_shot_for_any",
            "P_shot_against_any",
            "PV_for",
            "PV_against",
            "APV_net",
        ]

        group_cols = [c for c in group_cols if c in df_apv.columns]

        summary = (
            df_apv.groupby("A_name")[group_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        flat_cols: List[str] = []
        for col in summary.columns:
            if isinstance(col, tuple):
                parts = [str(c) for c in col if c not in ("", None)]
                flat_cols.append("_".join(parts))
            else:
                flat_cols.append(str(col))
        summary.columns = flat_cols

        print("======================================================================")
        print("apv summary by defensive archetype (A_name)")
        print("======================================================================")
        print(summary.to_string(index=False))
        print("")
    else:
        summary = pd.DataFrame()
        print("'A_name' not found, skipping archetype summary.")
        print("")

    if save:
        suffix = split.lower()
        apv_path = out_dir / f"df_apv_{suffix}.csv"
        summary_path = out_dir / f"apv_summary_{suffix}.csv"

        df_apv.to_csv(apv_path, index=False)
        if not summary.empty:
            summary.to_csv(summary_path, index=False)

        print("======================================================================")
        print("stage 7 outputs saved")
        print("======================================================================")
        print(f"corner level apv -> {apv_path}")
        if not summary.empty:
            print(f"archetype summary -> {summary_path}")
        print("")

    return df_apv, summary


