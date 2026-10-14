# stage 7: predictions and APV for each corner, in goals
#   PV_for     = alpha * xG we create after the corner
#   PV_against = beta  * xG the opponent creates in their next possession
#   APV_net    = PV_for - PV_against

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.constants import CLASSIFICATION_TARGETS, REGRESSION_TARGETS


def _load_model(models_dir: Path, target_name: str) -> object:
    path = models_dir / f"{target_name}_pooled.joblib"
    if not path.exists():
        raise FileNotFoundError(f"model file not found for outcome '{target_name}': {path}")
    return joblib.load(path)


def _model_input(model, df: pd.DataFrame) -> pd.DataFrame:
    # each model only sees the columns it was trained on
    cols = list(getattr(model, "feature_names_in_", df.columns))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns for prediction: {missing}")
    return df[cols]


def predict_outcomes(df: pd.DataFrame, models: Dict[str, object]) -> pd.DataFrame:
    # add P_<target> (probability) and E_<target> (expected xG) columns
    out = df.copy()
    for target in CLASSIFICATION_TARGETS:
        model = models.get(target)
        if model is not None:
            out[f"P_{target}"] = model.predict_proba(_model_input(model, out))[:, 1]
    for target in REGRESSION_TARGETS:
        model = models.get(target)
        if model is not None:
            out[f"E_{target}"] = model.predict(_model_input(model, out))
    return out


def add_apv_columns(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    xg_for_col: str = "E_xg_for",
    xg_against_col: str = "E_xg_against",
    pv_for_col: str = "PV_for",
    pv_against_col: str = "PV_against",
    apv_net_col: str = "APV_net",
) -> pd.DataFrame:
    # add the APV columns (needs the expected xG predictions first)
    missing = [c for c in (xg_for_col, xg_against_col) if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected-goals columns before apv computation: {missing}")

    df_out = df.copy()
    df_out[pv_for_col] = alpha * df_out[xg_for_col].astype(float)
    df_out[pv_against_col] = beta * df_out[xg_against_col].astype(float)
    df_out[apv_net_col] = df_out[pv_for_col] - df_out[pv_against_col]
    return df_out


def stage7_compute_apv(
    df_target: pd.DataFrame,
    cfg: Dict,
    models_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    split: str = "all",
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # stage 7: predictions and APV for every corner, plus a summary
    models_dir = Path(models_dir or cfg["paths"]["models"])
    out_dir = Path(out_dir or cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_target is None or df_target.empty:
        print("df_target is empty. stage 7 will return empty outputs.")
        return pd.DataFrame(), pd.DataFrame()

    models = {t: _load_model(models_dir, t) for t in CLASSIFICATION_TARGETS + REGRESSION_TARGETS}
    df = predict_outcomes(df_target, models)

    apv_cfg = cfg.get("apv", {}) or {}
    df_apv = add_apv_columns(df, alpha=float(apv_cfg.get("alpha", 1.0)), beta=float(apv_cfg.get("beta", 1.0)))

    print("------------------------------------------------------")
    print("stage 7: apv (expected goals)")
    print("------------------------------------------------------")
    print(df_apv[["PV_for", "PV_against", "APV_net"]].describe().to_string())
    print("")

    summary = pd.DataFrame()
    if "A_name" in df_apv.columns:
        group_cols: List[str] = [
            c
            for c in [*(f"P_{t}" for t in CLASSIFICATION_TARGETS), "PV_for", "PV_against", "APV_net", "xg_for", "xg_against"]
            if c in df_apv.columns
        ]
        summary = df_apv.groupby("A_name")[group_cols].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else str(col) for col in summary.columns]
        print("apv summary by defensive archetype (A_name)")
        print(summary.to_string(index=False))
        print("")

    if save:
        suffix = split.lower()
        df_apv.to_csv(out_dir / f"df_apv_{suffix}.csv", index=False)
        if not summary.empty:
            summary.to_csv(out_dir / f"apv_summary_{suffix}.csv", index=False)
        print(f"corner level apv -> {out_dir / f'df_apv_{suffix}.csv'}")

    return df_apv, summary
