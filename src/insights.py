from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import CLASSIFICATION_TARGETS, OUTCOME_FEATURES, all_model_features


# shared settings

PATTERN_GROUP_COLS: List[str] = [
    "A_name",
    "delivery_type",
    "end_zone",
    "short_corner_flag",
    "n_att_near",
    "n_att_central",
    "n_att_far",
    "n_blockers",
    "n_edge_of_box",
    "n_rest_defence",
]

APV_METRIC_COLS: List[str] = [
    "APV_net",
    "PV_for",
    "PV_against",
    "P_chance_for",
    "P_chance_against",
    "P_shot_for_any",
    "P_shot_against_any",
    "P_first_contact_attack",
    # real outcomes, shown next to the model's
    "first_contact_attack",
    "chance_for",
    "shot_against_any",
    "xg_for",
    "xg_against",
]

_DEFAULT_VALUE_RISK_THRESHOLDS: Dict[str, float] = {
    "VALUE_HIGH": 0.15,
    "VALUE_POSITIVE": 0.05,
    "VALUE_NEGATIVE": -0.05,
    "RISK_LOW": 0.25,
    "RISK_MEDIUM": 0.50,
    # within this much APV_net of the typical corner counts as "similar"
    "SIMILAR_BAND": 0.02,
}

_VALUE_RISK_THRESHOLDS: Dict[str, float] = dict(_DEFAULT_VALUE_RISK_THRESHOLDS)
_THRESHOLDS_LOADED: bool = False



def build_pattern_library(df_apv: pd.DataFrame) -> pd.DataFrame:
    # group corners into setups and sum up APV per setup
    df = df_apv.copy()

    missing = [c for c in PATTERN_GROUP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing pattern columns in df_apv: {missing}")

    if "A_confidence" in df.columns:
        weights = (
            df["A_confidence"]
            .astype(float)
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        weights = pd.Series(1.0, index=df.index, dtype=float)

    grouped = df.groupby(PATTERN_GROUP_COLS, dropna=False)

    rows: List[Dict[str, Any]] = []
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec: Dict[str, Any] = {}
        for col, val in zip(PATTERN_GROUP_COLS, keys):
            rec[col] = val

        idx = g.index
        w = weights.loc[idx].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w, dtype=float)

        rec["n_corners"] = int(len(g))

        for m in APV_METRIC_COLS:
            if m not in g.columns:
                continue

            vals = g[m].to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(w)
            if not mask.any():
                mean = np.nan
                std = np.nan
            else:
                v = vals[mask]
                w_m = w[mask]
                w_m_sum = w_m.sum()
                if w_m_sum <= 0:
                    mean = float(np.nanmean(v))
                    std = float(np.nanstd(v, ddof=0))
                else:
                    mean = float(np.average(v, weights=w_m))
                    diff = v - mean
                    var = float(np.average(diff**2, weights=w_m))
                    std = float(np.sqrt(var))

            rec[f"{m}_mean"] = mean
            rec[f"{m}_std"] = std

        rows.append(rec)

    df_patterns = pd.DataFrame(rows)

    keep_cols: List[str] = (
        PATTERN_GROUP_COLS
        + [
            "n_corners",
            "APV_net_mean",
            "APV_net_std",
            "PV_for_mean",
            "PV_against_mean",
            "P_chance_for_mean",
            "P_chance_against_mean",
            "P_shot_for_any_mean",
            "P_shot_against_any_mean",
            "P_first_contact_attack_mean",
            "first_contact_attack_mean",
            "chance_for_mean",
            "shot_against_any_mean",
            "xg_for_mean",
            "xg_against_mean",
        ]
    )
    keep_cols = [c for c in keep_cols if c in df_patterns.columns]
    df_patterns = df_patterns[keep_cols].copy()

    sort_cols = [
        "A_name",
        "delivery_type",
        "end_zone",
        "n_corners",
        "APV_net_mean",
    ]
    sort_cols = [c for c in sort_cols if c in df_patterns.columns]

    if sort_cols:
        df_patterns = df_patterns.sort_values(
            sort_cols,
            ascending=[True, True, True, False, False][: len(sort_cols)],
        ).reset_index(drop=True)

    df_patterns.insert(0, "pattern_id", np.arange(1, len(df_patterns) + 1, dtype=int))

    return df_patterns


def compute_apv_baselines(df_apv: pd.DataFrame) -> pd.DataFrame:
    # average APV per defensive shape
    if "A_name" not in df_apv.columns:
        raise ValueError("df_apv must contain A_name for baseline aggregation")

    metrics = ["PV_for", "PV_against", "APV_net", *(f"P_{t}" for t in CLASSIFICATION_TARGETS)]
    missing = [m for m in metrics if m not in df_apv.columns]
    if missing:
        raise ValueError(f"Missing APV metrics in df_apv: {missing}")

    grouped = df_apv.groupby("A_name")[metrics].agg(["mean", "std", "count"]).reset_index()

    flat_cols: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "":
                flat_cols.append(str(base))
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(str(col))
    grouped.columns = flat_cols

    return grouped


def compute_feature_baselines(df_apv: pd.DataFrame) -> pd.DataFrame:
    # average defensive and attacking features per shape
    if "A_name" not in df_apv.columns:
        raise ValueError("df_apv must contain A_name for feature baselines")

    feature_cols = [
        "def_line_height",
        "def_in_6yd",
        "gk_start_line",
        "players_on_posts",
        "n_att_near",
        "n_att_central",
        "n_att_far",
    ]
    available = [c for c in feature_cols if c in df_apv.columns]
    if not available:
        raise ValueError("No feature baseline columns found in df_apv")

    grouped = df_apv.groupby("A_name")[available].mean().reset_index()

    return grouped


def compute_archetype_defaults(df: pd.DataFrame) -> pd.DataFrame:
    # the typical opponent and match situation per defensive shape (medians, or the
    # most common value); the app starts from this and applies the coach's choices
    if "A_name" not in df.columns:
        raise ValueError("df must contain A_name for archetype defaults")
    cols = [c for c in all_model_features() if c in df.columns and c != "A_name"]
    rows = []
    for name, g in df.groupby("A_name"):
        rec: Dict[str, Any] = {"A_name": name}
        for c in cols:
            s = g[c].dropna()
            if s.empty:
                rec[c] = np.nan
            elif pd.api.types.is_numeric_dtype(s) and c not in ("short_corner_flag",):
                rec[c] = float(s.median())
            else:
                rec[c] = s.mode().iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_delivery_defaults(df: pd.DataFrame) -> pd.DataFrame:
    # typical depth and height for each delivery type and zone
    keys = ["delivery_type", "end_zone"]
    if any(k not in df.columns for k in keys + ["delivery_depth", "delivery_high"]):
        return pd.DataFrame(columns=keys + ["delivery_depth", "delivery_high", "n"])
    return (
        df.groupby(keys)
        .agg(
            delivery_depth=("delivery_depth", "median"),
            delivery_high=("delivery_high", lambda s: int(s.mean() >= 0.5)),
            n=("delivery_depth", "size"),
        )
        .reset_index()
    )


def compute_value_risk_thresholds(
    df_apv: pd.DataFrame,
    value_q_neg: float = 0.30,
    value_q_pos: float = 0.70,
    value_q_high: float = 0.85,
    risk_q_low: float = 0.30,
    risk_q_med: float = 0.70,
) -> Dict[str, float]:
    # value and risk thresholds from APV quantiles
    thresholds = dict(_DEFAULT_VALUE_RISK_THRESHOLDS)

    if "APV_net" in df_apv.columns:
        s_val = df_apv["APV_net"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s_val):
            thresholds["VALUE_NEGATIVE"] = float(s_val.quantile(value_q_neg))
            thresholds["VALUE_POSITIVE"] = float(s_val.quantile(value_q_pos))
            thresholds["VALUE_HIGH"] = float(s_val.quantile(value_q_high))
            thresholds["SIMILAR_BAND"] = float(0.25 * s_val.std(ddof=0))

    if "PV_against" in df_apv.columns:
        s_risk = df_apv["PV_against"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s_risk):
            thresholds["RISK_LOW"] = float(s_risk.quantile(risk_q_low))
            thresholds["RISK_MEDIUM"] = float(s_risk.quantile(risk_q_med))

    return thresholds



def build_shap_background(
    df_target: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    # sample of real corners: SHAP background and the app's list of model inputs
    feature_cols = all_model_features()

    missing = [c for c in feature_cols if c not in df_target.columns]
    if missing:
        raise ValueError(f"Missing SHAP feature columns in df_target: {missing}")

    df_feat = df_target[feature_cols].copy()
    df_feat = df_feat.dropna(axis=0, how="any")

    if len(df_feat) == 0:
        raise ValueError("No rows available for SHAP background after dropping missing values")

    n = min(sample_size, len(df_feat))
    df_bg = df_feat.sample(n=n, random_state=random_state).reset_index(drop=True)

    return df_bg



def build_runtime_artefacts(cfg: Dict[str, Any]) -> Dict[str, Path]:
    # stage 8: build the files the app loads
    paths_cfg = cfg.get("paths", {}) or {}
    outputs_root = Path(paths_cfg.get("outputs", "outputs")).expanduser()
    artefacts_root = Path(paths_cfg.get("artefacts", "artefacts")).expanduser()

    train_dir = outputs_root / "train"
    df_apv_path = train_dir / "df_apv_train.csv"
    df_target_path = train_dir / "df_target_train.csv"

    if not df_apv_path.exists():
        raise FileNotFoundError(f"df_apv_train.csv not found at {df_apv_path}")
    if not df_target_path.exists():
        raise FileNotFoundError(f"df_target_train.csv not found at {df_target_path}")

    patterns_dir = artefacts_root / "patterns"
    baselines_dir = artefacts_root / "baselines"
    explain_dir = artefacts_root / "explainability"

    for d in (patterns_dir, baselines_dir, explain_dir):
        d.mkdir(parents=True, exist_ok=True)

    df_apv = pd.read_csv(df_apv_path)
    df_target = pd.read_csv(df_target_path)

    df_patterns = build_pattern_library(df_apv)
    pattern_path = patterns_dir / "pattern_library.csv"
    df_patterns.to_csv(pattern_path, index=False)

    df_apv_baselines = compute_apv_baselines(df_apv)
    apv_baselines_path = baselines_dir / "apv_baselines.csv"
    df_apv_baselines.to_csv(apv_baselines_path, index=False)

    df_feature_baselines = compute_feature_baselines(df_apv)
    feature_baselines_path = baselines_dir / "feature_baselines.csv"
    df_feature_baselines.to_csv(feature_baselines_path, index=False)

    archetype_defaults_path = baselines_dir / "archetype_defaults.csv"
    compute_archetype_defaults(df_apv).to_csv(archetype_defaults_path, index=False)

    delivery_defaults_path = baselines_dir / "delivery_defaults.csv"
    compute_delivery_defaults(df_apv).to_csv(delivery_defaults_path, index=False)

    # one row per real corner (setup and outcome), for the app's evidence and charts
    outcome_cols = [
        "match_id", "competition_id", "season_id", "A_name", "delivery_type", "end_zone",
        "short_corner_flag", "delivery_route",
        "n_att_near", "n_att_central", "n_att_far", "n_blockers", "n_edge_of_box", "n_rest_defence",
        "def_spot", "free_att_near", "free_att_central", "free_att_far", "free_att_spot",
        "first_contact_attack", "first_contact_type", "chance_for", "shot_against_any",
        "goal_for", "xg_for", "xg_against",
    ]
    corner_outcomes_path = patterns_dir / "corner_outcomes.csv"
    df_apv[[c for c in outcome_cols if c in df_apv.columns]].to_csv(corner_outcomes_path, index=False)

    thresholds = compute_value_risk_thresholds(df_apv)
    thresholds_path = baselines_dir / "value_risk_thresholds.csv"
    pd.DataFrame([thresholds]).to_csv(thresholds_path, index=False)

    expl_cfg = cfg.get("explainability", {}) or {}
    project_cfg = cfg.get("project", {}) or {}

    shap_sample_size = int(expl_cfg.get("shap_sample_size", 500))
    random_seed = int(project_cfg.get("random_seed", 42))

    shap_bg = build_shap_background(
        df_target=df_target,
        sample_size=shap_sample_size,
        random_state=random_seed,
    )
    shap_bg_path = explain_dir / "shap_background.csv"
    shap_bg.to_csv(shap_bg_path, index=False)

    return {
        "pattern_library": pattern_path,
        "corner_outcomes": corner_outcomes_path,
        "apv_baselines": apv_baselines_path,
        "feature_baselines": feature_baselines_path,
        "archetype_defaults": archetype_defaults_path,
        "delivery_defaults": delivery_defaults_path,
        "value_risk_thresholds": thresholds_path,
        "shap_background": shap_bg_path,
    }



def _load_value_risk_thresholds_from_disk() -> None:
    # read the thresholds from csv, else use the defaults
    global _VALUE_RISK_THRESHOLDS, _THRESHOLDS_LOADED

    if _THRESHOLDS_LOADED:
        return

    default_path = Path("artefacts") / "baselines" / "value_risk_thresholds.csv"
    if default_path.exists():
        try:
            df_thr = pd.read_csv(default_path)
            if not df_thr.empty:
                row = df_thr.iloc[0].to_dict()
                for key in _VALUE_RISK_THRESHOLDS.keys():
                    if key in row and pd.notna(row[key]):
                        _VALUE_RISK_THRESHOLDS[key] = float(row[key])
        except Exception:
            # use the defaults if reading fails
            pass

    _THRESHOLDS_LOADED = True


def _get_value_risk_thresholds() -> Dict[str, float]:
    _load_value_risk_thresholds_from_disk()
    return _VALUE_RISK_THRESHOLDS


def _value_label(apv_net: float) -> str:
    # APV_net as a simple value label
    if pd.isna(apv_net):
        return "Unknown value"

    thr = _get_value_risk_thresholds()
    v_high = thr["VALUE_HIGH"]
    v_pos = thr["VALUE_POSITIVE"]
    v_neg = thr["VALUE_NEGATIVE"]

    if apv_net >= v_high:
        return "High value"
    if apv_net >= v_pos:
        return "Slightly positive"
    if apv_net <= v_neg:
        return "Negative value"
    return "Neutral value"


def _risk_label(pv_against: float) -> str:
    # PV_against as a simple risk label
    if pd.isna(pv_against):
        return "Unknown risk"

    thr = _get_value_risk_thresholds()
    r_low = thr["RISK_LOW"]
    r_med = thr["RISK_MEDIUM"]

    if pv_against <= r_low:
        return "Low risk"
    if pv_against <= r_med:
        return "Medium risk"
    return "High risk"


def _value_risk_tag(apv_net: float, pv_against: float) -> str:
    # value and risk as one short tag
    v = _value_label(apv_net)
    r = _risk_label(pv_against)
    if "Unknown" in v or "Unknown" in r:
        return f"{v}, {r.lower()}"
    return f"{v}, {r.lower()}"



def select_best_setups(
    pattern_df: pd.DataFrame,
    A_name: Optional[str] = None,
    min_samples: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    # setups against one defensive shape, ranked by APV
    if pattern_df.empty:
        return pattern_df.copy()

    df = pattern_df.copy()

    if A_name and A_name not in ("All", "ALL", "all"):
        df = df[df["A_name"] == A_name]

    if "n_corners" in df.columns:
        df = df[df["n_corners"] >= int(min_samples)]

    if df.empty:
        return df

    apv_col = "APV_net_mean"
    risk_col = "PV_against_mean"

    if apv_col not in df.columns or risk_col not in df.columns:
        return df

    df = df.copy()
    df["value_label"] = df[apv_col].apply(_value_label)
    df["risk_label"] = df[risk_col].apply(_risk_label)
    df["tag"] = df.apply(
        lambda row: _value_risk_tag(row[apv_col], row[risk_col]),
        axis=1,
    )

    df = df.sort_values(apv_col, ascending=False)

    return df.head(int(top_n)).reset_index(drop=True)



# features the coach controls; explanations only mention these (the opponent's
# come with the chosen defensive shape)
TACTICAL_FEATURE_KEYS: List[str] = [
    "n_att_near",
    "n_att_central",
    "n_att_far",
    "n_blockers",
    "n_edge_of_box",
    "n_rest_defence",
    "free_att_",
    "delivery_type",
    "end_zone",
]


def _is_tactical_feature(name: Any) -> bool:
    text = str(name)
    return any(key in text for key in TACTICAL_FEATURE_KEYS)


def _positive_sentence(label: str) -> str:
    base = label.lower()
    if "attackers" in base or "players" in base:
        return f"{label} looks like a strength compared with similar corners."
    if "rest defence" in base:
        return f"{label} has generally provided good cover against counters."
    if "delivery" in base:
        return f"This {label.lower()} has tended to work well in similar corners."
    if (
        "defensive line" in base
        or "goalkeeper" in base
        or "defenders in six yard" in base
        or "players on the posts" in base
    ):
        return f"The {label.lower()} has been associated with better outcomes in the data."
    return f"{label} has been a positive driver in similar corners."


def _risk_sentence(label: str) -> str:
    base = label.lower()
    if "rest defence" in base:
        return f"The current {label.lower()} may leave space for counters if the ball is cleared."
    if "attackers" in base or "players" in base:
        return f"The current {label.lower()} can expose space or reduce chance quality if execution is poor."
    if "delivery" in base:
        return f"Similar corners with this {label.lower()} have more often produced lower net value or higher risk."
    if (
        "defensive line" in base
        or "goalkeeper" in base
        or "defenders in six yard" in base
        or "players on the posts" in base
    ):
        return f"In the data, the {label.lower()} has sometimes been linked with higher risk in these situations."
    return f"{label} tends to be associated with higher risk or lower value in the data."


def build_coach_explanation(
    pattern_features: Dict[str, Any],
    apv: Dict[str, float],
    probs: Dict[str, float],
    baseline_apv: Optional[float],
    shap_df: Optional[pd.DataFrame] = None,
    max_bullets: int = 3,
) -> Dict[str, Any]:
    # turn the APV numbers and SHAP values into plain text
    if shap_df is not None and not shap_df.empty:
        if pattern_features.get("delivery_type", "").lower() != "short":
            shap_df = shap_df[shap_df["feature_label"] != "Short corner flag"]

    apv_net = float(apv.get("APV_net", np.nan))
    pv_for = float(apv.get("PV_for", np.nan))
    pv_against = float(apv.get("PV_against", np.nan))

    verdict_label = _value_risk_tag(apv_net, pv_against)

    archetype = pattern_features.get("A_name", "this archetype")
    if baseline_apv is None or pd.isna(baseline_apv):
        baseline_text = "No baseline available for this defensive shape."
    else:
        band = _get_value_risk_thresholds()["SIMILAR_BAND"]
        delta = apv_net - float(baseline_apv)
        if delta > band:
            baseline_text = f"Better than typical {archetype} corners."
        elif delta < -band:
            baseline_text = f"Worse than typical {archetype} corners."
        else:
            baseline_text = f"Similar to typical {archetype} corners."

    bullets_help: List[str] = []
    bullets_risk: List[str] = []
    # the feature labels behind each bullet, so the app can word them itself
    help_labels: List[str] = []
    risk_labels: List[str] = []

    if shap_df is not None and not shap_df.empty:
        df = shap_df.copy()

        if "feature" in df.columns:
            mask = df["feature"].apply(_is_tactical_feature)
            df_tactical = df[mask]
            if not df_tactical.empty:
                df = df_tactical

        if {"feature_label", "shap", "abs_shap"}.issubset(df.columns):
            grouped = (
                df.groupby("feature_label", as_index=False)
                .agg(total_abs=("abs_shap", "sum"), net=("shap", "sum"))
            )

            grouped["direction"] = np.where(
                grouped["net"] >= 0, "positive", "negative"
            )
            # rank by the net effect: one-hot parts of a factor can cancel out, so adding up
            # their sizes would overstate it
            grouped["net_abs"] = grouped["net"].abs()

            df_pos = (
                grouped[grouped["direction"] == "positive"]
                .sort_values("net_abs", ascending=False)
                .head(max_bullets)
            )
            df_neg = (
                grouped[grouped["direction"] == "negative"]
                .sort_values("net_abs", ascending=False)
                .head(max_bullets)
            )

            help_labels = [str(label) for label in df_pos["feature_label"].tolist()]
            risk_labels = [str(label) for label in df_neg["feature_label"].tolist()]

            for label in help_labels:
                bullets_help.append(_positive_sentence(label))

            for label in risk_labels:
                bullets_risk.append(_risk_sentence(label))

    if not bullets_help and not bullets_risk:
        n_central = int(pattern_features.get("n_att_central", 0) or 0)
        n_near = int(pattern_features.get("n_att_near", 0) or 0)
        n_far = int(pattern_features.get("n_att_far", 0) or 0)
        n_rest = int(pattern_features.get("n_rest_defence", 0) or 0)

        if n_central >= 4:
            bullets_help.append(
                "Several central targets increase the chance of a first contact."
            )
        if n_near >= 2:
            bullets_help.append(
                "Extra players near post help to attack flick ons and rebounds."
            )
        if n_far >= 2:
            bullets_help.append(
                "Far post presence improves chances from second balls."
            )

        if n_rest <= 1:
            bullets_risk.append(
                "Very few players staying back increase counter attack risk."
            )
        elif n_rest == 2:
            bullets_risk.append(
                "Limited rest defence can leave space for counters if delivery is poor."
            )

    result: Dict[str, Any] = {
        "verdict_label": verdict_label,
        "baseline_comparison": baseline_text,
        "apv": apv,
        "probabilities": probs,
        "bullets_help": bullets_help,
        "bullets_risk": bullets_risk,
        "help_labels": help_labels,
        "risk_labels": risk_labels,
    }

    return result



def _describe_pattern_row(row: pd.Series) -> str:
    # short text describing one setup
    parts: List[str] = []

    parts.append(str(row.get("A_name", "Unknown")))
    parts.append(str(row.get("delivery_type", "delivery")))
    parts.append(str(row.get("end_zone", "zone")))

    n_near = int(row.get("n_att_near", 0) or 0)
    n_central = int(row.get("n_att_central", 0) or 0)
    n_far = int(row.get("n_att_far", 0) or 0)
    parts.append(f"N/C/F {n_near}/{n_central}/{n_far}")

    n_blockers = int(row.get("n_blockers", 0) or 0)
    n_rest = int(row.get("n_rest_defence", 0) or 0)
    parts.append(f"blockers {n_blockers}")
    parts.append(f"rest defence {n_rest}")

    return ", ".join(parts)


def suggest_tweaks_simple(
    pattern_features: Dict[str, Any],
    apv_net: float,
    pattern_df: pd.DataFrame,
    min_samples: int = 10,
    max_suggestions: int = 3,
    improvement_threshold: float = 0.02,
) -> List[Dict[str, Any]]:
    # suggest simple alternative setups
    if pattern_df.empty:
        return []

    archetype = pattern_features.get("A_name")
    if not archetype:
        return []

    df = pattern_df.copy()
    df = df[df["A_name"] == archetype]

    if "n_corners" in df.columns:
        df = df[df["n_corners"] >= int(min_samples)]

    if df.empty:
        return []

    apv_col = "APV_net_mean"
    risk_col = "PV_against_mean"

    if apv_col not in df.columns or risk_col not in df.columns:
        return []

    df = df.copy()
    df["delta_apv"] = df[apv_col] - float(apv_net)
    df = df[df["delta_apv"] > float(improvement_threshold)]

    if df.empty:
        return []

    df = df.sort_values("delta_apv", ascending=False).head(int(max_suggestions))

    suggestions: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        new_apv = float(row[apv_col])
        delta = float(row["delta_apv"])
        tag = _value_risk_tag(new_apv, float(row[risk_col]))

        suggestions.append(
            {
                "change_label": _describe_pattern_row(row),
                "new_apv": new_apv,
                "delta_apv": delta,
                "tag": tag,
            }
        )

    return suggestions



def plot_apv_histogram(
    df_apv: pd.DataFrame,
    out_path: Path,
    bins: int = 30,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_apv is None or df_apv.empty:
        print("plot_apv_histogram: empty df_apv, skipping.")
        return

    if "APV_net" not in df_apv.columns:
        raise KeyError("plot_apv_histogram: df_apv missing 'APV_net' column.")

    values = df_apv["APV_net"].astype(float).dropna()
    if values.empty:
        print("plot_apv_histogram: no non-null APV_net values, skipping.")
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    counts, bin_edges, _ = ax.hist(values.to_numpy(), bins=bins, edgecolor="black")
    ax.set_xlabel("APV_net")
    ax.set_ylabel("Number of corners")
    ax.set_title("Histogram of APV_net (all corners)")

    mean_val = float(values.mean())
    ax.axvline(mean_val, linestyle="--", linewidth=1)
    ax.text(
        mean_val,
        ax.get_ylim()[1] * 0.9,
        f"mean = {mean_val:.3f}",
        ha="center",
        va="top",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved APV histogram -> {out_path}")

    bins_csv_path = out_path.parent / f"{out_path.stem}_bins.csv"
    hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": counts.astype(int),
        }
    )
    hist_df.to_csv(bins_csv_path, index=False)
    print(f"Saved APV histogram bins -> {bins_csv_path}")

    summary_csv_path = out_path.parent / f"{out_path.stem}_summary.csv"
    summary = {
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.quantile(0.50)),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
    }
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    print(f"Saved APV histogram summary -> {summary_csv_path}")


def plot_apv_by_archetype_boxplot(
    df_apv: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_apv is None or df_apv.empty:
        print("plot_apv_by_archetype_boxplot: empty df_apv, skipping.")
        return

    required = ["A_name", "APV_net"]
    missing = [c for c in required if c not in df_apv.columns]
    if missing:
        raise KeyError(
            f"plot_apv_by_archetype_boxplot: df_apv missing columns {missing}"
        )

    df_plot = df_apv[required].dropna()
    if df_plot.empty:
        print("plot_apv_by_archetype_boxplot: no non-null rows, skipping.")
        return

    archetypes = sorted(df_plot["A_name"].astype(str).unique())
    data = [
        df_plot.loc[df_plot["A_name"] == a, "APV_net"].astype(float).values
        for a in archetypes
    ]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot(
        data,
        labels=archetypes,
        showmeans=True,
    )
    ax.set_xlabel("Defensive archetype (A_name)")
    ax.set_ylabel("APV_net")
    ax.set_title("Distribution of APV_net by defensive archetype")

    ax.axhline(0.0, linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved APV by archetype boxplot -> {out_path}")

    stats_rows = []
    grouped = df_plot.groupby("A_name")["APV_net"]

    for a_name, s in grouped:
        s = s.astype(float)
        stats_rows.append(
            {
                "A_name": str(a_name),
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "q1": float(s.quantile(0.25)),
                "median": float(s.quantile(0.50)),
                "q3": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )

    stats_df = pd.DataFrame(stats_rows).sort_values("A_name")
    stats_csv_path = out_path.parent / f"{out_path.stem}_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved APV by archetype stats -> {stats_csv_path}")



