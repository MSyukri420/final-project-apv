from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from src.apv import add_apv_columns
from src.constants import MODELLING_TARGETS, OUTCOME_FEATURES
from src.insights import (
    build_coach_explanation,
    select_best_setups,
    suggest_tweaks_simple,
    PATTERN_GROUP_COLS,
    _value_label,
    _risk_label,
    _value_risk_tag,
)


# Coach friendly labels for key features used in SHAP explanations
_FEATURE_LABEL_MAP: Dict[str, str] = {
    "def_line_height": "Defensive line height",
    "def_in_6yd": "Defenders in six yard box",
    "gk_start_line": "Goalkeeper starting position",
    "players_on_posts": "Players on the posts",
    "n_att_near": "Attackers at near post",
    "n_att_central": "Attackers in central zone",
    "n_att_far": "Attackers at far post",
    "n_blockers": "Blockers around markers",
    "n_edge_of_box": "Players on edge of box",
    "n_rest_defence": "Players in rest defence",
    "delivery_type": "Delivery type",
    "end_zone": "Delivery end zone",
    "short_corner_flag": "Short corner flag",
}


class APVEngine:
    """
    Runtime engine that works only with trained models and artefacts.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}

        paths_cfg = self.cfg.get("paths", {}) or {}
        self.models_dir = Path(paths_cfg.get("models", "models")).expanduser()
        self.outputs_dir = Path(paths_cfg.get("outputs", "outputs")).expanduser()
        self.artefacts_dir = Path(paths_cfg.get("artefacts", "artefacts")).expanduser()

        apv_cfg = self.cfg.get("apv", {}) or {}
        self.alpha = float(apv_cfg.get("alpha", 1.0))
        self.beta = float(apv_cfg.get("beta", 1.0))

        reco_cfg = self.cfg.get("recommendations", {}) or {}
        self.default_min_samples = int(reco_cfg.get("min_samples", 10))

        project_cfg = self.cfg.get("project", {}) or {}
        self.random_seed = int(project_cfg.get("random_seed", 42))

        # Runtime state
        self.models: Dict[str, Any] = {}

        self.pattern_library: Optional[pd.DataFrame] = None
        self.apv_baselines: Optional[pd.DataFrame] = None
        self.baseline_by_archetype: Dict[str, float] = {}
        self.feature_baselines: Optional[pd.DataFrame] = None
        self.shap_background: Optional[pd.DataFrame] = None

        self.feature_columns: List[str] = []
        self.archetype_defaults: Dict[str, Dict[str, Any]] = {}

        # Runtime SHAP state (lazy initialised)
        self._shap_explainer: Any = None
        self._shap_feature_names: List[str] = []
        self._shap_preprocess: Optional[Any] = None

        # Load everything
        self._load_models()
        self._load_artefacts()
        self._build_archetype_defaults()

        if not self.models:
            raise RuntimeError(
                f"No pooled models found in {self.models_dir.resolve()}."
            )

        if not self.feature_columns:
            raise RuntimeError(
                "No feature columns available for runtime predictions."
            )

        print(
            f"[APVEngine] Initialised. "
            f"Models: {list(self.models.keys())}, "
            f"feature_cols: {len(self.feature_columns)}, "
            f"archetypes: {self.get_archetypes()}"
        )

    # ============================================================
    # Loading models and artefacts
    # ============================================================

    def _load_models(self) -> None:
        """
        Load pooled logistic regression pipelines for all modelling targets.
        """
        self.models = {}
        for target_name in MODELLING_TARGETS:
            path = self.models_dir / f"{target_name}_pooled.joblib"
            if not path.exists():
                print(f"[APVEngine] Model not found for {target_name}: {path}")
                continue
            self.models[target_name] = joblib.load(path)
            print(f"[APVEngine] Loaded model for {target_name} from {path}")

    def _load_artefacts(self) -> None:
        """
        Load pattern library, APV baselines, feature baselines and SHAP background.
        """
        # Pattern library
        patterns_path = self.artefacts_dir / "patterns" / "pattern_library.csv"
        if patterns_path.exists():
            self.pattern_library = pd.read_csv(patterns_path)
            print(
                "[APVEngine] Loaded pattern library from "
                f"{patterns_path} with shape {self.pattern_library.shape}"
            )
        else:
            print(
                "[APVEngine] Pattern library not found at "
                f"{patterns_path}. Best setups and tweaks will be empty."
            )
            self.pattern_library = pd.DataFrame()

        # APV baselines and feature baselines
        baselines_dir = self.artefacts_dir / "baselines"
        apv_baselines_path = baselines_dir / "apv_baselines.csv"
        if apv_baselines_path.exists():
            self.apv_baselines = pd.read_csv(apv_baselines_path)
            print(
                "[APVEngine] Loaded APV baselines from "
                f"{apv_baselines_path} with shape {self.apv_baselines.shape}"
            )
            if (
                "A_name" in self.apv_baselines.columns
                and "APV_net_mean" in self.apv_baselines.columns
            ):
                self.baseline_by_archetype = dict(
                    zip(
                        self.apv_baselines["A_name"],
                        self.apv_baselines["APV_net_mean"],
                    )
                )
        else:
            print("[APVEngine] APV baselines not found.")

        feature_baselines_path = baselines_dir / "feature_baselines.csv"
        if feature_baselines_path.exists():
            self.feature_baselines = pd.read_csv(feature_baselines_path)
            print(
                "[APVEngine] Loaded feature baselines from "
                f"{feature_baselines_path} with shape {self.feature_baselines.shape}"
            )
        else:
            print("[APVEngine] Feature baselines not found.")

        # SHAP background (used only for archetype defaults, SHAP itself is not used)
        explain_dir = self.artefacts_dir / "explainability"
        shap_bg_path = explain_dir / "shap_background_chance_for.csv"
        if shap_bg_path.exists():
            self.shap_background = pd.read_csv(shap_bg_path)
            print(
                "[APVEngine] Loaded SHAP background from "
                f"{shap_bg_path} with shape {self.shap_background.shape}"
            )
            self.feature_columns = list(self.shap_background.columns)
        else:
            print(
                "[APVEngine] SHAP background not found. "
                "Falling back to OUTCOME_FEATURES for feature list."
            )
            self.shap_background = None
            outcome_cfg = OUTCOME_FEATURES.get("outcome_chance_for", {}) or {}
            num_cols = outcome_cfg.get("numeric") or []
            cat_cols = outcome_cfg.get("categorical") or []
            self.feature_columns = list(dict.fromkeys(list(num_cols) + list(cat_cols)))

    def _ensure_pattern_library_loaded(self) -> None:
        """
        Make sure pattern_library is non empty. If it is empty, try to reload
        from artefacts/patterns/pattern_library.csv.
        """
        if self.pattern_library is not None and not self.pattern_library.empty:
            return

        patterns_path = self.artefacts_dir / "patterns" / "pattern_library.csv"
        if patterns_path.exists():
            self.pattern_library = pd.read_csv(patterns_path)
            print(
                "[APVEngine] Reloaded pattern library from "
                f"{patterns_path} with shape {self.pattern_library.shape}"
            )
        else:
            print(
                "[APVEngine] Pattern library still not found at "
                f"{patterns_path}"
            )
            self.pattern_library = pd.DataFrame()

    def _build_archetype_defaults(self) -> None:
        """
        Build default feature rows per defensive archetype from the SHAP background.
        """
        self.archetype_defaults = {}

        if self.shap_background is not None and "A_name" in self.shap_background.columns:
            for archetype, group in self.shap_background.groupby("A_name"):
                row = group.iloc[0]
                self.archetype_defaults[str(archetype)] = row.to_dict()

    # ============================================================
    # Public helpers
    # ============================================================

    def get_archetypes(self) -> List[str]:
        """
        List defensive archetypes known to the engine.
        """
        if self.apv_baselines is not None and "A_name" in self.apv_baselines.columns:
            return sorted(self.apv_baselines["A_name"].unique().tolist())

        if self.shap_background is not None and "A_name" in self.shap_background.columns:
            return sorted(self.shap_background["A_name"].unique().tolist())

        if self.pattern_library is not None and "A_name" in self.pattern_library.columns:
            return sorted(self.pattern_library["A_name"].unique().tolist())

        return []

    # ============================================================
    # Runtime SHAP helpers
    # ============================================================

    def _ensure_shap_explainer(self) -> None:
        """
        Lazily build a SHAP explainer for the pooled chance_for model.

        This uses the logistic regression step on top of the preprocessed,
        numeric feature space to avoid issues with string features.
        """
        if self._shap_explainer is not None:
            return

        if shap is None:
            print("[APVEngine] shap package not available - runtime SHAP disabled.")
            return

        model = self.models.get("chance_for")
        if model is None:
            print("[APVEngine] chance_for model not loaded - runtime SHAP disabled.")
            return

        if self.shap_background is None or self.shap_background.empty:
            print("[APVEngine] No SHAP background - runtime SHAP disabled.")
            return

        # We expect a Pipeline with 'preprocess' and 'clf' steps
        if not hasattr(model, "named_steps"):
            print(
                "[APVEngine] chance_for model is not a Pipeline with named_steps - "
                "runtime SHAP disabled."
            )
            return

        try:
            preprocess = model.named_steps["preprocess"]
            clf = model.named_steps["clf"]
        except KeyError:
            print(
                "[APVEngine] chance_for Pipeline does not have 'preprocess' and 'clf' "
                "steps - runtime SHAP disabled."
            )
            return

        # Use the same feature columns as training
        cols = self.feature_columns or list(self.shap_background.columns)
        try:
            X_bg_raw = self.shap_background[cols].copy()
        except KeyError:
            cols = [c for c in cols if c in self.shap_background.columns]
            if not cols:
                print(
                    "[APVEngine] SHAP background does not contain any feature columns - "
                    "runtime SHAP disabled."
                )
                return
            X_bg_raw = self.shap_background[cols].copy()

        # Downsample background for speed
        max_bg = 200
        if len(X_bg_raw) > max_bg:
            X_bg_raw = X_bg_raw.sample(
                n=max_bg, random_state=self.random_seed
            ).reset_index(drop=True)

        try:
            # Transform raw features to the numeric feature space used by the model
            X_bg_proc = preprocess.transform(X_bg_raw)
            if hasattr(X_bg_proc, "toarray"):
                X_bg_proc_dense = X_bg_proc.toarray()
            else:
                X_bg_proc_dense = np.asarray(X_bg_proc)

            # LinearExplainer is appropriate for LogisticRegression
            self._shap_explainer = shap.LinearExplainer(clf, X_bg_proc_dense)
            self._shap_preprocess = preprocess

            # Feature names from the preprocessor if available
            if hasattr(preprocess, "get_feature_names_out"):
                self._shap_feature_names = list(preprocess.get_feature_names_out())
            else:
                self._shap_feature_names = [
                    f"f_{i}" for i in range(X_bg_proc_dense.shape[1])
                ]

            print(
                "[APVEngine] Initialised runtime SHAP explainer with "
                f"{len(self._shap_feature_names)} features and {len(X_bg_proc_dense)} background rows."
            )
        except Exception as exc:  # pragma: no cover
            print(f"[APVEngine] Failed to initialise SHAP explainer: {exc}")
            self._shap_explainer = None
            self._shap_preprocess = None
            self._shap_feature_names = []



    def _compute_local_shap(
        self,
        X_row: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Compute local SHAP values for a single feature row and return a coach
        friendly summary DataFrame with columns:

        - feature
        - shap
        - abs_shap
        - direction ('positive' / 'negative')
        - feature_label (human readable label)

        Returns None if SHAP is not available.
        """
        if X_row is None or X_row.empty:
            return None

        if self._shap_explainer is None or self._shap_preprocess is None:
            self._ensure_shap_explainer()

        if self._shap_explainer is None or self._shap_preprocess is None:
            return None

        # Align raw row with training feature columns
        cols = self.feature_columns or list(X_row.columns)
        try:
            X_raw = X_row[cols].copy()
        except KeyError:
            cols = [c for c in cols if c in X_row.columns]
            if not cols:
                return None
            X_raw = X_row[cols].copy()

        # Transform to the numeric feature space used during training
        try:
            X_proc = self._shap_preprocess.transform(X_raw)
            if hasattr(X_proc, "toarray"):
                X_proc_dense = X_proc.toarray()
            else:
                X_proc_dense = np.asarray(X_proc)

            shap_values = self._shap_explainer(X_proc_dense)
        except Exception as exc:  # pragma: no cover
            print(f"[APVEngine] Failed to compute SHAP values: {exc}")
            return None

        # Extract raw SHAP array
        values_arr = getattr(shap_values, "values", None)
        if values_arr is None:
            try:
                values_arr = np.array(shap_values)
            except Exception:
                return None

        values_arr = np.asarray(values_arr)

        # Handle possible shapes:
        # (n_samples, n_features) or (n_samples, n_classes, n_features)
        if values_arr.ndim == 2:
            vals = values_arr[0]
        elif values_arr.ndim == 3:
            # For binary classification, take the positive class (last index)
            vals = values_arr[0, -1, :]
        else:
            vals = values_arr.reshape(-1)

        vals = np.asarray(vals, dtype=float).reshape(-1)

        # Use SHAP feature names if available, otherwise fall back
        feat_names = self._shap_feature_names or [
            f"f_{i}" for i in range(len(vals))
        ]
        if len(vals) != len(feat_names):
            n = min(len(vals), len(feat_names))
            vals = vals[:n]
            feat_names = feat_names[:n]

        df = pd.DataFrame(
            {
                "feature": feat_names,
                "shap": vals,
            }
        )
        df["abs_shap"] = df["shap"].abs()
        df["direction"] = np.where(df["shap"] >= 0, "positive", "negative")

        # Map to coach friendly labels by matching substrings to base features
        labels: List[str] = []
        for f in df["feature"]:
            label = None
            for base_name, human_label in _FEATURE_LABEL_MAP.items():
                if base_name in f:
                    label = human_label
                    break
            if label is None:
                # Fallback: strip common prefixes and tidy up
                label_clean = f
                for prefix in ("num__", "cat__", "preprocess__", "clf__"):
                    label_clean = label_clean.replace(prefix, "")
                label_clean = label_clean.replace("_", " ")
                label = label_clean
            labels.append(label)
        df["feature_label"] = labels

        # Remove exact zeros to keep explanations concise
        df = df[df["abs_shap"] > 0].reset_index(drop=True)

        if df.empty:
            return None

        return df

    def _lookup_pattern_stats(
        self,
        pattern_features: Dict[str, Any],
        min_corners: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find an exact match for this setup in the pattern library.

        Matching uses PATTERN_GROUP_COLS, restricted to columns that are
        actually present in pattern_library. Only patterns with at least
        min_corners examples are considered.

        Returns a dict with mean APV metrics if a match is found,
        otherwise None.
        """
        self._ensure_pattern_library_loaded()

        if self.pattern_library is None or self.pattern_library.empty:
            return None

        df = self.pattern_library

        required_cols = {
            "n_corners",
            "APV_net_mean",
            "PV_for_mean",
            "PV_against_mean",
        }
        if not required_cols.issubset(df.columns):
            return None

        # Build a mask for equality on all grouping columns that exist
        group_cols = [c for c in PATTERN_GROUP_COLS if c in df.columns]
        if not group_cols:
            return None

        mask = pd.Series(True, index=df.index)
        for col in group_cols:
            val = pattern_features.get(col)
            mask &= df[col] == val

        df_match = df[mask]

        if df_match.empty:
            return None

        # Enforce minimum sample size
        df_match = df_match[df_match["n_corners"] >= int(min_corners)]
        if df_match.empty:
            return None

        # If multiple matches, take the one with highest APV_net_mean
        row = df_match.sort_values("APV_net_mean", ascending=False).iloc[0]

        out: Dict[str, Any] = {
            "PV_for": float(row.get("PV_for_mean")),
            "PV_against": float(row.get("PV_against_mean")),
            "APV_net": float(row.get("APV_net_mean")),
            "P_chance_for": float(row.get("P_chance_for_mean", np.nan)),
            "P_chance_against": float(row.get("P_chance_against_mean", np.nan)),
            "P_shot_for_any": float(row.get("P_shot_for_any_mean", np.nan)),
            "P_shot_against_any": float(row.get("P_shot_against_any_mean", np.nan)),
            "n_corners": int(row.get("n_corners", 0)),
        }
        return out


    # ============================================================
    # Feature vector construction and prediction
    # ============================================================

    def _build_feature_row_from_coach_input(
        self,
        coach_input: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Build a single feature row from high level coach input.

        Starts from an archetype default row (from SHAP background) and
        overrides only the features the coach has specified.
        """
        archetype = str(coach_input.get("A_name", "Hybrid"))

        if archetype in self.archetype_defaults:
            base: Dict[str, Any] = dict(self.archetype_defaults[archetype])
        elif self.shap_background is not None and not self.shap_background.empty:
            base = self.shap_background.iloc[0].to_dict()
            base["A_name"] = archetype
        else:
            base = {col: 0 for col in self.feature_columns}
            if "A_name" in self.feature_columns:
                base["A_name"] = archetype
            if "delivery_type" in self.feature_columns:
                base["delivery_type"] = "inswing"
            if "end_zone" in self.feature_columns:
                base["end_zone"] = "central"

        # Override with coach input where the feature is modelled
        for key, value in coach_input.items():
            if key in self.feature_columns:
                base[key] = value

        # Derive short_corner_flag from delivery_type if needed
        if (
            "delivery_type" in self.feature_columns
            and "short_corner_flag" in self.feature_columns
            and "short_corner_flag" not in coach_input
        ):
            delivery_type = str(base.get("delivery_type", "")).lower()
            base["short_corner_flag"] = 1 if delivery_type == "short" else 0

        # Ensure integer counts for attacking distribution
        count_cols = [
            "n_att_near",
            "n_att_central",
            "n_att_far",
            "n_blockers",
            "n_edge_of_box",
            "n_rest_defence",
        ]
        for col in count_cols:
            if col in self.feature_columns:
                try:
                    base[col] = int(base.get(col, 0))
                except (TypeError, ValueError):
                    base[col] = 0

        row = {col: base.get(col) for col in self.feature_columns}
        return pd.DataFrame([row])

    def _predict_probs_and_apv(
        self,
        X_row: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run all pooled models on a single feature row and compute APV.
        """
        if not self.feature_columns:
            raise RuntimeError("Feature columns are not available for prediction.")

        target_to_proba_col = {
            "chance_for": "P_chance_for",
            "chance_against": "P_chance_against",
            "shot_for_any": "P_shot_for_any",
            "shot_against_any": "P_shot_against_any",
        }

        probs: Dict[str, float] = {}
        X_model = X_row[self.feature_columns].copy()

        for target_name, proba_col in target_to_proba_col.items():
            model = self.models.get(target_name)
            if model is None:
                probs[proba_col] = float("nan")
                continue

            try:
                proba = model.predict_proba(X_model)[:, 1]
                probs[proba_col] = float(proba[0])
            except Exception:
                # Fallback to predict if predict_proba is missing
                pred = model.predict(X_model)
                probs[proba_col] = float(pred[0])

        df_probs = pd.DataFrame([probs])

        df_apv = add_apv_columns(
            df_probs,
            alpha=self.alpha,
            beta=self.beta,
            p_chance_for_col="P_chance_for",
            p_shot_against_col="P_shot_against_any",
            pv_for_col="PV_for",
            pv_against_col="PV_against",
            apv_net_col="APV_net",
        )

        apv = {
            "PV_for": float(df_apv["PV_for"].iloc[0]),
            "PV_against": float(df_apv["PV_against"].iloc[0]),
            "APV_net": float(df_apv["APV_net"].iloc[0]),
        }

        return probs, apv

    # ============================================================
    # Coach facing helpers
    # ============================================================

    def get_best_setups_for_shape(
        self,
        archetype: str,
        min_samples: Optional[int] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top patterns for a defensive shape using the pattern library.

        This version filters directly from pattern_library and
        respects the min_samples filter. If no patterns satisfy
        the filter it returns an empty DataFrame instead of
        silently falling back.
        """
        self._ensure_pattern_library_loaded()

        if self.pattern_library is None or self.pattern_library.empty:
            print("[APVEngine] No pattern library available.")
            return pd.DataFrame()

        if not archetype:
            return pd.DataFrame()

        df = self.pattern_library.copy()

        # Filter by defensive shape
        if "A_name" in df.columns:
            df = df[df["A_name"] == archetype]

        if df.empty:
            return pd.DataFrame()

        # Apply min_samples if possible
        ms = int(min_samples) if min_samples is not None else int(self.default_min_samples)
        ms = max(ms, 1)

        if "n_corners" in df.columns:
            df = df[df["n_corners"] >= ms]

        # If the filter removes everything, return empty and let the UI
        # tell the coach that no patterns are available at this threshold.
        if df.empty:
            return pd.DataFrame()

        # Add value / risk tags if needed
        apv_col = "APV_net_mean"
        risk_col = "PV_against_mean"
        if apv_col in df.columns and risk_col in df.columns:
            df = df.copy()
            df["value_label"] = df[apv_col].apply(_value_label)
            df["risk_label"] = df[risk_col].apply(_risk_label)
            df["tag"] = df.apply(
                lambda row: _value_risk_tag(row[apv_col], row[risk_col]),
                axis=1,
            )

        # Sort by APV and keep top_n rows
        if apv_col in df.columns:
            df = df.sort_values(apv_col, ascending=False)

        if top_n is not None and top_n > 0:
            df = df.head(int(top_n))

        return df


    def evaluate_for_all_archetypes(
        self,
        coach_input: Dict[str, Any],
        max_explanation_bullets: int = 3,
    ) -> Dict[str, Any]:
        """
        Evaluate the same attacking setup against all defensive archetypes.

        Returns:
            {
                "per_archetype": [ list of evaluate_for_coach style dicts, one per A_name ],
                "summary": pd.DataFrame with one row per archetype,
                "best_index": index into per_archetype of the highest APV_net,
            }
        """
        archetypes = self.get_archetypes()
        results: List[Dict[str, Any]] = []

        if not archetypes:
            return {
                "per_archetype": [],
                "summary": pd.DataFrame(),
                "best_index": None,
            }

        # Take a copy of input and remove any existing A_name
        base_input: Dict[str, Any] = dict(coach_input)
        base_input.pop("A_name", None)

        best_index: Optional[int] = None
        best_apv: float = float("-inf")

        for idx, archetype in enumerate(archetypes):
            ci = dict(base_input)
            ci["A_name"] = archetype

            res = self.evaluate_for_coach(
                ci,
                max_explanation_bullets=max_explanation_bullets,
            )
            res["A_name"] = archetype
            results.append(res)

            apv_dict = res.get("apv", {}) or {}
            apv_net = float(apv_dict.get("APV_net", float("nan")))
            if not np.isnan(apv_net) and apv_net > best_apv:
                best_apv = apv_net
                best_index = idx

        # Build a simple comparison table
        rows: List[Dict[str, Any]] = []
        for res in results:
            archetype = res.get("A_name")
            apv_dict = res.get("apv", {}) or {}
            probs_dict = res.get("probabilities", {}) or {}
            explanation = res.get("explanation", {}) or {}

            rows.append(
                {
                    "A_name": archetype,
                    "P_chance_for": probs_dict.get("P_chance_for"),
                    "P_shot_against_any": probs_dict.get("P_shot_against_any"),
                    "PV_for": apv_dict.get("PV_for"),
                    "PV_against": apv_dict.get("PV_against"),
                    "APV_net": apv_dict.get("APV_net"),
                    "verdict_label": explanation.get("verdict_label"),
                }
            )

        summary_df = pd.DataFrame(rows)

        return {
            "per_archetype": results,
            "summary": summary_df,
            "best_index": best_index,
        }



    def evaluate_for_coach(
        self,
        coach_input: Dict[str, Any],
        max_explanation_bullets: int = 3,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for the coach tab.

        Takes a high level description of a corner setup and returns:
        - feature vector used for the model
        - probabilities of key outcomes
        - APV components
        - baseline comparison per archetype
        - bullet point explanation
        - simple tweak suggestions
        """
        self._ensure_pattern_library_loaded()

        # Build full feature row and pattern descriptor
        X_row = self._build_feature_row_from_coach_input(coach_input)
        pattern_features = X_row.iloc[0].to_dict()

        archetype = str(pattern_features.get("A_name", coach_input.get("A_name", "")))
        baseline_apv = self.baseline_by_archetype.get(archetype)

        # ------------------------------------------------------------------
        # 1) Try to use historical pattern stats if an exact match exists
        # ------------------------------------------------------------------
        pattern_stats = self._lookup_pattern_stats(
            pattern_features=pattern_features,
            min_corners=3,  # same intuition as the slider default in the UI
        )

        if pattern_stats is not None:
            # Use averages from pattern_library
            probs = {
                "P_chance_for": pattern_stats.get("P_chance_for", np.nan),
                "P_chance_against": pattern_stats.get("P_chance_against", np.nan),
                "P_shot_for_any": pattern_stats.get("P_shot_for_any", np.nan),
                "P_shot_against_any": pattern_stats.get("P_shot_against_any", np.nan),
            }
            apv = {
                "PV_for": pattern_stats.get("PV_for", np.nan),
                "PV_against": pattern_stats.get("PV_against", np.nan),
                "APV_net": pattern_stats.get("APV_net", np.nan),
            }
            # Still compute SHAP so we can explain the shape
            shap_df = self._compute_local_shap(X_row)
        else:
            # Fall back to model prediction for unseen or rare patterns
            probs, apv = self._predict_probs_and_apv(X_row)
            shap_df = self._compute_local_shap(X_row)


        explanation = build_coach_explanation(
            pattern_features=pattern_features,
            apv=apv,
            probs=probs,
            baseline_apv=baseline_apv,
            shap_df=shap_df,
            max_bullets=max_explanation_bullets,
        )

        # Simple tweak suggestions from pattern library (your existing logic)
        tweaks: List[Dict[str, Any]] = []
        try:
            if (
                self.pattern_library is not None
                and not self.pattern_library.empty
                and archetype
            ):
                df = self.pattern_library.copy()
                df = df[df.get("A_name") == archetype]

                if not df.empty and "APV_net_mean" in df.columns:
                    current_apv = float(apv.get("APV_net", np.nan))
                    current_risk = float(apv.get("PV_against", np.nan))

                    df_p = df.copy()
                    df_p["delta_apv"] = df_p["APV_net_mean"] - current_apv

                    candidates = pd.DataFrame()

                    # If we have risk info, enforce "similar or lower risk" first
                    if (
                        not np.isnan(current_risk)
                        and "PV_against_mean" in df_p.columns
                    ):
                        df_p["delta_risk"] = df_p["PV_against_mean"] - current_risk

                        # Primary rule: higher value, risk not more than +0.02
                        risk_tol = 0.02
                        candidates = df_p[
                            (df_p["delta_apv"] > 0.01)
                            & (df_p["delta_risk"] <= risk_tol)
                        ]

                        # Fallback: allow higher value regardless of risk
                        if candidates.empty:
                            candidates = df_p[df_p["delta_apv"] > 0.01]
                    else:
                        # No risk column, only higher value
                        candidates = df_p[df_p["delta_apv"] > 0.01]

                    # Final fallback: if still empty, just show the top patterns
                    if candidates.empty:
                        candidates = df_p

                    # Prefer big APV gain, then smaller extra risk
                    sort_cols = ["delta_apv"]
                    ascending = [False]
                    if "delta_risk" in candidates.columns:
                        sort_cols.append("delta_risk")
                        ascending.append(True)

                    candidates = candidates.sort_values(
                        sort_cols, ascending=ascending
                    ).head(3)

                    for _, row in candidates.iterrows():
                        label_parts = [
                            str(row.get("delivery_type", "")),
                            str(row.get("end_zone", "")),
                            f"N/C/F = {row.get('n_att_near', 0)}/"
                            f"{row.get('n_att_central', 0)}/"
                            f"{row.get('n_att_far', 0)}",
                        ]
                        change_label = ", ".join(label_parts)

                        tweaks.append(
                            {
                                "change_label": change_label,
                                "new_apv": float(row["APV_net_mean"]),
                                "delta_apv": float(row["delta_apv"]),
                                "tag": row.get("tag", ""),
                            }
                        )
        except Exception as exc:  # pragma: no cover
            print("[APVEngine] Error while computing tweaks:", exc)
            tweaks = []

        return {
            "features": pattern_features,
            "probabilities": probs,
            "apv": apv,
            "baseline_apv": baseline_apv,
            "explanation": explanation,
            "tweaks": tweaks,
        }

