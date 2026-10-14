from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from src.apv import add_apv_columns, predict_outcomes
from src.constants import CHAIN_TARGETS, MODELLING_TARGETS, SHAPE_ORDER, all_model_features
from src.feat_eng import add_derived_features
from src.insights import (
    build_coach_explanation,
    _value_label,
    _risk_label,
    _value_risk_tag,
)


# readable labels for the features shown in explanations
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
    "free_att_near": "Spare attackers at near post",
    "free_att_central": "Spare attackers in central zone",
    "free_att_far": "Spare attackers at far post",
    "free_att_spot": "Spare attackers around penalty spot",
    "def_near": "Defenders at near post",
    "def_central": "Defenders in central zone",
    "def_far": "Defenders at far post",
    "def_spot": "Defenders around penalty spot",
    "n_def_outside_box": "Opponents outside the box",
    "delivery_depth": "Delivery depth",
    "delivery_high": "High delivery",
    "delivery_type": "Delivery type",
    "end_zone": "Delivery end zone",
    "short_corner_flag": "Short corner flag",
}

# the explanation ("what the model sees") is about winning the first contact
EXPLAIN_TARGET = "first_contact_attack"

# estimates average over up to this many real situations per defensive shape
MAX_SITUATIONS = 50


class APVEngine:
    """Scores corner setups using the trained models and app files."""

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

        # state
        self.models: Dict[str, Any] = {}
        self.bootstrap: List[Dict[str, Any]] = []
        self.delivery_defaults: Optional[pd.DataFrame] = None
        # real corners with their outcomes, and the evaluation report
        self.corner_outcomes: pd.DataFrame = pd.DataFrame()
        self.evaluation: pd.DataFrame = pd.DataFrame()

        self.pattern_library: Optional[pd.DataFrame] = None
        self.apv_baselines: Optional[pd.DataFrame] = None
        self.baseline_by_archetype: Dict[str, float] = {}
        self.feature_baselines: Optional[pd.DataFrame] = None
        self.shap_background: Optional[pd.DataFrame] = None

        self.feature_columns: List[str] = []
        self.archetype_defaults: Dict[str, Dict[str, Any]] = {}
        self._archetype_defaults_table: Optional[pd.DataFrame] = None

        # SHAP, built on first use
        self._shap_explainer: Any = None
        self._shap_feature_names: List[str] = []
        self._shap_preprocess: Optional[Any] = None
        self._shap_columns: List[str] = []

        # load everything
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

    # loading models and app files

    def _load_models(self) -> None:
        """Load the outcome models and the bootstrap models for the ranges."""
        self.models = {}
        for target_name in MODELLING_TARGETS:
            path = self.models_dir / f"{target_name}_pooled.joblib"
            if not path.exists():
                print(f"[APVEngine] Model not found for {target_name}: {path}")
                continue
            self.models[target_name] = joblib.load(path)
        print(f"[APVEngine] Loaded models: {sorted(self.models)}")

        boot_path = self.models_dir / "bootstrap_ensemble.joblib"
        if boot_path.exists():
            self.bootstrap = joblib.load(boot_path)
            print(f"[APVEngine] Loaded {len(self.bootstrap)} bootstrap model pairs")
        else:
            print("[APVEngine] No bootstrap ensemble found - ratings will have no ranges.")

    def _load_artefacts(self) -> None:
        """Load the setup library, APV baselines, feature baselines and SHAP background."""
        # setup library
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

        # APV and feature baselines
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

        archetype_defaults_path = baselines_dir / "archetype_defaults.csv"
        self._archetype_defaults_table = (
            pd.read_csv(archetype_defaults_path) if archetype_defaults_path.exists() else None
        )

        outcomes_path = self.artefacts_dir / "patterns" / "corner_outcomes.csv"
        if outcomes_path.exists():
            self.corner_outcomes = pd.read_csv(outcomes_path)

        reports_dir = Path((self.cfg.get("paths", {}) or {}).get("reports", "reports")).expanduser()
        evaluation_path = reports_dir / "train" / "model_evaluation_train.csv"
        if evaluation_path.exists():
            self.evaluation = pd.read_csv(evaluation_path)

        delivery_defaults_path = baselines_dir / "delivery_defaults.csv"
        if delivery_defaults_path.exists():
            self.delivery_defaults = pd.read_csv(delivery_defaults_path)

        # SHAP background: real corners covering every model input
        explain_dir = self.artefacts_dir / "explainability"
        shap_bg_path = explain_dir / "shap_background.csv"
        if shap_bg_path.exists():
            self.shap_background = pd.read_csv(shap_bg_path)
            print(
                "[APVEngine] Loaded SHAP background from "
                f"{shap_bg_path} with shape {self.shap_background.shape}"
            )
        else:
            print("[APVEngine] SHAP background not found - explanations disabled.")
            self.shap_background = None
        self.feature_columns = all_model_features()

    def _ensure_pattern_library_loaded(self) -> None:
        """Reload the setup library from disk if it is empty."""
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
        Typical opponent and match situation per defensive shape (medians or most
        common values); the coach's choices are applied on top.
        """
        self.archetype_defaults = {}
        table = self._archetype_defaults_table
        if table is not None and "A_name" in table.columns:
            for rec in table.to_dict("records"):
                self.archetype_defaults[str(rec["A_name"])] = rec
            return

        # otherwise: build it from the SHAP background
        bg = self.shap_background
        if bg is not None and "A_name" in bg.columns:
            for archetype, group in bg.groupby("A_name"):
                rec: Dict[str, Any] = {}
                for col in group.columns:
                    s = group[col].dropna()
                    if s.empty:
                        continue
                    rec[col] = s.median() if pd.api.types.is_numeric_dtype(s) else s.mode().iloc[0]
                rec["A_name"] = archetype
                self.archetype_defaults[str(archetype)] = rec

    # public helpers

    def get_archetypes(self) -> List[str]:
        """The defensive shapes the engine knows."""
        if self.apv_baselines is not None and "A_name" in self.apv_baselines.columns:
            return sorted(self.apv_baselines["A_name"].unique().tolist())

        if self.archetype_defaults:
            return sorted(self.archetype_defaults)

        if self.shap_background is not None and "A_name" in self.shap_background.columns:
            return sorted(self.shap_background["A_name"].unique().tolist())

        if self.pattern_library is not None and "A_name" in self.pattern_library.columns:
            return sorted(self.pattern_library["A_name"].unique().tolist())

        return []

    # SHAP explanations

    def _ensure_shap_explainer(self) -> None:
        """
        Build the SHAP explainer for the explained model on first use, working on the
        model's numeric inputs (after encoding) to avoid problems with text features.
        """
        if self._shap_explainer is not None:
            return

        if shap is None:
            print("[APVEngine] shap package not available - runtime SHAP disabled.")
            return

        model = self.models.get(EXPLAIN_TARGET)
        if model is None:
            print(f"[APVEngine] {EXPLAIN_TARGET} model not loaded - runtime SHAP disabled.")
            return

        if self.shap_background is None or self.shap_background.empty:
            print("[APVEngine] No SHAP background - runtime SHAP disabled.")
            return

        # expects a pipeline with 'preprocess' and 'clf' steps
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

        # same input columns as in training
        cols = list(getattr(model, "feature_names_in_", self.feature_columns))
        missing = [c for c in cols if c not in self.shap_background.columns]
        if missing:
            print(f"[APVEngine] SHAP background lacks {missing} - runtime SHAP disabled.")
            return
        X_bg_raw = self.shap_background[cols].copy()
        self._shap_columns = cols

        # fewer background rows, for speed
        max_bg = 200
        if len(X_bg_raw) > max_bg:
            X_bg_raw = X_bg_raw.sample(
                n=max_bg, random_state=self.random_seed
            ).reset_index(drop=True)

        try:
            # encode the inputs the way the model sees them
            X_bg_proc = preprocess.transform(X_bg_raw)
            if hasattr(X_bg_proc, "toarray"):
                X_bg_proc_dense = X_bg_proc.toarray()
            else:
                X_bg_proc_dense = np.asarray(X_bg_proc)

            # LinearExplainer suits logistic regression
            self._shap_explainer = shap.LinearExplainer(clf, X_bg_proc_dense)
            self._shap_preprocess = preprocess

            # feature names from the preprocessor, if it has them
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
        SHAP values for the given rows, as a table with feature, shap, abs_shap,
        direction ('positive' / 'negative') and feature_label. None if SHAP isn't available.
        """
        if X_row is None or X_row.empty:
            return None

        if self._shap_explainer is None or self._shap_preprocess is None:
            self._ensure_shap_explainer()

        if self._shap_explainer is None or self._shap_preprocess is None:
            return None

        # same columns as the model was trained on
        cols = self._shap_columns
        if any(c not in X_row.columns for c in cols):
            return None
        X_raw = X_row[cols].copy()

        # encode them the way the model sees them
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

        # the raw SHAP values
        values_arr = getattr(shap_values, "values", None)
        if values_arr is None:
            try:
                values_arr = np.array(shap_values)
            except Exception:
                return None

        values_arr = np.asarray(values_arr)

        # shape is (rows, features) or (rows, classes, features); average over the rows,
        # which are the same setup in many real situations
        if values_arr.ndim == 2:
            vals = values_arr.mean(axis=0)
        elif values_arr.ndim == 3:
            # yes/no model: take the "yes" class
            vals = values_arr[:, -1, :].mean(axis=0)
        else:
            vals = values_arr.reshape(-1)

        vals = np.asarray(vals, dtype=float).reshape(-1)

        # SHAP's feature names if available
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

        # readable labels, matched on the base feature name
        labels: List[str] = []
        for f in df["feature"]:
            label = None
            for base_name, human_label in _FEATURE_LABEL_MAP.items():
                if base_name in f:
                    label = human_label
                    break
            if label is None:
                # otherwise tidy up the raw name
                label_clean = f
                for prefix in ("num__", "cat__", "preprocess__", "clf__"):
                    label_clean = label_clean.replace(prefix, "")
                label_clean = label_clean.replace("_", " ")
                label = label_clean
            labels.append(label)
        df["feature_label"] = labels

        # drop zeros to keep explanations short
        df = df[df["abs_shap"] > 0].reset_index(drop=True)

        if df.empty:
            return None

        return df

    # building inputs and predicting

    def _build_feature_row_from_coach_input(
        self,
        coach_input: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        One input row from the coach's setup: the typical row for the chosen shape, the
        coach's choices on top, the typical depth and height for the delivery and target,
        and the matchup features built the same way as in training.
        """
        archetype = str(coach_input.get("A_name", SHAPE_ORDER[0]))
        base: Dict[str, Any] = dict(self.archetype_defaults.get(archetype, {}))
        base["A_name"] = archetype

        for key, value in coach_input.items():
            if key in self.feature_columns:
                base[key] = value

        delivery_type = str(base.get("delivery_type", "")).lower()
        if "short_corner_flag" not in coach_input:
            base["short_corner_flag"] = 1 if delivery_type == "short" else 0

        dd = self.delivery_defaults
        if dd is not None and not dd.empty:
            match = dd[(dd["delivery_type"] == delivery_type) & (dd["end_zone"] == base.get("end_zone"))]
            if not match.empty:
                base["delivery_depth"] = float(match["delivery_depth"].iloc[0])
                base["delivery_high"] = int(match["delivery_high"].iloc[0])

        row = pd.DataFrame([{col: base.get(col) for col in self.feature_columns}])
        return add_derived_features(row)

    def build_rows(self, coach_input: Dict[str, Any]) -> pd.DataFrame:
        """
        The coach's setup placed into real situations against the chosen defence: each
        real corner's opponents and match state, with the coach's players, delivery and
        target. Estimates average over these rows, so "against a deep-line defence" means
        real deep-line situations, not one made-up average opponent.
        """
        shape = str(coach_input.get("A_name", ""))
        situations = self._situations(shape)
        if situations.empty:
            return self._build_feature_row_from_coach_input(coach_input)

        rows = situations.copy()
        for key, value in coach_input.items():
            if key in rows.columns:
                rows[key] = value
        delivery_type = str(coach_input.get("delivery_type", "")).lower()
        rows["short_corner_flag"] = 1 if delivery_type == "short" else 0

        dd = self.delivery_defaults
        if dd is not None and not dd.empty:
            match = dd[(dd["delivery_type"] == delivery_type) & (dd["end_zone"] == coach_input.get("end_zone"))]
            if not match.empty:
                rows["delivery_depth"] = float(match["delivery_depth"].iloc[0])
                rows["delivery_high"] = int(match["delivery_high"].iloc[0])
        return add_derived_features(rows).reset_index(drop=True)

    def _situations(self, shape: str) -> pd.DataFrame:
        # real corners against this defence (cached, capped for speed)
        cache = self.__dict__.setdefault("_situation_cache", {})
        if shape not in cache:
            bg = self.shap_background
            if bg is None or bg.empty or "A_name" not in bg.columns:
                cache[shape] = pd.DataFrame()
            else:
                rows = bg[bg["A_name"] == shape]
                if len(rows) > MAX_SITUATIONS:
                    rows = rows.sample(n=MAX_SITUATIONS, random_state=self.random_seed)
                cache[shape] = rows.reset_index(drop=True)
        return cache[shape]

    def _predict_probs_and_apv(
        self,
        X_row: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run every outcome model on the rows and work out APV in goals, averaged over the
        rows. Returns (outcome predictions, APV parts).
        """
        preds = predict_outcomes(X_row, self.models)
        df_apv = add_apv_columns(preds, alpha=self.alpha, beta=self.beta)

        outcomes = {
            col: float(preds[col].mean())
            for col in preds.columns
            if col.startswith("P_") or col.startswith("E_")
        }
        apv = {k: float(df_apv[k].mean()) for k in ("PV_for", "PV_against", "APV_net")}
        return outcomes, apv

    def predict_target(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """The final model's prediction for one target on one or more rows."""
        model = self.models[target]
        cols = list(getattr(model, "feature_names_in_", X.columns))
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X[cols])[:, 1]
        return model.predict(X[cols])

    def bootstrap_predict(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """Predictions from every bootstrap model, shape (n_models, n_rows)."""
        lin = self._linear_ensemble(target)
        if lin is None:
            return np.empty((0, len(X)))
        design, weights, bias, link = lin
        z = design(X) @ weights.T + bias  # (rows, models)
        pred = 1.0 / (1.0 + np.exp(-z)) if link == "logit" else np.exp(z)
        return pred.T

    def _linear_ensemble(self, target: str):
        """
        Each bootstrap model is scaling + one-hot encoding + a linear model, so it folds
        into one weight per raw input. Scoring them all is then one matrix product.
        """
        cache = self.__dict__.setdefault("_linear_cache", {})
        if target in cache:
            return cache[target]
        members = [m[target] for m in self.bootstrap if target in m]
        if not members:
            cache[target] = None
            return None

        folded = [self._fold_linear(m) for m in members]
        keys = sorted({k for weights, _ in folded for k in weights}, key=str)
        index = {k: i for i, k in enumerate(keys)}
        W = np.zeros((len(folded), len(keys)))
        for row, (weights, _) in enumerate(folded):
            for k, w in weights.items():
                W[row, index[k]] = w
        b = np.array([bias for _, bias in folded])
        link = "logit" if hasattr(members[0], "predict_proba") else "log"

        bg = self.shap_background if self.shap_background is not None else pd.DataFrame()
        medians = {k[1]: float(pd.to_numeric(bg[k[1]], errors="coerce").median()) for k in keys if k[0] == "num" and k[1] in bg}

        def design(X: pd.DataFrame) -> np.ndarray:
            cols = []
            for k in keys:
                if k[0] == "num":
                    cols.append(pd.to_numeric(X[k[1]], errors="coerce").fillna(medians.get(k[1], 0.0)).to_numpy(float))
                else:
                    cols.append((X[k[1]] == k[2]).to_numpy(float))
            return np.column_stack(cols)

        cache[target] = (design, W, b, link)
        return cache[target]

    @staticmethod
    def _fold_linear(pipe) -> Tuple[Dict[tuple, float], float]:
        # weights on the raw inputs: numeric w / scale (the bias absorbs the mean),
        # one weight per known category (unknown ones score 0, as in training)
        pre = pipe.named_steps["preprocess"]
        est = pipe.named_steps["clf"]
        coef = np.ravel(est.coef_)
        bias = float(np.ravel(est.intercept_)[0])
        weights: Dict[tuple, float] = {}
        i = 0
        for name, trans, cols in pre.transformers_:
            if name == "num":
                scaler = trans.named_steps["scaler"]
                for j, col in enumerate(cols):
                    w = coef[i + j] / scaler.scale_[j]
                    weights[("num", col)] = w
                    bias -= w * scaler.mean_[j]
                i += len(cols)
            elif name == "cat":
                ohe = trans.named_steps["ohe"]
                for j, col in enumerate(cols):
                    for level in ohe.categories_[j]:
                        weights[("cat", col, level)] = coef[i]
                        i += 1
        return weights, bias

    def chain_predict(self, X: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        The chance chain on each row:
            chance = P(win) * P(chance | won) + (1 - P(win)) * P(chance | lost)
        Returns each step and the chain per row, plus the same from every bootstrap model.
        """
        steps = ["first_contact_attack", *CHAIN_TARGETS]
        if any(t not in self.models for t in steps):
            return None
        win, won, lost = (self.predict_target(t, X) for t in steps)
        out: Dict[str, Any] = {"win": win, "won": won, "lost": lost, "chance": win * won + (1 - win) * lost}
        boots = [self.bootstrap_predict(t, X) for t in steps]
        if all(b.size for b in boots) and len({b.shape for b in boots}) == 1:
            b_win, b_won, b_lost = boots
            out["boot"] = {"win": b_win, "won": b_won, "lost": b_lost, "chance": b_win * b_won + (1 - b_win) * b_lost}
        return out

    def predict_ranges(self, X_row: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """90% ranges for PV_for, PV_against, APV_net and each step of the chain."""
        ranges: Dict[str, Tuple[float, float]] = {}
        xg_for = self.bootstrap_predict("xg_for", X_row)
        xg_against = self.bootstrap_predict("xg_against", X_row)
        if xg_for.size and xg_against.size:
            f = self.alpha * xg_for.mean(axis=1)
            a = self.beta * xg_against.mean(axis=1)
            for key, vals in (("PV_for", f), ("PV_against", a), ("APV_net", f - a)):
                ranges[key] = (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))
        chain = self.chain_predict(X_row)
        if chain is not None and "boot" in chain:
            names = {"win": "first_contact_attack", "won": "chance_if_won", "lost": "chance_if_lost", "chance": "chance_chain"}
            for step, key in names.items():
                per_model = chain["boot"][step].mean(axis=1)
                ranges[key] = (float(np.percentile(per_model, 5)), float(np.percentile(per_model, 95)))
        else:
            boot = self.bootstrap_predict("first_contact_attack", X_row)
            if boot.size:
                per_model = boot.mean(axis=1)
                ranges["first_contact_attack"] = (float(np.percentile(per_model, 5)), float(np.percentile(per_model, 95)))
        return ranges

    # older helpers for the setup library

    def get_best_setups_for_shape(
        self,
        archetype: str,
        min_samples: Optional[int] = None,
        top_n: Optional[int] = 10,
    ) -> pd.DataFrame:
        """
        Top setups against one defensive shape from the setup library, keeping only
        setups used at least min_samples times (empty if none are).
        """
        self._ensure_pattern_library_loaded()

        if self.pattern_library is None or self.pattern_library.empty:
            print("[APVEngine] No pattern library available.")
            return pd.DataFrame()

        if not archetype:
            return pd.DataFrame()

        df = self.pattern_library.copy()

        # this defensive shape only
        if "A_name" in df.columns:
            df = df[df["A_name"] == archetype]

        if df.empty:
            return pd.DataFrame()

        # setups used at least min_samples times
        ms = int(min_samples) if min_samples is not None else int(self.default_min_samples)
        ms = max(ms, 1)

        if "n_corners" in df.columns:
            df = df[df["n_corners"] >= ms]

        # nothing left: return empty and let the app say so
        if df.empty:
            return pd.DataFrame()

        # add value / risk tags if missing
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

        # best APV first, keep top_n
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
        Rate the same setup against every defensive shape. Returns per_archetype (one
        evaluate_for_coach result each), a summary table and best_index (highest APV_net).
        """
        archetypes = self.get_archetypes()
        results: List[Dict[str, Any]] = []

        if not archetypes:
            return {
                "per_archetype": [],
                "summary": pd.DataFrame(),
                "best_index": None,
            }

        # copy the input without A_name
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

        # simple comparison table
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
        Main entry point for the app: rate one corner setup. Returns the input row,
        outcome predictions, APV in goals with 90% ranges, and a plain-language
        explanation. Everything comes from the models, so it matches the quick wins.
        """
        X_row = self.build_rows(coach_input)
        # the coach's values are the same on every row, but the spare-attacker counts vary
        # with the real defenders, so report their average
        pattern_features = X_row.iloc[0].to_dict()
        for col in [c for c in X_row.columns if c.startswith("free_att_")]:
            pattern_features[col] = float(X_row[col].mean())

        archetype = str(pattern_features.get("A_name", coach_input.get("A_name", "")))
        baseline_apv = self.baseline_by_archetype.get(archetype)

        outcomes, apv = self._predict_probs_and_apv(X_row)
        chain = self.chain_predict(X_row)
        if chain is not None:
            outcomes["P_chance_chain"] = float(chain["chance"].mean())
        shap_df = self._compute_local_shap(X_row)

        explanation = build_coach_explanation(
            pattern_features=pattern_features,
            apv=apv,
            probs=outcomes,
            baseline_apv=baseline_apv,
            shap_df=shap_df,
            max_bullets=max_explanation_bullets,
        )

        return {
            "features": pattern_features,
            "probabilities": outcomes,
            "apv": apv,
            "ranges": self.predict_ranges(X_row),
            "explanation": explanation,
        }
