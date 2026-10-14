"""
What the app shows, built on APVEngine. In order of trust:
1. real corners: what happened in similar past corners, with 90% ranges
2. the model: an estimate for the exact setup, step by step, each step labelled
   by how well it did on competitions it never saw
3. quick wins: single changes the models agree help and real corners don't contradict
Plus the playbook and the insights page.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.apv_engine import _FEATURE_LABEL_MAP, APVEngine
from src.constants import SHAPE_ORDER

ALL_SHAPES = "All"
PLAYERS_TOTAL = 10
FIRST_CONTACT = "first_contact_attack"
RARE_SHARE = 0.05  # a player count in under 5% of real corners is rare
Z90 = 1.645  # two-sided 90% interval
MIN_EVIDENCE = 15  # widen the similar-corner search until this many contests
PLAYBOOK_PRIOR = 10  # how strongly small playbook groups are pulled to the average
MIN_GAIN = 0.01  # a quick win must add at least 1 point
# the chance chain: step -> (model target, evaluation target, evaluated model)
CHAIN_STEPS: Dict[str, Tuple[str, str, str]] = {
    "win": (FIRST_CONTACT, FIRST_CONTACT, "full_model"),
    "if_won": ("chance_if_won", "chance_if_won", "full_model"),
    "if_lost": ("chance_if_lost", "chance_if_lost", "full_model"),
    "chance": ("chance_chain", "chance_chain", "chain_model"),
}


class Slot(NamedTuple):
    id: str  # feature column
    label: str
    hint: str
    place: str  # "2 players at the near post"
    from_place: str  # "move one player from the near post ..."
    to_place: str  # "... to the near post"
    minimum: int
    maximum: int
    default: int


# zones follow feat_eng.py: n_edge_of_box is the 6-12 yard central area and
# n_rest_defence is everyone outside the box. Ranges cover what real teams did;
# defaults are the most common real setup.
SLOTS: List[Slot] = [
    Slot("n_att_near", "Near post", "In the box, on the side the corner comes from",
         "at the near post", "the near post", "the near post", 0, 4, 0),
    Slot("n_att_central", "Middle", "In the box, in line with the goal",
         "in the middle", "the middle", "the middle", 0, 8, 2),
    Slot("n_att_far", "Far post", "In the box, on the far side",
         "at the far post", "the far post", "the far post", 0, 5, 0),
    Slot("n_blockers", "On the keeper", "Within 3 yards of the goalkeeper",
         "on the keeper", "around the keeper", "block the keeper", 0, 5, 0),
    Slot("n_edge_of_box", "Penalty spot", "Central, 6 to 12 yards out",
         "around the penalty spot", "the penalty spot", "the penalty spot", 0, 7, 3),
    Slot("n_rest_defence", "Outside the box", "Waiting for second balls or guarding the counter",
         "outside the box", "outside the box", "outside the box", 1, 7, 5),
]
SLOT_BY_ID: Dict[str, Slot] = {slot.id: slot for slot in SLOTS}
SLOT_IDS: List[str] = [slot.id for slot in SLOTS]

# defensive shapes come from two rules on the freeze frame (clustering.py)
SHAPE_INFO: Dict[str, str] = {
    "Posts covered": "Someone on a post, usually with a packed 6-yard box.",
    "Deep line": "Nobody on the posts. Box defenders stay close to goal.",
    "High line": "Nobody on the posts. Box defenders step out towards the spot.",
}
# how each shape reads after "against"
SHAPE_PHRASE: Dict[str, str] = {
    "Posts covered": "a defence with the posts covered",
    "Deep line": "a deep-line defence",
    "High line": "a high-line defence",
}


def shape_phrase(shape: str) -> str:
    return SHAPE_PHRASE.get(shape, f"a {shape} defence")

DELIVERIES: Dict[str, str] = {"inswing": "Inswinger", "outswing": "Outswinger", "straight": "Straight"}
DELIVERY_PHRASES: Dict[str, str] = {
    "inswing": "an inswinger",
    "outswing": "an outswinger",
    "straight": "a straight delivery",
    "other": "another type of delivery",
}
ZONES: Dict[str, str] = {"near": "Near post", "central": "Middle", "far": "Far post"}
ZONE_PHRASES: Dict[str, str] = {"near": "the near post", "central": "the middle", "far": "the far post"}

LABEL_TO_COLUMN: Dict[str, str] = {label: column for column, label in _FEATURE_LABEL_MAP.items()}

# where the spare-attacker counts apply
FREE_ATTACKER_ZONES: Dict[str, str] = {
    "free_att_near": "at the near post",
    "free_att_central": "in the middle",
    "free_att_far": "at the far post",
    "free_att_spot": "around the penalty spot",
}

# playbook groups: delivery, target and how many attack the box (10 minus those outside)
BOX_BANDS: List[Tuple[int, int, str]] = [
    (0, 5, "5 or fewer in the box"),
    (6, 7, "6 or 7 in the box"),
    (8, 10, "8 or more in the box"),
]
PLAYBOOK_SORTS: Dict[str, str] = {"chance": "Good chance", "first_contact": "First ball won"}

# insights: the measures each factor is shown on, and how a takeaway reads
CHAIN_MEASURES: List[Tuple[str, str]] = [
    ("first_contact", "First ball won"),
    ("chance_if_won", "Good chance after winning it"),
    ("chance", "Good chance overall"),
]
TAKEAWAY_VERBS: Dict[str, Tuple[str, str]] = {
    # (singular subject, plural subject)
    "first_contact": ("wins the first ball most often", "win the first ball most often"),
    "chance_if_won": ("turns a won first ball into a good chance most often",
                      "turn a won first ball into a good chance most often"),
    "chance": ("creates the most good chances", "create the most good chances"),
}
MIN_TAKEAWAY_N = 30  # smaller groups are left out of takeaways

CONTACT_TYPES: Dict[str, str] = {
    "header_shot": "Attacker heads at goal",
    "shot": "Attacker shoots",
    "attacker_controls": "Attacker plays it on",
    "defender_clears": "Defender clears",
    "keeper_claims": "Keeper catches or punches",
    "defence_wins": "Defence wins it another way",
}


# helpers

def _num(value: Any, digits: int = 3) -> Optional[float]:
    # rounded float that is safe for JSON
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if math.isfinite(x) else None


def _rate(k: float, n: float) -> Dict[str, Optional[float]]:
    # rate with a 90% Wilson range
    k, n = int(k), int(n)
    if n <= 0:
        return {"k": k, "n": n, "rate": None, "low": None, "high": None}
    p = k / n
    denom = 1 + Z90**2 / n
    centre = (p + Z90**2 / (2 * n)) / denom
    half = Z90 * math.sqrt(p * (1 - p) / n + Z90**2 / (4 * n * n)) / denom
    return {"k": k, "n": n, "rate": _num(p), "low": _num(max(0.0, centre - half)), "high": _num(min(1.0, centre + half))}


def _outcome_rates(df: pd.DataFrame) -> Dict[str, Any]:
    # first contact (contested corners only), good chances and shots conceded
    contested = df[FIRST_CONTACT].dropna() if FIRST_CONTACT in df.columns else pd.Series(dtype=float)
    fc = df[FIRST_CONTACT] if FIRST_CONTACT in df.columns else pd.Series(np.nan, index=df.index)
    after_won = df.loc[fc == 1, "chance_for"]
    after_lost = df.loc[fc == 0, "chance_for"]
    return {
        "corners": int(len(df)),
        "first_contact": _rate(contested.sum(), len(contested)),
        "chance": _rate(df["chance_for"].sum(), len(df)),
        # step 2: good chances after winning / losing the first ball
        "chance_if_won": _rate(after_won.sum(), len(after_won)),
        "chance_if_lost": _rate(after_lost.sum(), len(after_lost)),
        "shot_conceded": _rate(df["shot_against_any"].sum(), len(df)),
        "goals": int(df.get("goal_for", pd.Series(dtype=float)).sum()),
    }


def _reliability(n: int) -> Dict[str, str]:
    if n >= 40:
        return {"label": "Solid evidence", "tone": "good"}
    if n >= MIN_EVIDENCE:
        return {"label": "Some evidence", "tone": "fair"}
    if n > 0:
        return {"label": "Rough guide", "tone": "warn"}
    return {"label": "No similar corners", "tone": "neutral"}


def _players_phrase(n: int, slot: Slot) -> str:
    if n == 0:
        return f"No one {slot.place}"
    return f"{n} {'player' if n == 1 else 'players'} {slot.place}"


def _pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{round(100 * x)}%"


class CoachService:
    def __init__(self, engine: APVEngine) -> None:
        self.engine = engine
        self.cfg = engine.cfg
        known = engine.get_archetypes() or list(SHAPE_ORDER)
        self.shapes: List[str] = [s for s in SHAPE_ORDER if s in known] + [s for s in known if s not in SHAPE_ORDER]
        self.min_used_default = int(getattr(engine, "default_min_samples", 10))

        corners = engine.corner_outcomes.copy()
        for col in SLOT_IDS + ["short_corner_flag", "free_att_spot", "free_att_far"]:
            if col in corners.columns:
                corners[col] = pd.to_numeric(corners[col], errors="coerce")
        self.corners = corners
        self.n_corners = int(len(corners))
        # the planner opens on the most common defence
        counts = corners["A_name"].value_counts() if "A_name" in corners.columns else pd.Series(dtype=int)
        self.default_shape: str = str(counts.idxmax()) if not counts.empty else self.shapes[0]

        # average rates for each shape, and overall
        self.base: Dict[str, Dict[str, Any]] = {ALL_SHAPES: _outcome_rates(corners)} if not corners.empty else {}
        for shape in self.shapes:
            if not corners.empty:
                self.base[shape] = _outcome_rates(corners[corners["A_name"] == shape])

        app_cfg = self.cfg.get("app", {}) or {}
        self.min_auc = float(app_cfg.get("estimate_min_auc", 0.60))
        self.min_skill = float(app_cfg.get("estimate_min_brier_skill", 0.02))
        self.agreement = float(app_cfg.get("suggestion_agreement", 0.9))
        self.model_check = self._model_check()

        self.count_shares = self._count_shares()

    # how trustworthy each model is, from the evaluation report

    def _model_check(self) -> Dict[str, Any]:
        # first contact, plus one check per step of the chain
        check = self._check_target(FIRST_CONTACT, "full_model")
        check["steps"] = {step: self._check_target(t, m) for step, (_, t, m) in CHAIN_STEPS.items()}
        return check

    def _check_target(self, target: str, model: str) -> Dict[str, Any]:
        ev = self.engine.evaluation
        check: Dict[str, Any] = {
            "level": "none",
            "auc": None,
            "brier_skill": None,
            "shape_only_brier_skill": None,
            "min_auc": self.min_auc,
            "min_brier_skill": self.min_skill,
            "agreement": self.agreement,
        }
        if ev.empty or FIRST_CONTACT not in self.engine.models:
            return check
        rows = ev[(ev["target"] == target) & (ev["scheme"] == "leave_one_competition_out")]
        full = rows[rows["model"] == model]
        shape = rows[rows["model"] == "shape_only"]
        if full.empty:
            return check
        auc = float(full["roc_auc"].iloc[0])
        skill = float(full["brier_skill"].iloc[0])
        shape_skill = float(shape["brier_skill"].iloc[0]) if not shape.empty else 0.0
        check.update({"auc": _num(auc), "brier_skill": _num(skill, 4), "shape_only_brier_skill": _num(shape_skill, 4)})
        if auc >= self.min_auc and skill >= self.min_skill and skill > shape_skill:
            check["level"] = "tested"
        elif skill > 0 and skill > shape_skill:
            check["level"] = "rough"
        return check

    # meta

    def meta(self) -> Dict[str, Any]:
        targets = (self.cfg.get("targets", {}) or {})
        overall = self.base.get(ALL_SHAPES, {})
        return {
            "rules": {
                "horizon_secs": int(targets.get("horizon_secs", 60)),
                "xg_threshold": float(targets.get("xg_threshold", 0.04)),
            },
            "shapes": [
                {"id": s, "name": s, "description": SHAPE_INFO.get(s, ""), "phrase": shape_phrase(s)}
                for s in self.shapes
            ],
            "all_shapes": ALL_SHAPES,
            "deliveries": [{"id": k, "name": v} for k, v in DELIVERIES.items()],
            "zones": [{"id": k, "name": v} for k, v in ZONES.items()],
            "slots": [
                {"id": s.id, "label": s.label, "hint": s.hint, "min": s.minimum, "max": s.maximum, "default": s.default}
                for s in SLOTS
            ],
            "players_total": PLAYERS_TOTAL,
            "defaults": {
                "shape": self.default_shape,
                "delivery": "inswing",
                "zone": "central",
                "players": {s.id: s.default for s in SLOTS},
            },
            "n_corners": self.n_corners,
            "overall": overall,
            "model": self.model_check,
            "playbook": {
                "min_used": self.min_used_default,
                "max_used": {shape: self._max_used(shape) for shape in [*self.shapes, ALL_SHAPES]},
                "sorts": [{"id": k, "name": v} for k, v in PLAYBOOK_SORTS.items()],
            },
        }

    # rating a setup

    def evaluate(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        engine_setup = self._validate(setup)
        shape = engine_setup["A_name"]

        if shape == ALL_SHAPES:
            per_shape = [self._assess({**engine_setup, "A_name": s}) for s in self.shapes]
            ranked = sorted(per_shape, key=lambda a: a["estimate"]["chain"]["chance"]["value"] or 0.0, reverse=True)
            best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None
            top = best["estimate"]["chain"]["chance"]
            second = runner_up["estimate"]["chain"]["chance"] if runner_up else {}
            clear = (
                runner_up is not None
                and top["low"] is not None
                and second.get("high") is not None
                and top["low"] > second["high"]
            )
            best_setup = {**engine_setup, "A_name": best["shape"]}
            return {
                "mode": "compare",
                "best_shape": best["shape"],
                "clear_winner": clear,
                "headline": (
                    f"Most good chances: against {shape_phrase(best['shape'])}"
                    if clear
                    else "Similar against every defence, within the uncertainty"
                ),
                "shapes": ranked,
                "why": best["why"],
                "suggestions": self.suggest(best_setup),
                "rare": self._rare_phrases(engine_setup),
            }

        assessment = self._assess(engine_setup)
        return {
            "mode": "single",
            **assessment,
            "suggestions": self.suggest(engine_setup),
            "rare": self._rare_phrases(engine_setup),
        }

    def _assess(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        # real corners, model estimate and explanation for one defensive shape
        shape = setup["A_name"]
        result = self.engine.evaluate_for_coach(setup)
        features = result.get("features") or {}
        evidence = self._evidence(setup)
        estimate = self._estimate(result)
        base = self.base.get(shape, {})
        chance = estimate["chain"]["chance"]
        source = self._source(evidence, estimate)
        return {
            "shape": shape,
            "source": source,
            "headline": self._headline(source, evidence, estimate, base),
            "verdict": self._verdict(source, evidence, estimate, base),
            "estimate_vs_average": self._compare(estimate["first_contact"], base.get("first_contact", {}).get("rate"))
            if estimate["level"] != "none"
            else "unknown",
            "chance_vs_average": self._compare(chance, base.get("chance", {}).get("rate"))
            if chance["level"] != "none"
            else "unknown",
            "evidence": evidence,
            "estimate": estimate,
            "base": self.base.get(shape, {}),
            "why": self._why(result.get("explanation", {}) or {}, {**setup, **features}),
        }

    def _validate(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        shape = str(setup.get("shape", ""))
        if shape not in self.shapes and shape != ALL_SHAPES:
            raise ValueError(f"Unknown defence: {shape}")
        delivery = str(setup.get("delivery", ""))
        if delivery not in DELIVERIES:
            raise ValueError(f"Unknown delivery: {delivery}")
        zone = str(setup.get("zone", ""))
        if zone not in ZONES:
            raise ValueError(f"Unknown target zone: {zone}")

        players = setup.get("players", {}) or {}
        counts: Dict[str, int] = {}
        for slot in SLOTS:
            n = int(players.get(slot.id, 0))
            if not slot.minimum <= n <= slot.maximum:
                raise ValueError(f"{slot.label} must be between {slot.minimum} and {slot.maximum}.")
            counts[slot.id] = n
        total = sum(counts.values())
        if total != PLAYERS_TOTAL:
            raise ValueError(f"Place exactly {PLAYERS_TOTAL} outfield players (you have {total}).")

        return {"A_name": shape, "delivery_type": delivery, "end_zone": zone, **counts}

    # real corners

    def _evidence(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        # widen the search step by step until there are enough contested corners
        df = self.corners
        if df.empty:
            return {"description": "", "reliability": _reliability(0), **_outcome_rates(df)}
        df = df[(df["A_name"] == setup["A_name"]) & (df["short_corner_flag"] == 0)]
        moves = (df[SLOT_IDS].sub(pd.Series({c: setup[c] for c in SLOT_IDS})).abs().sum(axis=1)) / 2
        same_delivery = df["delivery_type"] == setup["delivery_type"]
        same_zone = df["end_zone"] == setup["end_zone"]
        delivery = DELIVERY_PHRASES.get(setup["delivery_type"], "this delivery")
        zone = ZONE_PHRASES.get(setup["end_zone"], "this zone")

        tiers: List[Tuple[str, pd.Series]] = [
            (f"{delivery} to {zone}, with at most 1 player placed differently", same_delivery & same_zone & (moves <= 1)),
            (f"{delivery}, with at most 2 players placed differently", same_delivery & (moves <= 2)),
            ("any delivery, with at most 2 players placed differently", moves <= 2),
            ("any delivery, with at most 3 players placed differently", moves <= 3),
        ]
        chosen_text, chosen = tiers[-1]
        for text, mask in tiers:
            if int(df.loc[mask, FIRST_CONTACT].notna().sum()) >= MIN_EVIDENCE:
                chosen_text, chosen = text, mask
                break
        similar = df[chosen]
        rates = _outcome_rates(similar)
        return {
            "description": f"Real corners against {shape_phrase(setup['A_name'])}: {chosen_text}.",
            "reliability": _reliability(rates["first_contact"]["n"] or 0),
            **rates,
        }

    # model estimate

    def _estimate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        probs = result.get("probabilities", {}) or {}
        apv = result.get("apv", {}) or {}
        ranges = result.get("ranges", {}) or {}
        low, high = ranges.get(FIRST_CONTACT) or (None, None)

        def xg(key: str) -> Dict[str, Optional[float]]:
            lo_hi = ranges.get(key) or (None, None)
            return {"value": _num(apv.get(key)), "low": _num(lo_hi[0]), "high": _num(lo_hi[1])}

        def step(name: str) -> Dict[str, Any]:
            target = CHAIN_STEPS[name][0]
            lo_hi = ranges.get(target) or (None, None)
            return {
                "value": _num(probs.get(f"P_{target}")),
                "low": _num(lo_hi[0]),
                "high": _num(lo_hi[1]),
                "level": self.model_check["steps"][name]["level"],
            }

        return {
            "level": self.model_check["level"],
            "first_contact": {
                "value": _num(probs.get(f"P_{FIRST_CONTACT}")),
                "low": _num(low),
                "high": _num(high),
            },
            "chain": {name: step(name) for name in CHAIN_STEPS},
            "p_chance": _num(probs.get("P_chance_for")),
            "xg_for": xg("PV_for"),
            "xg_against": xg("PV_against"),
            "net": xg("APV_net"),
        }

    @staticmethod
    def _compare(span: Dict[str, Any], avg: Optional[float]) -> str:
        # better / worse only when the whole 90% range is above / below the average
        if avg is None or span.get("low") is None or span.get("high") is None:
            return "unknown"
        if span["low"] > avg:
            return "better"
        if span["high"] < avg:
            return "worse"
        return "similar"

    @staticmethod
    def _source(evidence: Dict[str, Any], estimate: Dict[str, Any]) -> str:
        # which source the headline uses: real corners if there are enough contested ones,
        # else the model's chain, else the average
        if (evidence["first_contact"]["n"] or 0) >= MIN_EVIDENCE and evidence["chance"]["rate"] is not None:
            return "real"
        chance = estimate["chain"]["chance"]
        if chance["level"] != "none" and chance["value"] is not None:
            return "model"
        return "average"

    def _verdict(self, source: str, evidence: Dict[str, Any], estimate: Dict[str, Any], base: Dict[str, Any]) -> str:
        avg = base.get("chance", {}).get("rate")
        if source == "real":
            return self._compare(evidence["chance"], avg)
        if source == "model":
            return self._compare(estimate["chain"]["chance"], avg)
        return "unknown"

    @staticmethod
    def _headline(source: str, evidence: Dict[str, Any], estimate: Dict[str, Any], base: Dict[str, Any]) -> str:
        if source == "real":
            ch = evidence["chance"]
            return f"Similar corners created a good chance {ch['k']} of {ch['n']} times ({_pct(ch['rate'])})"
        if source == "model":
            return f"Estimated to create a good chance about {_pct(estimate['chain']['chance']['value'])} of the time"
        avg = base.get("chance", {}).get("rate")
        return f"Corners like this create a good chance about {_pct(avg)} of the time on average"

    # reasons

    def _why(self, explanation: Dict[str, Any], setup: Dict[str, Any]) -> Dict[str, List[str]]:
        def phrases(labels: Any) -> List[str]:
            texts = [self._factor_text(str(label), setup) for label in (labels or [])]
            return [t for t in texts if t]

        return {
            "helps": phrases(explanation.get("help_labels")),
            "hurts": phrases(explanation.get("risk_labels")),
        }

    @staticmethod
    def _factor_text(label: str, setup: Dict[str, Any]) -> Optional[str]:
        column = LABEL_TO_COLUMN.get(label, "")
        if column == "delivery_type":
            return f"Using {DELIVERY_PHRASES.get(setup['delivery_type'], 'this delivery')}"
        if column == "end_zone":
            return f"Aiming at {ZONE_PHRASES.get(setup['end_zone'], 'this zone')}"
        slot = SLOT_BY_ID.get(column)
        if slot is not None:
            return _players_phrase(int(setup[slot.id]), slot)
        if column in FREE_ATTACKER_ZONES:
            spare = int(round(float(setup.get(column) or 0.0)))
            place = FREE_ATTACKER_ZONES[column]
            if spare <= 0:
                return f"No spare attacker {place}"
            return f"{spare} spare {'attacker' if spare == 1 else 'attackers'} {place}"
        # opponent factors (line height, keeper position...) come with the defence
        return None

    # rare setups

    def _count_shares(self) -> Dict[str, Dict[int, float]]:
        if self.corners.empty:
            return {}
        shares: Dict[str, Dict[int, float]] = {}
        for slot in SLOTS:
            if slot.id in self.corners.columns:
                counts = self.corners[slot.id].dropna().astype(int).value_counts(normalize=True)
                shares[slot.id] = dict(zip(counts.index.astype(int).tolist(), counts.tolist()))
        return shares

    def _rare_columns(self, setup: Dict[str, Any]) -> List[str]:
        if not self.count_shares:
            return []
        return [
            slot.id
            for slot in SLOTS
            if self.count_shares.get(slot.id, {}).get(int(setup[slot.id]), 0.0) < RARE_SHARE
        ]

    def _rare_phrases(self, setup: Dict[str, Any]) -> List[str]:
        return [_players_phrase(int(setup[c]), SLOT_BY_ID[c]).lower() for c in self._rare_columns(setup)]

    # quick wins: single changes the bootstrap models agree on

    def _single_changes(self, setup: Dict[str, Any]) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        for src in SLOTS:
            if setup[src.id] <= src.minimum:
                continue
            for dst in SLOTS:
                if dst.id == src.id or setup[dst.id] >= dst.maximum:
                    continue
                changed = dict(setup)
                changed[src.id] -= 1
                changed[dst.id] += 1
                options.append({
                    "text": f"Move one player from {src.from_place} to {dst.to_place}.",
                    "changes": [
                        {"text": f"−1 {src.label}", "kind": "remove"},
                        {"text": f"+1 {dst.label}", "kind": "add"},
                    ],
                    "setup": changed,
                })

        for value, name in DELIVERIES.items():
            if value != setup["delivery_type"]:
                options.append({
                    "text": f"Use {DELIVERY_PHRASES[value]} instead.",
                    "changes": [{"text": name, "kind": "switch"}],
                    "setup": {**setup, "delivery_type": value},
                })
        for value, name in ZONES.items():
            if value != setup["end_zone"]:
                options.append({
                    "text": f"Aim at {ZONE_PHRASES[value]} instead.",
                    "changes": [{"text": f"Aim: {name}", "kind": "switch"}],
                    "setup": {**setup, "end_zone": value},
                })
        return options

    def suggest(self, setup: Dict[str, Any], max_suggestions: int = 3) -> List[Dict[str, Any]]:
        if FIRST_CONTACT not in self.engine.models or self.model_check["level"] == "none":
            return []
        already_rare = set(self._rare_columns(setup))
        # don't push towards setups real teams hardly ever used
        options = [o for o in self._single_changes(setup) if not set(self._rare_columns(o["setup"])) - already_rare]
        if not options:
            return []

        # score every option in the same real situations, in one batch
        frames = []
        for i, s in enumerate([setup] + [o["setup"] for o in options]):
            frame = self.engine.build_rows(s)
            frame["_option"] = i
            frames.append(frame)
        rows = pd.concat(frames, ignore_index=True)
        option = rows["_option"].to_numpy()
        counts = np.bincount(option)

        def per_option(pred: np.ndarray) -> np.ndarray:
            return np.bincount(option, weights=pred) / counts

        chain = self.engine.chain_predict(rows)
        if chain is None or "boot" not in chain:
            return []

        def effect(step: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # gain over the current setup, and the share of models that see it go up / down
            point = per_option(chain[step])
            boot = np.vstack([per_option(b) for b in chain["boot"][step]])
            gain = point[1:] - point[0]
            up = (boot[:, 1:] > boot[:, :1]).mean(axis=0)
            down = (boot[:, 1:] < boot[:, :1]).mean(axis=0)
            return gain, up, down

        chance_gain, chance_up, chance_down = effect("chance")
        win_gain, win_up, win_down = effect("win")
        agree = self.agreement

        picked = []
        for i, option in enumerate(options):
            chance_better = chance_gain[i] >= MIN_GAIN and chance_up[i] >= agree
            win_better = win_gain[i] >= MIN_GAIN and win_up[i] >= agree
            # hold back a change that clearly hurts the other measure
            chance_worse = chance_gain[i] <= -MIN_GAIN and chance_down[i] >= agree
            win_worse = win_gain[i] <= -MIN_GAIN and win_down[i] >= agree
            if (chance_better and not win_worse) or (win_better and not chance_worse):
                picked.append((chance_better, chance_gain[i], win_gain[i], i, option))

        # good chances are the goal, so those gains come first
        picked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        # the good-chance step is only a rough guide, so check each suggestion against
        # real corners and hold it back if they point the other way
        real: Dict[int, Optional[Dict[str, Any]]] = {}
        checked = []
        for item in picked:
            chance_better, i, option = item[0], item[3], item[4]
            check_on = "chance" if chance_better else "first_contact"
            check = self._real_check(setup, option["setup"], check_on)
            if check is not None and check["after"]["rate"] < check["now"]["rate"]:
                continue
            real[i] = check
            checked.append(item)
            if len(checked) == max_suggestions:
                break
        picked = checked

        def measure(gain: float, up: float, down: float) -> Dict[str, Any]:
            points = int(round(100 * gain))
            return {"points": points, "agreement": int(round(100 * (up if points >= 0 else down)))}

        return [
            {
                "text": option["text"],
                "changes": option["changes"],
                "chance": measure(chance_gain[i], chance_up[i], chance_down[i]),
                "first_ball": measure(win_gain[i], win_up[i], win_down[i]),
                "trade_off": bool(chance_gain[i] <= -MIN_GAIN or win_gain[i] <= -MIN_GAIN),
                "real": real.get(i),
                "setup": self._to_client(option["setup"]),
            }
            for _, _, _, i, option in picked
        ]

    def _real_check(self, before: Dict[str, Any], after: Dict[str, Any], measure: str) -> Optional[Dict[str, Any]]:
        # real corners before and after one change: same delivery and target, same defence
        # and similar players, widened to any players and then any defence if too few.
        # None if both groups are too small or are the same corners.
        if self.corners.empty:
            return None
        direct = self.corners[self.corners["short_corner_flag"] == 0]

        def group(setup: Dict[str, Any], max_moves: Optional[int], same_defence: bool) -> pd.DataFrame:
            g = direct[direct["A_name"] == before["A_name"]] if same_defence else direct
            g = g[(g["delivery_type"] == setup["delivery_type"]) & (g["end_zone"] == setup["end_zone"])]
            if max_moves is not None:
                moves = g[SLOT_IDS].sub(pd.Series({c: setup[c] for c in SLOT_IDS})).abs().sum(axis=1) / 2
                g = g[moves <= max_moves]
            return g

        def sized(g: pd.DataFrame) -> int:
            return int(g[FIRST_CONTACT].notna().sum()) if measure == "first_contact" else len(g)

        scopes = [(2, True, "similar placement, same defence"), (None, True, "same defence"), (None, False, "any defence")]
        for max_moves, same_defence, scope in scopes:
            g_before, g_after = group(before, max_moves, same_defence), group(after, max_moves, same_defence)
            if min(sized(g_before), sized(g_after)) >= MIN_EVIDENCE:
                break
        else:
            return None
        if g_before.index.equals(g_after.index):
            return None
        now, later = _outcome_rates(g_before)[measure], _outcome_rates(g_after)[measure]
        if now["rate"] is None or later["rate"] is None:
            return None
        return {"measure": measure, "now": now, "after": later, "scope": scope}

    @staticmethod
    def _to_client(setup: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "shape": setup["A_name"],
            "delivery": setup["delivery_type"],
            "zone": setup["end_zone"],
            "players": {slot.id: int(setup[slot.id]) for slot in SLOTS},
        }

    # playbook: groups of real corners ranked by what happened

    @staticmethod
    def _box_band(n_rest: Any) -> Optional[str]:
        try:
            in_box = PLAYERS_TOTAL - int(n_rest)
        except (TypeError, ValueError):
            return None
        for low, high, label in BOX_BANDS:
            if low <= in_box <= high:
                return label
        return None

    def _patterns(self, shape: str = ALL_SHAPES) -> pd.DataFrame:
        # one row per group (delivery, target, players in the box). Exact player setups are
        # almost never repeated, so groups give enough corners to learn from.
        df = self.corners
        if df.empty:
            return pd.DataFrame()
        if shape != ALL_SHAPES:
            df = df[df["A_name"] == shape]
        if df.empty:
            return pd.DataFrame()
        df = df.assign(
            route=np.where(df["short_corner_flag"] == 1, "short", df["delivery_type"].astype(str)),
            box=df["n_rest_defence"].map(self._box_band),
        ).dropna(subset=["box"])

        rows: List[Dict[str, Any]] = []
        for (route, zone, box), g in df.groupby(["route", "end_zone", "box"]):
            # the group's average setup (drawn on the card, opened in the planner) and the
            # defence it was most often used against
            players = self._average_setup(g)
            typical_shape = str(g["A_name"].mode().iloc[0])
            rows.append({
                "route": route,
                "end_zone": zone,
                "box": box,
                "used": len(g),
                "contested": int(g[FIRST_CONTACT].notna().sum()),
                "first_contact_wins": int(g[FIRST_CONTACT].sum()),
                "chances": int(g["chance_for"].sum()),
                "players": players,
                "typical_shape": typical_shape,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        # pull small groups towards the average so a lucky few can't top the list
        p_chance = float(df["chance_for"].mean())
        p_first = float(df[FIRST_CONTACT].mean())
        out["score_chance"] = (out["chances"] + PLAYBOOK_PRIOR * p_chance) / (out["used"] + PLAYBOOK_PRIOR)
        out["score_first_contact"] = (out["first_contact_wins"] + PLAYBOOK_PRIOR * p_first) / (out["contested"] + PLAYBOOK_PRIOR)
        out["plannable"] = out["route"].isin(list(DELIVERIES)) & out["end_zone"].isin(list(ZONES))
        return out

    @staticmethod
    def _average_setup(g: pd.DataFrame) -> Dict[str, int]:
        # average players per zone, rounded to add up to 10 and kept within limits
        mean = g[SLOT_IDS].mean().fillna(0.0)
        counts = {c: int(np.floor(mean[c])) for c in SLOT_IDS}
        order = sorted(SLOT_IDS, key=lambda c: mean[c] - counts[c], reverse=True)
        for c in order[: max(0, PLAYERS_TOTAL - sum(counts.values()))]:
            counts[c] += 1
        for slot in SLOTS:
            counts[slot.id] = min(max(counts[slot.id], slot.minimum), slot.maximum)
        # keeping within limits can leave the total off: fix on the biggest zone with room
        while sum(counts.values()) != PLAYERS_TOTAL:
            step = 1 if sum(counts.values()) < PLAYERS_TOTAL else -1
            room = [sl for sl in SLOTS if sl.minimum <= counts[sl.id] + step <= sl.maximum]
            target = max(room, key=lambda sl: mean[sl.id] * step)
            counts[target.id] += step
        return counts

    def _max_used(self, shape: str) -> int:
        df = self._patterns(shape)
        if df.empty:
            return 1
        df = df[df["plannable"]]
        return max(int(df["used"].max()), 1) if not df.empty else 1

    def playbook(self, shape: str, min_used: int, limit: int, sort: str = "chance", include_other: bool = False) -> Dict[str, Any]:
        df = self._patterns(shape)
        if df.empty:
            return {"items": [], "total": 0}
        sort = sort if sort in PLAYBOOK_SORTS else "chance"
        if not include_other:
            df = df[df["plannable"]]
        df = df[df["used"] >= int(min_used)].sort_values([f"score_{sort}", "used"], ascending=False)

        items: List[Dict[str, Any]] = []
        for rec in df.head(int(limit)).to_dict("records"):
            route, zone = str(rec["route"]), str(rec["end_zone"])
            first = _rate(rec["first_contact_wins"], rec["contested"])
            chance = _rate(rec["chances"], rec["used"])
            main = chance if sort == "chance" else first
            items.append({
                "shape": shape if shape != ALL_SHAPES else str(rec["typical_shape"]),
                "delivery": route,
                "delivery_name": "Short corner" if route == "short" else DELIVERIES.get(route, "Other delivery"),
                "zone": zone,
                "zone_name": ZONES.get(zone, zone),
                "box": str(rec["box"]),
                "players": rec["players"],
                "used": int(rec["used"]),
                "first_contact": first,
                "chance": chance,
                "reliability": _reliability(int(main["n"] or 0)),
                "plannable": bool(rec["plannable"]),
            })
        return {"items": items, "total": int(len(df)), "sort": sort}

    # insights: what real corners show, no model involved

    def insights(self) -> Dict[str, Any]:
        df = self.corners
        if df.empty:
            return {"overall": {}, "takeaways": [], "factors": [], "little": [], "contact_types": []}

        def groups(column: pd.Series, labels: Dict[Any, Tuple[str, str, bool]], measures: List[str]) -> List[Dict[str, Any]]:
            # labels: value -> (row label, phrase for a sentence, phrase is plural)
            rows = []
            for value, (label, phrase, plural) in labels.items():
                part = df[column == value]
                if part.empty:
                    continue
                rates = _outcome_rates(part)
                rows.append({
                    "label": label,
                    "phrase": phrase,
                    "plural": plural,
                    "corners": rates["corners"],
                    **{m: rates[m] for m in measures},
                })
            return rows

        chain = [m for m, _ in CHAIN_MEASURES]
        route = df["delivery_type"].where(df["short_corner_flag"] == 0, "short")
        spare = df["free_att_spot"].clip(upper=2).round()
        rest = df["n_rest_defence"].clip(lower=2, upper=5)

        factors = [
            {
                "id": "zone",
                "title": "Target zone",
                "subtitle": "Where the delivery was aimed.",
                "rows": groups(df["end_zone"], {
                    "near": ("Near post", "Aiming at the near post", False),
                    "central": ("Middle", "Aiming at the middle", False),
                    "far": ("Far post", "Aiming at the far post", False),
                }, chain),
            },
            {
                "id": "delivery",
                "title": "Delivery type",
                "subtitle": "How the ball was struck. Short routines count the first ball into the box.",
                "rows": groups(route, {
                    "inswing": ("Inswinger", "Inswingers", True),
                    "outswing": ("Outswinger", "Outswingers", True),
                    "straight": ("Straight", "Straight deliveries", True),
                    "short": ("Short routine", "Short routines", True),
                }, chain),
            },
            {
                "id": "spare_spot",
                "title": "Spare attackers around the penalty spot",
                "subtitle": "Attackers left over once each defender in the 6 to 12 yard central zone takes one.",
                "rows": groups(spare, {
                    0.0: ("None", "No spare attacker at the penalty spot", False),
                    1.0: ("1 spare", "1 spare attacker at the penalty spot", False),
                    2.0: ("2 or more", "2 or more spare attackers at the penalty spot", True),
                }, chain),
            },
        ]

        little = [
            {
                "id": "shape",
                "title": "Defensive shape",
                "subtitle": "How the opponent defended.",
                "note": (
                    "Posts covered can look better, but mostly because it is common in women's "
                    "tournaments, where every kind of corner creates more chances. Once the "
                    "competition is accounted for, the shape makes no clear difference."
                ),
                "measures": [m for m in CHAIN_MEASURES if m[0] != "chance_if_won"],
                "rows": groups(df["A_name"], {s: (s, s, False) for s in self.shapes}, ["first_contact", "chance"]),
            },
            {
                "id": "rest",
                "title": "Players staying outside the box",
                "subtitle": "Does keeping more players back reduce shots conceded?",
                "note": "Shots conceded stay around 3 to 6% however many players stay back, so the planner doesn't show them.",
                "measures": [("shot_conceded", "Shot conceded")],
                "rows": groups(rest, {
                    2.0: ("2 or fewer", "2 or fewer", True),
                    3.0: ("3", "3", True),
                    4.0: ("4", "4", True),
                    5.0: ("5 or more", "5 or more", True),
                }, ["shot_conceded"]),
            },
        ]

        types = df.loc[df[FIRST_CONTACT].notna(), "first_contact_type"].value_counts()
        total_types = int(types.sum())

        return {
            "overall": _outcome_rates(df),
            "competitions": int(df["competition_id"].nunique()) if "competition_id" in df.columns else None,
            "measures": [{"id": m, "name": n} for m, n in CHAIN_MEASURES],
            "takeaways": self._takeaways(factors),
            "factors": factors,
            "little": little,
            "contact_types": [
                {"label": CONTACT_TYPES.get(str(k), str(k)), "share": _num(v / total_types), "count": int(v)}
                for k, v in types.items()
                if total_types
            ],
        }

    @staticmethod
    def _takeaways(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # one sentence per factor and measure where the best and worst groups clearly
        # differ (90% ranges don't overlap, enough corners in both)
        out: List[Dict[str, Any]] = []
        for factor in factors:
            found = []
            for measure, measure_name in CHAIN_MEASURES:
                rows = [r for r in factor["rows"] if (r[measure]["n"] or 0) >= MIN_TAKEAWAY_N and r[measure]["rate"] is not None]
                if len(rows) < 2:
                    continue
                best = max(rows, key=lambda r: r[measure]["rate"])
                worst = min(rows, key=lambda r: r[measure]["rate"])
                if best[measure]["low"] <= worst[measure]["high"]:
                    continue
                verb = TAKEAWAY_VERBS[measure][1 if best["plural"] else 0]
                b, w = best[measure], worst[measure]
                found.append({
                    "factor": factor["title"],
                    "measure": measure_name,
                    "winner": best["label"],
                    "text": (
                        f"{best['phrase']} {verb}: {_pct(b['rate'])} ({b['k']} of {b['n']}), "
                        f"against {_pct(w['rate'])} for {worst['phrase'][0].lower() + worst['phrase'][1:]}."
                    ),
                })
            if len({f["winner"] for f in found}) > 1:
                # the easiest ball to win isn't the one that leads to chances
                for f in found:
                    f["trade_off"] = True
            out += found
        return out
