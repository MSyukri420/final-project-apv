from typing import Dict, List


# outcome labels from stage 5 (target_eng.py): "for" counts our possession after
# the corner, "against" the opponent's next possession, both within the time window
LABEL_COLUMNS: List[str] = [
    "goal_for",
    "goal_against",
    "chance_for",
    "chance_against",
    "shot_for_any",
    "shot_against_any",
    "xg_for",
    "xg_against",
    # 1 = an attacker plays the ball first, 0 = the defence or keeper does,
    # missing = no contest (ball out, foul, short corner that never reached the box)
    "first_contact_attack",
]

# the two-step "chance chain":
#   P(good chance) = P(win) * P(chance | won) + (1 - P(win)) * P(chance | lost)
# chance_if_won / chance_if_lost are chance_for for contested corners we won / lost
# (missing otherwise), built by add_chain_targets in target_eng.py
CHAIN_TARGETS: List[str] = ["chance_if_won", "chance_if_lost"]

# yes/no outcomes: calibrated logistic regression
CLASSIFICATION_TARGETS: List[str] = [
    "first_contact_attack",
    *CHAIN_TARGETS,
    "chance_for",
    "chance_against",
    "shot_for_any",
    "shot_against_any",
]

# expected-goals outcomes: Poisson regression (these feed APV)
REGRESSION_TARGETS: List[str] = [
    "xg_for",
    "xg_against",
]

MODELLING_TARGETS: List[str] = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

PROBA_COLUMNS: List[str] = [f"P_{name}" for name in CLASSIFICATION_TARGETS]
XG_PRED_COLUMNS: List[str] = [f"E_{name}" for name in REGRESSION_TARGETS]


# rule-based defensive shapes, in display order (see src/clustering.py)
SHAPE_ORDER: List[str] = ["Posts covered", "Deep line", "High line"]


DEFENSIVE_FEATURES: List[str] = [
    "def_line_height",
    "def_spread_y",
    "def_in_6yd",
    "players_on_posts",
    "marking_ratio_pen",
    "def_edge_density",
    "max_pairs_per_def",
    "gk_start_line",
]


ATTACKING_NUMERIC_FEATURES: List[str] = [
    "n_att_near",
    "n_att_central",
    "n_att_far",
    "n_blockers",
    "n_edge_of_box",
    "n_rest_defence",
]

# defenders counted in the same zones as attackers (feat_eng.py)
DEFENDER_ZONE_FEATURES: List[str] = [
    "def_near",
    "def_central",
    "def_far",
    "def_spot",
]

# spare attackers per zone: max(0, attackers - defenders)
MATCHUP_FEATURES: List[str] = [
    "free_att_near",
    "free_att_central",
    "free_att_far",
    "free_att_spot",
]

# zone pairs used to build MATCHUP_FEATURES
MATCHUP_ZONES: Dict[str, tuple] = {
    "free_att_near": ("n_att_near", "def_near"),
    "free_att_central": ("n_att_central", "def_central"),
    "free_att_far": ("n_att_far", "def_far"),
    "free_att_spot": ("n_edge_of_box", "def_spot"),
}

# delivery details beyond inswing / outswing and target zone
DELIVERY_FEATURES: List[str] = [
    "delivery_depth",  # yards from the goal line where the ball is met
    "delivery_high",  # 1 if StatsBomb marks it a high pass
]

ATTACKING_CATEGORICAL_FEATURES: List[str] = [
    "delivery_type",
    "short_corner_flag",
    "end_zone",
    "score_state",
    "minute_bucket",
    "competition_id",
    "team_id",
    "corner_side",
]


# chance models use the whole setup: attackers, defenders, matchups, delivery
_FOR_FEATURES: Dict[str, List[str]] = {
    "numeric": ATTACKING_NUMERIC_FEATURES
    + ["def_line_height", "def_in_6yd", "gk_start_line"]
    + DEFENDER_ZONE_FEATURES
    + MATCHUP_FEATURES
    + DELIVERY_FEATURES,
    "categorical": [
        "delivery_type",
        "short_corner_flag",
        "end_zone",
        "score_state",
        "minute_bucket",
        "A_name",
    ],
}

# counter-attacks are rare, so the "against" models only get how many players stay
# back and whether the corner is short; nothing richer beat the average in testing
_AGAINST_FEATURES: Dict[str, List[str]] = {
    "numeric": ["n_rest_defence"],
    "categorical": ["short_corner_flag"],
}

OUTCOME_FEATURES: Dict[str, Dict[str, List[str]]] = {
    "outcome_first_contact_attack": _FOR_FEATURES,
    "outcome_chance_for": _FOR_FEATURES,
    "outcome_chance_if_won": _FOR_FEATURES,
    "outcome_chance_if_lost": _FOR_FEATURES,
    "outcome_shot_for_any": _FOR_FEATURES,
    "outcome_xg_for": _FOR_FEATURES,
    "outcome_chance_against": _AGAINST_FEATURES,
    "outcome_shot_against_any": _AGAINST_FEATURES,
    "outcome_xg_against": _AGAINST_FEATURES,
}


def all_model_features() -> List[str]:
    # every model's inputs combined, in a stable order
    cols: List[str] = []
    for spec in OUTCOME_FEATURES.values():
        for col in list(spec.get("numeric", [])) + list(spec.get("categorical", [])):
            if col not in cols:
                cols.append(col)
    return cols
