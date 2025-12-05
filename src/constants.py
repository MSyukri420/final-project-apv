from typing import Dict, List



LABEL_COLUMNS: List[str] = [
    "goal_for",
    "goal_against",
    "chance_for",
    "chance_against",
    "shot_for_any",
    "shot_against_any",
]

MODELLING_TARGETS: List[str] = [
    "chance_for",
    "chance_against",
    "shot_for_any",
    "shot_against_any",
]

PROBA_COLUMNS: List[str] = [f"P_{name}" for name in MODELLING_TARGETS]


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


OUTCOME_FEATURES: Dict[str, Dict[str, List[str]]] = {
    "outcome_chance_for": {
        "numeric": ATTACKING_NUMERIC_FEATURES + [
            "def_line_height",
            "def_in_6yd",
            "gk_start_line",
        ],
        "categorical": [
            "delivery_type",
            "short_corner_flag",
            "end_zone",
            "score_state",
            "minute_bucket",
            "A_name",
        ],
    },
    "outcome_chance_against": {
        "numeric": ATTACKING_NUMERIC_FEATURES + [
            "def_line_height",
            "def_in_6yd",
            "gk_start_line",
        ],
        "categorical": [
            "delivery_type",
            "short_corner_flag",
            "end_zone",
            "score_state",
            "minute_bucket",
            "A_name",
        ],
    },
    "outcome_shot_for_any": {
        "numeric": ATTACKING_NUMERIC_FEATURES + [
            "def_line_height",
            "def_in_6yd",
            "gk_start_line",
        ],
        "categorical": [
            "delivery_type",
            "short_corner_flag",
            "end_zone",
            "score_state",
            "minute_bucket",
            "A_name",
        ],
    },
    "outcome_shot_against_any": {
        "numeric": ATTACKING_NUMERIC_FEATURES + [
            "def_line_height",
            "def_in_6yd",
            "gk_start_line",
        ],
        "categorical": [
            "delivery_type",
            "short_corner_flag",
            "end_zone",
            "score_state",
            "minute_bucket",
            "A_name",
        ],
    },
}
