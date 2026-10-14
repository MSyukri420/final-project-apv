from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle

from src.constants import (
    ATTACKING_NUMERIC_FEATURES,
    DEFENDER_ZONE_FEATURES,
    DEFENSIVE_FEATURES,
    MATCHUP_ZONES,
)

# pitch measurements (StatsBomb), attacking towards x = 120

GOAL_X = 120.0
GOAL_Y_LOW, GOAL_Y_HIGH = 36.0, 44.0
GOAL_Y_CENTER = (GOAL_Y_LOW + GOAL_Y_HIGH) / 2.0

SIX_YARD_X_MIN = 114.0

PEN_X_MIN = 102.0
PEN_Y_MIN, PEN_Y_MAX = 18.0, 62.0

EDGE_X_MIN, EDGE_X_MAX = 108.0, 114.0
EDGE_Y_MIN, EDGE_Y_MAX = 32.0, 48.0

POST_L = (GOAL_X, GOAL_Y_LOW)
POST_R = (GOAL_X, GOAL_Y_HIGH)
POST_RADIUS = 1.5

GK_MIN_X_IN_BOX = PEN_X_MIN


# match context: minute bucket, delivery type, short or direct, target zone

def minute_bucket(minute: float) -> str:
    # group the minute into a phase of the game
    try:
        m = float(minute)
    except Exception:
        return "unknown"

    if m < 15:
        return "0-15"
    if m < 30:
        return "15-30"
    if m < 45:
        return "30-45"
    if m < 60:
        return "45-60"
    if m < 75:
        return "60-75"
    return "75-90+"  # includes added time and extra time


def infer_delivery_type(row: pd.Series) -> str:
    # turn the pass details into inswing / outswing / straight / other
    if row.get("pass.inswinging") is True:
        return "inswing"
    if row.get("pass.outswinging") is True:
        return "outswing"
    if row.get("pass.straight") is True:
        return "straight"
    return "other"


def infer_short_corner(row: pd.Series) -> int:
    # short corner: the pass ends before the penalty area
    loc = row.get("pass.end_location")
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return 0

    try:
        x = float(loc[0])
    except Exception:
        return 0

    return 1 if x < PEN_X_MIN else 0


def infer_end_zone(row: pd.Series) -> str:
    # where the delivery lands: near / central / far / unknown
    loc = row.get("pass.end_location")
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return "unknown"

    try:
        x, y = float(loc[0]), float(loc[1])
    except Exception:
        return "unknown"

    # short deliveries that never reach the box
    if x < PEN_X_MIN:
        return "central"

    dy = y - GOAL_Y_CENTER

    # middle band in line with the goal
    if abs(dy) <= 6.0:
        return "central"

    side = row.get("corner_side", None)

    # near / far depend on which side the corner is taken from
    if side == "left":
        return "near" if dy < -6.0 else "far"
    elif side == "right":
        return "near" if dy > 6.0 else "far"

    # if the side is unknown
    return "near" if dy < -6.0 else "far"


def infer_delivery_depth(row: pd.Series) -> float:
    # yards from the goal line to where the delivery ends
    loc = row.get("pass.end_location")
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return math.nan
    try:
        x = float(loc[0])
    except Exception:
        return math.nan
    return float(min(max(GOAL_X - x, 0.0), 40.0))


def infer_delivery_high(row: pd.Series) -> int:
    # 1 for a lofted delivery, 0 for a low one
    return 1 if row.get("pass.height.name") == "High Pass" else 0


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # spare attackers per zone, once each defender there takes one. Used by both
    # training and the app, so both build them the same way.
    out = df.copy()
    for col, (att_col, def_col) in MATCHUP_ZONES.items():
        if att_col in out.columns and def_col in out.columns:
            att = pd.to_numeric(out[att_col], errors="coerce")
            dfn = pd.to_numeric(out[def_col], errors="coerce")
            out[col] = (att - dfn).clip(lower=0.0)
    return out


def _lateral_bands(y: np.ndarray, corner_side: str):
    # near / far: more than 6 yards either side of the goal centre, by corner side
    if corner_side == "right":
        return y > GOAL_Y_CENTER + 6.0, y < GOAL_Y_CENTER - 6.0
    return y < GOAL_Y_CENTER - 6.0, y > GOAL_Y_CENTER + 6.0


def add_score_state(
    df_events_time: Optional[pd.DataFrame],
    df_corners: pd.DataFrame,
) -> pd.DataFrame:
    # add score_state (winning / drawing / losing) for the attacking team
    df = df_corners.copy()

    if df_events_time is None or df_events_time.empty:
        df["score_state"] = "drawing"
        return df

    if "shot.outcome.name" not in df_events_time.columns:
        df["score_state"] = "unknown"
        return df

    df_goals = df_events_time.loc[
        df_events_time["shot.outcome.name"] == "Goal",
        ["match_id", "t_abs", "team.id"],
    ].copy()

    if df_goals.empty:
        df["score_state"] = "drawing"
        return df

    df_goals = df_goals.rename(columns={"team.id": "scoring_team_id"})
    goals_by_match: Dict[int, pd.DataFrame] = {
        mid: g.sort_values("t_abs").reset_index(drop=True)
        for mid, g in df_goals.groupby("match_id")
    }

    states: List[str] = []

    for _, row in df.iterrows():
        mid = row["match_id"]
        t_corner = row.get("t_abs", row.get("_t", np.nan))
        att_id = row.get("attacking_team.id")
        opp_id = row.get("opponent_team.id")

        gmatch = goals_by_match.get(mid)
        if gmatch is None or gmatch.empty or pd.isna(t_corner):
            states.append("drawing")
            continue

        gbefore = gmatch[gmatch["t_abs"] < t_corner]
        if gbefore.empty:
            states.append("drawing")
            continue

        att_goals = (gbefore["scoring_team_id"] == att_id).sum()
        opp_goals = (gbefore["scoring_team_id"] == opp_id).sum()

        if att_goals > opp_goals:
            states.append("winning")
        elif att_goals < opp_goals:
            states.append("losing")
        else:
            states.append("drawing")

    df["score_state"] = states
    return df


# player positions to features

def _compute_def_att_features_for_corner(
    df_ff: pd.DataFrame,
    corner_side: str,
) -> Dict[str, float]:
    # defensive and attacking features for one corner's freeze frame
    df_ff = df_ff.dropna(subset=["x", "y"])
    if df_ff.empty:
        return {
            "def_line_height": math.nan,
            "def_spread_y": 0.0,
            "def_in_6yd": 0,
            "players_on_posts": 0,
            "marking_ratio_pen": math.nan,
            "def_edge_density": 0,
            "max_pairs_per_def": 0,
            "gk_start_line": math.nan,
            "n_att_near": 0,
            "n_att_central": 0,
            "n_att_far": 0,
            "n_blockers": 0,
            "n_edge_of_box": 0,
            "n_rest_defence": 0,
            **{col: 0 for col in DEFENDER_ZONE_FEATURES},
            "n_def_outside_box": 0,
        }

    # split into attackers and defenders
    teammate_mask = df_ff["teammate"].astype(bool)
    atk = df_ff[teammate_mask].copy()
    deff = df_ff[~teammate_mask].copy()

    for col in ("x", "y"):
        atk[col] = pd.to_numeric(atk[col], errors="coerce")
        deff[col] = pd.to_numeric(deff[col], errors="coerce")

    atk = atk.dropna(subset=["x", "y"])
    deff = deff.dropna(subset=["x", "y"])

    x_a = atk["x"].to_numpy()
    y_a = atk["y"].to_numpy()
    x_d = deff["x"].to_numpy()
    y_d = deff["y"].to_numpy()

    # which defender is the keeper
    is_gk_mask = np.zeros_like(x_d, dtype=bool)

    n_attackers = len(x_a)

    # keeper position (for gk_start_line and blockers)
    gk_start_line = math.nan
    gk_x = float(GOAL_X)
    gk_y = float(GOAL_Y_CENTER)

    if len(deff) > 0:
        gk_idx: Optional[int] = None

        # use StatsBomb's keeper flag if there is one
        if "keeper" in deff.columns:
            keeper_mask = deff["keeper"].fillna(False).astype(bool).to_numpy()
            if keeper_mask.any():
                gk_idx = int(np.nonzero(keeper_mask)[0][0])

        # otherwise: the deepest defender near goal
        if gk_idx is None and x_d.size > 0:
            mask_gk_zone = x_d >= GK_MIN_X_IN_BOX
            if not mask_gk_zone.any():
                mask_gk_zone = np.ones_like(x_d, dtype=bool)
            zone_indices = np.nonzero(mask_gk_zone)[0]
            if zone_indices.size > 0:
                deepest_idx_in_zone = int(np.argmax(x_d[mask_gk_zone]))
                gk_idx = int(zone_indices[deepest_idx_in_zone])

        if gk_idx is not None:
            gk_x = float(x_d[gk_idx])
            gk_y = float(y_d[gk_idx])
            is_gk_mask[gk_idx] = True

        gk_start_line = float(GOAL_X - gk_x)

    # attackers: one zone each, then scaled to 10 players
    if n_attackers == 0:
        # no attackers visible: treat them all as staying back
        n_att_near = 0
        n_att_central = 0
        n_att_far = 0
        n_blockers = 0
        n_edge_of_box = 0
        n_rest_defence = 10
        in_box_a = np.zeros(0, dtype=bool)
    else:
        # starting masks
        in_box_a = (x_a >= PEN_X_MIN) & (y_a >= PEN_Y_MIN) & (y_a <= PEN_Y_MAX)

        # blockers: close to the keeper
        dist_to_gk = np.sqrt((x_a - gk_x) ** 2 + (y_a - gk_y) ** 2)
        GK_BLOCK_RADIUS = 3.0
        block_mask = in_box_a & (dist_to_gk <= GK_BLOCK_RADIUS)

        # penalty-spot rectangle, not counting blockers
        edge_mask_a = (
            (x_a >= EDGE_X_MIN)
            & (x_a <= EDGE_X_MAX)
            & (y_a >= EDGE_Y_MIN)
            & (y_a <= EDGE_Y_MAX)
        )
        effective_edge_mask = edge_mask_a & (~block_mask)

        # the rest of the box, split into near / central / far
        remaining_in_box = in_box_a & (~block_mask) & (~effective_edge_mask)

        # sideways bands for those players
        near_band, far_band = _lateral_bands(y_a, corner_side)

        central_band = (~near_band) & (~far_band)

        near_mask = remaining_in_box & near_band
        far_mask = remaining_in_box & far_band
        central_mask = remaining_in_box & central_band

        # raw counts per zone
        n_blockers_raw = int(block_mask.sum())
        n_edge_raw = int(effective_edge_mask.sum())
        n_near_raw = int(near_mask.sum())
        n_central_raw = int(central_mask.sum())
        n_far_raw = int(far_mask.sum())

        # everyone else is outside the box
        n_rest_raw = int(
            n_attackers
            - (n_blockers_raw + n_edge_raw + n_near_raw + n_central_raw + n_far_raw)
        )
        if n_rest_raw < 0:
            n_rest_raw = 0

        raw_counts = {
            "n_att_near": n_near_raw,
            "n_att_central": n_central_raw,
            "n_att_far": n_far_raw,
            "n_blockers": n_blockers_raw,
            "n_edge_of_box": n_edge_raw,
            "n_rest_defence": n_rest_raw,
        }

        # scale to 10 outfield players
        total_roles = sum(raw_counts.values())
        if total_roles <= 0:
            n_att_near = 0
            n_att_central = 0
            n_att_far = 0
            n_blockers = 0
            n_edge_of_box = 0
            n_rest_defence = 10
        else:
            quotas = {k: 10.0 * v / float(total_roles) for k, v in raw_counts.items()}
            base = {k: int(math.floor(q)) for k, q in quotas.items()}
            remainder = 10 - sum(base.values())

            if remainder > 0:
                # add to the zones with the largest remainders
                frac = {k: quotas[k] - base[k] for k in raw_counts.keys()}
                for k in sorted(frac.keys(), key=lambda kk: frac[kk], reverse=True):
                    if remainder <= 0:
                        break
                    base[k] += 1
                    remainder -= 1
            elif remainder < 0:
                # take from the zones with the smallest remainders
                frac = {k: quotas[k] - base[k] for k in raw_counts.keys()}
                for k in sorted(frac.keys(), key=lambda kk: frac[kk]):
                    if remainder >= 0:
                        break
                    if base[k] > 0:
                        base[k] -= 1
                        remainder += 1

            n_att_near = base["n_att_near"]
            n_att_central = base["n_att_central"]
            n_att_far = base["n_att_far"]
            n_blockers = base["n_blockers"]
            n_edge_of_box = base["n_edge_of_box"]
            n_rest_defence = base["n_rest_defence"]

    # defenders
    in_box_d = (x_d >= PEN_X_MIN) & (y_d >= PEN_Y_MIN) & (y_d <= PEN_Y_MAX)

    # defenders in the attackers' zones (not the keeper), to see where we outnumber them
    spot_d = (
        (x_d >= EDGE_X_MIN)
        & (x_d <= EDGE_X_MAX)
        & (y_d >= EDGE_Y_MIN)
        & (y_d <= EDGE_Y_MAX)
        & (~is_gk_mask)
    )
    zoned_d = in_box_d & (~is_gk_mask) & (~spot_d)
    near_d, far_d = _lateral_bands(y_d, corner_side)
    def_near = int((zoned_d & near_d).sum())
    def_far = int((zoned_d & far_d).sum())
    def_central = int((zoned_d & ~near_d & ~far_d).sum())
    def_spot = int(spot_d.sum())

    # visible opponents outside their box: where a counter-attack starts
    n_def_outside_box = int(((~in_box_d) & (~is_gk_mask)).sum())

    # defenders in the six-yard box
    in_6yd = in_box_d & (x_d >= SIX_YARD_X_MIN)
    def_in_6yd = int(in_6yd.sum())

    # defenders on the posts
    dist_l = np.sqrt((x_d - POST_L[0]) ** 2 + (y_d - POST_L[1]) ** 2)
    dist_r = np.sqrt((x_d - POST_R[0]) ** 2 + (y_d - POST_R[1]) ** 2)
    on_posts = (dist_l <= POST_RADIUS) | (dist_r <= POST_RADIUS)
    players_on_posts = int(on_posts.sum())

    # how deep the defenders stand, and how spread out
    in_box_outfield = in_box_d & (~is_gk_mask)

    if in_box_outfield.any():
        mean_x_box = float(x_d[in_box_outfield].mean())
        def_line_height = float(GOAL_X - mean_x_box)
        def_spread_y = float(
            y_d[in_box_outfield].max() - y_d[in_box_outfield].min()
        )
    elif in_box_d.any():
        mean_x_box = float(x_d[in_box_d].mean())
        def_line_height = float(GOAL_X - mean_x_box)
        def_spread_y = float(y_d[in_box_d].max() - y_d[in_box_d].min())
    else:
        def_line_height = math.nan
        def_spread_y = 0.0

    # defenders around the penalty spot
    edge_mask_d = (
        (x_d >= EDGE_X_MIN)
        & (x_d <= EDGE_X_MAX)
        & (y_d >= EDGE_Y_MIN)
        & (y_d <= EDGE_Y_MAX)
    )
    def_edge_density = int(edge_mask_d.sum())

    # defenders per attacker in the penalty area
    n_def_pen = int(in_box_d.sum())
    n_att_pen = int(in_box_a.sum()) if n_attackers > 0 else 0
    marking_ratio_pen = (
        float(n_def_pen) / float(n_att_pen) if n_att_pen > 0 else math.nan
    )

    # most attackers paired to one defender (nearest neighbour)
    if n_def_pen > 0 and n_att_pen > 0:
        A = np.column_stack([x_d[in_box_d], y_d[in_box_d]])
        B = np.column_stack([x_a[in_box_a], y_a[in_box_a]])
        diff = B[:, None, :] - A[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        nearest_def_idx = dist2.argmin(axis=1)
        counts = np.bincount(nearest_def_idx, minlength=A.shape[0])
        max_pairs_per_def = int(counts.max())
    else:
        max_pairs_per_def = 0

    return {
        "def_line_height": def_line_height,
        "def_spread_y": def_spread_y,
        "def_in_6yd": def_in_6yd,
        "players_on_posts": players_on_posts,
        "marking_ratio_pen": marking_ratio_pen,
        "def_edge_density": def_edge_density,
        "max_pairs_per_def": max_pairs_per_def,
        "gk_start_line": gk_start_line,
        "n_att_near": n_att_near,
        "n_att_central": n_att_central,
        "n_att_far": n_att_far,
        "n_blockers": n_blockers,
        "n_edge_of_box": n_edge_of_box,
        "n_rest_defence": n_rest_defence,
        "def_near": def_near,
        "def_central": def_central,
        "def_far": def_far,
        "def_spot": def_spot,
        "n_def_outside_box": n_def_outside_box,
    }


# stage 3: one row per corner with defensive, attacking and context features

def build_defensive_features(
    df_corners: pd.DataFrame,
    df_360_clean: pd.DataFrame,
    df_events_time: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_corners is None or df_corners.empty:
        print("build_defensive_features: empty df_corners, returning empty frame")
        return pd.DataFrame()

    if df_360_clean is None or df_360_clean.empty:
        print("build_defensive_features: empty df_360_clean, defensive geometry missing")
        df = df_corners.copy()
        for col in DEFENSIVE_FEATURES + ATTACKING_NUMERIC_FEATURES:
            df[col] = np.nan
        return df

    df_corners_ctx = add_score_state(df_events_time, df_corners)

    df_corners_ctx = df_corners_ctx.copy()
    if "minute" not in df_corners_ctx.columns:
        raise ValueError("build_defensive_features: df_corners must contain 'minute'")

    df_corners_ctx["team_id"] = df_corners_ctx.get("attacking_team.id")
    df_corners_ctx["minute_bucket"] = df_corners_ctx["minute"].apply(minute_bucket)
    df_corners_ctx["delivery_type"] = df_corners_ctx.apply(infer_delivery_type, axis=1)
    df_corners_ctx["short_corner_flag"] = (
        df_corners_ctx.apply(infer_short_corner, axis=1).astype("int64")
    )
    df_corners_ctx["end_zone"] = df_corners_ctx.apply(infer_end_zone, axis=1)
    df_corners_ctx["delivery_depth"] = df_corners_ctx.apply(infer_delivery_depth, axis=1)
    df_corners_ctx["delivery_high"] = (
        df_corners_ctx.apply(infer_delivery_high, axis=1).astype("int64")
    )

    required = ["match_id", "corner_uuid", "corner_side"]
    for col in required:
        if col not in df_corners_ctx.columns:
            raise ValueError(f"build_defensive_features: df_corners missing '{col}'")

    df_ff = df_360_clean.merge(
        df_corners_ctx[["match_id", "corner_uuid", "corner_side"]],
        left_on=["match_id", "event_uuid"],
        right_on=["match_id", "corner_uuid"],
        how="inner",
    )

    if df_ff.empty:
        print("build_defensive_features: no 360 rows matched to corners")
        return pd.DataFrame()

    rows: List[Dict] = []
    for (mid, cuid), grp in df_ff.groupby(["match_id", "corner_uuid"], sort=False):
        side = grp["corner_side"].iloc[0] if "corner_side" in grp.columns else "unknown"
        feats = _compute_def_att_features_for_corner(grp, side)
        feats["match_id"] = mid
        feats["corner_uuid"] = cuid
        rows.append(feats)

    df_geom = pd.DataFrame(rows)

    df_features = df_corners_ctx.merge(
        df_geom,
        on=["match_id", "corner_uuid"],
        how="inner",
    )
    df_features = add_derived_features(df_features)

    expected = set(DEFENSIVE_FEATURES + ATTACKING_NUMERIC_FEATURES)
    missing = [c for c in expected if c not in df_features.columns]
    if missing:
        raise ValueError(f"build_defensive_features: missing engineered features {missing}")

    print("------------------------------------------------------")
    print("stage 3: defensive + attacking feature engineering")
    print("------------------------------------------------------")
    print(f"corners input:    {len(df_corners):>5}")
    print(f"corners with 360: {len(df_features):>5}")

    return df_features



# pitch drawing for the figures

def _draw_full_pitch(ax) -> None:
    # draw a full StatsBomb pitch (120 x 80)
    x_min, x_max = 0.0, 120.0
    y_min, y_max = 0.0, 80.0

    # outer pitch
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color="black",
        linewidth=1.0,
    )

    # halfway line
    mid_x = (x_min + x_max) / 2.0
    ax.plot([mid_x, mid_x], [y_min, y_max], color="black", linewidth=1.0)

    # centre circle
    centre = (mid_x, (y_min + y_max) / 2.0)
    centre_circle = Circle(centre, radius=9.15, fill=False, linewidth=1.0)
    ax.add_patch(centre_circle)

    # penalty areas
    box_width = GOAL_X - PEN_X_MIN
    box_height = PEN_Y_MAX - PEN_Y_MIN

    # left penalty box (goal at x = 0)
    ax.add_patch(
        Rectangle(
            (0.0, PEN_Y_MIN),
            box_width,
            box_height,
            fill=False,
            linewidth=1.0,
        )
    )

    # right penalty box (goal at x = GOAL_X)
    ax.add_patch(
        Rectangle(
            (PEN_X_MIN, PEN_Y_MIN),
            box_width,
            box_height,
            fill=False,
            linewidth=1.0,
        )
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")


def plot_attacking_direction_pitch(
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    # full pitch showing which way corners are attacked
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_full_pitch(ax)

    # corners from the left and the right
    corner_from_right = (GOAL_X, 0.0)
    target_from_right = (GOAL_X - 18.0, GOAL_Y_LOW - 4.0)

    corner_from_left = (GOAL_X, 80.0)
    target_from_left = (GOAL_X - 18.0, GOAL_Y_HIGH + 4.0)

    ax.annotate(
        "",
        xy=target_from_right,
        xytext=corner_from_right,
        arrowprops=dict(arrowstyle="->", linewidth=2.0, color="red"),
    )
    ax.annotate(
        "",
        xy=target_from_left,
        xytext=corner_from_left,
        arrowprops=dict(arrowstyle="->", linewidth=2.0, color="green"),
    )

    # arrow for the attacking direction
    attack_arrow_start = (5.0, GOAL_Y_CENTER)
    attack_arrow_end = (GOAL_X - 5.0, GOAL_Y_CENTER)

    ax.annotate(
        "",
        xy=attack_arrow_end,
        xytext=attack_arrow_start,
        arrowprops=dict(arrowstyle="->", linewidth=2.5, color="black"),
    )

    # invisible lines for the legend
    ax.plot([], [], color="red", label="Corner from right")
    ax.plot([], [], color="green", label="Corner from left")
    ax.plot([], [], color="black", label="Attacking direction")

    ax.legend(loc="upper right")
    ax.set_title("Attacking Direction")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved attacking direction pitch to {out_path}")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def plot_zone_definitions_pitch(
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    # full pitch with the three target zones used by infer_end_zone
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_full_pitch(ax)

    # band limits (same as infer_end_zone)
    y_near_min = PEN_Y_MIN
    y_near_max = GOAL_Y_CENTER - 6.0
    y_central_min = GOAL_Y_CENTER - 6.0
    y_central_max = GOAL_Y_CENTER + 6.0
    y_far_min = GOAL_Y_CENTER + 6.0
    y_far_max = PEN_Y_MAX

    # how far the zones reach into the box
    x_min = PEN_X_MIN
    x_max = GOAL_X

    zones = [
        ("near", (x_min, x_max, y_near_min, y_near_max)),
        ("central", (x_min, x_max, y_central_min, y_central_max)),
        ("far", (x_min, x_max, y_far_min, y_far_max)),
    ]

    colors = {
        "near": "red",
        "central": "blue",
        "far": "green",
    }

    handles = []

    for label, (zx_min, zx_max, zy_min, zy_max) in zones:
        width = zx_max - zx_min
        height = zy_max - zy_min

        rect = Rectangle(
            (zx_min, zy_min),
            width,
            height,
            linewidth=1.5,
            edgecolor="black",
            facecolor=colors[label],
            alpha=0.6,
        )
        ax.add_patch(rect)

        cx = zx_min + width / 2.0
        cy = zy_min + height / 2.0
        ax.text(
            cx,
            cy,
            label.capitalize(),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
        )

        handles.append(rect)

    ax.set_title("Delivery Zones on Pitch (Near / Central / Far)")

    ax.legend(
        handles,
        [f"{lbl.capitalize()} zone" for lbl, _ in zones],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved zone definitions pitch to {out_path}")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def plot_archetype_example_freeze_frames(
    df_corners: pd.DataFrame,
    df_360_clean: pd.DataFrame,
    df_clustered: pd.DataFrame,
    out_dir: Optional[str | Path] = None,
    prefix: str = "archetype_example_freeze_frames",
) -> pd.DataFrame:
    # one example freeze frame for each defensive shape
    if df_corners is None or df_corners.empty:
        print("plot_archetype_example_freeze_frames: df_corners is empty, skipping")
        return pd.DataFrame()

    if df_360_clean is None or df_360_clean.empty:
        print("plot_archetype_example_freeze_frames: df_360_clean is empty, skipping")
        return pd.DataFrame()

    if df_clustered is None or df_clustered.empty:
        print("plot_archetype_example_freeze_frames: df_clustered is empty, skipping")
        return pd.DataFrame()

    required_cluster_cols = {"match_id", "corner_uuid", "A_name"}
    missing_cluster = [c for c in required_cluster_cols if c not in df_clustered.columns]
    if missing_cluster:
        raise ValueError(
            "plot_archetype_example_freeze_frames: df_clustered missing columns "
            f"{missing_cluster}"
        )

    required_corner_cols = {"match_id", "event_uuid", "corner_uuid"}
    missing_corner = [c for c in required_corner_cols if c not in df_corners.columns]
    if missing_corner:
        raise ValueError(
            "plot_archetype_example_freeze_frames: df_corners missing columns "
            f"{missing_corner}"
        )

    required_360_cols = {"match_id", "event_uuid", "x", "y", "teammate"}
    missing_360 = [c for c in required_360_cols if c not in df_360_clean.columns]
    if missing_360:
        raise ValueError(
            "plot_archetype_example_freeze_frames: df_360_clean missing columns "
            f"{missing_360}"
        )

    # work on copies
    df_cluster = df_clustered.copy()
    df_corner_map = df_corners[["match_id", "event_uuid", "corner_uuid"]].drop_duplicates()

    # keep rows with a proper shape name
    df_cluster = df_cluster[df_cluster["A_name"].notna()]
    df_cluster = df_cluster[df_cluster["A_name"] != "Unknown"]

    if df_cluster.empty:
        print("plot_archetype_example_freeze_frames: no rows with non empty A_name")
        return pd.DataFrame()

    if "A_confidence" in df_cluster.columns:
        df_cluster = df_cluster.dropna(subset=["A_confidence"])
        if df_cluster.empty:
            print("plot_archetype_example_freeze_frames: no rows with A_confidence")
            return pd.DataFrame()

    # order of the panels
    archetype_values = df_cluster["A_name"].unique().tolist()
    preferred_order = ["Posts covered", "Deep line", "High line"]
    ordered = [a for a in preferred_order if a in archetype_values]
    ordered += [a for a in sorted(archetype_values) if a not in ordered]
    archetypes = ordered

    # one example corner per shape
    example_rows = []
    for a_name in archetypes:
        df_a = df_cluster[df_cluster["A_name"] == a_name]
        if df_a.empty:
            continue

        if "A_confidence" in df_a.columns:
            df_a = df_a.sort_values("A_confidence", ascending=False)
        else:
            df_a = df_a.sort_values("corner_uuid", ascending=True)

        row = df_a.iloc[0]
        example_rows.append(
            {
                "A_name": str(a_name),
                "match_id": int(row["match_id"]),
                "corner_uuid": str(row["corner_uuid"]),
            }
        )

    if not example_rows:
        print("plot_archetype_example_freeze_frames: no example rows selected")
        return pd.DataFrame()

    df_examples = pd.DataFrame(example_rows)

    # add event_uuid for the chosen corners
    df_examples = df_examples.merge(
        df_corner_map,
        on=["match_id", "corner_uuid"],
        how="left",
    )

    if df_examples["event_uuid"].isna().any():
        print(
            "plot_archetype_example_freeze_frames: some examples missing event_uuid "
            "after merge"
        )
        df_examples = df_examples.dropna(subset=["event_uuid"])

    if df_examples.empty:
        print("plot_archetype_example_freeze_frames: no examples with event_uuid")
        return df_examples

    # their 360 rows
    df_examples["event_uuid"] = df_examples["event_uuid"].astype(str)
    df_360_clean = df_360_clean.copy()
    df_360_clean["event_uuid"] = df_360_clean["event_uuid"].astype(str)

    df_ff = df_360_clean.merge(
        df_examples[["match_id", "event_uuid", "A_name"]],
        on=["match_id", "event_uuid"],
        how="inner",
    )

    if df_ff.empty:
        print(
            "plot_archetype_example_freeze_frames: no 360 rows matched to selected "
            "examples"
        )
        return df_examples

    # draw the figure
    n_arch = len(df_examples["A_name"].unique())
    fig, axes = plt.subplots(
        1,
        n_arch,
        figsize=(5 * n_arch, 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, a_name in zip(axes, archetypes):
        _draw_full_pitch(ax)

        subset = df_ff[df_ff["A_name"] == a_name].copy()
        if subset.empty:
            ax.set_title(f"{a_name} (no data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        teammate_mask = subset["teammate"].astype(bool)
        attackers = subset[teammate_mask]
        defenders = subset[~teammate_mask]

        ax.scatter(
            attackers["x"],
            attackers["y"],
            s=25,
            marker="o",
            label="Attackers",
        )
        ax.scatter(
            defenders["x"],
            defenders["y"],
            s=25,
            marker="s",
            label="Defenders",
        )

        # highlight the keeper if known
        if "keeper" in subset.columns:
            gk_mask = (~teammate_mask) & subset["keeper"].fillna(False).astype(bool)
            gk = subset[gk_mask]
            if not gk.empty:
                ax.scatter(
                    gk["x"],
                    gk["y"],
                    s=60,
                    marker="D",
                    label="Goalkeeper",
                )

        ax.set_title(f"{a_name} example")
        ax.set_xticks([])
        ax.set_yticks([])

    # one legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.15)

    fig.suptitle("Example freeze frame for each defensive archetype", fontsize=12)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"{prefix}.png"
        csv_path = out_dir / f"{prefix}_examples.csv"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        df_examples.to_csv(csv_path, index=False)
        print(f"saved archetype freeze frame figure to {fig_path}")
        print(f"saved archetype example mapping to {csv_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return df_examples
