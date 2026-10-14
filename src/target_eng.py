from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import LABEL_COLUMNS

# outcomes are also labelled at each sensitivity window, as "<label>_h<seconds>"
SENSITIVITY_LABELS: List[str] = ["chance_for", "shot_against_any", "xg_for", "xg_against"]


# helper: absolute time column

def _ensure_time_column(df: pd.DataFrame, name: str = "t_abs") -> pd.DataFrame:
    # make sure df has an absolute time column called `name`
    df = df.copy()
    if name in df.columns:
        return df
    if "_t" in df.columns:
        df[name] = df["_t"]
        return df
    raise ValueError(f"table is missing both '{name}' and '_t' time columns.")


# helpers: possessions and shots for one match

def _possession_table(events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # possession ids in order, and which team had the ball in each
    cols = ["possession", "possession_team.id"]
    if events.empty or any(c not in events.columns for c in cols):
        return np.array([], dtype=int), np.array([], dtype=int)
    poss = events[cols].apply(pd.to_numeric, errors="coerce").dropna()
    first = poss.groupby("possession")["possession_team.id"].first().sort_index()
    return first.index.to_numpy(dtype=int), first.to_numpy(dtype=int)


def _phase_possessions(
    poss_ids: np.ndarray,
    poss_teams: np.ndarray,
    corner_possession: int,
    att_id: int,
    opp_id: int,
) -> Tuple[Set[int], Set[int]]:
    # attacking phase: the corner's possession plus any straight after it by the same team.
    # counter phase: the opponent's possessions after that, until we have the ball again.
    attack: Set[int] = set()
    counter: Set[int] = set()
    j = int(np.searchsorted(poss_ids, corner_possession))
    while j < len(poss_ids) and poss_teams[j] == att_id:
        attack.add(int(poss_ids[j]))
        j += 1
    while j < len(poss_ids) and poss_teams[j] == opp_id:
        counter.add(int(poss_ids[j]))
        j += 1
    return attack, counter


def _shot_table(events: pd.DataFrame) -> pd.DataFrame:
    # shots with their order, possession, team, time, xg and whether they scored
    if events.empty or "type.name" not in events.columns:
        return pd.DataFrame(columns=["index", "possession", "team", "t_abs", "xg", "goal"])
    shots = events[events["type.name"] == "Shot"]
    return pd.DataFrame(
        {
            "index": pd.to_numeric(shots.get("index"), errors="coerce"),
            "possession": pd.to_numeric(shots.get("possession"), errors="coerce"),
            "team": pd.to_numeric(shots.get("team.id"), errors="coerce"),
            "t_abs": pd.to_numeric(shots.get("t_abs"), errors="coerce"),
            "xg": pd.to_numeric(shots.get("shot.statsbomb_xg"), errors="coerce").fillna(0.0),
            "goal": (shots.get("shot.outcome.name") == "Goal").astype(int),
        }
    ).dropna(subset=["index", "possession", "team", "t_abs"])


def _phase_outcomes(shots: pd.DataFrame, xg_threshold: float) -> Dict[str, float]:
    # sum up the shots in one phase
    if shots.empty:
        return {"goal": 0, "chance": 0, "shot": 0, "xg": 0.0}
    return {
        "goal": int(shots["goal"].max()),
        "chance": int((shots["xg"] >= xg_threshold).any()),
        "shot": 1,
        "xg": float(shots["xg"].sum()),
    }


# helpers: first contact
#
# StatsBomb marks a pass "complete" when a teammate plays the ball first and
# "Incomplete" when an opponent or the keeper does, so the delivery's outcome says
# who won the first contact. Out of play, offside and fouls aren't contests.
# Short corners are followed to the first ball into the box (up to SHORT_MAX_PASSES
# passes within SHORT_MAX_SECS), and that cross counts as the delivery.

SHORT_MAX_PASSES = 3
SHORT_MAX_SECS = 15.0
_BOX_X_MIN, _BOX_Y_MIN, _BOX_Y_MAX = 102.0, 18.0, 62.0
_NO_CONTEST = {"Out", "Pass Offside", "Unknown", "Injury Clearance"}


def _in_box(loc) -> bool:
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return False
    return float(loc[0]) >= _BOX_X_MIN and _BOX_Y_MIN <= float(loc[1]) <= _BOX_Y_MAX


def _first_contact(events: pd.DataFrame, pos: int, att_id: int, t0: float) -> Dict[str, object]:
    # who played the ball first after the delivery (pos = the corner's row)
    corner = events.iloc[pos]
    out: Dict[str, object] = {"first_contact_attack": np.nan, "first_contact_type": "", "delivery_route": "direct"}
    delivery = corner

    if not _in_box(corner.get("pass.end_location")):
        out["delivery_route"] = "short_routine"
        delivery = None
        passes = 0
        for _, ev in events.iloc[pos + 1 :].iterrows():
            if float(ev["t_abs"]) > t0 + SHORT_MAX_SECS:
                break
            if ev.get("possession_team.id") != att_id:
                out["first_contact_type"] = "routine_broken"
                return out
            if ev.get("type.name") != "Pass" or ev.get("team.id") != att_id:
                continue
            passes += 1
            if _in_box(ev.get("pass.end_location")) or ev.get("pass.cross") is True:
                delivery = ev
                break
            if passes >= SHORT_MAX_PASSES:
                break
        if delivery is None:
            out["first_contact_type"] = "no_cross"
            return out

    outcome = delivery.get("pass.outcome.name")
    if isinstance(outcome, str) and outcome in _NO_CONTEST:
        out["first_contact_type"] = outcome.lower().replace(" ", "_")
        return out

    won = not isinstance(outcome, str)  # no outcome recorded means complete
    out["first_contact_attack"] = int(won)

    # what the first contact was, for the charts and the label check
    nxt = events[(events["index"] > delivery["index"]) & (events["type.name"] != "Pressure")].head(3)
    types = nxt["type.name"].tolist()
    out["first_contact_events"] = " > ".join(
        f"{'A' if team == att_id else 'D'}:{kind}" for team, kind in zip(nxt["team.id"], types)
    )
    if won:
        shot = nxt[(nxt["type.name"] == "Shot") & (nxt["team.id"] == att_id)]
        if not shot.empty:
            out["first_contact_type"] = "header_shot" if shot["shot.body_part.name"].iloc[0] == "Head" else "shot"
        else:
            out["first_contact_type"] = "attacker_controls"
    elif "Goal Keeper" in types[:2]:
        out["first_contact_type"] = "keeper_claims"
    elif "Clearance" in types[:2]:
        out["first_contact_type"] = "defender_clears"
    else:
        out["first_contact_type"] = "defence_wins"
    return out


# helper: label every corner in one match

def _label_match_corners(
    df_corners_match: pd.DataFrame,
    df_events_match: pd.DataFrame,
    horizon_secs: float,
    xg_threshold: float,
    sensitivity_horizons: Sequence[float] = (),
) -> pd.DataFrame:
    # label what happened in our phase ("for") and their next one ("against")
    corners = _ensure_time_column(df_corners_match, name="t_abs").copy()
    for col in ["attacking_team.id", "opponent_team.id", "possession", "index"]:
        if col not in corners.columns:
            raise ValueError(f"_label_match_corners: corners missing '{col}'.")

    events = df_events_match.copy() if df_events_match is not None else pd.DataFrame()
    if not events.empty:
        events = _ensure_time_column(events, name="t_abs")

    poss_ids, poss_teams = _possession_table(events)
    shots = _shot_table(events)

    # events in order, for the first-contact labels
    contact_cols = [
        "index", "type.name", "team.id", "possession_team.id", "t_abs",
        "pass.outcome.name", "pass.end_location", "pass.cross", "shot.body_part.name",
    ]
    ordered = events.reindex(columns=contact_cols).copy() if not events.empty else pd.DataFrame(columns=contact_cols)
    for col in ["index", "team.id", "possession_team.id", "t_abs"]:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    ordered = ordered.dropna(subset=["index"]).sort_values("index").reset_index(drop=True)
    position_of = {int(ix): pos for pos, ix in enumerate(ordered["index"].to_numpy())}

    horizons = [float(horizon_secs)] + [float(h) for h in sensitivity_horizons]
    records: List[Dict[str, float]] = []

    for _, row in corners.iterrows():
        t0 = float(row["t_abs"])
        idx0 = float(row["index"])
        att_id = int(row["attacking_team.id"])
        opp_id = int(row["opponent_team.id"])
        attack, counter = _phase_possessions(
            poss_ids, poss_teams, int(row["possession"]), att_id, opp_id
        )

        after = shots[shots["index"] > idx0]
        shots_for = after[(after["team"] == att_id) & after["possession"].isin(attack)]
        shots_against = after[(after["team"] == opp_id) & after["possession"].isin(counter)]

        rec: Dict[str, float] = {}
        for i, h in enumerate(horizons):
            out_for = _phase_outcomes(shots_for[shots_for["t_abs"] <= t0 + h], xg_threshold)
            out_against = _phase_outcomes(shots_against[shots_against["t_abs"] <= t0 + h], xg_threshold)
            labels = {
                "goal_for": out_for["goal"],
                "goal_against": out_against["goal"],
                "chance_for": out_for["chance"],
                "chance_against": out_against["chance"],
                "shot_for_any": out_for["shot"],
                "shot_against_any": out_against["shot"],
                "xg_for": out_for["xg"],
                "xg_against": out_against["xg"],
            }
            if i == 0:
                rec.update(labels)
            else:
                rec.update({f"{k}_h{int(h)}": labels[k] for k in SENSITIVITY_LABELS})

        pos = position_of.get(int(idx0))
        if pos is not None:
            rec.update(_first_contact(ordered, pos, att_id, t0))
        else:
            rec.update({"first_contact_attack": np.nan, "first_contact_type": "corner_event_missing", "delivery_route": ""})
        records.append(rec)

    labelled = pd.DataFrame(records, index=corners.index)
    return pd.concat([corners, labelled], axis=1)


# stage 5: outcome labels and the master table

def build_target_table(
    df_events_time: pd.DataFrame,
    df_features_clustered: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    # stage 5: build the master table of corners and their outcomes
    if df_features_clustered is None or df_features_clustered.empty:
        print("build_target_table: df_features_clustered is empty, returning empty frame")
        return pd.DataFrame()

    cfg_targets = (cfg or {}).get("targets", {}) or {}
    horizon_secs = float(cfg_targets.get("horizon_secs", 60.0))
    xg_threshold = float(cfg_targets.get("xg_threshold", 0.04))
    sensitivity = [
        float(h) for h in (cfg_targets.get("sensitivity_horizons", []) or []) if float(h) != horizon_secs
    ]

    print("------------------------------------------------------")
    print("stage 5: target engineering configuration")
    print("------------------------------------------------------")
    print(f"horizon (seconds):     {horizon_secs}")
    print(f"sensitivity horizons:  {sensitivity}")
    print(f"good chance xg >=      {xg_threshold}")
    print("for = attacking possession phase, against = opponent's next phase")
    print("")

    df_corners = df_features_clustered.copy()
    for col in ["match_id", "attacking_team.id", "opponent_team.id"]:
        if col not in df_corners.columns:
            raise ValueError(f"build_target_table: df_features_clustered missing '{col}'.")

    if df_events_time is None or df_events_time.empty:
        print("build_target_table: df_events_time is empty, all labels set to zero")
        df_target = df_corners.copy()
        for col in LABEL_COLUMNS:
            df_target[col] = 0
        return df_target

    df_events = _ensure_time_column(df_events_time, name="t_abs")
    events_by_match: Dict[int, pd.DataFrame] = {
        int(mid): g for mid, g in df_events.groupby("match_id")
    }

    chunks: List[pd.DataFrame] = []
    for mid, df_c_m in df_corners.groupby("match_id"):
        chunks.append(
            _label_match_corners(
                df_corners_match=df_c_m,
                df_events_match=events_by_match.get(int(mid), pd.DataFrame()),
                horizon_secs=horizon_secs,
                xg_threshold=xg_threshold,
                sensitivity_horizons=sensitivity,
            )
        )

    if not chunks:
        print("build_target_table: no labelled corners produced, returning empty frame")
        return pd.DataFrame()

    df_target = pd.concat(chunks).sort_index().reset_index(drop=True)

    print("------------------------------------------------------")
    print("stage 5: label summary")
    print("------------------------------------------------------")
    print(f"total samples (rows): {len(df_target)}")
    for col in LABEL_COLUMNS:
        if col.startswith("xg_"):
            print(f"mean {col}: {df_target[col].mean():.4f}  (non-zero: {int((df_target[col] > 0).sum())})")
        elif col == "first_contact_attack":
            labelled = df_target[col].dropna()
            print(f"first contact won by attack: {int(labelled.sum())} of {len(labelled)} contested ({labelled.mean():.1%})")
        else:
            print(f"positive {col}: {int(df_target[col].sum())}")
    if "first_contact_type" in df_target.columns:
        print("first contact types:")
        print(df_target.groupby(["delivery_route", "first_contact_type"]).size().to_string())
    print("")

    return df_target


# stage 5 wrapper, with an optional csv save

def add_chain_targets(df: pd.DataFrame) -> pd.DataFrame:
    # chance_for split by who won the first ball (see CHAIN_TARGETS in constants.py)
    out = df.copy()
    if not {"first_contact_attack", "chance_for"}.issubset(out.columns):
        return out
    fc = out["first_contact_attack"]
    chance = out["chance_for"].astype(float)
    out["chance_if_won"] = chance.where(fc == 1)
    out["chance_if_lost"] = chance.where(fc == 0)
    return out


def stage5_target_engineering(
    df_events_time: pd.DataFrame,
    df_features_clustered: pd.DataFrame,
    cfg: dict,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    # run stage 5 and optionally save df_target as csv
    df_target = add_chain_targets(
        build_target_table(
            df_events_time=df_events_time,
            df_features_clustered=df_features_clustered,
            cfg=cfg,
        )
    )

    if out_path is not None and not df_target.empty:
        df_target.to_csv(out_path, index=False)
        print("------------------------------------------------------")
        print("stage 5: target table saved")
        print("------------------------------------------------------")
        print(f"saved to: {out_path}")
        print("")

    return df_target


# figure: label balance

def plot_label_balance(
    df_target: pd.DataFrame,
    out_dir: Optional[str] = None,
    prefix: str = "labels_stage5",
):
    # share of yes and no for the main outcome labels
    if df_target is None or df_target.empty:
        print("plot_label_balance: empty df_target, skipping")
        return None

    labels_to_plot: List[str] = ["chance_for", "shot_against_any"]
    labels = [col for col in labels_to_plot if col in df_target.columns]

    if not labels:
        print("plot_label_balance: no expected labels found, skipping")
        return None

    n = len(df_target)
    if n == 0:
        print("plot_label_balance: zero rows in df_target, skipping")
        return None

    # counts and shares
    counts_pos = df_target[labels].sum()
    counts_neg = n - counts_pos

    props_pos = counts_pos / n
    props_neg = counts_neg / n

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x, props_neg.values, label="negative (0)")
    ax.bar(x, props_pos.values, bottom=props_neg.values, label="positive (1)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("proportion of corners")
    ax.set_title("proportion of positive and negative outcomes for APV labels")
    ax.legend()

    fig.tight_layout()

    if out_dir is not None:
        import os

        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"{prefix}_proportions.png")
        fig.savefig(fig_path, dpi=300)
        print(f"saved label balance figure to {fig_path}")
        plt.close(fig)
    else:
        plt.show()

    # summary table of counts and shares
    df_summary = pd.DataFrame(
        {
            "n_positive": counts_pos,
            "n_negative": counts_neg,
            "prop_positive": props_pos,
            "prop_negative": props_neg,
        }
    )

    if out_dir is not None:
        import os

        csv_path = os.path.join(out_dir, f"{prefix}_summary.csv")
        df_summary.to_csv(csv_path, index=True)
        print(f"saved label summary table to {csv_path}")

    return df_summary
