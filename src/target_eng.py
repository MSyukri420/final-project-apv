from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import LABEL_COLUMNS


# helper: ensure absolute time column

def _ensure_time_column(df: pd.DataFrame, name: str = "t_abs") -> pd.DataFrame:
    # make sure df has an absolute time column named `name`
    df = df.copy()
    if name in df.columns:
        return df
    if "_t" in df.columns:
        df[name] = df["_t"]
        return df
    raise ValueError(f"table is missing both '{name}' and '_t' time columns.")


# helper: build timelines of events per team and match

def _build_timelines(
    df_events_match: pd.DataFrame,
    kind: str,
    xg_threshold: float,
) -> Dict[int, np.ndarray]:
    # build team_id -> sorted array of event times (t_abs) for one match
    # kind: "goal", "chance", or "shot"
    if df_events_match is None or df_events_match.empty:
        return {}

    df = df_events_match.copy()

    # filter by event type according to kind
    if kind in {"chance", "shot"}:
        if "type.name" not in df.columns:
            return {}
        df = df[df["type.name"] == "Shot"]

    if kind == "goal":
        if "shot.outcome.name" not in df.columns:
            return {}
        df = df[df["shot.outcome.name"] == "Goal"]
    elif kind == "chance":
        if "shot.statsbomb_xg" not in df.columns:
            return {}
        df["shot.statsbomb_xg"] = pd.to_numeric(
            df["shot.statsbomb_xg"],
            errors="coerce",
        )
        df = df[df["shot.statsbomb_xg"] >= float(xg_threshold)]
    elif kind == "shot":
        # already filtered by type.name
        pass
    else:
        raise ValueError(f"unknown timeline kind: {kind}")

    if df.empty:
        return {}

    if "team.id" not in df.columns:
        return {}

    df = _ensure_time_column(df, name="t_abs")
    df["team.id"] = pd.to_numeric(df["team.id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["team.id", "t_abs"])

    timelines: Dict[int, np.ndarray] = {}
    for tid, g in df.groupby("team.id"):
        times = np.sort(g["t_abs"].to_numpy(dtype=float))
        timelines[int(tid)] = times

    return timelines


# helper: check if a team has an event in the horizon

def _has_event_in_window(
    timelines: Dict[int, np.ndarray],
    team_id: int,
    t0: float,
    horizon_secs: float,
) -> int:
    # return 1 if team_id has any event in (t0, t0 + horizon_secs], else 0
    if not timelines:
        return 0

    if pd.isna(team_id):
        return 0

    tid = int(team_id)
    events = timelines.get(tid)
    if events is None or len(events) == 0:
        return 0

    # first event strictly after t0
    idx = np.searchsorted(events, t0, side="right")
    if idx >= len(events):
        return 0

    t_event = events[idx]
    return int(t_event <= t0 + horizon_secs)


# helper: label all corners for a single match

def _label_match_corners(
    df_corners_match: pd.DataFrame,
    df_events_match: pd.DataFrame,
    horizon_secs: float,
    xg_threshold: float,
) -> pd.DataFrame:
    # label all corners in a single match with goal/chance/shot outcomes
    if df_corners_match is None or df_corners_match.empty:
        return df_corners_match.copy()

    # ensure time and team id columns
    corners = _ensure_time_column(df_corners_match, name="t_abs").copy()

    for col in ["attacking_team.id", "opponent_team.id"]:
        if col not in corners.columns:
            raise ValueError(
                f"_label_match_corners: df_corners_match missing '{col}'."
            )
        corners[col] = pd.to_numeric(corners[col], errors="coerce").astype("Int64")

    # normalise events table
    events = df_events_match.copy() if df_events_match is not None else pd.DataFrame()
    if not events.empty:
        events = _ensure_time_column(events, name="t_abs")

    goal_tl = _build_timelines(events, kind="goal", xg_threshold=xg_threshold)
    chance_tl = _build_timelines(events, kind="chance", xg_threshold=xg_threshold)
    shot_tl = _build_timelines(events, kind="shot", xg_threshold=xg_threshold)

    # initialise label arrays
    goal_for = pd.Series(0, index=corners.index, dtype="int64")
    goal_against = pd.Series(0, index=corners.index, dtype="int64")
    chance_for = pd.Series(0, index=corners.index, dtype="int64")
    chance_against = pd.Series(0, index=corners.index, dtype="int64")
    shot_for_any = pd.Series(0, index=corners.index, dtype="int64")
    shot_against_any = pd.Series(0, index=corners.index, dtype="int64")

    # per corner labelling
    for idx, row in corners.iterrows():
        t0 = float(row["t_abs"])
        att_id = row["attacking_team.id"]
        opp_id = row["opponent_team.id"]

        goal_for.loc[idx] = _has_event_in_window(goal_tl, att_id, t0, horizon_secs)
        goal_against.loc[idx] = _has_event_in_window(goal_tl, opp_id, t0, horizon_secs)

        chance_for.loc[idx] = _has_event_in_window(chance_tl, att_id, t0, horizon_secs)
        chance_against.loc[idx] = _has_event_in_window(
            chance_tl,
            opp_id,
            t0,
            horizon_secs,
        )

        shot_for_any.loc[idx] = _has_event_in_window(shot_tl, att_id, t0, horizon_secs)
        shot_against_any.loc[idx] = _has_event_in_window(
            shot_tl,
            opp_id,
            t0,
            horizon_secs,
        )

    corners = corners.copy()
    corners["goal_for"] = goal_for
    corners["goal_against"] = goal_against
    corners["chance_for"] = chance_for
    corners["chance_against"] = chance_against
    corners["shot_for_any"] = shot_for_any
    corners["shot_against_any"] = shot_against_any

    return corners


# stage 5: target engineering and master table

def build_target_table(
    df_events_time: pd.DataFrame,
    df_features_clustered: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    # stage 5: target engineering and master table assembly
    if df_features_clustered is None or df_features_clustered.empty:
        print("build_target_table: df_features_clustered is empty, returning empty frame")
        return pd.DataFrame()

    # time horizon and xg threshold
    cfg_targets = (cfg or {}).get("targets", {}) or {}
    horizon_secs = float(cfg_targets.get("horizon_secs", 60.0))
    xg_threshold = float(cfg_targets.get("xg_threshold", 0.05))

    print("------------------------------------------------------")
    print("stage 5: target engineering configuration")
    print("------------------------------------------------------")
    print(f"horizon (seconds): {horizon_secs}")
    print(f"xg threshold:      {xg_threshold}")
    print("")

    # use all clustered corners (no A_reliable filter any more)
    df_corners = df_features_clustered.copy()
    n_corners = len(df_corners)
    print("stage 5: input corners")
    print("----------------------")
    print(f"rows received from stage 4: {n_corners}")
    print("")

    if df_corners.empty:
        print("build_target_table: no corners to label, returning empty frame")
        return pd.DataFrame()

    # basic column checks
    for col in ["match_id", "attacking_team.id", "opponent_team.id"]:
        if col not in df_corners.columns:
            raise ValueError(
                f"build_target_table: df_features_clustered missing '{col}'."
            )

    # if we have no events, label everything as zero
    if df_events_time is None or df_events_time.empty:
        print("build_target_table: df_events_time is empty, all labels set to zero")
        df_target = df_corners.copy()
        for col in LABEL_COLUMNS:
            df_target[col] = 0
        return df_target

    if "match_id" not in df_events_time.columns:
        raise ValueError("build_target_table: df_events_time is missing 'match_id'.")

    # ensure absolute time in events
    df_events = _ensure_time_column(df_events_time, name="t_abs")

    # group events by match for efficiency
    events_by_match: Dict[int, pd.DataFrame] = {
        int(mid): g.copy() for mid, g in df_events.groupby("match_id")
    }

    # process per match
    unique_matches = (
        df_corners["match_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print("------------------------------------------------------")
    print("stage 5: labelling per match")
    print("------------------------------------------------------")
    print(f"matches with corners: {len(unique_matches)}")
    print("")

    labelled_chunks: List[pd.DataFrame] = []

    for mid in unique_matches:
        df_c_m = df_corners[df_corners["match_id"] == mid]
        df_e_m = events_by_match.get(int(mid), pd.DataFrame())

        if df_c_m.empty:
            continue

        chunk = _label_match_corners(
            df_corners_match=df_c_m,
            df_events_match=df_e_m,
            horizon_secs=horizon_secs,
            xg_threshold=xg_threshold,
        )
        labelled_chunks.append(chunk)

    if not labelled_chunks:
        print("build_target_table: no labelled corners produced, returning empty frame")
        return pd.DataFrame()

    df_target = pd.concat(labelled_chunks, ignore_index=True)

    # summary
    print("------------------------------------------------------")
    print("stage 5: label summary")
    print("------------------------------------------------------")
    print(f"total samples (rows): {len(df_target)}")
    for col in LABEL_COLUMNS:
        if col in df_target.columns:
            pos = int(df_target[col].sum())
            print(f"positive {col}: {pos}")
    print("")

    if "A_name" in df_target.columns:
        for label in ["chance_for", "chance_against"]:
            if label in df_target.columns:
                print(f"by archetype (A_name) for {label} = 1:")
                print(df_target[df_target[label] == 1]["A_name"].value_counts())
                print("")

        for label in ["shot_for_any", "shot_against_any"]:
            if label in df_target.columns:
                print(f"by archetype (A_name) for {label} = 1:")
                print(df_target[df_target[label] == 1]["A_name"].value_counts())
                print("")

    return df_target


# wrapper around build_target_table, with optional csv save

def stage5_target_engineering(
    df_events_time: pd.DataFrame,
    df_features_clustered: pd.DataFrame,
    cfg: dict,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    # run stage 5 and optionally save df_target to csv
    df_target = build_target_table(
        df_events_time=df_events_time,
        df_features_clustered=df_features_clustered,
        cfg=cfg,
    )

    if out_path is not None and not df_target.empty:
        df_target.to_csv(out_path, index=False)
        print("------------------------------------------------------")
        print("stage 5: target table saved")
        print("------------------------------------------------------")
        print(f"saved to: {out_path}")
        print("")

    return df_target


# diagnostics: label balance figure

def plot_label_balance(
    df_target: pd.DataFrame,
    out_dir: Optional[str] = None,
    prefix: str = "labels_stage5",
):
    # plot proportion of positive and negative outcomes for selected apv labels
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

    # positive and negative counts / proportions
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

    # summary table: counts and proportions for both classes
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
