from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MIN_PLAYERS_IN_FRAME = 20
MIN_ATTACKERS_IN_FRAME = 9
MIN_DEFENDERS_IN_FRAME = 9
MIN_TOTAL_PLAYERS_IN_FRAME = 18


def _get_nested(obj, *keys):
    # safely read nested dict keys
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


def flatten_fields(df_events_raw: pd.DataFrame) -> pd.DataFrame:
    # create flat events table with key statsbomb fields
    if df_events_raw is None or df_events_raw.empty:
        print("flatten_fields: received empty events dataframe")
        return pd.DataFrame()

    df = df_events_raw.copy()

    def fill(col, fn):
        if col not in df.columns:
            df[col] = df.apply(fn, axis=1)

    # event types and pass detail
    fill("type.name", lambda r: _get_nested(r.get("type"), "name"))
    fill("pass.type.name", lambda r: _get_nested(r.get("pass"), "type", "name"))
    fill("pass.end_location", lambda r: _get_nested(r.get("pass"), "end_location"))
    fill(
        "pass.body_part.name",
        lambda r: _get_nested(r.get("pass"), "body_part", "name"),
    )
    fill("pass.height.name", lambda r: _get_nested(r.get("pass"), "height", "name"))

    # team / player
    fill("team.id", lambda r: _get_nested(r.get("team"), "id"))
    fill("team.name", lambda r: _get_nested(r.get("team"), "name"))
    fill("player.id", lambda r: _get_nested(r.get("player"), "id"))
    fill("player.name", lambda r: _get_nested(r.get("player"), "name"))
    fill(
        "possession_team.id",
        lambda r: _get_nested(r.get("possession_team"), "id"),
    )
    fill(
        "possession_team.name",
        lambda r: _get_nested(r.get("possession_team"), "name"),
    )
    fill("shot.outcome.name", lambda r: _get_nested(r.get("shot"), "outcome", "name"))
    fill("shot.statsbomb_xg", lambda r: _get_nested(r.get("shot"), "statsbomb_xg"))

    # match_id
    if "match_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "match_id"})
    if "match_id" not in df.columns:
        raise ValueError("flatten_fields: events dataframe has no 'match_id' column")

    # event_uuid
    if "event_uuid" not in df.columns:
        if "id" in df.columns:
            df["event_uuid"] = df["id"].astype(str)
        else:
            raise ValueError(
                "flatten_fields: events dataframe has no 'event_uuid' and no 'id'"
            )
    df["event_uuid"] = df["event_uuid"].astype(str)

    # time fields
    for col in ["period", "minute", "second"]:
        if col not in df.columns:
            raise ValueError(
                f"flatten_fields: expected '{col}' column in events dataframe"
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"flatten_fields: {len(df_events_raw)} -> {len(df)} events (raw -> flat)")
    return df


def attach_event_time(df_events_flat: pd.DataFrame) -> pd.DataFrame:
    # attach absolute event time per match
    if df_events_flat is None or df_events_flat.empty:
        print("attach_event_time: received empty events dataframe")
        return pd.DataFrame()

    df = df_events_flat.copy()

    period_offsets_min = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    period_offsets_sec = {p: m * 60 for p, m in period_offsets_min.items()}

    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["second"] = pd.to_numeric(df["second"], errors="coerce")

    df["seconds_in_period"] = df["minute"] * 60.0 + df["second"]
    df["period_offset"] = df["period"].map(period_offsets_sec).fillna(0).astype(float)
    df["t_abs"] = df["period_offset"] + df["seconds_in_period"]
    df["_t"] = df["t_abs"]

    sort_cols = [c for c in ["match_id", "period", "minute", "second"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    t_min = df["t_abs"].min()
    t_max = df["t_abs"].max()
    print(
        f"attach_event_time: {len(df)} events, t_abs range {t_min:.1f}s–{t_max:.1f}s"
    )
    return df


def build_corners(df_events_time: pd.DataFrame, df_matches: pd.DataFrame) -> pd.DataFrame:
    # extract corner events and attach basic match context
    if df_events_time is None or df_events_time.empty:
        print("build_corners: received empty events dataframe")
        return pd.DataFrame()

    for col in ["type.name", "pass.type.name"]:
        if col not in df_events_time.columns:
            raise ValueError(f"build_corners: expected '{col}' in events dataframe")

    corners = df_events_time[
        (df_events_time["type.name"] == "Pass")
        & (df_events_time["pass.type.name"] == "Corner")
    ].copy()

    if corners.empty:
        print("build_corners: no corner events found")
        return pd.DataFrame()

    corners["corner_id"] = range(1, len(corners) + 1)
    corners["corner_uuid"] = corners["event_uuid"].astype(str)

    if "_t" not in corners.columns and "t_abs" in corners.columns:
        corners["_t"] = corners["t_abs"]
    corners["_t"] = corners["_t"].astype(float)

    def _side(loc):
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            try:
                return "left" if float(loc[1]) < 40.0 else "right"
            except Exception:
                return "unknown"
        return "unknown"

    if "location" in corners.columns:
        corners["corner_side"] = corners["location"].apply(_side)
    else:
        corners["corner_side"] = "unknown"

    if "team.id" in corners.columns:
        corners["attacking_team.id"] = corners["team.id"]
    else:
        corners["attacking_team.id"] = np.nan

    if df_matches is not None and not df_matches.empty:
        home_id = next(
            (c for c in df_matches.columns if "home_team" in c and "id" in c), None
        )
        away_id = next(
            (c for c in df_matches.columns if "away_team" in c and "id" in c), None
        )
        home_nm = next(
            (c for c in df_matches.columns if "home_team" in c and "name" in c), None
        )
        away_nm = next(
            (c for c in df_matches.columns if "away_team" in c and "name" in c), None
        )

        cols = ["match_id", "competition_id", "season_id"]
        for c in [home_id, away_id, home_nm, away_nm]:
            if c and c not in cols:
                cols.append(c)

        matches_sub = df_matches[cols].copy()
        merged = corners.merge(matches_sub, on="match_id", how="left")

        rename = {}
        if home_id:
            rename[home_id] = "home_team.id"
        if away_id:
            rename[away_id] = "away_team.id"
        if home_nm:
            rename[home_nm] = "home_team.name"
        if away_nm:
            rename[away_nm] = "away_team.name"
        merged.rename(columns=rename, inplace=True)

        merged["opponent_team.id"] = np.where(
            merged["attacking_team.id"] == merged.get("home_team.id"),
            merged.get("away_team.id"),
            merged.get("home_team.id"),
        )
        merged["opponent_team.name"] = np.where(
            merged["attacking_team.id"] == merged.get("home_team.id"),
            merged.get("away_team.name"),
            merged.get("home_team.name"),
        )
    else:
        merged = corners.copy()
        merged["competition_id"] = merged.get("competition_id")
        merged["season_id"] = merged.get("season_id")
        merged["home_team.id"] = np.nan
        merged["away_team.id"] = np.nan
        merged["home_team.name"] = np.nan
        merged["away_team.name"] = np.nan
        merged["opponent_team.id"] = np.nan
        merged["opponent_team.name"] = np.nan

    merged = merged.reset_index(drop=True)

    print(f"build_corners: {len(merged)} corners")
    return merged


def rescale_ff_x_to_statsbomb(df_ff: pd.DataFrame) -> pd.DataFrame:
    # rescale x to 0–120 if coordinates look compressed
    if df_ff is None or df_ff.empty:
        return df_ff
    if "x" not in df_ff.columns:
        return df_ff

    df = df_ff.copy()
    x = pd.to_numeric(df["x"], errors="coerce")
    max_abs_x = x.abs().max()

    if pd.isna(max_abs_x):
        return df

    if max_abs_x >= 60.0:
        return df

    if max_abs_x > 0:
        scale = 120.0 / float(max_abs_x)
        df["x"] = x * scale
        print(
            f"rescale_ff_x_to_statsbomb: compressed x (max={max_abs_x:.2f}), "
            f"scale={scale:.3f}"
        )
    else:
        print("rescale_ff_x_to_statsbomb: max_abs_x == 0, cannot rescale")

    return df


def normalize_360(
    df_360_raw: pd.DataFrame,
    min_players_in_frame: Optional[int] = None,
) -> pd.DataFrame:
    # clean and filter 360 freeze frames
    if df_360_raw is None or df_360_raw.empty:
        print("normalize_360: received empty 360 dataframe")
        return pd.DataFrame()

    df = df_360_raw.copy()

    if "match_id" not in df.columns or "event_uuid" not in df.columns:
        raise ValueError("normalize_360: expected 'match_id' and 'event_uuid' columns")

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    df["event_uuid"] = df["event_uuid"].astype(str)

    for col in ["x", "y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if min_players_in_frame is None:
        min_players_in_frame = MIN_PLAYERS_IN_FRAME

    df["n_players_frame"] = df.groupby(["match_id", "event_uuid"])["player.id"].transform(
        "size"
    )

    if "teammate" in df.columns:
        df["n_att_frame"] = df.groupby(["match_id", "event_uuid"])["teammate"].transform(
            lambda s: s.astype(bool).sum()
        )
        df["n_def_frame"] = df["n_players_frame"] - df["n_att_frame"]
    else:
        df["n_att_frame"] = pd.NA
        df["n_def_frame"] = pd.NA

    n_frames_before = df[["match_id", "event_uuid"]].drop_duplicates().shape[0]

    if "teammate" in df.columns:
        mask_quality = (
            (df["n_players_frame"] >= min_players_in_frame)
            & (df["n_att_frame"].between(MIN_ATTACKERS_IN_FRAME, 11))
            & (df["n_def_frame"].between(MIN_DEFENDERS_IN_FRAME, 11))
            & ((df["n_att_frame"] + df["n_def_frame"]) >= MIN_TOTAL_PLAYERS_IN_FRAME)
        )
    else:
        mask_quality = df["n_players_frame"] >= min_players_in_frame

    df_clean = df.loc[mask_quality].copy()
    n_frames_after = df_clean[["match_id", "event_uuid"]].drop_duplicates().shape[0]

    df_clean = rescale_ff_x_to_statsbomb(df_clean)

    print(
        "normalize_360: "
        f"frames_all={n_frames_before}, frames_kept={n_frames_after}, "
        f"rows_raw={len(df)}, rows_kept={len(df_clean)}"
    )
    return df_clean


def stage2_preprocess(
    df_events_raw: pd.DataFrame,
    df_matches: pd.DataFrame,
    df_360_raw: pd.DataFrame,
    min_players_in_frame: int = MIN_PLAYERS_IN_FRAME,
):
    # run stage 2 preprocessing and return events_time, corners, 360_clean
    df_events_flat = flatten_fields(df_events_raw)
    df_events_time = attach_event_time(df_events_flat)
    df_corners = build_corners(df_events_time, df_matches)
    df_360_clean = normalize_360(df_360_raw, min_players_in_frame=min_players_in_frame)

    print(
        "stage2_preprocess: "
        f"events_flat={len(df_events_flat)}, "
        f"corners={len(df_corners)}, "
        f"ff_clean={len(df_360_clean)}"
    )
    return df_events_time, df_corners, df_360_clean


def dataset_summary(df_events_time: pd.DataFrame, df_corners: pd.DataFrame) -> pd.DataFrame:
    # return a one-row summary of dataset counts
    n_events = 0 if df_events_time is None or df_events_time.empty else len(df_events_time)
    n_corners = 0 if df_corners is None or df_corners.empty else len(df_corners)

    if df_corners is None or df_corners.empty or "match_id" not in df_corners.columns:
        n_matches_with_corner = 0
    else:
        n_matches_with_corner = df_corners["match_id"].nunique()

    return pd.DataFrame(
        [
            {
                "n_events_total": n_events,
                "n_corners_total": n_corners,
                "n_matches_with_corner": n_matches_with_corner,
            }
        ]
    )


def plot_matches_and_corners_by_comp_season(
    df_matches: pd.DataFrame,
    df_corners: pd.DataFrame,
    out_dir: Optional[str] = None,
) -> None:
    # bar chart of matches and corners per competition-season
    if df_matches is None or df_matches.empty or df_corners is None or df_corners.empty:
        print("plot_matches_and_corners_by_comp_season: empty input, skipping")
        return

    required_match_cols = ["competition_id", "season_id", "match_id"]
    for col in required_match_cols:
        if col not in df_matches.columns:
            raise KeyError(f"df_matches is missing expected column '{col}'")

    for col in ["competition_id", "season_id", "corner_id"]:
        if col not in df_corners.columns:
            raise KeyError(f"df_corners is missing expected column '{col}'")

    matches_per_cs = (
        df_matches.groupby(["competition_id", "season_id"])["match_id"]
        .nunique()
        .reset_index(name="n_matches")
    )

    corners_per_cs = (
        df_corners.groupby(["competition_id", "season_id"])["corner_id"]
        .count()
        .reset_index(name="n_corners")
    )

    df_plot = matches_per_cs.merge(
        corners_per_cs,
        on=["competition_id", "season_id"],
        how="outer",
    ).fillna(0)

    df_plot["label"] = (
        df_plot["competition_id"].astype(str)
        + "-"
        + df_plot["season_id"].astype(str)
    )

    x = range(len(df_plot))
    width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar([p - width / 2 for p in x], df_plot["n_matches"], width=width, label="Matches")
    plt.bar([p + width / 2 for p in x], df_plot["n_corners"], width=width, label="Corners")
    plt.xticks(list(x), df_plot["label"], rotation=45, ha="right")
    plt.xlabel("Competition season")
    plt.ylabel("Count")
    plt.title("Matches and corners per competition season")
    plt.legend()
    plt.tight_layout()

    if out_dir is not None:
        import os

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            "dataset_matches_corners_by_comp_season.png",
        )
        plt.savefig(out_path, dpi=300)
        print(
            f"plot_matches_and_corners_by_comp_season: saved figure to {out_path}"
        )
        plt.close()
    else:
        plt.show()
