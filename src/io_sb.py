import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd

Config = dict[str, Any]


def _load_json(path: Path) -> Any:
    # read a json file; None if it fails
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {path}")
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {path}: {exc}")
    except Exception as exc:  # pragma: no cover
        print(f"Could not load JSON from {path}: {exc}")
    return None


def load_competitions(cfg: Config) -> pd.DataFrame:
    # the StatsBomb competitions table
    path = Path(cfg["paths"]["competitions"])
    data = _load_json(path)

    if not data:
        print(f"No competitions data loaded from {path}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def load_matches(cfg: Config, comp_seasons: Sequence[Sequence[int]]) -> pd.DataFrame:
    # match files for a list of (competition_id, season_id) pairs
    matches_root = Path(cfg["paths"]["matches"])
    all_matches: List[dict] = []

    print(f"Reading matches from {matches_root}")

    for comp_id, season_id in comp_seasons:
        match_file = matches_root / str(comp_id) / f"{season_id}.json"
        if not match_file.exists():
            print(f"Match file missing: {match_file}")
            continue

        data = _load_json(match_file)
        if not data:
            print(f"Empty or invalid match file: {match_file}")
            continue

        for match in data:
            match["competition_id"] = comp_id
            match["season_id"] = season_id

        all_matches.extend(data)
        print(
            f"Loaded {len(data)} matches for competition {comp_id}, season {season_id}"
        )

    if not all_matches:
        print("No match data loaded.")
        return pd.DataFrame()

    df = pd.json_normalize(all_matches)

    # make sure the column is called match_id
    if "match_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "match_id"})

    return df


def load_events_by_ids(cfg: Config, match_ids: Iterable[int]) -> pd.DataFrame:
    # event files for the given match ids
    root = Path(cfg["paths"]["events"])
    frames: List[pd.DataFrame] = []

    for mid in match_ids:
        path = root / f"{int(mid)}.json"
        if not path.exists():
            # skip missing matches
            continue

        data = _load_json(path)
        if not data:
            continue

        for ev in data:
            # every event needs a string event_uuid
            if "event_uuid" not in ev:
                ev["event_uuid"] = str(ev.get("id"))
            else:
                ev["event_uuid"] = str(ev["event_uuid"])
            ev["match_id"] = int(mid)

        df = pd.json_normalize(data)
        frames.append(df)

    if not frames:
        print("No events loaded for the given match_ids.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    if "id" in df_all.columns:
        df_all["id"] = df_all["id"].astype(str)
    if "event_uuid" in df_all.columns:
        df_all["event_uuid"] = df_all["event_uuid"].astype(str)

    return df_all


def load_360_by_ids(cfg: Config, match_ids: Iterable[int]) -> pd.DataFrame:
    # 360 freeze frames for the given match ids
    base = Path(cfg["paths"]["three_sixty"])
    frames: List[pd.DataFrame] = []

    match_ids_clean = sorted({int(mid) for mid in match_ids})
    print(f"Reading 360 data from {base}")

    for mid in match_ids_clean:
        path = base / f"{mid}.json"
        if not path.exists():
            print(f"360 file missing for match_id={mid}: {path}")
            continue

        data = _load_json(path)
        if data is None:
            print(f"Could not parse 360 JSON for match_id={mid}: {path}")
            continue

        if isinstance(data, list):
            frames_json = data
        else:
            frames_json = data.get("frames") or data.get("freeze_frames") or []

        rows: List[dict] = []

        for frame in frames_json:
            if not isinstance(frame, dict):
                continue

            ev_uuid = (
                frame.get("event_uuid")
                or frame.get("id")
                or frame.get("event_id")
            )
            if ev_uuid is None:
                continue

            freeze_list = frame.get("freeze_frame") or []
            if not isinstance(freeze_list, list):
                continue

            for player_obj in freeze_list:
                if not isinstance(player_obj, dict):
                    continue

                teammate_raw = player_obj.get("teammate")
                teammate_flag = bool(teammate_raw) if teammate_raw is not None else False

                keeper_raw = player_obj.get("keeper")
                keeper_flag = bool(keeper_raw) if keeper_raw is not None else False

                player = (
                    player_obj.get("player")
                    if isinstance(player_obj.get("player"), dict)
                    else {}
                )
                pid = player.get("id")
                pname = player.get("name")

                loc = player_obj.get("location") or [None, None]
                x = loc[0] if len(loc) >= 1 else None
                y = loc[1] if len(loc) >= 2 else None

                rows.append(
                    {
                        "event_uuid": str(ev_uuid),
                        "match_id": int(mid),
                        "teammate": teammate_flag,
                        "keeper": keeper_flag,
                        "player.id": pid,
                        "player.name": pname,
                        "x": x,
                        "y": y,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            df["event_uuid"] = df["event_uuid"].astype(str)
            df["match_id"] = df["match_id"].astype(int)
            frames.append(df)

    if not frames:
        print("No 360 data loaded for the given match_ids.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def load_lineups_by_ids(cfg: Config, match_ids: Iterable[int]) -> pd.DataFrame:
    # lineup files for the given match ids
    root = Path(cfg["paths"]["lineups"])
    frames: List[pd.DataFrame] = []

    for mid in match_ids:
        path = root / f"{int(mid)}.json"
        if not path.exists():
            continue

        data = _load_json(path)
        if not data:
            continue

        df = pd.DataFrame(data)
        df["match_id"] = int(mid)
        frames.append(df)

    if not frames:
        print("No lineups loaded for the given match_ids.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    return df_all
