from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

CONFIG_PATH = "config.yaml"
TRAIN_SPLIT = "train"


def load_config(path: str = CONFIG_PATH) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_comp_seasons(cfg: dict, split: str) -> list[tuple[int, int]]:
    cs_cfg = cfg.get("dataset", {}).get("comp_seasons", {}) or {}
    split_lower = split.lower()
    if split_lower == "train":
        comp_seasons = cs_cfg.get("train", [])
    elif split_lower == "test":
        comp_seasons = cs_cfg.get("test", [])
    elif split_lower == "all":
        comp_seasons = (cs_cfg.get("train", []) or []) + (cs_cfg.get("test", []) or [])
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'test', or 'all'.")
    return [(int(pair[0]), int(pair[1])) for pair in comp_seasons]


def main(models_only: bool = False) -> None:
    try:
        from src.io_sb import (
            load_competitions,
            load_matches,
            load_events_by_ids,
            load_360_by_ids,
            load_lineups_by_ids,
        )
        from src.preprocess import (
            stage2_preprocess,
            dataset_summary,
            plot_matches_and_corners_by_comp_season,
        )
        from src.feat_eng import (
            build_defensive_features,
            plot_attacking_direction_pitch,
            plot_zone_definitions_pitch,
            plot_archetype_example_freeze_frames,
        )
        from src.clustering import (
            assign_defensive_shapes,
            cluster_defensive_archetypes,
            compare_shape_methods,
            plot_cluster_counts,
            plot_cluster_feature_profiles,
            plot_cluster_silhouette,
        )
        from src.target_eng import stage5_target_engineering, plot_label_balance
        from src.models import stage6_model_training
        from src.apv import stage7_compute_apv
        import src.insights as insights
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import pipeline modules from src/. "
            "Run train.py from the project root with src/ on PYTHONPATH."
        ) from e

    cfg = load_config(CONFIG_PATH)
    paths_cfg = cfg.get("paths", {}) or {}
    outputs_root = Path(paths_cfg.get("outputs", "outputs"))
    reports_root = Path(paths_cfg.get("reports", "reports"))
    outputs_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    split = TRAIN_SPLIT.lower()
    split_outputs_dir = outputs_root / split
    split_reports_dir = reports_root / split
    split_outputs_dir.mkdir(parents=True, exist_ok=True)
    split_reports_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = split_reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # competition names for the per-competition report
    df_comp = load_competitions(cfg)

    if models_only:
        df_target_path = split_outputs_dir / "df_target_train.csv"
        if not df_target_path.exists():
            raise FileNotFoundError(
                f"{df_target_path} not found. Run the full pipeline (python train.py) once first."
            )
        print(f"\n=== Stages 1-5 skipped (--models-only): using {df_target_path} ===")
        df_master = pd.read_csv(df_target_path, low_memory=False)
        # redo the defensive shapes, so changing the rules needs no full rerun
        df_master = assign_defensive_shapes(df_master, cfg)
        df_master.to_csv(df_target_path, index=False)
    else:
        comp_seasons = _resolve_comp_seasons(cfg, TRAIN_SPLIT)
        print(f"\n=== Stage 1: data ingestion ({TRAIN_SPLIT.upper()}) ===")
        print(f"Competition seasons: {comp_seasons}")

        df_matches = load_matches(cfg, comp_seasons)

        if df_matches is None or df_matches.empty:
            raise RuntimeError("No matches found for TRAIN split.")

        match_ids = sorted(df_matches["match_id"].unique().tolist())
        print(f"Matches loaded: {len(match_ids)}")

        df_events_raw = load_events_by_ids(cfg, match_ids)
        df_360_raw = load_360_by_ids(cfg, match_ids)
        df_lineups = load_lineups_by_ids(cfg, match_ids)

        n_events_raw = 0 if df_events_raw is None else len(df_events_raw)
        n_360_raw = 0 if df_360_raw is None else len(df_360_raw)
        n_lineups = 0 if df_lineups is None else len(df_lineups)
        print(f"Events rows:  {n_events_raw}")
        print(f"360 frames:   {n_360_raw}")
        print(f"Lineups rows: {n_lineups}")

        print("\n=== Stage 2: preprocessing ===")

        df_events_time, df_corners, df_360_clean = stage2_preprocess(
            df_events_raw=df_events_raw,
            df_matches=df_matches,
            df_360_raw=df_360_raw,
        )

        df_ds_summary = dataset_summary(df_events_time, df_corners)
        ds_summary_path = split_reports_dir / "dataset_summary_train.csv"
        df_ds_summary.to_csv(ds_summary_path, index=False)
        print(f"Dataset summary saved to {ds_summary_path}")

        plot_matches_and_corners_by_comp_season(
            df_matches=df_matches,
            df_corners=df_corners,
            out_dir=str(figures_dir),
        )
        print(f"Match and corner figures saved to {figures_dir}")

        print("\n=== Stage 3: feature engineering ===")

        df_features = build_defensive_features(
            df_corners=df_corners,
            df_360_clean=df_360_clean,
            df_events_time=df_events_time,
        )

        atk_cols = [
            "n_att_near",
            "n_att_central",
            "n_att_far",
            "n_blockers",
            "n_edge_of_box",
            "n_rest_defence",
        ]
        if all(col in df_features.columns for col in atk_cols):
            sums = df_features[atk_cols].sum(axis=1)
            print(
                "Attacking allocation sanity check "
                f"(min={sums.min()}, max={sums.max()}, mean={sums.mean():.2f})"
            )

        attacking_fig_path = figures_dir / "attacking_direction_pitch.png"
        plot_attacking_direction_pitch(out_path=attacking_fig_path)

        zones_fig_path = figures_dir / "zone_definitions_pitch.png"
        plot_zone_definitions_pitch(out_path=zones_fig_path)

        print("Pitch diagrams saved.")

        print("\n=== Stage 4: defensive shapes ===")

        # k-means only as the benchmark for the rules
        df_kmeans, kmeans, scaler, used_features, cluster_label_map = (
            cluster_defensive_archetypes(
                df_features=df_features,
                cfg=cfg,
                return_model=True,
            )
        )
        df_clustered = assign_defensive_shapes(df_features, cfg)

        print(f"Labelled corners: {len(df_clustered)}")

        shape_eval_df = compare_shape_methods(df_clustered, df_kmeans, used_features, scaler, cfg)
        cluster_eval_path = split_reports_dir / "cluster_eval_train.csv"
        shape_eval_df.to_csv(cluster_eval_path, index=False)
        print(shape_eval_df.to_string(index=False))
        print(f"Shape method comparison saved to {cluster_eval_path}")

        cluster_fig_dir = figures_dir / "clustering"
        cluster_fig_dir.mkdir(parents=True, exist_ok=True)

        plot_cluster_counts(df_clustered=df_clustered, out_dir=str(cluster_fig_dir))
        plot_cluster_feature_profiles(
            df_clustered=df_clustered,
            used_features=used_features,
            out_dir=str(cluster_fig_dir),
        )
        plot_cluster_silhouette(
            df_clustered=df_clustered,
            scaler=scaler,
            used_features=used_features,
            out_dir=str(cluster_fig_dir),
        )
        plot_archetype_example_freeze_frames(
            df_corners=df_corners,
            df_360_clean=df_360_clean,
            df_clustered=df_clustered,
            out_dir=str(cluster_fig_dir),
            prefix="archetype_example_freeze_frames",
        )
        print(f"Clustering figures saved to {cluster_fig_dir}")

        print("\n=== Stage 5: target engineering ===")

        df_target_path = split_outputs_dir / "df_target_train.csv"
        df_target = stage5_target_engineering(
            df_events_time=df_events_time,
            df_features_clustered=df_clustered,
            cfg=cfg,
            out_path=str(df_target_path),
        )

        print(f"Target rows after filtering: {len(df_target)}")
        print(f"Target table saved to {df_target_path}")

        label_fig_dir = figures_dir / "labels"
        label_fig_dir.mkdir(parents=True, exist_ok=True)
        plot_label_balance(df_target=df_target, out_dir=str(label_fig_dir))
        print(f"Label balance figures saved to {label_fig_dir}")

        # a sample of first-contact labels with the events behind them, to check by hand
        check_cols = [
            "match_id", "minute", "second", "delivery_route", "pass.outcome.name",
            "first_contact_attack", "first_contact_type", "first_contact_events",
        ]
        if "first_contact_attack" in df_target.columns:
            check = df_target[[c for c in check_cols if c in df_target.columns]]
            check_path = split_reports_dir / "first_contact_labels_check.csv"
            check.sample(n=min(40, len(check)), random_state=15).to_csv(check_path, index=False)
            print(f"First-contact label sample saved to {check_path}")

        df_master = df_target.copy()

    print("\n=== Stage 6: model training (TRAIN pooled models) ===")

    model_fig_dir = figures_dir / "models"
    model_fig_dir.mkdir(parents=True, exist_ok=True)

    comp_names: Dict[str, str] = {}
    if df_comp is not None and {"competition_id", "season_id"}.issubset(df_comp.columns):
        for rec in df_comp.to_dict("records"):
            key = f"{rec['competition_id']}-{rec['season_id']}"
            comp_names[key] = f"{rec.get('competition_name', '')} {rec.get('season_name', '')}".strip()

    model_reports: Dict[str, pd.DataFrame] = stage6_model_training(
        df_target=df_master,
        cfg=cfg,
        out_dir=str(model_fig_dir),
        holdout_names=comp_names,
    )

    for name, table in model_reports.items():
        if table is None or table.empty:
            continue
        report_path = split_reports_dir / f"model_{name}_train.csv"
        table.to_csv(report_path, index=False)
        print(f"Model {name.replace('_', ' ')} saved to {report_path}")

    print("\n=== Stage 7: APV computation (TRAIN) ===")

    df_apv, apv_summary = stage7_compute_apv(
        df_target=df_master,
        cfg=cfg,
        models_dir=None,
        out_dir=split_outputs_dir,
        split=TRAIN_SPLIT,
        save=True,
    )

    print(f"APV rows (TRAIN): {len(df_apv)}")
    if apv_summary is not None and not apv_summary.empty:
        apv_summary_path = split_reports_dir / "apv_summary_train.csv"
        apv_summary.to_csv(apv_summary_path, index=False)
        print(f"APV summary saved to {apv_summary_path}")
    else:
        print("No APV summary table produced.")

    apv_fig_dir = figures_dir / "apv"
    apv_fig_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(insights, "plot_apv_histogram"):
        insights.plot_apv_histogram(
            df_apv=df_apv,
            out_path=apv_fig_dir / "apv_histogram.png",
        )
    if hasattr(insights, "plot_apv_by_archetype_boxplot"):
        insights.plot_apv_by_archetype_boxplot(
            df_apv=df_apv,
            out_path=apv_fig_dir / "apv_by_archetype_boxplot.png",
        )
    print(f"APV figures saved to {apv_fig_dir}")

    print("\n=== Stage 8: build runtime artefacts for app ===")

    try:
        artefact_paths = insights.build_runtime_artefacts(cfg)
        print(f"Runtime artefacts built ({len(artefact_paths)} files).")
    except Exception:
        print("Failed to build runtime artefacts. Raising error.")
        raise

    print("\nTRAIN pipeline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the corner APV models and build app artefacts.")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="skip data loading and feature/target engineering; retrain from the saved target table",
    )
    main(models_only=parser.parse_args().models_only)
