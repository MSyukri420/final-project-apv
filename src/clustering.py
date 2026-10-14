from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.constants import DEFENSIVE_FEATURES, SHAPE_ORDER


# helper: softmax with temperature

def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # softmax over the last axis, with a temperature
    if temperature <= 0:
        temperature = 1.0

    z = scores / float(temperature)
    # subtract the max to avoid overflow
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_z / denom


# helper: readable names for the clusters

def _assign_cluster_labels(centroids: pd.DataFrame) -> Dict[int, str]:
    # name each cluster from its centroid
    cols = set(centroids.columns)
    labels_default = {idx: f"Cluster_{idx}" for idx in centroids.index}

    if "def_in_6yd" not in cols:
        return labels_default

    # zonal: most players in the six-yard box, plus anyone on the posts
    zonal_score = centroids["def_in_6yd"].copy()
    if "players_on_posts" in cols:
        zonal_score = zonal_score + 0.5 * centroids["players_on_posts"]
    idx_zonal = int(zonal_score.idxmax())

    # man: high line and high marking ratio, if available
    if "marking_ratio_pen" in cols and "def_line_height" in cols:
        man_score = centroids["marking_ratio_pen"] + 0.3 * centroids["def_line_height"]
    elif "marking_ratio_pen" in cols:
        man_score = centroids["marking_ratio_pen"].copy()
    elif "def_line_height" in cols:
        man_score = centroids["def_line_height"].copy()
    else:
        return labels_default

    # don't pick the same cluster twice
    man_score = man_score.copy()
    man_score.loc[idx_zonal] = man_score.min() - 1.0
    idx_man = int(man_score.idxmax())

    # hybrid: the cluster that's left
    remaining = [idx for idx in centroids.index if idx not in (idx_zonal, idx_man)]
    if not remaining:
        return labels_default
    idx_hybrid = int(remaining[0])

    labels: Dict[int, str] = {}
    for idx in centroids.index:
        if idx == idx_zonal:
            labels[idx] = "Zonal"
        elif idx == idx_man:
            labels[idx] = "Man"
        elif idx == idx_hybrid:
            labels[idx] = "Hybrid"
        else:
            labels[idx] = f"Cluster_{idx}"

    return labels


# k-means defensive clusters (stage 4), kept as the benchmark for the rules

@overload
def cluster_defensive_archetypes(
    df_features: pd.DataFrame,
    cfg: Optional[dict] = ...,
    return_model: Literal[False] = ...,
) -> pd.DataFrame: ...


@overload
def cluster_defensive_archetypes(
    df_features: pd.DataFrame,
    cfg: Optional[dict] = ...,
    *,
    return_model: Literal[True],
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, List[str], Dict[int, str]]: ...


def cluster_defensive_archetypes(
    df_features: pd.DataFrame,
    cfg: Optional[dict] = None,
    return_model: bool = False,
):
    # cluster defensive shapes with k-means and attach the labels
    if df_features is None or df_features.empty:
        raise ValueError("cluster_defensive_archetypes: df_features is empty.")

    df = df_features.copy()

    # settings
    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    k = int(clustering_cfg.get("k", 3))
    max_iter = int(clustering_cfg.get("max_iter", 500))
    n_init = int(clustering_cfg.get("n_init", 20))
    random_state = int(clustering_cfg.get("random_state", 15))
    temperature = float(clustering_cfg.get("temperature", 0.3))

    # features
    feat_names = [f for f in DEFENSIVE_FEATURES if f in df.columns]
    if not feat_names:
        raise ValueError(
            "cluster_defensive_archetypes: no defensive features found in df."
        )

    X = df[feat_names].copy()

    # drop all-nan columns
    X = X.loc[:, X.notna().any(axis=0)]

    # drop near-constant columns
    stds = X.std(axis=0, ddof=0)
    used_features = list(stds[stds > 1e-6].index)
    if not used_features:
        raise ValueError(
            "cluster_defensive_archetypes: all defensive features are near constant."
        )

    # fill gaps with the median, then scale
    medians = X[used_features].median(axis=0)
    X_imputed = X[used_features].fillna(medians)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # kmeans clustering
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    hard_labels = kmeans.fit_predict(X_scaled)

    # confidence: softmax over the distances to each centroid
    dists = kmeans.transform(X_scaled)
    scores = -dists
    probs = _softmax(scores, temperature=temperature)
    confidence = probs.max(axis=1)
    hard_labels_soft = probs.argmax(axis=1)

    # keep the plain k-means label if the two disagree
    if not np.array_equal(hard_labels, hard_labels_soft):
        hard_final = hard_labels
    else:
        hard_final = hard_labels_soft

    # attach the cluster columns
    df["A_id"] = hard_final
    df["A_confidence"] = confidence

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=used_features)
    cluster_label_map = _assign_cluster_labels(centroids)
    df["A_name"] = df["A_id"].map(cluster_label_map).fillna("Unknown")

    # print a summary
    print("------------------------------------------------------")
    print("stage 4: defensive archetype clustering")
    print("------------------------------------------------------")
    print(f"samples clustered:               {df.shape[0]:>7}")
    print(f"number of clusters (k):          {k}")
    print(f"features used ({len(used_features)}): {', '.join(used_features)}")
    print(f"softmax temperature:             {temperature:.3f}")
    print("")
    print("A_confidence summary:")
    print(f"  mean:  {float(confidence.mean()):.3f}")
    print(f"  std:   {float(confidence.std()):.3f}")
    print(f"  min:   {float(confidence.min()):.3f}")
    print(f"  max:   {float(confidence.max()):.3f}")
    print("")
    print("cluster sizes (by A_id):")
    print(df["A_id"].value_counts().sort_index())
    print("")
    print("cluster label mapping:")
    for cid, name in cluster_label_map.items():
        print(f"  {cid} -> {name}")
    print("")

    if return_model:
        return df, kmeans, scaler, used_features, cluster_label_map

    return df


# rule-based defensive shapes (stage 4)
#
# k-means finds weak, unstable groups here, and one freeze frame can't show who
# marks whom. Two simple rules give almost the same groups, don't change when
# new corners are added, and can be checked by eye:
#   1. anyone on a post                                -> Posts covered
#   2. else box defenders within high_line_yards of goal -> Deep line
#   3. else                                            -> High line

def _shape_rules(clustering_cfg: dict) -> Tuple[int, float]:
    min_on_posts = int(clustering_cfg.get("min_on_posts", 1))
    high_line_yards = float(clustering_cfg.get("high_line_yards", 7.5))
    return min_on_posts, high_line_yards


def assign_defensive_shapes(df_features: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    # give every corner a rule-based defensive shape (A_name)
    if df_features is None or df_features.empty:
        raise ValueError("assign_defensive_shapes: df_features is empty.")
    missing = [c for c in ("players_on_posts", "def_line_height") if c not in df_features.columns]
    if missing:
        raise KeyError(f"assign_defensive_shapes: missing columns {missing}")

    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    min_on_posts, high_line_yards = _shape_rules(clustering_cfg)

    df = df_features.copy()
    on_posts = df["players_on_posts"].fillna(0) >= min_on_posts
    # no defenders in the box (no line height) counts as deep
    deep = ~(df["def_line_height"] > high_line_yards)
    names = np.where(on_posts, SHAPE_ORDER[0], np.where(deep, SHAPE_ORDER[1], SHAPE_ORDER[2]))

    df["A_name"] = names
    df["A_id"] = pd.Series(names, index=df.index).map({n: i for i, n in enumerate(SHAPE_ORDER)})
    # rules leave no doubt, so every label is certain
    df["A_confidence"] = 1.0

    print("------------------------------------------------------")
    print("stage 4: defensive shapes (rules)")
    print("------------------------------------------------------")
    print(f"posts covered: {min_on_posts}+ defender(s) within the post radius")
    print(f"deep line:     box defenders on average <= {high_line_yards:g} yards from goal")
    print(df["A_name"].value_counts().reindex(SHAPE_ORDER).to_string())
    print("")
    return df


def compare_shape_methods(
    df_rules: pd.DataFrame,
    df_kmeans: pd.DataFrame,
    used_features: List[str],
    scaler: StandardScaler,
    cfg: Optional[dict] = None,
    n_boot: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    # compare the rules with k-means: how well each separates the defences, and how
    # stable each is when the data is resampled
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score

    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    k = int(clustering_cfg.get("k", 3))
    min_on_posts, _ = _shape_rules(clustering_cfg)

    X = df_kmeans[used_features].copy()
    X_scaled = scaler.transform(X.fillna(X.median()))
    rng = np.random.default_rng(seed)
    n = len(df_kmeans)

    labels_k = df_kmeans["A_id"].to_numpy()
    labels_r = df_rules["A_id"].to_numpy()
    no_post = (df_rules["players_on_posts"].fillna(0) < min_on_posts).to_numpy()
    line = df_rules["def_line_height"].to_numpy()

    stab_k: List[float] = []
    stab_r: List[float] = []
    for i in range(n_boot):
        idx = rng.choice(n, n)
        refit = KMeans(n_clusters=k, n_init=10, random_state=i).fit(X_scaled[idx])
        stab_k.append(adjusted_rand_score(labels_k, refit.predict(X_scaled)))
        # redo the rules on the resample: line threshold at its no-post median
        threshold = float(np.nanmedian(line[idx][no_post[idx]]))
        relabel = np.where(~no_post, 0, np.where(~(line > threshold), 1, 2))
        stab_r.append(adjusted_rand_score(labels_r, relabel))

    def row(method: str, labels: np.ndarray, names: pd.Series, stab: List[float]) -> Dict:
        sizes = names.value_counts()
        return {
            "method": method,
            "groups": int(sizes.size),
            "sizes": "; ".join(f"{k_}={v}" for k_, v in sizes.items()),
            "silhouette": float(silhouette_score(X_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            "stability_ari_mean": float(np.mean(stab)),
            "stability_ari_min": float(np.min(stab)),
        }

    out = pd.DataFrame(
        [
            row("rules", labels_r, df_rules["A_name"], stab_r),
            row("kmeans", labels_k, df_kmeans["A_name"], stab_k),
        ]
    )
    out["agreement_ari"] = float(adjusted_rand_score(labels_r, labels_k))
    return out


# figures for the report

def plot_cluster_counts(
    df_clustered: pd.DataFrame,
    out_dir: Optional[str] = None,
    prefix: str = "cluster_counts",
):
    # bar chart of how many corners each shape has, plus a csv
    if df_clustered is None or df_clustered.empty:
        print("plot_cluster_counts: empty df_clustered, skipping")
        return None

    if "A_name" not in df_clustered.columns:
        raise KeyError("plot_cluster_counts: expected column 'A_name'")

    counts_all = df_clustered["A_name"].value_counts().sort_index()
    x = np.arange(len(counts_all))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, counts_all.values, width=0.6, label="all")
    ax.set_xticks(x)
    ax.set_xticklabels(counts_all.index, rotation=30, ha="right")
    ax.set_ylabel("number of corners")
    ax.set_title("cluster sizes")
    ax.legend()
    fig.tight_layout()

    summary = pd.DataFrame(
        {
            "cluster": counts_all.index,
            "count": counts_all.values,
        }
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"{prefix}.png"
        csv_path = out_dir / f"{prefix}_summary.csv"
        fig.savefig(fig_path, dpi=300)
        summary.to_csv(csv_path, index=False)
        print(f"saved cluster count figure to {fig_path}")
        print(f"saved cluster count summary to {csv_path}")
        plt.close(fig)

    return summary


def plot_cluster_feature_profiles(
    df_clustered: pd.DataFrame,
    used_features: List[str],
    out_dir: Optional[str] = None,
    prefix: str = "cluster_profiles",
):
    # average defensive features per shape, plus attacker counts by zone
    if df_clustered is None or df_clustered.empty:
        print("plot_cluster_feature_profiles: empty df_clustered, skipping")
        return None

    if "A_name" not in df_clustered.columns:
        raise KeyError("plot_cluster_feature_profiles: expected column 'A_name'")

    df = df_clustered.copy()

    # a coach-friendly order where possible
    archetype_order = SHAPE_ORDER
    present = [a for a in archetype_order if a in df["A_name"].unique()]
    if not present:
        present = sorted(df["A_name"].unique())
    archetypes = present

    # defensive features to show
    defensive_features = [
        "def_line_height",
        "def_in_6yd",
        "gk_start_line",
        "players_on_posts",
    ]
    defensive_features = [f for f in defensive_features if f in df.columns]

    # attacker counts by zone
    attacker_zone_features = [
        "n_att_near",
        "n_att_central",
        "n_att_far",
    ]
    attacker_zone_features = [f for f in attacker_zone_features if f in df.columns]

    if not defensive_features and not attacker_zone_features:
        print("plot_cluster_feature_profiles: no required features present, skipping")
        return None

    cols_for_mean = defensive_features + attacker_zone_features
    df_mean = (
        df.groupby("A_name")[cols_for_mean]
        .mean(numeric_only=True)
        .loc[archetypes]
    )

    # defensive features
    if defensive_features:
        n_plots = min(len(defensive_features), 4)
        fig_def, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.flatten()

        pretty_def_names = {
            "def_line_height": "Defensive line height",
            "def_in_6yd": "Defenders in six-yard box",
            "gk_start_line": "Goalkeeper starting line",
            "players_on_posts": "Players on posts",
        }

        for idx in range(4):
            ax = axes[idx]
            if idx < n_plots:
                feat = defensive_features[idx]
                means = df_mean[feat].reindex(archetypes)
                ax.bar(archetypes, means.values)
                ax.set_title(pretty_def_names.get(feat, feat))
                ax.set_ylabel("mean value")
                ax.tick_params(axis="x", rotation=30)
            else:
                ax.axis("off")

        fig_def.suptitle("defensive feature profiles by archetype", y=0.98)
        fig_def.tight_layout(rect=[0, 0, 1, 0.95])

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path_def = out_dir / f"{prefix}.png"
            fig_def.savefig(fig_path_def, dpi=300)
            print(f"saved defensive cluster profile figure to {fig_path_def}")
            plt.close(fig_def)

    # attacker counts by zone
    if attacker_zone_features:
        zones = attacker_zone_features
        zone_labels = {
            "n_att_near": "Near",
            "n_att_central": "Central",
            "n_att_far": "Far",
        }
        x = np.arange(len(zones))
        n_arch = len(archetypes)
        width = 0.8 / max(n_arch, 1)

        fig_att, ax_att = plt.subplots(figsize=(8, 3))

        for i, archetype in enumerate(archetypes):
            heights = df_mean.loc[archetype, zones].values
            offset = (i - (n_arch - 1) / 2.0) * width
            ax_att.bar(x + offset, heights, width, label=archetype)

        ax_att.set_xticks(x)
        ax_att.set_xticklabels([zone_labels.get(z, z) for z in zones])
        ax_att.set_ylabel("mean attacker count")
        ax_att.set_title("attacker counts by zone")
        ax_att.legend(title="archetype")
        fig_att.tight_layout()

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path_att = out_dir / f"{prefix}_attacker_zones.png"
            fig_att.savefig(fig_path_att, dpi=300)
            print(f"saved attacker zone figure to {fig_path_att}")
            plt.close(fig_att)

    return df_mean



def plot_cluster_silhouette(
    df_clustered: pd.DataFrame,
    scaler: StandardScaler,
    used_features: List[str],
    out_dir: Optional[str] = None,
    prefix: str = "cluster_silhouette",
):
    # average silhouette score per cluster, on the scaled defensive features
    if df_clustered is None or df_clustered.empty:
        print("plot_cluster_silhouette: empty df_clustered, skipping")
        return None

    if "A_id" not in df_clustered.columns:
        raise KeyError("plot_cluster_silhouette: expected column 'A_id'")

    if not used_features:
        print("plot_cluster_silhouette: used_features is empty, skipping")
        return None

    df_sub = df_clustered.copy()
    if df_sub.empty:
        print("plot_cluster_silhouette: no rows after filtering, skipping")
        return None

    # keep only features that still exist
    used_features = [f for f in used_features if f in df_sub.columns]
    if not used_features:
        print(
            "plot_cluster_silhouette: none of used_features are in df_clustered, "
            "skipping"
        )
        return None

    X = df_sub[used_features].copy()
    medians = X.median(axis=0)
    X_imputed = X.fillna(medians)
    X_scaled = scaler.transform(X_imputed)
    labels = df_sub["A_id"].to_numpy()

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        print("plot_cluster_silhouette: fewer than 2 clusters present, skipping")
        return None

    sil_values = silhouette_samples(X_scaled, labels)
    sil_mean = silhouette_score(X_scaled, labels)

    df_summary = (
        pd.DataFrame({"A_id": labels, "silhouette": sil_values})
        .groupby("A_id")["silhouette"]
        .mean()
        .rename("silhouette_mean")
        .reset_index()
        .sort_values("A_id")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_summary["A_id"].astype(str), df_summary["silhouette_mean"].values)
    ax.axhline(sil_mean, linestyle="--")
    ax.set_xlabel("cluster id")
    ax.set_ylabel("mean silhouette")
    ax.set_title("silhouette score per cluster")
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"{prefix}.png"
        csv_path = out_dir / f"{prefix}_summary.csv"
        fig.savefig(fig_path, dpi=300)
        df_summary.to_csv(csv_path, index=False)
        print(f"saved silhouette figure to {fig_path}")
        print(f"saved silhouette summary to {csv_path}")
        plt.close(fig)

    return df_summary
