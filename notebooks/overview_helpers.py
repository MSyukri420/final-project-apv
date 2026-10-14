"""Data loading, rate tables and charts for the stakeholder overview notebook."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.target_eng import add_chain_targets  # noqa: E402

# chart colours: background, text, one hue for rates, a second for "before"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
BLUE = "#2a78d6"  # the current system, or the measured rate
ORANGE = "#eb6834"  # the original system

Z90 = 1.645
COIN_FLIP = "0.5\ncoin flip"  # tick label for the no-skill line

COMPETITIONS: Dict[int, Tuple[str, str, str]] = {
    # id: (name, men's or women's, club or international)
    43: ("FIFA World Cup", "Men's", "International"),
    55: ("UEFA Euro", "Men's", "International"),
    9: ("Bundesliga", "Men's", "Club"),
    7: ("Ligue 1", "Men's", "Club"),
    11: ("La Liga", "Men's", "Club"),
    72: ("FIFA Women's World Cup", "Women's", "International"),
    53: ("UEFA Women's Euro", "Women's", "International"),
}

# the two steps of a corner and the overall result
CHAIN: List[Tuple[str, str]] = [
    ("first_contact_attack", "1 · First ball won"),
    ("chance_if_won", "2 · Good chance after winning it"),
    ("chance_for", "= Good chance overall"),
]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK_3,
        "ytick.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.dpi": 110,
    })


# ------------------------------------------------------------------ data

def load_corners() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs" / "train" / "df_target_train.csv", low_memory=False)
    df = add_chain_targets(df)
    info = df["competition_id"].map(COMPETITIONS)
    df["competition"] = info.map(lambda t: t[0] if isinstance(t, tuple) else "Other")
    df["football"] = info.map(lambda t: t[1] if isinstance(t, tuple) else "Other")
    df["level"] = info.map(lambda t: t[2] if isinstance(t, tuple) else "Other")
    df["route"] = np.where(df["short_corner_flag"] == 1, "short", df["delivery_type"])
    return df


def _reports_dir() -> Path:
    # where train.py writes its evaluation tables (paths.reports in config.yaml)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    return ROOT / (cfg.get("paths", {}) or {}).get("reports", "reports") / "train"


def load_evaluation() -> pd.DataFrame:
    return pd.read_csv(_reports_dir() / "model_evaluation_train.csv")


def load_per_competition() -> pd.DataFrame:
    return pd.read_csv(_reports_dir() / "model_per_competition_train.csv")


def wilson(k: float, n: float) -> Tuple[float, float, float]:
    # rate with a 90% Wilson range
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + Z90**2 / n
    centre = (p + Z90**2 / (2 * n)) / denom
    half = Z90 * math.sqrt(p * (1 - p) / n + Z90**2 / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rate_table(df: pd.DataFrame, by: str, measures: Sequence[str], order: Optional[Dict] = None) -> pd.DataFrame:
    # one row per group and measure: rate, 90% range and counts
    rows = []
    groups = order.items() if order else [(v, str(v)) for v in sorted(df[by].dropna().unique())]
    for value, label in groups:
        part = df[df[by] == value]
        for m in measures:
            y = part[m].dropna()
            p, lo, hi = wilson(y.sum(), len(y))
            rows.append({"group": label, "measure": m, "rate": p, "low": lo, "high": hi, "k": int(y.sum()), "n": len(y)})
    return pd.DataFrame(rows)


def readable(table: pd.DataFrame, names: Dict[str, str]) -> pd.DataFrame:
    # a readable table: "38% (89 of 233)" per group and measure
    t = table.copy()
    t["cell"] = [f"{r:.0%} ({k} of {n})" if n else "–" for r, k, n in zip(t["rate"], t["k"], t["n"])]
    out = t.pivot(index="group", columns="measure", values="cell")
    out = out.reindex(index=list(dict.fromkeys(t["group"])), columns=[m for m in names if m in out.columns])
    return out.rename(columns=names).rename_axis(None, axis=0).rename_axis(None, axis=1)


# ------------------------------------------------------------------ charts

def _bars(ax, labels, rates, lows, highs, ref=None, color=BLUE):
    # bars with 90% whiskers; each value sits just past its whisker, never on a bar
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rates, height=0.5, color=color, zorder=2)
    ax.hlines(y, lows, highs, color=INK, linewidth=1.5, zorder=3)
    for yy, lo, hi in zip(y, lows, highs):
        ax.vlines([lo, hi], yy - 0.13, yy + 0.13, color=INK, linewidth=1.5, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=INK_3, linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
    for yy, r, hi in zip(y, rates, highs):
        ax.text(hi + 0.025, yy, f"{r:.0%}", va="center", ha="left", fontsize=9.5, color=INK, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)


def plot_chain(table: pd.DataFrame, overall: Dict[str, float], title: str, measures=CHAIN):
    # one factor through the two steps, all panels on the same 0-100% scale
    groups = list(dict.fromkeys(table["group"]))
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 0.55 * len(groups) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (m, name) in zip(axes, measures):
        t = table[table["measure"] == m].set_index("group").reindex(groups)
        _bars(ax, groups, t["rate"], t["low"], t["high"], ref=overall.get(m))
        ax.set_title(name, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.text(0.01, -0.03, "Bars: rate in real corners · whiskers: 90% range · dashed line: all corners · counts in the table below",
             fontsize=8.5, color=INK_3)
    return fig


def plot_model_steps(evaluation: pd.DataFrame):
    # how well each step is predicted on unseen competitions
    steps = [
        ("first_contact_attack", "full_model", "1 · First ball won"),
        ("chance_if_won", "full_model", "2 · Good chance after winning it"),
        ("chance_if_lost", "full_model", "2 · Good chance after losing it"),
        ("chance_chain", "chain_model", "= Good chance overall"),
    ]
    rows = []
    for target, model, label in steps:
        r = evaluation[(evaluation.target == target) & (evaluation.scheme == "leave_one_competition_out") & (evaluation.model == model)]
        rows.append((label, float(r.roc_auc.iloc[0]), float(r.brier_skill.iloc[0])))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] - 0.5 for r in rows], left=0.5, height=0.5, color=BLUE, zorder=2)
    for yy, (_, auc, skill) in zip(y, rows):
        ax.text(auc + 0.005, yy, f"{auc:.2f}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.5, color=INK_3, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.45, 0.75)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], ["", COIN_FLIP, "0.55", "0.6", "0.65", "0.7", "0.75"])
    ax.set_xlabel("AUC on competitions the model never saw")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title("How well each step can be predicted")
    fig.tight_layout()
    return fig, pd.DataFrame(rows, columns=["Step", "AUC (unseen competitions)", "Brier skill"])


def plot_per_competition(per_comp: pd.DataFrame, target: str = "first_contact_attack"):
    t = per_comp[per_comp.target == target].sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t.roc_auc - 0.5, left=0.5, height=0.5, color=BLUE, zorder=2)
    for yy, auc, n in zip(y, t.roc_auc, t.n):
        ax.text(max(auc, 0.5) + 0.005, yy, f"{auc:.2f}  ({n} corners)", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.5, color=INK_3, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.set_yticks(y, t.held_out)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["0.3", "0.4", COIN_FLIP, "0.6", "0.7", "0.8", "0.9"])
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("AUC when this competition was left out of training")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title("First ball won, tested on each competition in turn")
    fig.tight_layout()
    return fig


def plot_old_vs_new(rows: List[Tuple[str, Optional[float], float]]):
    # the original system against the current one, on unseen competitions
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    h = 0.32
    for yy, (label, old, new) in zip(y, rows):
        if old is not None:
            ax.barh(yy + h / 2 + 0.02, old - 0.5, left=0.5, height=h, color=ORANGE, zorder=2)
            ax.text(max(old, 0.5) + 0.005, yy + h / 2 + 0.02, f"{old:.2f}", va="center", fontsize=9, color=INK_2)
        else:
            ax.text(0.505, yy + h / 2 + 0.02, "not measured by the original system", va="center", fontsize=8.5, color=INK_3)
        ax.barh(yy - h / 2 - 0.02, new - 0.5, left=0.5, height=h, color=BLUE, zorder=2)
        ax.text(max(new, 0.5) + 0.005, yy - h / 2 - 0.02, f"{new:.2f}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.5, color=INK_3, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.45, 0.75)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], ["", COIN_FLIP, "0.55", "0.6", "0.65", "0.7", "0.75"])
    ax.set_xlabel("AUC on competitions the model never saw")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="Original system"), Patch(color=BLUE, label="Current system")],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Original against current system, measured the same way")
    fig.tight_layout()
    return fig
