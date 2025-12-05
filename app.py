from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import yaml

from src.apv_engine import APVEngine





def load_config(path: Path = Path("config.yaml")) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG: Dict[str, Any] = load_config()
ENGINE = APVEngine(CFG)

ARCHETYPES: List[str] = ENGINE.get_archetypes()
if not ARCHETYPES:
    ARCHETYPES = ["Hybrid", "Man", "Zonal"]

ALL_SHAPES_LABEL: str = "All archetypes"

DEFAULT_ARCHETYPE: str = ARCHETYPES[0]
DEFAULT_MIN_SAMPLES: int = getattr(ENGINE, "default_min_samples", 5)

# Historical tab and scenario editor can both use an "All" option
BEST_SHAPES: List[str] = ARCHETYPES + [ALL_SHAPES_LABEL]
SCENARIO_SHAPES: List[str] = ARCHETYPES + [ALL_SHAPES_LABEL]





def build_apv_summary_table(
    apv: Dict[str, float],
    probs: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a compact APV table, rounded to 3 decimals.
    """
    rows: List[Dict[str, Any]] = []

    entries: List[Tuple[str, Dict[str, float], str]] = [
        ("PV_for", apv, "PV_for"),
        ("PV_against", apv, "PV_against"),
        ("APV_net", apv, "APV_net"),
    ]

    for label, source, key in entries:
        value = source.get(key)
        if isinstance(value, (int, float)):
            value = round(float(value), 3)
        rows.append({"metric": label, "value": value})

    return pd.DataFrame(rows)



def build_explanation_markdown(explanation: Dict[str, Any]) -> Tuple[str, str]:
    verdict = explanation.get("verdict_label", "")
    baseline_text = explanation.get("baseline_comparison", "")
    bullets_help: List[str] = explanation.get("bullets_help", []) or []
    bullets_risk: List[str] = explanation.get("bullets_risk", []) or []

    # Headline
    headline_lines: List[str] = []
    if verdict:
        headline_lines.append("### Verdict")
        headline_lines.append(verdict)
        headline_lines.append("")
    if baseline_text:
        headline_lines.append(baseline_text)

    headline_md = "\n".join(headline_lines).strip() or "No verdict available."

    # Detail blocks
    expl_lines: List[str] = []

    expl_lines.append("### What helps this setup")
    if bullets_help:
        for b in bullets_help:
            expl_lines.append(f"- {b}")
    else:
        expl_lines.append("- No clear positive drivers identified.")

    expl_lines.append("")
    expl_lines.append("### What to watch")
    if bullets_risk:
        for b in bullets_risk:
            expl_lines.append(f"- {b}")
    else:
        expl_lines.append("- No specific risks highlighted by the model.")

    explanation_md = "\n".join(expl_lines)

    return headline_md, explanation_md


def build_best_setups_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build historical pattern view, with APV and PV rounded to 3 decimals.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_preferred = [
        "delivery_type",
        "end_zone",
        "n_att_near",
        "n_att_central",
        "n_att_far",
        "n_blockers",
        "n_edge_of_box",
        "n_rest_defence",
        "n_corners",
        "APV_net_mean",
        "PV_for_mean",
        "PV_against_mean",
        "tag",
    ]
    cols_present = [c for c in cols_preferred if c in df.columns]
    view = df[cols_present].copy()

    rename_map = {
        "APV_net_mean": "APV_net",
        "PV_for_mean": "PV_for",
        "PV_against_mean": "PV_against",
    }
    view = view.rename(columns={k: v for k, v in rename_map.items() if k in view.columns})

    # Round APV and PV for readability
    for col in ["APV_net", "PV_for", "PV_against"]:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(3)

    return view





def _max_n_corners_for_archetype(archetype: str) -> int:
    """
    Inspect the pattern library and return the maximum n_corners
    for the given archetype. Falls back to DEFAULT_MIN_SAMPLES if
    anything is missing.
    """
    try:
        ENGINE._ensure_pattern_library_loaded()
    except Exception:
        return DEFAULT_MIN_SAMPLES

    df = getattr(ENGINE, "pattern_library", None)
    if df is None or df.empty:
        return DEFAULT_MIN_SAMPLES

    if archetype and "A_name" in df.columns:
        df = df[df["A_name"] == archetype]

    if df.empty or "n_corners" not in df.columns:
        return DEFAULT_MIN_SAMPLES

    max_n = int(df["n_corners"].max())
    return max(max_n, 1)


def update_min_samples_slider(
    archetype: str,
    current_value: int,
):
    """
    Dynamically adjust the 'Minimum corners in pattern' slider maximum
    when the archetype changes.
    """
    max_n = _max_n_corners_for_archetype(archetype)
    new_value = min(int(current_value), max_n)
    return gr.update(maximum=max_n, value=new_value)



def best_setups_handler(
    archetype: str,
    min_samples: int,
    top_n: int,
) -> pd.DataFrame:
    if not archetype:
        return pd.DataFrame()

    min_samples_int = int(min_samples)
    top_n_int = int(top_n)

    is_all = isinstance(archetype, str) and archetype.lower().startswith("all")

    if is_all:
        shapes = ARCHETYPES or ENGINE.get_archetypes() or []
        frames: List[pd.DataFrame] = []

        for shape in shapes:
            df_shape = ENGINE.get_best_setups_for_shape(
                archetype=shape,
                min_samples=min_samples_int,
                top_n=None, 
            )
            if df_shape is not None and not df_shape.empty:
                frames.append(df_shape)

        if not frames:
            return pd.DataFrame(
                [
                    {
                        "delivery_type": "",
                        "end_zone": "",
                        "n_att_near": "",
                        "n_att_central": "",
                        "n_att_far": "",
                        "n_blockers": "",
                        "n_edge_of_box": "",
                        "n_rest_defence": "",
                        "n_corners": "",
                        "APV_net": "",
                        "PV_for": "",
                        "PV_against": "",
                        "tag": (
                            f"No patterns with at least {min_samples_int} corners "
                            "for any defensive shape."
                        ),
                    }
                ]
            )

        df_all = pd.concat(frames, ignore_index=True)

        apv_col = "APV_net_mean"
        if apv_col in df_all.columns:
            df_all = df_all.sort_values(apv_col, ascending=False)

        if top_n_int > 0:
            df_all = df_all.head(top_n_int)

        return build_best_setups_table(df_all)

    df = ENGINE.get_best_setups_for_shape(
        archetype=archetype,
        min_samples=min_samples_int,
        top_n=top_n_int,
    )

    if df is None or df.empty:
        return pd.DataFrame(
            [
                {
                    "delivery_type": "",
                    "end_zone": "",
                    "n_att_near": "",
                    "n_att_central": "",
                    "n_att_far": "",
                    "n_blockers": "",
                    "n_edge_of_box": "",
                    "n_rest_defence": "",
                    "n_corners": "",
                    "APV_net": "",
                    "PV_for": "",
                    "PV_against": "",
                    "tag": (
                        f"No patterns with at least {min_samples_int} corners "
                        f"for {archetype}."
                    ),
                }
            ]
        )

    return build_best_setups_table(df)



def evaluate_corner_handler(
    A_name: str,
    delivery_type: str,
    end_zone: str,
    n_att_near: int,
    n_att_central: int,
    n_att_far: int,
    n_blockers: int,
    n_edge_of_box: int,
    n_rest_defence: int,
) -> Tuple[str, pd.DataFrame, str, pd.DataFrame]:
    total_players = (
        int(n_att_near)
        + int(n_att_central)
        + int(n_att_far)
        + int(n_blockers)
        + int(n_edge_of_box)
        + int(n_rest_defence)
    )
    if total_players != 10:
        headline_md = (
            f"Total outfield players = {total_players}. "
            "Please keep the total at 10."
        )
        explanation_md = (
            "Adjust the counts so that the total number of outfield players "
            "is exactly 10. For example reduce attackers in the box or increase "
            "players staying back."
        )
        return headline_md, pd.DataFrame(), explanation_md, pd.DataFrame()

    coach_input: Dict[str, Any] = {
        "A_name": A_name,
        "delivery_type": delivery_type,
        "end_zone": end_zone,
        "n_att_near": int(n_att_near),
        "n_att_central": int(n_att_central),
        "n_att_far": int(n_att_far),
        "n_blockers": int(n_blockers),
        "n_edge_of_box": int(n_edge_of_box),
        "n_rest_defence": int(n_rest_defence),
    }

    is_all_shapes = isinstance(A_name, str) and A_name.lower().startswith("all")

    try:
        if is_all_shapes:
            multi = ENGINE.evaluate_for_all_archetypes(coach_input)
            summary_df: pd.DataFrame = multi.get("summary", pd.DataFrame())
            per_archetype: List[Dict[str, Any]] = multi.get("per_archetype", []) or []

            if summary_df is None or summary_df.empty or not per_archetype:
                headline_md = "No predictions available for any defensive shape."
                return headline_md, pd.DataFrame(), "", pd.DataFrame()

            df = summary_df.copy()

            numeric_cols = [
                "P_chance_for",
                "P_shot_against_any",
                "PV_for",
                "PV_against",
                "APV_net",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "APV_net" in df.columns and df["APV_net"].notna().any():
                best_idx_num = int(df["APV_net"].idxmax())
                worst_idx_num = int(df["APV_net"].idxmin())
                best_row = df.iloc[best_idx_num]
                worst_row = df.iloc[worst_idx_num]
                mean_apv = float(df["APV_net"].mean())

                best_shape = str(best_row.get("A_name", ""))
                worst_shape = str(worst_row.get("A_name", ""))

                headline_lines: List[str] = ["### Verdict across defensive shapes"]
                headline_lines.append(
                    f"Best against **{best_shape}** (APV {best_row.get('APV_net', float('nan')):.3f})."
                )
                headline_lines.append(
                    f"Worst against **{worst_shape}** (APV {worst_row.get('APV_net', float('nan')):.3f})."
                )
                headline_lines.append(
                    f"Average APV across shapes: {mean_apv:.3f}."
                )
                headline_md = "\n".join(headline_lines)
            else:
                headline_md = (
                    "### Verdict across defensive shapes\n"
                    "APV values are not available for this setup."
                )

            apv_table = df.copy()
            for col in numeric_cols:
                if col in apv_table.columns:
                    apv_table[col] = apv_table[col].round(3)


            best_index = multi.get("best_index")
            if not isinstance(best_index, int) or best_index < 0 or best_index >= len(per_archetype):
                if "APV_net" in df.columns and df["APV_net"].notna().any():
                    best_index = int(df["APV_net"].idxmax())
                else:
                    best_index = 0

            best_result = per_archetype[best_index]
            best_explanation: Dict[str, Any] = best_result.get("explanation", {}) or {}
            _, explanation_md = build_explanation_markdown(best_explanation)

            tweaks_list = best_result.get("tweaks", []) or []
            tweaks_df = pd.DataFrame(tweaks_list)
            if not tweaks_df.empty:
                cols_pref = ["change_label", "new_apv", "delta_apv"]
                cols_exist = [c for c in cols_pref if c in tweaks_df.columns]
                tweaks_df = tweaks_df[cols_exist]

                for col in ["new_apv", "delta_apv"]:
                    if col in tweaks_df.columns:
                        tweaks_df[col] = pd.to_numeric(
                            tweaks_df[col], errors="coerce"
                        ).round(3)

                tweaks_df = tweaks_df.rename(columns={"delta_apv": "APV_change"})
            else:
                tweaks_df = pd.DataFrame(
                    [
                        {
                            "change_label": (
                                "No clearly better historical setups found "
                                "for any defensive shape."
                            ),
                            "new_apv": None,
                            "delta_apv": 0.0,
                        }
                    ]
                )
                tweaks_df = tweaks_df.rename(columns={"delta_apv": "APV_change"})


            return headline_md, apv_table, explanation_md, tweaks_df

        result = ENGINE.evaluate_for_coach(coach_input)
    except Exception as exc:  
        print("Error in evaluate_for_coach:", exc)
        headline = "Error: could not evaluate this corner."
        return headline, pd.DataFrame(), "", pd.DataFrame()

    apv: Dict[str, float] = result.get("apv", {}) or {}
    probs: Dict[str, float] = result.get("probabilities", {}) or {}
    explanation: Dict[str, Any] = result.get("explanation", {}) or {}
    tweaks_list = result.get("tweaks", []) or []

    headline_md, explanation_md = build_explanation_markdown(explanation)
    apv_table = build_apv_summary_table(apv, probs)

    tweaks_df = pd.DataFrame(tweaks_list)
    if not tweaks_df.empty:
        cols_pref = ["change_label", "new_apv", "delta_apv"]
        cols_exist = [c for c in cols_pref if c in tweaks_df.columns]
        tweaks_df = tweaks_df[cols_exist]

        for col in ["new_apv", "delta_apv"]:
            if col in tweaks_df.columns:
                tweaks_df[col] = pd.to_numeric(
                    tweaks_df[col], errors="coerce"
                ).round(3)

        tweaks_df = tweaks_df.rename(columns={"delta_apv": "APV_change"})
    else:
        tweaks_df = pd.DataFrame(
            [
                {
                    "change_label": (
                        "No clearly better historical setups found for this "
                        "defensive shape."
                    ),
                    "new_apv": None,
                    "delta_apv": 0.0,
                }
            ]
        )
        tweaks_df = tweaks_df.rename(columns={"delta_apv": "APV_change"})


    return headline_md, apv_table, explanation_md, tweaks_df




with gr.Blocks(title="Corner APV Coach Tool") as demo:
    gr.Markdown(
        """
        # Corner APV Coach Tool

        This tool summarises your trained APV models into two coach friendly views:

        - **Historical setups**: best patterns for each defensive shape.
        - **Scenario editor**: test your own corner idea and see value, risk, and suggestions.
        """
    )

    # Tab 1: Historical setups
    with gr.Tab("Historical setups"):
        gr.Markdown("## Best setups for a defensive shape")

        with gr.Row():
            dd_shape_best = gr.Dropdown(
                label="Defensive shape",
                choices=BEST_SHAPES,
                value=DEFAULT_ARCHETYPE,
            )
            slider_min_samples = gr.Slider(
                label="Minimum corners in pattern",
                minimum=1,
                maximum=20,
                step=1,
                value=DEFAULT_MIN_SAMPLES,
            )
            slider_top_n = gr.Slider(
                label="Number of setups to show",
                minimum=3,
                maximum=20,
                step=1,
                value=5,
            )
            btn_show_best = gr.Button("Show best setups")

        best_table = gr.Dataframe(
            label="Best historical setups",
            interactive=False,
        )

    # Tab 2: Scenario editor
    with gr.Tab("Scenario editor"):
        gr.Markdown("## Design and evaluate a corner setup")

        with gr.Row():
            with gr.Column(scale=1):
                dd_shape_scenario = gr.Dropdown(
                    label="Defensive shape",
                    choices=SCENARIO_SHAPES,
                    value=DEFAULT_ARCHETYPE,
                )
                dd_delivery = gr.Dropdown(
                    label="Delivery type",
                    choices=["inswing", "outswing", "short"],
                    value="inswing",
                )
                dd_zone = gr.Dropdown(
                    label="Delivery zone",
                    choices=["near", "central", "far"],
                    value="central",
                )

                slider_near = gr.Slider(
                    label="Attackers at near post",
                    minimum=0,
                    maximum=6,
                    step=1,
                    value=2,
                )
                slider_central = gr.Slider(
                    label="Attackers in central zone",
                    minimum=0,
                    maximum=6,
                    step=1,
                    value=3,
                )
                slider_far = gr.Slider(
                    label="Attackers at far post",
                    minimum=0,
                    maximum=6,
                    step=1,
                    value=1,
                )

                slider_blockers = gr.Slider(
                    label="Blockers around goalkeeper",
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=1,
                )
                slider_edge = gr.Slider(
                    label="Players on edge of box",
                    minimum=0,
                    maximum=4,
                    step=1,
                    value=1,
                )
                slider_rest = gr.Slider(
                    label="Players staying back",
                    minimum=0,
                    maximum=5,
                    step=1,
                    value=2,
                )

                btn_evaluate = gr.Button("Evaluate this corner")

            with gr.Column(scale=1):
                headline_out = gr.Markdown(label="Verdict")
                apv_table_out = gr.Dataframe(
                    label="APV summary",
                    interactive=False,
                )
                explanation_out = gr.Markdown(label="Explanation")
                tweaks_out = gr.Dataframe(
                    label="Suggested tweaks",
                    interactive=False,
                )

    # Historical setups
    dd_shape_best.change(
        fn=update_min_samples_slider,
        inputs=[dd_shape_best, slider_min_samples],
        outputs=[slider_min_samples],
    )

    btn_show_best.click(
        fn=best_setups_handler,
        inputs=[dd_shape_best, slider_min_samples, slider_top_n],
        outputs=[best_table],
    )

    # Scenario editor
    btn_evaluate.click(
        fn=evaluate_corner_handler,
        inputs=[
            dd_shape_scenario,
            dd_delivery,
            dd_zone,
            slider_near,
            slider_central,
            slider_far,
            slider_blockers,
            slider_edge,
            slider_rest,
        ],
        outputs=[headline_out, apv_table_out, explanation_out, tweaks_out],
    )


if __name__ == "__main__":
    demo.launch()
