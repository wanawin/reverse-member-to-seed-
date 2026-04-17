#!/usr/bin/env python3
"""
BUILD: core025_decision_optimization_lab__2026-04-16_v1

Purpose
-------
Decision optimization engine for Core025 using precomputed artifacts.

This app:
1) loads reusable precompute artifacts,
2) runs many scoring/decision configurations quickly,
3) evaluates both diagnostic and operational outcomes,
4) ranks experiments by a real objective aligned to the user's goal:
   - maximize Top1
   - retain needed Top2
   - reduce waste Top2
   - reduce misses
   - improve play efficiency

Required uploads from precompute builder:
- prepared_training_rows__...csv
- rule_metadata__...csv
- match_matrix__...csv
- precompute_manifest__...csv

Full file. No placeholders.
"""

from __future__ import annotations

import io
import itertools
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_decision_optimization_lab__2026-04-16_v1"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")

MEMBERS = ["0025", "0225", "0255"]
OUTCOMES = ["TOP1_WIN", "WASTE", "NEEDED", "MISS"]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int = 100) -> pd.DataFrame:
    return df.head(int(rows)).copy()


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for part in re.split(r"[,\n]+", text):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def product_size(lists: List[List[object]]) -> int:
    total = 1
    for lst in lists:
        total *= max(1, len(lst))
    return total


def ensure_required_columns(df: pd.DataFrame, needed: Sequence[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_metadata(rule_metadata: pd.DataFrame) -> pd.DataFrame:
    out = rule_metadata.copy()
    ensure_required_columns(out, ["rule_id", "column_name", "target", "kind"], "rule metadata")
    for col in ["support", "gap", "lift", "hit_rate_true"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["rule_id"] = pd.to_numeric(out["rule_id"], errors="coerce").astype(int)
    out["column_name"] = out["column_name"].astype(str)
    out["target"] = out["target"].astype(str)
    out["kind"] = out["kind"].astype(str)
    return out


def normalize_prepared(prepared: pd.DataFrame) -> pd.DataFrame:
    out = prepared.copy()
    ensure_required_columns(out, ["row_id", "seed", "WinningMember"], "prepared training rows")
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype(int)
    out["seed"] = out["seed"].astype(str)
    out["WinningMember"] = out["WinningMember"].astype(str)
    return out


def normalize_match_matrix(match_matrix: pd.DataFrame) -> pd.DataFrame:
    out = match_matrix.copy()
    ensure_required_columns(out, ["row_id"], "match matrix")
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype(int)
    for c in out.columns:
        if c == "row_id":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.uint8)
    return out


def build_experiment_grid(
    single_ws: List[float],
    filtered_ws: List[float],
    sep_ws: List[float],
    stacked_ws: List[float],
    miss_penalties: List[float],
    needed_penalties: List[float],
    waste_bonuses: List[float],
    top1win_bonuses: List[float],
    gap_thresholds: List[float],
    risk_thresholds: List[float],
    waste_penalty_weights: List[float],
    miss_penalty_weights: List[float],
    top1_reward_weights: List[float],
    needed_reward_weights: List[float],
    max_tests: int,
    run_full_grid: bool,
) -> tuple[list[dict], int]:
    axes = [
        single_ws,
        filtered_ws,
        sep_ws,
        stacked_ws,
        miss_penalties,
        needed_penalties,
        waste_bonuses,
        top1win_bonuses,
        gap_thresholds,
        risk_thresholds,
        waste_penalty_weights,
        miss_penalty_weights,
        top1_reward_weights,
        needed_reward_weights,
    ]
    total_possible = product_size(axes)
    combos_iter = itertools.product(*axes)
    combos = list(combos_iter) if run_full_grid else list(itertools.islice(combos_iter, int(max_tests)))

    cfgs = []
    for i, combo in enumerate(combos, start=1):
        cfgs.append({
            "experiment_name": f"exp_{i:05d}",
            "single_w": float(combo[0]),
            "filtered_w": float(combo[1]),
            "sep_w": float(combo[2]),
            "stacked_w": float(combo[3]),
            "miss_penalty": float(combo[4]),
            "needed_penalty": float(combo[5]),
            "waste_top1_bonus": float(combo[6]),
            "top1win_bonus": float(combo[7]),
            "gap_threshold": float(combo[8]),
            "risk_threshold": float(combo[9]),
            "waste_penalty_weight": float(combo[10]),
            "miss_penalty_weight": float(combo[11]),
            "top1_reward_weight": float(combo[12]),
            "needed_reward_weight": float(combo[13]),
        })
    return cfgs, total_possible


def build_weight_vectors(rule_metadata: pd.DataFrame, cfg: Dict[str, float]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    md = rule_metadata.copy()

    kind_mult = np.select(
        [
            md["kind"].str.lower().eq("single"),
            md["kind"].str.lower().eq("filtered"),
            md["kind"].str.lower().eq("separator"),
            md["kind"].str.lower().eq("stacked"),
        ],
        [
            cfg["single_w"],
            cfg["filtered_w"],
            cfg["sep_w"],
            cfg["stacked_w"],
        ],
        default=cfg["single_w"],
    )

    base_strength = np.maximum(0.05, md["gap"].to_numpy(dtype=float) + 0.35 * md["hit_rate_true"].to_numpy(dtype=float))
    support_factor = np.minimum(1.5, np.log1p(np.maximum(md["support"].to_numpy(dtype=float), 0.0)) / 3.0 + 0.5)
    lift_factor = np.minimum(1.5, 0.5 + np.maximum(md["lift"].to_numpy(dtype=float), 0.0) / 4.0)
    final_weight = kind_mult * base_strength * support_factor * lift_factor

    member_weights: Dict[str, np.ndarray] = {}
    outcome_weights: Dict[str, np.ndarray] = {}
    for target in MEMBERS:
        member_weights[target] = np.where(md["target"].eq(target).to_numpy(), final_weight, 0.0)
    for target in OUTCOMES:
        outcome_weights[target] = np.where(md["target"].eq(target).to_numpy(), final_weight, 0.0)

    return member_weights, outcome_weights


def evaluate_config(
    prepared: pd.DataFrame,
    rule_metadata: pd.DataFrame,
    match_matrix: pd.DataFrame,
    cfg: Dict[str, float],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    col_names = rule_metadata["column_name"].tolist()
    X = match_matrix[col_names].to_numpy(dtype=float)

    member_weights, outcome_weights = build_weight_vectors(rule_metadata, cfg)

    member_scores = {m: X @ member_weights[m] for m in MEMBERS}
    outcome_scores = {o: X @ outcome_weights[o] for o in OUTCOMES}

    top1_bonus = cfg["top1win_bonus"] * outcome_scores["TOP1_WIN"]
    waste_bonus = cfg["waste_top1_bonus"] * outcome_scores["WASTE"]
    total_risk = cfg["miss_penalty"] * outcome_scores["MISS"] + cfg["needed_penalty"] * outcome_scores["NEEDED"]

    adjusted_member_scores = {
        m: member_scores[m] + top1_bonus + waste_bonus
        for m in MEMBERS
    }

    score_df = pd.DataFrame({
        "member_score_0025": adjusted_member_scores["0025"],
        "member_score_0225": adjusted_member_scores["0225"],
        "member_score_0255": adjusted_member_scores["0255"],
        "outcome_score_TOP1_WIN": outcome_scores["TOP1_WIN"],
        "outcome_score_WASTE": outcome_scores["WASTE"],
        "outcome_score_NEEDED": outcome_scores["NEEDED"],
        "outcome_score_MISS": outcome_scores["MISS"],
        "total_risk": total_risk,
        "matched_rule_count": X.sum(axis=1),
    })

    member_cols = ["member_score_0025", "member_score_0225", "member_score_0255"]
    member_names = np.array(["0025", "0225", "0255"])

    arr = score_df[member_cols].to_numpy(dtype=float)
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    top1_scores = arr[np.arange(len(arr)), top1_idx]
    top2_scores = arr[np.arange(len(arr)), top2_idx]
    pred_gap12 = top1_scores - top2_scores

    pred_top1 = member_names[top1_idx]
    pred_top2 = member_names[top2_idx]

    play_top2_mask = (total_risk > cfg["risk_threshold"]) | (pred_gap12 < cfg["gap_threshold"])
    play_mode = np.where(play_top2_mask, "PLAY_TOP2", "PLAY_TOP1")

    out = prepared.copy()
    out["pred_top1"] = pred_top1
    out["pred_top2"] = pred_top2
    out["pred_gap12"] = pred_gap12
    out["play_mode"] = play_mode
    out["member_score_0025"] = score_df["member_score_0025"].to_numpy()
    out["member_score_0225"] = score_df["member_score_0225"].to_numpy()
    out["member_score_0255"] = score_df["member_score_0255"].to_numpy()
    out["outcome_score_TOP1_WIN"] = score_df["outcome_score_TOP1_WIN"].to_numpy()
    out["outcome_score_WASTE"] = score_df["outcome_score_WASTE"].to_numpy()
    out["outcome_score_NEEDED"] = score_df["outcome_score_NEEDED"].to_numpy()
    out["outcome_score_MISS"] = score_df["outcome_score_MISS"].to_numpy()
    out["total_risk"] = total_risk
    out["matched_rule_count"] = score_df["matched_rule_count"].to_numpy()

    winner = out["WinningMember"].astype(str).to_numpy()
    top1 = out["pred_top1"].astype(str).to_numpy()
    top2 = out["pred_top2"].astype(str).to_numpy()
    mode = out["play_mode"].astype(str).to_numpy()

    app_top1_hit = (top1 == winner).astype(int)
    app_top2_hit = ((top1 == winner) | (top2 == winner)).astype(int)

    your_top1_win = ((mode == "PLAY_TOP1") & (top1 == winner)).astype(int)
    your_top2_win = ((mode == "PLAY_TOP2") & (top1 != winner) & (top2 == winner)).astype(int)
    your_top3_win = np.zeros(len(out), dtype=int)

    needed_top2 = ((mode == "PLAY_TOP2") & (top1 != winner) & (top2 == winner)).astype(int)
    waste_top2 = ((mode == "PLAY_TOP2") & (top1 == winner)).astype(int)
    misses = (((mode == "PLAY_TOP1") & (top1 != winner)) | ((mode == "PLAY_TOP2") & (top1 != winner) & (top2 != winner))).astype(int)

    plays_used = np.where(mode == "PLAY_TOP2", 2, 1).astype(int)
    wins_captured = your_top1_win + your_top2_win + your_top3_win

    out["app_top1_hit"] = app_top1_hit
    out["app_top2_hit"] = app_top2_hit
    out["your_top1_win"] = your_top1_win
    out["your_top2_win"] = your_top2_win
    out["your_top3_win"] = your_top3_win
    out["needed_top2"] = needed_top2
    out["waste_top2"] = waste_top2
    out["miss"] = misses
    out["plays_used"] = plays_used
    out["wins_captured"] = wins_captured
    out["BuildMarker"] = BUILD_SLUG

    total_plays = int(plays_used.sum())
    total_wins = int(wins_captured.sum())
    plays_per_win = float(total_plays / total_wins) if total_wins > 0 else float("inf")

    objective_score = (
        cfg["top1_reward_weight"] * int(your_top1_win.sum())
        + cfg["needed_reward_weight"] * int(needed_top2.sum())
        - cfg["waste_penalty_weight"] * int(waste_top2.sum())
        - cfg["miss_penalty_weight"] * int(misses.sum())
    )

    metrics = {
        "experiment_name": cfg["experiment_name"],
        "rows": int(len(out)),
        "app_top1_hits": int(app_top1_hit.sum()),
        "app_top2_hits": int(app_top2_hit.sum()),
        "misses": int(misses.sum()),
        "your_top1_wins": int(your_top1_win.sum()),
        "your_top2_wins": int(your_top2_win.sum()),
        "your_top3_wins": int(your_top3_win.sum()),
        "needed_top2": int(needed_top2.sum()),
        "waste_top2": int(waste_top2.sum()),
        "pred_play_top1_rows": int((mode == "PLAY_TOP1").sum()),
        "pred_play_top2_rows": int((mode == "PLAY_TOP2").sum()),
        "total_plays_used": total_plays,
        "plays_per_win": plays_per_win,
        "avg_pred_gap12": float(pd.to_numeric(out["pred_gap12"], errors="coerce").mean()),
        "avg_matched_rule_count": float(pd.to_numeric(out["matched_rule_count"], errors="coerce").mean()),
        "objective_score": objective_score,
        "single_w": cfg["single_w"],
        "filtered_w": cfg["filtered_w"],
        "sep_w": cfg["sep_w"],
        "stacked_w": cfg["stacked_w"],
        "miss_penalty": cfg["miss_penalty"],
        "needed_penalty": cfg["needed_penalty"],
        "waste_top1_bonus": cfg["waste_top1_bonus"],
        "top1win_bonus": cfg["top1win_bonus"],
        "gap_threshold": cfg["gap_threshold"],
        "risk_threshold": cfg["risk_threshold"],
        "waste_penalty_weight": cfg["waste_penalty_weight"],
        "miss_penalty_weight": cfg["miss_penalty_weight"],
        "top1_reward_weight": cfg["top1_reward_weight"],
        "needed_reward_weight": cfg["needed_reward_weight"],
        "BuildMarker": BUILD_SLUG,
    }

    return out, metrics


def main() -> None:
    st.set_page_config(page_title="Core025 Decision Optimization Lab", layout="wide")
    st.title("Core025 Decision Optimization Lab")
    st.caption(BUILD_MARKER)

    if "decision_opt_outputs" not in st.session_state:
        st.session_state["decision_opt_outputs"] = None

    with st.sidebar:
        st.write(BUILD_MARKER)

        prepared_file = st.file_uploader("Upload prepared training rows CSV", type=["csv","txt","tsv","xlsx","xls"], key="prepared_file")
        rule_metadata_file = st.file_uploader("Upload rule metadata CSV", type=["csv","txt","tsv","xlsx","xls"], key="rule_metadata_file")
        match_matrix_file = st.file_uploader("Upload match matrix CSV", type=["csv","txt","tsv","xlsx","xls"], key="match_matrix_file")
        manifest_file = st.file_uploader("Upload precompute manifest CSV", type=["csv","txt","tsv","xlsx","xls"], key="manifest_file")

        rows_to_show = st.number_input("Rows to preview", min_value=20, max_value=500, value=100, step=20)

        st.markdown("### Sweep values")
        single_ws_text = st.text_area("Single trait weights", value="0.4,0.8,1.2")
        filtered_ws_text = st.text_area("Filtered trait weights", value="0.8,1.2,1.6")
        sep_ws_text = st.text_area("Separator trait weights", value="1.2,1.6,2.0")
        stacked_ws_text = st.text_area("Stacked trait weights", value="0.0")
        miss_penalties_text = st.text_area("MISS penalties", value="0.8,1.0,1.2")
        needed_penalties_text = st.text_area("NEEDED penalties", value="0.6,0.8,1.0")
        waste_bonuses_text = st.text_area("WASTE Top1 bonuses", value="0.2,0.4,0.6")
        top1win_bonuses_text = st.text_area("TOP1_WIN bonuses", value="0.2,0.4")
        gap_thresholds_text = st.text_area("Top1/Top2 gap thresholds", value="0.5,0.75,1.0")
        risk_thresholds_text = st.text_area("Risk thresholds", value="1.0,1.5,2.0")

        st.markdown("### Objective weights")
        waste_penalty_weights_text = st.text_area("Waste Top2 penalty weights", value="1.0,1.5")
        miss_penalty_weights_text = st.text_area("Miss penalty weights", value="2.0,3.0")
        top1_reward_weights_text = st.text_area("Top1 reward weights", value="2.0,3.0")
        needed_reward_weights_text = st.text_area("Needed Top2 reward weights", value="1.0,1.5")

        run_full_grid = st.checkbox("Run full plausible grid", value=True)
        max_tests = st.number_input("Maximum experiments when full grid is OFF", min_value=1, max_value=50000, value=200, step=1)

        run_btn = st.button("Run decision optimization sweep", type="primary", use_container_width=True)

    if run_btn:
        if prepared_file is None or rule_metadata_file is None or match_matrix_file is None or manifest_file is None:
            st.error("Upload all 4 required precompute files.")
            st.session_state["decision_opt_outputs"] = None
        else:
            try:
                prepared = normalize_prepared(load_table(prepared_file))
                rule_metadata = normalize_metadata(load_table(rule_metadata_file))
                match_matrix = normalize_match_matrix(load_table(match_matrix_file))
                manifest = load_table(manifest_file)

                if len(prepared) != len(match_matrix):
                    raise ValueError(f"Prepared rows ({len(prepared)}) do not match matrix rows ({len(match_matrix)}).")

                grid, total_possible = build_experiment_grid(
                    single_ws=parse_float_list(single_ws_text),
                    filtered_ws=parse_float_list(filtered_ws_text),
                    sep_ws=parse_float_list(sep_ws_text),
                    stacked_ws=parse_float_list(stacked_ws_text),
                    miss_penalties=parse_float_list(miss_penalties_text),
                    needed_penalties=parse_float_list(needed_penalties_text),
                    waste_bonuses=parse_float_list(waste_bonuses_text),
                    top1win_bonuses=parse_float_list(top1win_bonuses_text),
                    gap_thresholds=parse_float_list(gap_thresholds_text),
                    risk_thresholds=parse_float_list(risk_thresholds_text),
                    waste_penalty_weights=parse_float_list(waste_penalty_weights_text),
                    miss_penalty_weights=parse_float_list(miss_penalty_weights_text),
                    top1_reward_weights=parse_float_list(top1_reward_weights_text),
                    needed_reward_weights=parse_float_list(needed_reward_weights_text),
                    max_tests=int(max_tests),
                    run_full_grid=bool(run_full_grid),
                )

                summary_rows = []
                detail_map: Dict[str, pd.DataFrame] = {}

                progress = st.progress(0.0, text="Running decision optimization experiments...")
                total_tests = len(grid)
                for idx, cfg in enumerate(grid, start=1):
                    detail_df, metrics = evaluate_config(prepared, rule_metadata, match_matrix, cfg)
                    summary_rows.append(metrics)
                    detail_map[cfg["experiment_name"]] = detail_df
                    progress.progress(idx / total_tests, text=f"Running decision optimization experiments... {idx}/{total_tests}")
                progress.empty()

                summary_df = pd.DataFrame(summary_rows).sort_values(
                    ["objective_score", "misses", "your_top1_wins", "plays_per_win"],
                    ascending=[False, True, False, True]
                ).reset_index(drop=True)

                run_meta = pd.DataFrame([{
                    "build_marker": BUILD_SLUG,
                    "prepared_rows": len(prepared),
                    "rule_metadata_rows": len(rule_metadata),
                    "match_matrix_rows": len(match_matrix),
                    "total_possible_experiments": int(total_possible),
                    "experiments_run": int(len(grid)),
                    "full_grid_enabled": bool(run_full_grid),
                }])

                st.session_state["decision_opt_outputs"] = {
                    "prepared": prepared,
                    "rule_metadata": rule_metadata,
                    "match_matrix": match_matrix,
                    "manifest": manifest,
                    "summary_df": summary_df,
                    "detail_map": detail_map,
                    "run_meta": run_meta,
                }

            except Exception as e:
                st.session_state["decision_opt_outputs"] = None
                st.error(f"Failed to run decision optimization lab: {e}")

    outputs = st.session_state.get("decision_opt_outputs")
    if outputs is None:
        st.info("Upload the 4 precompute artifacts and click Run decision optimization sweep.")
        return

    st.success("Decision optimization sweep complete")

    st.subheader("Run metadata")
    st.dataframe(outputs["run_meta"], use_container_width=True, hide_index=True)

    st.subheader("Experiment ranking")
    st.dataframe(outputs["summary_df"], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download decision ranking",
            data=df_to_csv_bytes(outputs["summary_df"]),
            file_name=f"decision_ranking__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_decision_ranking",
        )
    with c2:
        st.download_button(
            "Download run metadata",
            data=df_to_csv_bytes(outputs["run_meta"]),
            file_name=f"run_metadata__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_run_metadata",
        )

    st.markdown("---")
    st.subheader("Experiment detail preview")
    exp_names = outputs["summary_df"]["experiment_name"].tolist()
    picked = st.selectbox("Choose an experiment to inspect", exp_names)
    detail_df = outputs["detail_map"][picked]

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            f"Download {picked} scored rows",
            data=df_to_csv_bytes(detail_df),
            file_name=f"{picked}__scored_rows__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_scored_rows_{picked}",
        )
    with d2:
        mode_counts = (
            detail_df["play_mode"]
            .value_counts(dropna=False)
            .rename_axis("play_mode")
            .reset_index(name="count")
        )
        st.download_button(
            f"Download {picked} play-mode counts",
            data=df_to_csv_bytes(mode_counts),
            file_name=f"{picked}__play_mode_counts__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_play_mode_counts_{picked}",
        )

    tab1, tab2 = st.tabs(["Scored rows preview", "Play mode counts"])
    with tab1:
        cols = [
            "PlayDate", "StreamKey", "seed", "WinningMember",
            "pred_top1", "pred_top2", "pred_gap12", "play_mode",
            "app_top1_hit", "app_top2_hit",
            "your_top1_win", "your_top2_win", "your_top3_win",
            "needed_top2", "waste_top2", "miss", "plays_used",
            "matched_rule_count"
        ]
        cols = [c for c in cols if c in detail_df.columns]
        st.dataframe(safe_display_df(detail_df[cols], rows_to_show), use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(mode_counts, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
