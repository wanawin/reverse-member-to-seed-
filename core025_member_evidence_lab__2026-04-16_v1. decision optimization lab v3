#!/usr/bin/env python3
"""
BUILD: core025_decision_optimization_lab__2026-04-16_v3_member_normalization_fix

Decision optimization engine using precomputed artifacts.

This version fixes:
- member label normalization across the full comparison chain
- robust handling of 25/225/255 vs 0025/0225/0255
- keeps memory-safe top-N detail storage
- keeps staged/full/manual search modes

Required uploads:
- prepared_training_rows__...csv
- rule_metadata__...csv
- match_matrix__...csv
- precompute_manifest__...csv
"""

from __future__ import annotations

import io
import itertools
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_decision_optimization_lab__2026-04-16_v3_member_normalization_fix"
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
        if p:
            out.append(float(p))
    return out


def ensure_required_columns(df: pd.DataFrame, needed: Sequence[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_member_value(x) -> str:
    if pd.isna(x):
        return ""
    s = re.sub(r"\D", "", str(x).strip())
    if not s:
        return ""
    if s in {"25", "225", "255"}:
        return s.zfill(4)
    if len(s) <= 4:
        return s.zfill(4)
    return s[-4:]


def normalize_metadata(rule_metadata: pd.DataFrame) -> pd.DataFrame:
    out = rule_metadata.copy()
    ensure_required_columns(out, ["rule_id", "column_name", "target", "kind"], "rule metadata")
    for col in ["support", "gap", "lift", "hit_rate_true"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["rule_id"] = pd.to_numeric(out["rule_id"], errors="coerce").astype(int)
    out["column_name"] = out["column_name"].astype(str)
    out["target"] = out["target"].astype(str).apply(normalize_member_value).where(
        out["target"].astype(str).isin(MEMBERS), out["target"].astype(str)
    )
    out["kind"] = out["kind"].astype(str)
    return out


def normalize_prepared(prepared: pd.DataFrame) -> pd.DataFrame:
    out = prepared.copy()
    ensure_required_columns(out, ["row_id", "seed", "WinningMember"], "prepared training rows")
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype(int)
    out["seed"] = out["seed"].astype(str).str.zfill(4)
    out["WinningMember"] = out["WinningMember"].apply(normalize_member_value)
    if "Top1_actual" in out.columns:
        out["Top1_actual"] = out["Top1_actual"].apply(normalize_member_value)
    if "Top2_actual" in out.columns:
        out["Top2_actual"] = out["Top2_actual"].apply(normalize_member_value)
    return out


def normalize_match_matrix(match_matrix: pd.DataFrame) -> pd.DataFrame:
    out = match_matrix.copy()
    ensure_required_columns(out, ["row_id"], "match matrix")
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype(int)
    for c in out.columns:
        if c != "row_id":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.uint8)
    return out


def product_size(lists: List[List[object]]) -> int:
    total = 1
    for lst in lists:
        total *= max(1, len(lst))
    return total


def build_weight_vectors(rule_metadata: pd.DataFrame, cfg: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    md = rule_metadata.copy()
    kind_mult = np.select(
        [
            md["kind"].str.lower().eq("single"),
            md["kind"].str.lower().eq("filtered"),
            md["kind"].str.lower().eq("separator"),
            md["kind"].str.lower().eq("stacked"),
        ],
        [cfg["single_w"], cfg["filtered_w"], cfg["sep_w"], cfg["stacked_w"]],
        default=cfg["single_w"],
    )
    base_strength = np.maximum(0.05, md["gap"].to_numpy(float) + 0.35 * md["hit_rate_true"].to_numpy(float))
    support_factor = np.minimum(1.5, np.log1p(np.maximum(md["support"].to_numpy(float), 0.0)) / 3.0 + 0.5)
    lift_factor = np.minimum(1.5, 0.5 + np.maximum(md["lift"].to_numpy(float), 0.0) / 4.0)
    final_weight = kind_mult * base_strength * support_factor * lift_factor

    member_weights, outcome_weights = {}, {}
    for t in MEMBERS:
        member_weights[t] = np.where(md["target"].eq(t).to_numpy(), final_weight, 0.0)
    for t in OUTCOMES:
        outcome_weights[t] = np.where(md["target"].eq(t).to_numpy(), final_weight, 0.0)
    return member_weights, outcome_weights


def evaluate_config(
    prepared: pd.DataFrame,
    rule_metadata: pd.DataFrame,
    X: np.ndarray,
    cfg: Dict[str, float],
    return_detail: bool = False,
):
    member_weights, outcome_weights = build_weight_vectors(rule_metadata, cfg)

    member_scores = {m: X @ member_weights[m] for m in MEMBERS}
    outcome_scores = {o: X @ outcome_weights[o] for o in OUTCOMES}
    top1_bonus = cfg["top1win_bonus"] * outcome_scores["TOP1_WIN"]
    waste_bonus = cfg["waste_top1_bonus"] * outcome_scores["WASTE"]
    total_risk = cfg["miss_penalty"] * outcome_scores["MISS"] + cfg["needed_penalty"] * outcome_scores["NEEDED"]

    adjusted = {m: member_scores[m] + top1_bonus + waste_bonus for m in MEMBERS}
    arr = np.column_stack([adjusted["0025"], adjusted["0225"], adjusted["0255"]])
    member_names = np.array(["0025", "0225", "0255"])
    order = np.argsort(-arr, axis=1)
    top1_idx, top2_idx = order[:, 0], order[:, 1]
    pred_top1, pred_top2 = member_names[top1_idx], member_names[top2_idx]
    pred_gap12 = arr[np.arange(len(arr)), top1_idx] - arr[np.arange(len(arr)), top2_idx]

    play_top2_mask = (total_risk > cfg["risk_threshold"]) | (pred_gap12 < cfg["gap_threshold"])
    play_mode = np.where(play_top2_mask, "PLAY_TOP2", "PLAY_TOP1")

    winner = prepared["WinningMember"].astype(str).apply(normalize_member_value).to_numpy()
    pred_top1 = np.array([normalize_member_value(x) for x in pred_top1])
    pred_top2 = np.array([normalize_member_value(x) for x in pred_top2])
    mode = play_mode.astype(str)

    app_top1_hit = (pred_top1 == winner).astype(int)
    app_top2_hit = ((pred_top1 == winner) | (pred_top2 == winner)).astype(int)
    your_top1_win = ((mode == "PLAY_TOP1") & (pred_top1 == winner)).astype(int)
    your_top2_win = ((mode == "PLAY_TOP2") & (pred_top1 != winner) & (pred_top2 == winner)).astype(int)
    your_top3_win = np.zeros(len(prepared), dtype=int)
    needed_top2 = ((mode == "PLAY_TOP2") & (pred_top1 != winner) & (pred_top2 == winner)).astype(int)
    waste_top2 = ((mode == "PLAY_TOP2") & (pred_top1 == winner)).astype(int)
    misses = (((mode == "PLAY_TOP1") & (pred_top1 != winner)) | ((mode == "PLAY_TOP2") & (pred_top1 != winner) & (pred_top2 != winner))).astype(int)
    plays_used = np.where(mode == "PLAY_TOP2", 2, 1).astype(int)
    wins_captured = your_top1_win + your_top2_win + your_top3_win
    matched_rule_count = X.sum(axis=1)

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
        "rows": int(len(prepared)),
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
        "avg_pred_gap12": float(np.mean(pred_gap12)),
        "avg_matched_rule_count": float(np.mean(matched_rule_count)),
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

    if not return_detail:
        return metrics, None

    detail = prepared.copy()
    detail["WinningMember"] = detail["WinningMember"].apply(normalize_member_value)
    detail["pred_top1"] = pred_top1
    detail["pred_top2"] = pred_top2
    detail["pred_gap12"] = pred_gap12
    detail["play_mode"] = mode
    detail["app_top1_hit"] = app_top1_hit
    detail["app_top2_hit"] = app_top2_hit
    detail["your_top1_win"] = your_top1_win
    detail["your_top2_win"] = your_top2_win
    detail["your_top3_win"] = your_top3_win
    detail["needed_top2"] = needed_top2
    detail["waste_top2"] = waste_top2
    detail["miss"] = misses
    detail["plays_used"] = plays_used
    detail["wins_captured"] = wins_captured
    detail["matched_rule_count"] = matched_rule_count
    detail["member_score_0025"] = adjusted["0025"]
    detail["member_score_0225"] = adjusted["0225"]
    detail["member_score_0255"] = adjusted["0255"]
    detail["outcome_score_TOP1_WIN"] = outcome_scores["TOP1_WIN"]
    detail["outcome_score_WASTE"] = outcome_scores["WASTE"]
    detail["outcome_score_NEEDED"] = outcome_scores["NEEDED"]
    detail["outcome_score_MISS"] = outcome_scores["MISS"]
    detail["total_risk"] = total_risk
    detail["BuildMarker"] = BUILD_SLUG
    return metrics, detail


def dedupe_sorted(vals: List[float]) -> List[float]:
    return sorted(set(float(v) for v in vals))


def nearest_values(center: float, pool: List[float], take: int = 2) -> List[float]:
    return dedupe_sorted(sorted(pool, key=lambda v: abs(v - center))[:take])


def build_stage_a_axes() -> Dict[str, List[float]]:
    return {
        "single_w": [0.4, 0.8, 1.2],
        "filtered_w": [0.8, 1.2, 1.6],
        "sep_w": [1.2, 1.6, 2.0],
        "stacked_w": [0.0],
        "miss_penalty": [0.8, 1.0, 1.2],
        "needed_penalty": [0.6, 0.8, 1.0],
        "waste_top1_bonus": [0.2, 0.4, 0.6],
        "top1win_bonus": [0.2, 0.4],
        "gap_threshold": [0.5, 0.75, 1.0],
        "risk_threshold": [1.0, 1.5, 2.0],
        "waste_penalty_weight": [1.0, 1.5],
        "miss_penalty_weight": [2.0, 3.0],
        "top1_reward_weight": [2.0, 3.0],
        "needed_reward_weight": [1.0, 1.5],
    }


def build_grid_from_axes(axes: Dict[str, List[float]], limit: int | None = None) -> Tuple[List[Dict[str, float]], int]:
    keys = list(axes.keys())
    pools = [axes[k] for k in keys]
    total_possible = product_size(pools)
    combos_iter = itertools.product(*pools)
    if limit is not None:
        combos_iter = itertools.islice(combos_iter, int(limit))
    cfgs = []
    for i, combo in enumerate(combos_iter, start=1):
        cfg = {k: float(v) for k, v in zip(keys, combo)}
        cfg["experiment_name"] = f"exp_{i:05d}"
        cfgs.append(cfg)
    return cfgs, total_possible


def build_stage_b_axes(top_df: pd.DataFrame, master_axes: Dict[str, List[float]], top_k: int = 5) -> Dict[str, List[float]]:
    top = top_df.head(top_k).copy()
    axes = {}
    for col in master_axes.keys():
        vals = []
        for c in top[col].tolist():
            vals.extend(nearest_values(float(c), master_axes[col], take=2))
        axes[col] = dedupe_sorted(vals)
    return axes


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

        st.markdown("### Search mode")
        search_mode = st.selectbox("Search strategy", ["staged_search", "full_grid", "manual_grid"], index=0)

        st.markdown("### Manual grid values")
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

        stage_a_limit = st.number_input("Stage A limit (used only if staged search needs a safety cap)", min_value=1, max_value=50000, value=2000, step=1)
        top_k_expand = st.number_input("Top K configs to expand in Stage B", min_value=1, max_value=100, value=5, step=1)
        top_n_details = st.number_input("Store detail for top N experiments only", min_value=1, max_value=100, value=10, step=1)
        max_tests_when_not_full = st.number_input("Max tests when using manual_grid safety cap", min_value=1, max_value=50000, value=500, step=1)
        apply_manual_cap = st.checkbox("Apply manual cap in manual_grid", value=True)

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

                col_names = rule_metadata["column_name"].tolist()
                missing_cols = [c for c in col_names if c not in match_matrix.columns]
                if missing_cols:
                    raise ValueError(f"Match matrix is missing rule columns referenced by rule metadata: {missing_cols[:10]}")

                X = match_matrix[col_names].to_numpy(dtype=float)

                master_axes = {
                    "single_w": dedupe_sorted(parse_float_list(single_ws_text)),
                    "filtered_w": dedupe_sorted(parse_float_list(filtered_ws_text)),
                    "sep_w": dedupe_sorted(parse_float_list(sep_ws_text)),
                    "stacked_w": dedupe_sorted(parse_float_list(stacked_ws_text)),
                    "miss_penalty": dedupe_sorted(parse_float_list(miss_penalties_text)),
                    "needed_penalty": dedupe_sorted(parse_float_list(needed_penalties_text)),
                    "waste_top1_bonus": dedupe_sorted(parse_float_list(waste_bonuses_text)),
                    "top1win_bonus": dedupe_sorted(parse_float_list(top1win_bonuses_text)),
                    "gap_threshold": dedupe_sorted(parse_float_list(gap_thresholds_text)),
                    "risk_threshold": dedupe_sorted(parse_float_list(risk_thresholds_text)),
                    "waste_penalty_weight": dedupe_sorted(parse_float_list(waste_penalty_weights_text)),
                    "miss_penalty_weight": dedupe_sorted(parse_float_list(miss_penalty_weights_text)),
                    "top1_reward_weight": dedupe_sorted(parse_float_list(top1_reward_weights_text)),
                    "needed_reward_weight": dedupe_sorted(parse_float_list(needed_reward_weights_text)),
                }

                summary_rows = []
                progress = st.progress(0.0, text="Planning search...")
                total_possible = 0
                experiments_run = 0

                if search_mode == "staged_search":
                    stage_a_axes = build_stage_a_axes()
                    stage_a_cfgs, stage_a_possible = build_grid_from_axes(stage_a_axes, limit=int(stage_a_limit))
                    total_tests_a = len(stage_a_cfgs)
                    for idx, cfg in enumerate(stage_a_cfgs, start=1):
                        metrics, _ = evaluate_config(prepared, rule_metadata, X, cfg, return_detail=False)
                        metrics["search_stage"] = "A"
                        summary_rows.append(metrics)
                        progress.progress(0.02 + 0.48 * (idx / max(1, total_tests_a)), text=f"Running Stage A ({idx}/{total_tests_a})...")
                    stage_a_df = pd.DataFrame(summary_rows).sort_values(
                        ["objective_score", "misses", "your_top1_wins", "plays_per_win"],
                        ascending=[False, True, False, True]
                    ).reset_index(drop=True)
                    stage_b_axes = build_stage_b_axes(stage_a_df, master_axes, top_k=int(top_k_expand))
                    stage_b_cfgs, stage_b_possible = build_grid_from_axes(stage_b_axes, limit=None)
                    total_tests_b = len(stage_b_cfgs)
                    for idx, cfg in enumerate(stage_b_cfgs, start=1):
                        metrics, _ = evaluate_config(prepared, rule_metadata, X, cfg, return_detail=False)
                        metrics["search_stage"] = "B"
                        summary_rows.append(metrics)
                        progress.progress(0.52 + 0.48 * (idx / max(1, total_tests_b)), text=f"Running Stage B ({idx}/{total_tests_b})...")
                    total_possible = stage_a_possible + stage_b_possible
                else:
                    limit = int(max_tests_when_not_full) if (search_mode == "manual_grid" and apply_manual_cap) else None
                    cfgs, total_possible = build_grid_from_axes(master_axes, limit=limit)
                    total_tests = len(cfgs)
                    for idx, cfg in enumerate(cfgs, start=1):
                        metrics, _ = evaluate_config(prepared, rule_metadata, X, cfg, return_detail=False)
                        metrics["search_stage"] = "FULL" if search_mode == "full_grid" else "MANUAL"
                        summary_rows.append(metrics)
                        progress.progress(0.05 + 0.95 * (idx / max(1, total_tests)), text=f"Running {idx}/{total_tests} experiments...")
                progress.empty()

                summary_df = pd.DataFrame(summary_rows).drop_duplicates(
                    subset=[
                        "single_w","filtered_w","sep_w","stacked_w","miss_penalty","needed_penalty",
                        "waste_top1_bonus","top1win_bonus","gap_threshold","risk_threshold",
                        "waste_penalty_weight","miss_penalty_weight","top1_reward_weight","needed_reward_weight"
                    ],
                    keep="first"
                ).sort_values(
                    ["objective_score", "misses", "your_top1_wins", "plays_per_win"],
                    ascending=[False, True, False, True]
                ).reset_index(drop=True)
                experiments_run = len(summary_df)

                top_n = int(top_n_details)
                top_configs_df = summary_df.head(top_n).copy()
                top_detail_map = {}
                detail_progress = st.progress(0.0, text=f"Building detail for top {top_n} experiments only...")
                for idx, row in enumerate(top_configs_df.itertuples(index=False), start=1):
                    cfg = {
                        "experiment_name": getattr(row, "experiment_name"),
                        "single_w": float(getattr(row, "single_w")),
                        "filtered_w": float(getattr(row, "filtered_w")),
                        "sep_w": float(getattr(row, "sep_w")),
                        "stacked_w": float(getattr(row, "stacked_w")),
                        "miss_penalty": float(getattr(row, "miss_penalty")),
                        "needed_penalty": float(getattr(row, "needed_penalty")),
                        "waste_top1_bonus": float(getattr(row, "waste_top1_bonus")),
                        "top1win_bonus": float(getattr(row, "top1win_bonus")),
                        "gap_threshold": float(getattr(row, "gap_threshold")),
                        "risk_threshold": float(getattr(row, "risk_threshold")),
                        "waste_penalty_weight": float(getattr(row, "waste_penalty_weight")),
                        "miss_penalty_weight": float(getattr(row, "miss_penalty_weight")),
                        "top1_reward_weight": float(getattr(row, "top1_reward_weight")),
                        "needed_reward_weight": float(getattr(row, "needed_reward_weight")),
                    }
                    _, detail_df = evaluate_config(prepared, rule_metadata, X, cfg, return_detail=True)
                    top_detail_map[cfg["experiment_name"]] = detail_df
                    detail_progress.progress(idx / max(1, top_n), text=f"Building detail for top {top_n} experiments only... {idx}/{top_n}")
                detail_progress.empty()

                run_meta = pd.DataFrame([{
                    "build_marker": BUILD_SLUG,
                    "prepared_rows": len(prepared),
                    "rule_metadata_rows": len(rule_metadata),
                    "match_matrix_rows": len(match_matrix),
                    "total_possible_experiments": int(total_possible),
                    "experiments_run": int(experiments_run),
                    "search_mode": search_mode,
                    "top_detail_count": top_n,
                }])

                st.session_state["decision_opt_outputs"] = {
                    "summary_df": summary_df,
                    "top_detail_map": top_detail_map,
                    "run_meta": run_meta,
                    "manifest": manifest,
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
    st.subheader("Top-experiment detail preview")
    exp_names = list(outputs["top_detail_map"].keys())
    picked = st.selectbox("Choose a top experiment to inspect", exp_names)
    detail_df = outputs["top_detail_map"][picked]

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
