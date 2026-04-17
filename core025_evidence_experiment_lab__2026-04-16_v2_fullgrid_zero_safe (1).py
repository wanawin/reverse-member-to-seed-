#!/usr/bin/env python3
"""
BUILD: core025_evidence_experiment_lab__2026-04-16_v2_fullgrid_zero_safe

Purpose
-------
Unified experiment runner for Core025.

This app lets you:
1) upload a pure per-event training dataset,
2) upload a merged trait bank,
3) sweep many parameter combinations in one run,
4) score each experiment automatically,
5) compare all tests in one ranked table,
6) download overall summaries and per-test row outputs.

This version fixes:
- leading-zero seed preservation
- full-grid execution by default (unless user caps it)

Full file. No placeholders.
"""

from __future__ import annotations

import io
import itertools
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_evidence_experiment_lab__2026-04-16_v2_fullgrid_zero_safe"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")

MEMBERS = ["0025", "0225", "0255"]
OUTCOMES = ["TOP1_WIN", "WASTE", "NEEDED", "MISS"]


# ============================================================
# Generic helpers
# ============================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int = 100) -> pd.DataFrame:
    return df.head(int(rows)).copy()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", dtype=str)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, dtype=str)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for k, c in nmap.items():
            if key and key in k:
                return c
    if required:
        raise KeyError(f"Required column not found. Tried {list(candidates)}. Available columns: {cols}")
    return None


def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    if not s:
        return None
    if len(s) > 4:
        s = s[:4]
    return s.zfill(4)


def coerce_member_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if nums:
        for token in reversed(nums):
            v = token.zfill(4)
            if v in set(MEMBERS):
                return v
            if token in {"25", "225", "255"}:
                return {"25": "0025", "225": "0225", "255": "0255"}[token]
    s_up = s.upper()
    return s_up if s_up in set(MEMBERS) else None


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


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


# ============================================================
# Feature engineering from seed
# ============================================================
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}


def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in seed]


def as_pair_tokens(seed: str) -> List[str]:
    ds = list(seed)
    out = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            out.append("".join(sorted((ds[i], ds[j]))))
    return out


def as_ordered_adj_pairs(seed: str) -> List[str]:
    return [seed[i:i + 2] for i in range(len(seed) - 1)]


def as_unordered_adj_pairs(seed: str) -> List[str]:
    return ["".join(sorted(seed[i:i + 2])) for i in range(len(seed) - 1)]


def compute_features(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    parity = "".join("E" if x % 2 == 0 else "O" for x in d)
    highlow = "".join("H" if x >= 5 else "L" for x in d)
    ordered_adj = as_ordered_adj_pairs(seed)
    consec_links = 0
    unique_sorted = sorted(set(d))
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    mirrorpair_cnt = sum(1 for a, b in MIRROR_PAIRS if a in cnt and b in cnt)
    pairwise_absdiff = [abs(d[i] - d[j]) for i in range(4) for j in range(i + 1, 4)]
    adj_absdiff = [abs(d[i] - d[i + 1]) for i in range(3)]
    features: Dict[str, object] = {
        "seed_sum": s,
        "seed_sum_lastdigit": s % 10,
        "seed_sum_mod3": s % 3,
        "seed_sum_mod4": s % 4,
        "seed_sum_mod5": s % 5,
        "seed_spread": spread,
        "seed_unique_digits": len(cnt),
        "seed_has_pair": int(max(cnt.values()) >= 2),
        "seed_no_pair": int(max(cnt.values()) == 1),
        "seed_has_trip": int(max(cnt.values()) >= 3),
        "seed_has_quad": int(max(cnt.values()) >= 4),
        "seed_even_cnt": int(sum(x % 2 == 0 for x in d)),
        "seed_odd_cnt": int(sum(x % 2 == 1 for x in d)),
        "seed_high_cnt": int(sum(x >= 5 for x in d)),
        "seed_low_cnt": int(sum(x <= 4 for x in d)),
        "seed_consec_links": consec_links,
        "seed_mirrorpair_cnt": mirrorpair_cnt,
        "seed_pairwise_absdiff_sum": int(sum(pairwise_absdiff)),
        "seed_pairwise_absdiff_max": int(max(pairwise_absdiff)),
        "seed_pairwise_absdiff_min": int(min(pairwise_absdiff)),
        "seed_adj_absdiff_sum": int(sum(adj_absdiff)),
        "seed_adj_absdiff_max": int(max(adj_absdiff)),
        "seed_adj_absdiff_min": int(min(adj_absdiff)),
        "seed_pos1": d[0],
        "seed_pos2": d[1],
        "seed_pos3": d[2],
        "seed_pos4": d[3],
        "seed_first_last_sum": d[0] + d[3],
        "seed_middle_sum": d[1] + d[2],
        "seed_absdiff_outer_inner": abs((d[0] + d[3]) - (d[1] + d[2])),
        "seed_parity_pattern": parity,
        "seed_highlow_pattern": highlow,
        "seed_sorted": "".join(map(str, sorted(d))),
        "seed_pair_tokens": "|".join(sorted(as_pair_tokens(seed))),
        "seed_adj_pairs_ordered": "|".join(ordered_adj),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
        "seed_outer_equal": int(d[0] == d[3]),
        "seed_inner_equal": int(d[1] == d[2]),
        "seed_palindrome_like": int(d[0] == d[3] and d[1] == d[2]),
        "seed_same_adjacent_count": int(sum(d[i] == d[i + 1] for i in range(3))),
        "seed_pos1_lt_pos2": int(d[0] < d[1]),
        "seed_pos2_lt_pos3": int(d[1] < d[2]),
        "seed_pos3_lt_pos4": int(d[2] < d[3]),
        "seed_pos1_lt_pos3": int(d[0] < d[2]),
        "seed_pos2_lt_pos4": int(d[1] < d[3]),
        "seed_pos1_eq_pos2": int(d[0] == d[1]),
        "seed_pos2_eq_pos3": int(d[1] == d[2]),
        "seed_pos3_eq_pos4": int(d[2] == d[3]),
        "seed_outer_gt_inner": int((d[0] + d[3]) > (d[1] + d[2])),
        "seed_sum_even": int(s % 2 == 0),
        "seed_sum_high_20plus": int(s >= 20),
    }
    for k in DIGITS:
        features[f"seed_has{k}"] = int(k in cnt)
        features[f"seed_cnt{k}"] = int(cnt.get(k, 0))
    shape = "".join(map(str, sorted(cnt.values(), reverse=True)))
    features["seed_repeat_shape"] = {
        "1111": "all_unique",
        "211": "one_pair",
        "22": "two_pair",
        "31": "trip",
        "4": "quad",
    }.get(shape, f"shape_{shape}")
    features["cnt_0_3"] = int(sum(0 <= x <= 3 for x in d))
    features["cnt_4_6"] = int(sum(4 <= x <= 6 for x in d))
    features["cnt_7_9"] = int(sum(7 <= x <= 9 for x in d))
    pair_counts = Counter(as_pair_tokens(seed))
    for a in range(10):
        for b in range(a, 10):
            tok = f"{a}{b}"
            features[f"pair_has_{tok}"] = int(pair_counts.get(tok, 0) > 0)
    for a in range(10):
        for b in range(10):
            tok = f"{a}{b}"
            features[f"adj_ord_has_{tok}"] = int(tok in ordered_adj)
    return features


def prepare_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    seed_col = find_col(df_raw, ["seed", "PrevSeed", "prev_seed"], required=True)
    top1_col = find_col(df_raw, ["Top1", "top1"], required=False)
    top2_col = find_col(df_raw, ["Top2", "top2"], required=False)
    win_col = find_col(df_raw, ["winning_member", "WinningMember"], required=False)
    date_col = find_col(df_raw, ["transition_date", "PlayDate", "date"], required=False)
    stream_col = find_col(df_raw, ["stream", "StreamKey", "stream_id"], required=False)

    out = pd.DataFrame()
    out["seed"] = df_raw[seed_col].apply(canonical_seed)
    out["Top1_actual"] = df_raw[top1_col].apply(coerce_member_text) if top1_col else None
    out["Top2_actual"] = df_raw[top2_col].apply(coerce_member_text) if top2_col else None
    out["WinningMember"] = df_raw[win_col].apply(coerce_member_text) if win_col else None
    out["PlayDate"] = df_raw[date_col].apply(safe_str) if date_col else ""
    out["StreamKey"] = df_raw[stream_col].apply(safe_str) if stream_col else ""
    out = out.dropna(subset=["seed"]).reset_index(drop=True)

    feat = out["seed"].apply(compute_features).apply(pd.Series)
    out = pd.concat([out.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    out["BuildMarker"] = BUILD_SLUG
    return out


# ============================================================
# Rule parsing / normalization
# ============================================================
def parse_condition(token: str) -> Tuple[str, Optional[str], Optional[str]]:
    t = token.strip()
    for op in ["<=", ">=", "=="]:
        if op in t:
            left, right = t.split(op, 1)
            return left.strip(), op, right.strip()
    return t, None, None


def coerce_compare_value(raw: str):
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _truthy(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"", "0", "false", "none", "nan"}:
            return False
        return True
    try:
        return float(val) != 0.0
    except Exception:
        return bool(val)


def match_single_trait(row: pd.Series, trait: str) -> bool:
    col, op, raw_val = parse_condition(trait)
    if col not in row.index:
        return False
    left = row[col]

    if op is None:
        return _truthy(left)

    right = coerce_compare_value(raw_val)
    try:
        if op == "==":
            return str(left) == str(right) if isinstance(right, str) else float(left) == float(right)
        if op == "<=":
            return float(left) <= float(right)
        if op == ">=":
            return float(left) >= float(right)
    except Exception:
        return False
    return False


def match_rule(row: pd.Series, rule_text: str) -> bool:
    parts = [p.strip() for p in str(rule_text).split(" AND ") if p.strip()]
    if not parts:
        return False
    return all(match_single_trait(row, p) for p in parts)


def infer_target_and_kind(filename: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    fname = Path(filename).name.upper()

    target = None
    for x in MEMBERS + OUTCOMES:
        if x in fname:
            target = x
            break
    if target is None and "target" in df.columns and df["target"].notna().sum():
        vals = sorted(set(df["target"].astype(str).dropna().tolist()))
        if len(vals) == 1:
            target = vals[0]
    if target is None and "target_value" in df.columns and df["target_value"].notna().sum():
        vals = sorted(set(df["target_value"].astype(str).dropna().tolist()))
        if len(vals) == 1:
            target = vals[0]

    kind = None
    if "kind" in df.columns and df["kind"].notna().sum():
        vals = sorted(set(df["kind"].astype(str).dropna().tolist()))
        if len(vals) == 1:
            kind = vals[0]
    if kind is None:
        if "STACKED_BUCKETS" in fname or "stack" in df.columns:
            kind = "stacked"
        elif "SEPARATOR_TRAITS" in fname or "separator_strength" in df.columns or "separator_flag" in df.columns:
            kind = "separator"
        elif "FILTERED_CANDIDATES" in fname:
            kind = "filtered"
        elif "SINGLE_TRAITS" in fname or "trait" in df.columns:
            kind = "single"

    return target, kind


def normalize_trait_table(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    target, kind = infer_target_and_kind(filename, df)

    out = pd.DataFrame()
    if "rule_text" in df.columns:
        out["rule_text"] = df["rule_text"].astype(str)
    elif "trait" in df.columns:
        out["rule_text"] = df["trait"].astype(str)
    elif "stack" in df.columns:
        out["rule_text"] = df["stack"].astype(str)
    else:
        raise ValueError(f"Trait file has no usable rule column: {filename}")

    if "target" in df.columns:
        out["target"] = df["target"].astype(str)
    elif "target_value" in df.columns:
        out["target"] = df["target_value"].astype(str)
    else:
        out["target"] = target

    if "kind" in df.columns:
        out["kind"] = df["kind"].astype(str)
    else:
        out["kind"] = kind or "unknown"

    out["support"] = pd.to_numeric(df["support"], errors="coerce").fillna(0) if "support" in df.columns else 0
    out["hit_rate_true"] = pd.to_numeric(df["hit_rate_true"], errors="coerce").fillna(0) if "hit_rate_true" in df.columns else 0
    out["gap"] = pd.to_numeric(df["gap"], errors="coerce").fillna(0) if "gap" in df.columns else 0
    out["lift"] = pd.to_numeric(df["lift"], errors="coerce").fillna(0) if "lift" in df.columns else 0
    out["separator_flag"] = pd.to_numeric(df["separator_flag"], errors="coerce").fillna(0) if "separator_flag" in df.columns else 0
    out["source_file"] = Path(filename).name
    out["BuildMarker"] = BUILD_SLUG
    out = out.drop_duplicates(subset=["target", "kind", "rule_text"]).reset_index(drop=True)
    return out


# ============================================================
# Experiment scoring
# ============================================================
def weight_rule(rule: pd.Series, cfg: Dict[str, float]) -> float:
    base_gap = max(0.0, float(rule.get("gap", 0)))
    base_support = float(rule.get("support", 0))
    base_lift = max(0.0, float(rule.get("lift", 0)))
    support_factor = min(1.5, math.log1p(max(base_support, 0)) / 3.0 + 0.5)
    lift_factor = min(1.5, 0.5 + base_lift / 4.0)

    kind = str(rule.get("kind", "single")).lower()
    kind_weight = {
        "single": cfg["single_w"],
        "filtered": cfg["filtered_w"],
        "separator": cfg["sep_w"],
        "stacked": cfg["stacked_w"],
    }.get(kind, cfg["single_w"])

    return kind_weight * max(0.05, base_gap + 0.35 * float(rule.get("hit_rate_true", 0))) * support_factor * lift_factor


def score_one_row(row: pd.Series, trait_bank: pd.DataFrame, cfg: Dict[str, float]) -> Dict[str, object]:
    member_scores = {m: 0.0 for m in MEMBERS}
    outcome_scores = {o: 0.0 for o in OUTCOMES}
    matched_count = 0

    for _, rule in trait_bank.iterrows():
        if not match_rule(row, str(rule["rule_text"])):
            continue
        matched_count += 1
        wt = weight_rule(rule, cfg)
        target = str(rule.get("target", ""))

        if target in MEMBERS:
            member_scores[target] += wt
        elif target in OUTCOMES:
            outcome_scores[target] += wt

    adjusted_member_scores = dict(member_scores)
    adjusted_member_scores = {k: v + cfg["top1win_bonus"] * outcome_scores["TOP1_WIN"] for k, v in adjusted_member_scores.items()}
    adjusted_member_scores = {k: v + cfg["waste_top1_bonus"] * outcome_scores["WASTE"] for k, v in adjusted_member_scores.items()}
    total_top1_risk = cfg["miss_penalty"] * outcome_scores["MISS"] + cfg["needed_penalty"] * outcome_scores["NEEDED"]

    ranked_members = sorted(adjusted_member_scores.items(), key=lambda kv: kv[1], reverse=True)
    top1_member, top1_score = ranked_members[0]
    top2_member, top2_score = ranked_members[1]
    gap12 = top1_score - top2_score

    play_mode = "PLAY_TOP1"
    if total_top1_risk > outcome_scores["WASTE"] + outcome_scores["TOP1_WIN"]:
        play_mode = "PLAY_TOP2"
    elif gap12 < cfg["gap_threshold"]:
        play_mode = "PLAY_TOP2"

    return {
        "pred_top1": top1_member,
        "pred_top2": top2_member,
        "pred_gap12": float(gap12),
        "play_mode": play_mode,
        "matched_rule_count": int(matched_count),
    }


def evaluate_experiment(prepared: pd.DataFrame, trait_bank: pd.DataFrame, cfg: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    scored_rows = []
    for _, row in prepared.iterrows():
        scored_rows.append(score_one_row(row, trait_bank, cfg))

    out = pd.concat([prepared.reset_index(drop=True), pd.DataFrame(scored_rows)], axis=1)

    app_top1 = 0
    app_top2 = 0
    misses = 0
    your_top1 = 0
    your_top2 = 0
    your_top3 = 0

    if "WinningMember" in out.columns and out["WinningMember"].notna().sum() > 0:
        for _, r in out.iterrows():
            winner = r["WinningMember"]
            top1 = r["pred_top1"]
            top2 = r["pred_top2"]
            mode = r["play_mode"]

            if top1 == winner:
                app_top1 += 1
                if mode == "PLAY_TOP1":
                    your_top1 += 1
                else:
                    app_top2 += 1
            elif top2 == winner:
                app_top2 += 1
                if mode == "PLAY_TOP2":
                    your_top2 += 1
            else:
                misses += 1

    metrics = {
        "experiment_name": cfg["experiment_name"],
        "rows": int(len(out)),
        "app_top1_hits": int(app_top1),
        "app_top2_hits": int(app_top2),
        "misses": int(misses),
        "your_top1_wins": int(your_top1),
        "your_top2_wins": int(your_top2),
        "your_top3_wins": int(your_top3),
        "pred_play_top1_rows": int((out["play_mode"] == "PLAY_TOP1").sum()),
        "pred_play_top2_rows": int((out["play_mode"] == "PLAY_TOP2").sum()),
        "avg_pred_gap12": float(pd.to_numeric(out["pred_gap12"], errors="coerce").mean()),
        "avg_matched_rule_count": float(pd.to_numeric(out["matched_rule_count"], errors="coerce").mean()),
        "single_w": cfg["single_w"],
        "filtered_w": cfg["filtered_w"],
        "sep_w": cfg["sep_w"],
        "stacked_w": cfg["stacked_w"],
        "miss_penalty": cfg["miss_penalty"],
        "needed_penalty": cfg["needed_penalty"],
        "waste_top1_bonus": cfg["waste_top1_bonus"],
        "top1win_bonus": cfg["top1win_bonus"],
        "gap_threshold": cfg["gap_threshold"],
        "use_member_traits": cfg["use_member_traits"],
        "use_outcome_traits": cfg["use_outcome_traits"],
        "use_stacked_traits": cfg["use_stacked_traits"],
        "BuildMarker": BUILD_SLUG,
    }
    return out, metrics


def apply_trait_filters(
    trait_bank: pd.DataFrame,
    member_gap: float,
    member_support: int,
    outcome_gap: float,
    outcome_support: int,
    use_member_traits: bool,
    use_outcome_traits: bool,
    use_stacked_traits: bool,
) -> pd.DataFrame:
    bank = trait_bank.copy()

    bank["support"] = pd.to_numeric(bank["support"], errors="coerce").fillna(0)
    bank["gap"] = pd.to_numeric(bank["gap"], errors="coerce").fillna(0)
    bank["target"] = bank["target"].astype(str)
    bank["kind"] = bank["kind"].astype(str)

    keep = pd.Series(False, index=bank.index)

    member_mask = bank["target"].isin(MEMBERS)
    outcome_mask = bank["target"].isin(OUTCOMES)

    if use_member_traits:
        keep = keep | (member_mask & (bank["gap"] >= float(member_gap)) & (bank["support"] >= int(member_support)))
    if use_outcome_traits:
        keep = keep | (outcome_mask & (bank["gap"] >= float(outcome_gap)) & (bank["support"] >= int(outcome_support)))

    bank = bank[keep].copy()

    if not use_stacked_traits:
        bank = bank[bank["kind"].str.lower() != "stacked"].copy()

    return bank.reset_index(drop=True)


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
    use_member_traits: bool,
    use_outcome_traits: bool,
    use_stacked_traits: bool,
    max_tests: int,
    run_full_grid: bool,
) -> Tuple[List[Dict[str, object]], int]:
    axes = [
        single_ws, filtered_ws, sep_ws, stacked_ws,
        miss_penalties, needed_penalties, waste_bonuses, top1win_bonuses, gap_thresholds
    ]
    total_possible = product_size(axes)

    combos_iter = itertools.product(*axes)
    combos = list(combos_iter) if run_full_grid else list(itertools.islice(combos_iter, int(max_tests)))

    out = []
    for i, combo in enumerate(combos, start=1):
        cfg = {
            "single_w": float(combo[0]),
            "filtered_w": float(combo[1]),
            "sep_w": float(combo[2]),
            "stacked_w": float(combo[3]),
            "miss_penalty": float(combo[4]),
            "needed_penalty": float(combo[5]),
            "waste_top1_bonus": float(combo[6]),
            "top1win_bonus": float(combo[7]),
            "gap_threshold": float(combo[8]),
            "use_member_traits": bool(use_member_traits),
            "use_outcome_traits": bool(use_outcome_traits),
            "use_stacked_traits": bool(use_stacked_traits),
            "experiment_name": f"exp_{i:04d}",
        }
        out.append(cfg)
    return out, total_possible


# ============================================================
# App
# ============================================================
def main() -> None:
    st.set_page_config(page_title="Core025 Evidence Experiment Lab", layout="wide")
    st.title("Core025 Evidence Experiment Lab")
    st.caption(BUILD_MARKER)

    if "experiment_lab_outputs" not in st.session_state:
        st.session_state["experiment_lab_outputs"] = None

    with st.sidebar:
        st.write(BUILD_MARKER)

        per_event_file = st.file_uploader(
            "Upload pure per-event training CSV",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="per_event_file",
        )
        trait_file = st.file_uploader(
            "Upload merged trait bank CSV",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="trait_file",
        )

        rows_to_show = st.number_input("Rows to preview", min_value=20, max_value=500, value=100, step=20)

        st.markdown("### Trait inclusion")
        use_member_traits = st.checkbox("Use member traits", value=True)
        use_outcome_traits = st.checkbox("Use outcome traits", value=True)
        use_stacked_traits = st.checkbox("Use stacked traits", value=False)

        st.markdown("### Pre-filter thresholds")
        member_gap = st.number_input("Member min gap", min_value=0.0, max_value=5.0, value=0.15, step=0.01, format="%.2f")
        member_support = st.number_input("Member min support", min_value=1, max_value=5000, value=8, step=1)
        outcome_gap = st.number_input("Outcome min gap", min_value=0.0, max_value=5.0, value=0.10, step=0.01, format="%.2f")
        outcome_support = st.number_input("Outcome min support", min_value=1, max_value=5000, value=10, step=1)

        st.markdown("### Sweep values")
        single_ws_text = st.text_area("Single trait weights", value="0.4,0.8,1.2")
        filtered_ws_text = st.text_area("Filtered trait weights", value="0.8,1.2,1.6")
        sep_ws_text = st.text_area("Separator trait weights", value="1.2,1.6,2.0")
        stacked_ws_text = st.text_area("Stacked trait weights", value="1.5,2.0")
        miss_penalties_text = st.text_area("MISS penalties", value="0.8,1.0,1.2")
        needed_penalties_text = st.text_area("NEEDED penalties", value="0.6,0.8,1.0")
        waste_bonuses_text = st.text_area("WASTE Top1 bonuses", value="0.2,0.4,0.6")
        top1win_bonuses_text = st.text_area("TOP1_WIN bonuses", value="0.2,0.4")
        gap_thresholds_text = st.text_area("Top1/Top2 gap thresholds", value="0.5,0.75,1.0")

        run_full_grid = st.checkbox("Run full plausible grid", value=True)
        max_tests = st.number_input("Maximum experiments to run when full grid is OFF", min_value=1, max_value=5000, value=48, step=1)

        run_btn = st.button("Run experiment sweep", type="primary", use_container_width=True)

    if run_btn:
        if per_event_file is None:
            st.error("Upload the pure per-event training CSV.")
            st.session_state["experiment_lab_outputs"] = None
        elif trait_file is None:
            st.error("Upload the merged trait bank CSV.")
            st.session_state["experiment_lab_outputs"] = None
        else:
            try:
                per_event_raw = load_table(per_event_file)
                prepared = prepare_rows(per_event_raw)

                trait_raw = load_table(trait_file)
                trait_bank = normalize_trait_table(trait_raw, trait_file.name)

                filtered_trait_bank = apply_trait_filters(
                    trait_bank=trait_bank,
                    member_gap=float(member_gap),
                    member_support=int(member_support),
                    outcome_gap=float(outcome_gap),
                    outcome_support=int(outcome_support),
                    use_member_traits=bool(use_member_traits),
                    use_outcome_traits=bool(use_outcome_traits),
                    use_stacked_traits=bool(use_stacked_traits),
                )

                if len(filtered_trait_bank) == 0:
                    raise ValueError("Trait bank is empty after filtering.")

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
                    use_member_traits=bool(use_member_traits),
                    use_outcome_traits=bool(use_outcome_traits),
                    use_stacked_traits=bool(use_stacked_traits),
                    max_tests=int(max_tests),
                    run_full_grid=bool(run_full_grid),
                )

                summary_rows = []
                detail_map: Dict[str, pd.DataFrame] = {}

                progress = st.progress(0.0, text="Running experiments...")
                total_tests = len(grid)

                for idx, cfg in enumerate(grid, start=1):
                    detail_df, metrics = evaluate_experiment(prepared, filtered_trait_bank, cfg)
                    summary_rows.append(metrics)
                    detail_map[cfg["experiment_name"]] = detail_df
                    progress.progress(idx / total_tests, text=f"Running experiments... {idx}/{total_tests}")

                progress.empty()

                summary_df = pd.DataFrame(summary_rows).sort_values(
                    ["misses", "app_top1_hits", "app_top2_hits"],
                    ascending=[True, False, False]
                ).reset_index(drop=True)

                trait_inventory = (
                    filtered_trait_bank.groupby(["target", "kind"], dropna=False)
                    .size()
                    .reset_index(name="rows")
                    .sort_values(["target", "kind"])
                    .reset_index(drop=True)
                )
                trait_inventory["BuildMarker"] = BUILD_SLUG

                run_meta = pd.DataFrame([{
                    "build_marker": BUILD_SLUG,
                    "prepared_rows": len(prepared),
                    "filtered_trait_rows": len(filtered_trait_bank),
                    "total_possible_experiments": int(total_possible),
                    "experiments_run": int(len(grid)),
                    "full_grid_enabled": bool(run_full_grid),
                }])

                st.session_state["experiment_lab_outputs"] = {
                    "prepared_df": prepared,
                    "trait_bank_filtered": filtered_trait_bank,
                    "trait_inventory": trait_inventory,
                    "summary_df": summary_df,
                    "detail_map": detail_map,
                    "run_meta": run_meta,
                    "source_file": per_event_file.name,
                    "trait_source_file": trait_file.name,
                }

            except Exception as e:
                st.session_state["experiment_lab_outputs"] = None
                st.error(f"Failed to run experiment lab: {e}")

    outputs = st.session_state.get("experiment_lab_outputs")
    if outputs is None:
        st.info("Upload the pure per-event training CSV and merged trait bank, then click Run experiment sweep.")
        return

    st.success("Experiment sweep complete")
    st.write(f"**Per-event source:** {outputs['source_file']}")
    st.write(f"**Trait bank source:** {outputs['trait_source_file']}")

    st.subheader("Run metadata")
    st.dataframe(outputs["run_meta"], use_container_width=True, hide_index=True)

    st.subheader("Filtered trait inventory used in all tests")
    st.dataframe(outputs["trait_inventory"], use_container_width=True, hide_index=True)

    st.subheader("Experiment ranking")
    st.dataframe(outputs["summary_df"], use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download experiment ranking",
            data=df_to_csv_bytes(outputs["summary_df"]),
            file_name=f"experiment_ranking__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_experiment_ranking",
        )
    with c2:
        st.download_button(
            "Download filtered trait bank",
            data=df_to_csv_bytes(outputs["trait_bank_filtered"]),
            file_name=f"filtered_trait_bank__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_filtered_trait_bank",
        )
    with c3:
        st.download_button(
            "Download trait inventory",
            data=df_to_csv_bytes(outputs["trait_inventory"]),
            file_name=f"trait_inventory__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_trait_inventory",
        )
    with c4:
        st.download_button(
            "Download run metadata",
            data=df_to_csv_bytes(outputs["run_meta"]),
            file_name=f"run_metadata__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_run_meta",
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
        play_counts = (
            detail_df["play_mode"]
            .value_counts(dropna=False)
            .rename_axis("play_mode")
            .reset_index(name="count")
        )
        st.download_button(
            f"Download {picked} play-mode counts",
            data=df_to_csv_bytes(play_counts),
            file_name=f"{picked}__play_mode_counts__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_play_counts_{picked}",
        )

    tab1, tab2 = st.tabs(["Scored rows preview", "Play mode counts"])
    with tab1:
        cols = [
            "PlayDate", "StreamKey", "seed", "WinningMember",
            "pred_top1", "pred_top2", "pred_gap12", "play_mode",
            "app_top1_hit", "app_top2_hit", "miss", "matched_rule_count"
        ]
        cols = [c for c in cols if c in detail_df.columns]
        st.dataframe(safe_display_df(detail_df[cols], rows_to_show), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(play_counts, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
