#!/usr/bin/env python3
"""
BUILD: core025_member_evidence_lab__2026-04-16_v2_master_file_support

Purpose
-------
Training / development lab for Core025.

This app is NOT the final daily predictor. It is a reverse-analysis tool that:
1) takes mined trait files for members and outcome groups,
2) scores each row directly from trait evidence,
3) shows which evidence fired,
4) lets you compare that evidence-driven member choice to the actual winner.

This version adds direct support for merged master trait files that use `rule_text`.

Full file. No placeholders.
"""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_member_evidence_lab__2026-04-16_v2_master_file_support"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")

MEMBERS = ["0025", "0225", "0255"]
OUTCOMES = ["TOP1_WIN", "WASTE", "NEEDED", "MISS"]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int = 100) -> pd.DataFrame:
    return df.head(int(rows)).copy()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


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


def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    return s[:4] if len(s) >= 4 else None


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
        "seed_pos1": d[0], "seed_pos2": d[1], "seed_pos3": d[2], "seed_pos4": d[3],
        "seed_first_last_sum": d[0] + d[3], "seed_middle_sum": d[1] + d[2],
        "seed_absdiff_outer_inner": abs((d[0] + d[3]) - (d[1] + d[2])),
        "seed_parity_pattern": parity, "seed_highlow_pattern": highlow,
        "seed_sorted": "".join(map(str, sorted(d))),
        "seed_pair_tokens": "|".join(sorted(as_pair_tokens(seed))),
        "seed_adj_pairs_ordered": "|".join(ordered_adj),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
        "seed_outer_equal": int(d[0] == d[3]),
        "seed_inner_equal": int(d[1] == d[2]),
        "seed_palindrome_like": int(d[0] == d[3] and d[1] == d[2]),
        "seed_same_adjacent_count": int(sum(d[i] == d[i + 1] for i in range(3))),
        "seed_pos1_lt_pos2": int(d[0] < d[1]), "seed_pos2_lt_pos3": int(d[1] < d[2]),
        "seed_pos3_lt_pos4": int(d[2] < d[3]), "seed_pos1_lt_pos3": int(d[0] < d[2]),
        "seed_pos2_lt_pos4": int(d[1] < d[3]), "seed_pos1_eq_pos2": int(d[0] == d[1]),
        "seed_pos2_eq_pos3": int(d[1] == d[2]), "seed_pos3_eq_pos4": int(d[2] == d[3]),
        "seed_outer_gt_inner": int((d[0] + d[3]) > (d[1] + d[2])),
        "seed_sum_even": int(s % 2 == 0), "seed_sum_high_20plus": int(s >= 20),
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


def parse_condition(token: str) -> Tuple[str, str, str]:
    t = token.strip()
    for op in ["<=", ">=", "=="]:
        if op in t:
            left, right = t.split(op, 1)
            return left.strip(), op, right.strip()
    raise ValueError(f"Unsupported trait token: {token}")


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


def match_single_trait(row: pd.Series, trait: str) -> bool:
    col, op, raw_val = parse_condition(trait)
    if col not in row.index:
        return False
    left = row[col]
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


def weight_rule(row: pd.Series, single_w: float, filtered_w: float, sep_w: float, stacked_w: float) -> float:
    base_gap = max(0.0, float(row.get("gap", 0)))
    base_support = float(row.get("support", 0))
    base_lift = max(0.0, float(row.get("lift", 0)))
    support_factor = min(1.5, math.log1p(max(base_support, 0)) / 3.0 + 0.5)
    lift_factor = min(1.5, 0.5 + base_lift / 4.0)

    kind = str(row.get("kind", "single")).lower()
    kind_weight = {
        "single": single_w,
        "filtered": filtered_w,
        "separator": sep_w,
        "stacked": stacked_w,
    }.get(kind, single_w)

    return kind_weight * max(0.05, base_gap + 0.35 * float(row.get("hit_rate_true", 0))) * support_factor * lift_factor


def score_row(
    row: pd.Series,
    trait_bank: pd.DataFrame,
    single_w: float,
    filtered_w: float,
    sep_w: float,
    stacked_w: float,
    miss_penalty: float,
    needed_penalty: float,
    waste_top1_bonus: float,
    top1win_bonus: float,
) -> Dict[str, object]:
    matched: List[Dict[str, object]] = []
    member_scores = {m: 0.0 for m in MEMBERS}
    outcome_scores = {o: 0.0 for o in OUTCOMES}

    for _, rule in trait_bank.iterrows():
        text = str(rule["rule_text"])
        if not match_rule(row, text):
            continue
        wt = weight_rule(rule, single_w, filtered_w, sep_w, stacked_w)
        target = str(rule.get("target", ""))
        kind = str(rule.get("kind", ""))

        if target in MEMBERS:
            member_scores[target] += wt
        elif target in OUTCOMES:
            outcome_scores[target] += wt

        matched.append({
            "target": target,
            "kind": kind,
            "rule_text": text,
            "weight": wt,
            "gap": float(rule.get("gap", 0)),
            "support": float(rule.get("support", 0)),
            "source_file": str(rule.get("source_file", "")),
        })

    adjusted_member_scores = dict(member_scores)
    adjusted_member_scores = {k: v + top1win_bonus * outcome_scores["TOP1_WIN"] for k, v in adjusted_member_scores.items()}
    adjusted_member_scores = {k: v + waste_top1_bonus * outcome_scores["WASTE"] for k, v in adjusted_member_scores.items()}
    total_top1_risk = miss_penalty * outcome_scores["MISS"] + needed_penalty * outcome_scores["NEEDED"]

    ranked_members = sorted(adjusted_member_scores.items(), key=lambda kv: kv[1], reverse=True)
    top1_member, top1_score = ranked_members[0]
    top2_member, top2_score = ranked_members[1]
    gap12 = top1_score - top2_score

    play_mode = "PLAY_TOP1"
    play_reason = "member evidence dominant"
    if total_top1_risk > outcome_scores["WASTE"] + outcome_scores["TOP1_WIN"]:
        play_mode = "PLAY_TOP2"
        play_reason = "outcome risk > top1 confidence"
    elif gap12 < 0.75:
        play_mode = "PLAY_TOP2"
        play_reason = "member gap too small"

    matched_sorted = sorted(matched, key=lambda x: x["weight"], reverse=True)
    top_evidence = matched_sorted[:8]

    return {
        "pred_top1": top1_member,
        "pred_top2": top2_member,
        "pred_top1_score": round(top1_score, 6),
        "pred_top2_score": round(top2_score, 6),
        "pred_gap12": round(gap12, 6),
        "play_mode": play_mode,
        "play_reason": play_reason,
        "member_score_0025": round(adjusted_member_scores["0025"], 6),
        "member_score_0225": round(adjusted_member_scores["0225"], 6),
        "member_score_0255": round(adjusted_member_scores["0255"], 6),
        "outcome_score_TOP1_WIN": round(outcome_scores["TOP1_WIN"], 6),
        "outcome_score_WASTE": round(outcome_scores["WASTE"], 6),
        "outcome_score_NEEDED": round(outcome_scores["NEEDED"], 6),
        "outcome_score_MISS": round(outcome_scores["MISS"], 6),
        "top_evidence_text": " || ".join(
            [f"{x['target']}|{x['kind']}|w={x['weight']:.3f}|{x['rule_text']}" for x in top_evidence]
        ),
        "matched_rule_count": len(matched_sorted),
    }


def evaluate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "WinningMember" not in out.columns or out["WinningMember"].notna().sum() == 0:
        return out

    app_top1 = []
    app_top2 = []
    misses = []
    your_top1 = []
    your_top2 = []
    your_top3 = []

    for _, r in out.iterrows():
        winner = r["WinningMember"]
        top1 = r["pred_top1"]
        top2 = r["pred_top2"]
        mode = r["play_mode"]

        top1_hit = int(top1 == winner)
        top2_hit = int(top1 == winner or top2 == winner)
        miss = 0
        yt1 = 0
        yt2 = 0
        yt3 = 0

        if mode == "PLAY_TOP1":
            if top1 == winner:
                yt1 = 1
            else:
                miss = 1
        else:
            if top1 == winner:
                pass
            elif top2 == winner:
                yt2 = 1
            else:
                miss = 1

        app_top1.append(top1_hit)
        app_top2.append(top2_hit)
        misses.append(miss)
        your_top1.append(yt1)
        your_top2.append(yt2)
        your_top3.append(yt3)

    out["app_top1_hit"] = app_top1
    out["app_top2_hit"] = app_top2
    out["miss"] = misses
    out["your_top1_win"] = your_top1
    out["your_top2_win"] = your_top2
    out["your_top3_win"] = your_top3
    return out


def build_summary(df_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "rows", "value": int(len(df_eval))})
    if "WinningMember" in df_eval.columns and df_eval["WinningMember"].notna().sum() > 0:
        rows.append({"metric": "app_top1_hits", "value": int(df_eval["app_top1_hit"].sum())})
        rows.append({"metric": "app_top2_hits", "value": int(df_eval["app_top2_hit"].sum())})
        rows.append({"metric": "misses", "value": int(df_eval["miss"].sum())})
        rows.append({"metric": "your_top1_wins", "value": int(df_eval["your_top1_win"].sum())})
        rows.append({"metric": "your_top2_wins", "value": int(df_eval["your_top2_win"].sum())})
        rows.append({"metric": "your_top3_wins", "value": int(df_eval["your_top3_win"].sum())})
    rows.append({"metric": "pred_play_top1_rows", "value": int((df_eval["play_mode"] == "PLAY_TOP1").sum())})
    rows.append({"metric": "pred_play_top2_rows", "value": int((df_eval["play_mode"] == "PLAY_TOP2").sum())})
    rows.append({"metric": "avg_pred_gap12", "value": float(pd.to_numeric(df_eval["pred_gap12"], errors="coerce").mean())})
    rows.append({"metric": "avg_matched_rule_count", "value": float(pd.to_numeric(df_eval["matched_rule_count"], errors="coerce").mean())})
    out = pd.DataFrame(rows)
    out["BuildMarker"] = BUILD_SLUG
    return out


def main() -> None:
    st.set_page_config(page_title="Core025 Member Evidence Lab", layout="wide")
    st.title("Core025 Member Evidence Lab")
    st.caption(BUILD_MARKER)

    if "member_lab_outputs" not in st.session_state:
        st.session_state["member_lab_outputs"] = None

    with st.sidebar:
        st.write(BUILD_MARKER)
        per_event_file = st.file_uploader("Upload raw per-event CSV", type=["csv", "txt", "tsv", "xlsx", "xls"], key="per_event")
        trait_files = st.file_uploader(
            "Upload mined trait CSVs (member + outcome files, multiple allowed)",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="trait_files",
        )
        rows_to_show = st.number_input("Rows to preview", min_value=20, max_value=500, value=100, step=20)

        st.markdown("### Rule weighting")
        single_w = st.number_input("Single trait weight", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
        filtered_w = st.number_input("Filtered candidate weight", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        sep_w = st.number_input("Separator trait weight", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        stacked_w = st.number_input("Stacked bucket weight", min_value=0.0, max_value=10.0, value=1.8, step=0.1)

        st.markdown("### Outcome guidance")
        miss_penalty = st.number_input("MISS penalty", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        needed_penalty = st.number_input("NEEDED penalty", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
        waste_top1_bonus = st.number_input("WASTE Top1 bonus", min_value=0.0, max_value=10.0, value=0.6, step=0.1)
        top1win_bonus = st.number_input("TOP1_WIN bonus", min_value=0.0, max_value=10.0, value=0.4, step=0.1)

        run_btn = st.button("Run member evidence lab", type="primary", use_container_width=True)

    if run_btn:
        if per_event_file is None:
            st.error("Upload the raw per-event CSV.")
            st.session_state["member_lab_outputs"] = None
        elif not trait_files:
            st.error("Upload the mined trait files.")
            st.session_state["member_lab_outputs"] = None
        else:
            try:
                per_event_raw = load_table(per_event_file)
                prepared = prepare_rows(per_event_raw)

                trait_tables = []
                issues = []
                for f in trait_files:
                    try:
                        raw = load_table(f)
                        norm = normalize_trait_table(raw, f.name)
                        trait_tables.append(norm)
                    except Exception as e:
                        issues.append(f"{f.name}: {e}")

                if not trait_tables:
                    raise ValueError("None of the uploaded trait files could be parsed.")

                trait_bank = pd.concat(trait_tables, ignore_index=True).drop_duplicates(
                    subset=["target", "kind", "rule_text"]
                ).reset_index(drop=True)

                scored_rows = []
                for _, row in prepared.iterrows():
                    scored_rows.append(
                        score_row(
                            row=row,
                            trait_bank=trait_bank,
                            single_w=float(single_w),
                            filtered_w=float(filtered_w),
                            sep_w=float(sep_w),
                            stacked_w=float(stacked_w),
                            miss_penalty=float(miss_penalty),
                            needed_penalty=float(needed_penalty),
                            waste_top1_bonus=float(waste_top1_bonus),
                            top1win_bonus=float(top1win_bonus),
                        )
                    )

                scored_df = pd.concat([prepared.reset_index(drop=True), pd.DataFrame(scored_rows)], axis=1)
                eval_df = evaluate_predictions(scored_df)
                summary_df = build_summary(eval_df)

                trait_inventory = (
                    trait_bank.groupby(["target", "kind"], dropna=False)
                    .size()
                    .reset_index(name="rows")
                    .sort_values(["target", "kind"])
                    .reset_index(drop=True)
                )
                trait_inventory["BuildMarker"] = BUILD_SLUG

                st.session_state["member_lab_outputs"] = {
                    "prepared_df": prepared,
                    "trait_bank": trait_bank,
                    "trait_inventory": trait_inventory,
                    "eval_df": eval_df,
                    "summary_df": summary_df,
                    "issues": pd.DataFrame({"issue": issues}) if issues else pd.DataFrame(columns=["issue"]),
                    "source_file": per_event_file.name,
                }

            except Exception as e:
                st.session_state["member_lab_outputs"] = None
                st.error(f"Failed to run lab: {e}")

    outputs = st.session_state.get("member_lab_outputs")
    if outputs is None:
        st.info("Upload the per-event CSV and mined trait files, then click Run member evidence lab.")
        return

    st.success("Member evidence lab complete")
    st.write(f"**Source file:** {outputs['source_file']}")

    st.subheader("Summary")
    st.dataframe(outputs["summary_df"], use_container_width=True, hide_index=True)

    st.subheader("Trait inventory actually used")
    st.dataframe(outputs["trait_inventory"], use_container_width=True, hide_index=True)

    if len(outputs["issues"]) > 0:
        st.subheader("Trait-file parsing issues")
        st.dataframe(outputs["issues"], use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download scored rows",
            data=df_to_csv_bytes(outputs["eval_df"]),
            file_name=f"scored_rows__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_scored_rows",
        )
    with c2:
        st.download_button(
            "Download summary",
            data=df_to_csv_bytes(outputs["summary_df"]),
            file_name=f"summary__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_summary",
        )
    with c3:
        st.download_button(
            "Download trait bank",
            data=df_to_csv_bytes(outputs["trait_bank"]),
            file_name=f"trait_bank__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_trait_bank",
        )
    with c4:
        st.download_button(
            "Download trait inventory",
            data=df_to_csv_bytes(outputs["trait_inventory"]),
            file_name=f"trait_inventory__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_trait_inventory",
        )

    tab1, tab2, tab3 = st.tabs(["Scored rows preview", "Top evidence preview", "Trait bank preview"])

    with tab1:
        cols = [
            "PlayDate", "StreamKey", "seed", "WinningMember",
            "pred_top1", "pred_top2", "pred_top1_score", "pred_top2_score", "pred_gap12",
            "play_mode", "play_reason",
            "member_score_0025", "member_score_0225", "member_score_0255",
            "outcome_score_TOP1_WIN", "outcome_score_WASTE", "outcome_score_NEEDED", "outcome_score_MISS",
            "app_top1_hit", "app_top2_hit", "miss",
        ]
        cols = [c for c in cols if c in outputs["eval_df"].columns]
        st.dataframe(safe_display_df(outputs["eval_df"][cols], rows_to_show), use_container_width=True, hide_index=True)

    with tab2:
        cols = [
            "seed", "WinningMember", "pred_top1", "pred_top2",
            "play_mode", "play_reason", "matched_rule_count", "top_evidence_text",
        ]
        cols = [c for c in cols if c in outputs["eval_df"].columns]
        st.dataframe(safe_display_df(outputs["eval_df"][cols], rows_to_show), use_container_width=True, hide_index=True)

    with tab3:
        st.dataframe(safe_display_df(outputs["trait_bank"], rows_to_show), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
