#!/usr/bin/env python3
"""
BUILD: core025_precompute_builder__2026-04-16_v1

Purpose
-------
Create reusable precomputed artifacts for Core025 optimization.

This app:
1) loads the pure per-event training dataset,
2) loads the merged trait bank,
3) preserves leading-zero seeds,
4) computes all seed features once,
5) filters the trait bank once,
6) builds the rule-match matrix once,
7) exports reusable artifacts for later fast experiment runs.

Artifacts exported:
- prepared_training_rows__...csv
- filtered_trait_bank__...csv
- rule_metadata__...csv
- match_matrix__...csv
- precompute_manifest__...csv

Full file. No placeholders.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_precompute_builder__2026-04-16_v1"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")

MEMBERS = ["0025", "0225", "0255"]
OUTCOMES = ["TOP1_WIN", "WASTE", "NEEDED", "MISS"]
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}


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
    out["row_id"] = np.arange(len(df_raw), dtype=int)
    out["seed"] = df_raw[seed_col].apply(canonical_seed)
    out["Top1_actual"] = df_raw[top1_col].apply(coerce_member_text) if top1_col else None
    out["Top2_actual"] = df_raw[top2_col].apply(coerce_member_text) if top2_col else None
    out["WinningMember"] = df_raw[win_col].apply(coerce_member_text) if win_col else None
    out["PlayDate"] = df_raw[date_col].apply(safe_str) if date_col else ""
    out["StreamKey"] = df_raw[stream_col].apply(safe_str) if stream_col else ""
    out = out.dropna(subset=["seed"]).reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)

    feat = out["seed"].apply(compute_features).apply(pd.Series)
    out = pd.concat([out.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    out["BuildMarker"] = BUILD_SLUG
    return out


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

    out["support"] = pd.to_numeric(df["support"], errors="coerce").fillna(0)
    out["hit_rate_true"] = pd.to_numeric(df["hit_rate_true"], errors="coerce").fillna(0)
    out["gap"] = pd.to_numeric(df["gap"], errors="coerce").fillna(0)
    out["lift"] = pd.to_numeric(df["lift"], errors="coerce").fillna(0)
    out["separator_flag"] = pd.to_numeric(df["separator_flag"], errors="coerce").fillna(0) if "separator_flag" in df.columns else 0
    out["source_file"] = Path(filename).name
    out["BuildMarker"] = BUILD_SLUG
    out = out.drop_duplicates(subset=["target", "kind", "rule_text"]).reset_index(drop=True)
    return out


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

    bank = bank.reset_index(drop=True)
    bank["rule_id"] = np.arange(len(bank), dtype=int)
    return bank


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


def build_mask_for_token(prepared: pd.DataFrame, token: str) -> np.ndarray:
    col, op, raw_val = parse_condition(token)
    if col not in prepared.columns:
        return np.zeros(len(prepared), dtype=bool)

    series = prepared[col]

    if op is None:
        if series.dtype == object:
            vals = series.fillna("").astype(str).str.strip().str.lower()
            return ~vals.isin(["", "0", "false", "none", "nan"]).to_numpy()
        try:
            return (pd.to_numeric(series, errors="coerce").fillna(0) != 0).to_numpy()
        except Exception:
            return series.notna().to_numpy()

    right = coerce_compare_value(raw_val)
    try:
        if op == "==":
            if isinstance(right, str):
                return series.astype(str).to_numpy() == str(right)
            return pd.to_numeric(series, errors="coerce").fillna(np.nan).to_numpy() == float(right)
        if op == "<=":
            return pd.to_numeric(series, errors="coerce").fillna(np.nan).to_numpy() <= float(right)
        if op == ">=":
            return pd.to_numeric(series, errors="coerce").fillna(np.nan).to_numpy() >= float(right)
    except Exception:
        return np.zeros(len(prepared), dtype=bool)

    return np.zeros(len(prepared), dtype=bool)


def build_match_matrix(prepared: pd.DataFrame, filtered_trait_bank: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    token_cache: Dict[str, np.ndarray] = {}
    matrix_cols = {}
    meta_rows = []

    for _, rule in filtered_trait_bank.iterrows():
        rule_id = int(rule["rule_id"])
        rule_text = str(rule["rule_text"])
        parts = [p.strip() for p in rule_text.split(" AND ") if p.strip()]
        if not parts:
            mask = np.zeros(len(prepared), dtype=np.uint8)
        else:
            part_masks = []
            for p in parts:
                if p not in token_cache:
                    token_cache[p] = build_mask_for_token(prepared, p)
                part_masks.append(token_cache[p])
            combined = part_masks[0].copy()
            for pm in part_masks[1:]:
                combined = combined & pm
            mask = combined.astype(np.uint8)

        col_name = f"r_{rule_id}"
        matrix_cols[col_name] = mask
        meta_rows.append({
            "rule_id": rule_id,
            "column_name": col_name,
            "target": str(rule["target"]),
            "kind": str(rule["kind"]),
            "rule_text": rule_text,
            "support": float(rule["support"]),
            "gap": float(rule["gap"]),
            "lift": float(rule["lift"]),
            "hit_rate_true": float(rule["hit_rate_true"]),
            "source_file": str(rule["source_file"]),
            "BuildMarker": BUILD_SLUG,
        })

    matrix = pd.DataFrame(matrix_cols)
    matrix.insert(0, "row_id", prepared["row_id"].astype(int).to_numpy())
    metadata = pd.DataFrame(meta_rows)
    return matrix, metadata


def build_manifest(prepared: pd.DataFrame, filtered_trait_bank: pd.DataFrame, match_matrix: pd.DataFrame, rule_metadata: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    rows = [
        {"metric": "build_marker", "value": BUILD_SLUG},
        {"metric": "prepared_rows", "value": int(len(prepared))},
        {"metric": "filtered_trait_rows", "value": int(len(filtered_trait_bank))},
        {"metric": "match_matrix_rows", "value": int(len(match_matrix))},
        {"metric": "match_matrix_rule_columns", "value": int(max(0, match_matrix.shape[1] - 1))},
        {"metric": "rule_metadata_rows", "value": int(len(rule_metadata))},
    ]
    for k, v in params.items():
        rows.append({"metric": f"param__{k}", "value": v})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Core025 Precompute Builder", layout="wide")
    st.title("Core025 Precompute Builder")
    st.caption(BUILD_MARKER)

    if "precompute_outputs" not in st.session_state:
        st.session_state["precompute_outputs"] = None

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

        run_btn = st.button("Build precompute artifacts", type="primary", use_container_width=True)

    if run_btn:
        if per_event_file is None:
            st.error("Upload the pure per-event training CSV.")
            st.session_state["precompute_outputs"] = None
        elif trait_file is None:
            st.error("Upload the merged trait bank CSV.")
            st.session_state["precompute_outputs"] = None
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

                progress = st.progress(0.0, text="Building match matrix...")
                match_matrix, rule_metadata = build_match_matrix(prepared, filtered_trait_bank)
                progress.progress(1.0, text="Precompute complete.")
                progress.empty()

                params = {
                    "member_gap": member_gap,
                    "member_support": member_support,
                    "outcome_gap": outcome_gap,
                    "outcome_support": outcome_support,
                    "use_member_traits": use_member_traits,
                    "use_outcome_traits": use_outcome_traits,
                    "use_stacked_traits": use_stacked_traits,
                    "per_event_source": per_event_file.name,
                    "trait_source": trait_file.name,
                }
                manifest = build_manifest(prepared, filtered_trait_bank, match_matrix, rule_metadata, params)

                trait_inventory = (
                    filtered_trait_bank.groupby(["target", "kind"], dropna=False)
                    .size()
                    .reset_index(name="rows")
                    .sort_values(["target", "kind"])
                    .reset_index(drop=True)
                )
                trait_inventory["BuildMarker"] = BUILD_SLUG

                st.session_state["precompute_outputs"] = {
                    "prepared": prepared,
                    "filtered_trait_bank": filtered_trait_bank,
                    "rule_metadata": rule_metadata,
                    "match_matrix": match_matrix,
                    "manifest": manifest,
                    "trait_inventory": trait_inventory,
                    "per_event_source": per_event_file.name,
                    "trait_source": trait_file.name,
                }

            except Exception as e:
                st.session_state["precompute_outputs"] = None
                st.error(f"Failed to build precompute artifacts: {e}")

    outputs = st.session_state.get("precompute_outputs")
    if outputs is None:
        st.info("Upload the pure per-event training CSV and merged trait bank, then click Build precompute artifacts.")
        return

    st.success("Precompute artifacts built")
    st.write(f"**Per-event source:** {outputs['per_event_source']}")
    st.write(f"**Trait bank source:** {outputs['trait_source']}")

    st.subheader("Manifest")
    st.dataframe(outputs["manifest"], use_container_width=True, hide_index=True)

    st.subheader("Trait inventory")
    st.dataframe(outputs["trait_inventory"], use_container_width=True, hide_index=True)

    st.subheader("Artifact previews")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prepared training rows",
        "Filtered trait bank",
        "Rule metadata",
        "Match matrix preview",
    ])

    with tab1:
        st.dataframe(safe_display_df(outputs["prepared"], rows_to_show), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(safe_display_df(outputs["filtered_trait_bank"], rows_to_show), use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(safe_display_df(outputs["rule_metadata"], rows_to_show), use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(safe_display_df(outputs["match_matrix"], rows_to_show), use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download prepared training rows",
            data=df_to_csv_bytes(outputs["prepared"]),
            file_name=f"prepared_training_rows__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_prepared_training_rows",
        )
    with c2:
        st.download_button(
            "Download filtered trait bank",
            data=df_to_csv_bytes(outputs["filtered_trait_bank"]),
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

    c4, c5, c6 = st.columns(3)
    with c4:
        st.download_button(
            "Download rule metadata",
            data=df_to_csv_bytes(outputs["rule_metadata"]),
            file_name=f"rule_metadata__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rule_metadata",
        )
    with c5:
        st.download_button(
            "Download match matrix",
            data=df_to_csv_bytes(outputs["match_matrix"]),
            file_name=f"match_matrix__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_match_matrix",
        )
    with c6:
        st.download_button(
            "Download precompute manifest",
            data=df_to_csv_bytes(outputs["manifest"]),
            file_name=f"precompute_manifest__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_precompute_manifest",
        )


if __name__ == "__main__":
    main()
