#!/usr/bin/env python3
"""
BUILD: core025_trait_filter_layer__2026-04-16_v1

Purpose:
--------
Reduce noise in trait bank BEFORE evidence scoring.

Input:
- MASTER_TRAIT_FILE or raw trait files

Output:
- CLEANED trait file (ready for evidence lab)
- Removal stats

NO PLACEHOLDERS
"""

import pandas as pd
import streamlit as st
import io

BUILD = "core025_trait_filter_layer__2026-04-16_v1"

MEMBERS = ["0025","0225","0255"]
OUTCOMES = ["TOP1_WIN","WASTE","NEEDED","MISS"]

def load_file(f):
    return pd.read_csv(f)

def filter_traits(df, member_gap, member_support, outcome_gap, outcome_support, remove_stacked):

    initial = len(df)

    # Ensure columns
    df["gap"] = pd.to_numeric(df.get("gap", 0), errors="coerce").fillna(0)
    df["support"] = pd.to_numeric(df.get("support", 0), errors="coerce").fillna(0)
    df["kind"] = df.get("kind","unknown")
    df["target"] = df.get("target","unknown")

    # Member filter
    member_mask = df["target"].isin(MEMBERS)
    member_keep = (df["gap"] >= member_gap) & (df["support"] >= member_support)

    # Outcome filter
    outcome_mask = df["target"].isin(OUTCOMES)
    outcome_keep = (df["gap"] >= outcome_gap) & (df["support"] >= outcome_support)

    keep_mask = (member_mask & member_keep) | (outcome_mask & outcome_keep)

    df = df[keep_mask].copy()

    # Remove stacked if selected
    if remove_stacked:
        df = df[df["kind"] != "stacked"]

    df = df.drop_duplicates(subset=["target","kind","rule_text"])

    final = len(df)

    stats = pd.DataFrame([
        {"metric":"initial_rules","value":initial},
        {"metric":"final_rules","value":final},
        {"metric":"removed_rules","value":initial-final}
    ])

    return df, stats


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(layout="wide")
    st.title("Core025 Trait Filter Layer")
    st.caption(BUILD)

    with st.sidebar:

        trait_file = st.file_uploader("Upload MASTER trait file", type=["csv"])

        st.markdown("### Member Filters")
        member_gap = st.number_input("Member min gap", value=0.15, step=0.01)
        member_support = st.number_input("Member min support", value=8)

        st.markdown("### Outcome Filters")
        outcome_gap = st.number_input("Outcome min gap", value=0.10, step=0.01)
        outcome_support = st.number_input("Outcome min support", value=10)

        st.markdown("### Structure Control")
        remove_stacked = st.checkbox("Remove stacked rules", value=True)

        run = st.button("Run Filter")

    if not run:
        st.info("Upload file and run")
        return

    if trait_file is None:
        st.error("Upload trait file")
        return

    df = load_file(trait_file)

    filtered, stats = filter_traits(
        df,
        member_gap,
        member_support,
        outcome_gap,
        outcome_support,
        remove_stacked
    )

    st.subheader("Filter Stats")
    st.dataframe(stats)

    st.subheader("Preview Filtered Traits")
    st.dataframe(filtered.head(100))

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download CLEAN trait file",
            data=to_csv(filtered),
            file_name=f"filtered_traits__{BUILD}.csv"
        )

    with col2:
        st.download_button(
            "Download stats",
            data=to_csv(stats),
            file_name=f"filter_stats__{BUILD}.csv"
        )


if __name__ == "__main__":
    main()
