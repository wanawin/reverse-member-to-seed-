# BUILD: core025_precompute_builder__2026-04-17_v2_feature_fixed

import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(layout="wide")

st.title("Core025 Precompute Builder")
st.write("BUILD: core025_precompute_builder__2026-04-17_v2_feature_fixed")

# ================================
# FEATURE ENGINE (EMBEDDED — NO IMPORTS)
# ================================

def _digits_from_seed(seed):
    seed = str(seed).zfill(4)
    return [int(x) for x in seed]

def _parity_pattern(digits):
    return "".join(["E" if d % 2 == 0 else "O" for d in digits])

def _highlow_pattern(digits):
    return "".join(["H" if d >= 5 else "L" for d in digits])

def _repeat_shape(digits):
    counts = Counter(digits).values()
    counts = sorted(counts, reverse=True)

    if counts == [1,1,1,1]:
        return "all_unique"
    elif counts == [2,1,1]:
        return "one_pair"
    elif counts == [2,2]:
        return "two_pair"
    elif counts == [3,1]:
        return "triple"
    elif counts == [4]:
        return "quad"
    else:
        return "other"

def _unique_even_odd(digits):
    evens = set([d for d in digits if d % 2 == 0])
    odds = set([d for d in digits if d % 2 == 1])
    return len(evens), len(odds)

def apply_feature_extensions(df):
    if "seed" not in df.columns:
        st.error("Missing 'seed' column")
        st.stop()

    df["seed"] = df["seed"].astype(str).str.zfill(4)

    df["x_repeatshape_parity"] = ""
    df["x_repeatshape_highlow"] = ""
    df["x_unique_even"] = ""

    for i in range(len(df)):
        digits = _digits_from_seed(df.iloc[i]["seed"])

        shape = _repeat_shape(digits)
        parity = _parity_pattern(digits)
        hl = _highlow_pattern(digits)
        even_count, odd_count = _unique_even_odd(digits)

        df.at[i, "x_repeatshape_parity"] = f"{shape}|{parity}"
        df.at[i, "x_repeatshape_highlow"] = f"{shape}|{hl}"
        df.at[i, "x_unique_even"] = f"{even_count}|{odd_count}"

    return df

# ================================
# FILE UPLOAD
# ================================

uploaded = st.file_uploader("Upload base dataset CSV", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    st.write("Original rows:", len(df))

    # ================================
    # APPLY FEATURE ENGINE
    # ================================
    df = apply_feature_extensions(df)

    st.success("Feature extension applied")

    st.write(df.head())

    # ================================
    # DOWNLOAD OUTPUTS
    # ================================

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download prepared_training_rows__core025.csv",
        data=csv_data,
        file_name="prepared_training_rows__core025__2026-04-17_v2.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download prepared_training_rows__core025.txt",
        data=csv_data,
        file_name="prepared_training_rows__core025__2026-04-17_v2.txt",
        mime="text/plain"
    )

else:
    st.info("Upload a dataset to begin.")
