# BUILD: feature_engine_extension__core025__2026-04-17_v2

import io
from collections import Counter

import pandas as pd
import streamlit as st


# ================================
# PAGE CONFIG
# ================================

st.set_page_config(page_title="Core025 Feature Engine Extension", layout="wide")

BUILD_LABEL = "BUILD: feature_engine_extension__core025__2026-04-17_v2"


# ================================
# CORE HELPERS
# ================================

def _digits_from_seed(seed: object) -> list[int]:
    seed_str = str(seed).strip().zfill(4)
    if not seed_str.isdigit() or len(seed_str) != 4:
        raise ValueError(f"Invalid seed value: {seed!r}")
    return [int(x) for x in seed_str]


def _parity_pattern(digits: list[int]) -> str:
    return "".join("E" if d % 2 == 0 else "O" for d in digits)


def _highlow_pattern(digits: list[int]) -> str:
    return "".join("H" if d >= 5 else "L" for d in digits)


def _repeat_shape(digits: list[int]) -> str:
    counts = sorted(Counter(digits).values(), reverse=True)

    if counts == [1, 1, 1, 1]:
        return "all_unique"
    if counts == [2, 1, 1]:
        return "one_pair"
    if counts == [2, 2]:
        return "two_pair"
    if counts == [3, 1]:
        return "triple"
    if counts == [4]:
        return "quad"
    return "other"


def _unique_even_odd(digits: list[int]) -> tuple[int, int]:
    evens = {d for d in digits if d % 2 == 0}
    odds = {d for d in digits if d % 2 == 1}
    return len(evens), len(odds)


# ================================
# FEATURE BUILDERS
# ================================

def build_x_repeatshape_parity(digits: list[int]) -> str:
    return f"{_repeat_shape(digits)}|{_parity_pattern(digits)}"


def build_x_repeatshape_highlow(digits: list[int]) -> str:
    return f"{_repeat_shape(digits)}|{_highlow_pattern(digits)}"


def build_x_unique_even(digits: list[int]) -> str:
    even_count, odd_count = _unique_even_odd(digits)
    return f"{even_count}|{odd_count}"


# ================================
# MAIN APPLY FUNCTION
# ================================

def apply_feature_extensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - x_repeatshape_parity
    - x_repeatshape_highlow
    - x_unique_even

    Required:
    - df must contain column: 'seed'
    """

    if "seed" not in df.columns:
        raise ValueError("Prepared dataframe must contain 'seed' column")

    out = df.copy()
    out["seed"] = out["seed"].astype(str).str.strip().str.zfill(4)

    digits_series = out["seed"].apply(_digits_from_seed)

    out["x_repeatshape_parity"] = digits_series.apply(build_x_repeatshape_parity)
    out["x_repeatshape_highlow"] = digits_series.apply(build_x_repeatshape_highlow)
    out["x_unique_even"] = digits_series.apply(build_x_unique_even)

    return out


# ================================
# FILE HELPERS
# ================================

def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype={"seed": str})

    if name.endswith(".txt"):
        # First try tab-delimited, then comma-delimited fallback
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep="\t", dtype={"seed": str})
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, dtype={"seed": str})

    raise ValueError("Unsupported file type. Please upload a CSV or TXT file.")


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"feature": "x_repeatshape_parity", "non_null": int(df["x_repeatshape_parity"].notna().sum()), "unique_values": int(df["x_repeatshape_parity"].nunique(dropna=True))},
        {"feature": "x_repeatshape_highlow", "non_null": int(df["x_repeatshape_highlow"].notna().sum()), "unique_values": int(df["x_repeatshape_highlow"].nunique(dropna=True))},
        {"feature": "x_unique_even", "non_null": int(df["x_unique_even"].notna().sum()), "unique_values": int(df["x_unique_even"].nunique(dropna=True))},
    ]
    return pd.DataFrame(rows)


# ================================
# UI
# ================================

st.title("Core025 Feature Engine Extension")
st.caption(BUILD_LABEL)

st.write(
    """
Uploads a prepared dataset with a `seed` column and adds:

- `x_repeatshape_parity`
- `x_repeatshape_highlow`
- `x_unique_even`
"""
)

uploaded_file = st.file_uploader("Upload prepared dataset (.csv or .txt)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        raw_df = _read_uploaded_file(uploaded_file)

        st.subheader("Input Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        extended_df = apply_feature_extensions(raw_df)

        st.subheader("Extended Output Preview")
        st.dataframe(extended_df.head(50), use_container_width=True)

        st.subheader("Feature Summary")
        st.dataframe(_summary_table(extended_df), use_container_width=True)

        output_name = f"feature_engine_extension_output__core025__2026-04-17_v2.csv"
        st.download_button(
            label="Download extended CSV",
            data=_to_csv_bytes(extended_df),
            file_name=output_name,
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Processing failed: {e}")

else:
    st.info("Upload a file containing a 'seed' column to begin.")
