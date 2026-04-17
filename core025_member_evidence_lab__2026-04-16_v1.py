# BUILD: feature_engine_extension__core025__2026-04-16_v1

import pandas as pd
from collections import Counter


# ================================
# CORE HELPERS
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


# ================================
# FEATURE BUILDERS
# ================================

def build_x_repeatshape_parity(digits):
    shape = _repeat_shape(digits)
    parity = _parity_pattern(digits)
    return f"{shape}|{parity}"


def build_x_repeatshape_highlow(digits):
    shape = _repeat_shape(digits)
    hl = _highlow_pattern(digits)
    return f"{shape}|{hl}"


def build_x_unique_even(digits):
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

    # Ensure string format
    df["seed"] = df["seed"].astype(str).str.zfill(4)

    # Pre-allocate columns
    df["x_repeatshape_parity"] = ""
    df["x_repeatshape_highlow"] = ""
    df["x_unique_even"] = ""

    for i in range(len(df)):
        digits = _digits_from_seed(df.iloc[i]["seed"])

        df.at[i, "x_repeatshape_parity"] = build_x_repeatshape_parity(digits)
        df.at[i, "x_repeatshape_highlow"] = build_x_repeatshape_highlow(digits)
        df.at[i, "x_unique_even"] = build_x_unique_even(digits)

    return df
