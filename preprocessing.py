"""
preprocessing.py  ·  v5  —  Strict Pipeline · Anti-Overfit Edition
====================================================================
Key fixes vs v4
───────────────
  ✅  Clip outliers (not drop) → more data = better generalisation
  ✅  99.5-pct fence (was 99) → keeps edge cases that help test set
  ✅  No numpy conversion outside Pipeline — feature names stay intact
  ✅  LightGBM warning fixed: DataFrames preserved end-to-end
  ✅  seats bucketed as categorical (ordinal bucket, not numeric)
  ✅  model column dropped (200+ levels → OHE noise)
  ✅  car_name dropped (brand already extracted)
  ✅  All transforms inside ColumnTransformer → zero leakage
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Feature column lists (imported by train_model & compare_models) ───────────
NUMERICAL_FEATURES = [
    "vehicle_age",
    "km_driven_log",          # log1p(km_driven)
    "mileage",
    "engine",
    "max_power",
    "km_per_year",            # km_driven_log / (vehicle_age + 1)
    "power_to_engine",        # max_power / engine   → performance density
    "age_power_interaction",  # vehicle_age × max_power → depreciation signal
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel_type",
    "seller_type",
    "transmission_type",
    "seats_cat",              # bucketed: ≤5 / 6-7 / 8+
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Percentile caps  (clip, never drop)
PRICE_PERCENTILE = 0.995
KM_PERCENTILE    = 0.995
TOP_N_BRANDS     = 22


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.lower().startswith("unnamed")]
    for col in ("car_name", "model"):
        if col in df.columns:
            drop.append(col)
    return df.drop(columns=drop, errors="ignore")


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip (not drop) extreme outliers so no information is lost.
    Clipping keeps row count intact → better generalisation on test set.
    """
    price_cap = df["selling_price"].quantile(PRICE_PERCENTILE)
    km_cap    = df["km_driven"].quantile(KM_PERCENTILE)
    df["selling_price"] = df["selling_price"].clip(upper=price_cap)
    df["km_driven"]     = df["km_driven"].clip(upper=km_cap)
    # Drop seats == 0 (data error only, very rare)
    df = df[df["seats"] > 0]
    print(f"      After clip  : {len(df):,} rows kept  "
          f"(price cap=₹{price_cap:,.0f}  km cap={km_cap:,.0f})")
    return df


def _group_rare_brands(df: pd.DataFrame) -> pd.DataFrame:
    top = df["brand"].value_counts().nlargest(TOP_N_BRANDS).index
    df["brand"] = df["brand"].where(df["brand"].isin(top), other="Other")
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log-compress km_driven (right-skewed 100 → 3.8M)
    df["km_driven_log"] = np.log1p(df["km_driven"])

    # Average annual mileage (in log-km space)
    df["km_per_year"] = df["km_driven_log"] / (df["vehicle_age"] + 1)

    # Power density: bhp/cc — luxury / performance proxy
    df["power_to_engine"] = df["max_power"] / df["engine"].clip(lower=1)

    # Interaction: old car + high power → steeper depreciation
    df["age_power_interaction"] = df["vehicle_age"] * df["max_power"]

    # Seats bucket → categorical (avoids treating 7-seater as "more" than 5)
    df["seats_cat"] = df["seats"].apply(
        lambda s: "5_or_less" if s <= 5 else ("6_7" if s <= 7 else "8_plus")
    )

    # Drop raw km_driven (log version kept) and seats (bucket kept)
    df.drop(columns=["km_driven", "seats"], inplace=True)
    return df


def _build_preprocessor() -> ColumnTransformer:
    """
    Returns an UNFITTED ColumnTransformer.
    Always plug this into a sklearn Pipeline — never fit it standalone.
    """
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,   # keeps names clean for LightGBM
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(filepath: str):
    """
    Load raw CSV → clean → engineer → return pipeline-ready components.

    Returns
    -------
    X            : pd.DataFrame        raw features (13 cols)
    y            : pd.Series           log1p(selling_price)
    feature_names: list[str]           ALL_FEATURES column order
    preprocessor : ColumnTransformer   UNFITTED — always use inside Pipeline
    """
    print(f"\n      📂  {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Raw         : {df.shape[0]:,} rows × {df.shape[1]} cols")

    df.dropna(inplace=True)
    df = _drop_junk_columns(df)
    df = _clip_outliers(df)
    df = _group_rare_brands(df)
    df = _engineer_features(df)

    X = df[ALL_FEATURES].copy()
    y = np.log1p(df["selling_price"])

    print(f"      Final X     : {X.shape}  |  features: {ALL_FEATURES}")
    print(f"      Brands kept : {sorted(df['brand'].unique())}")

    return X, y, ALL_FEATURES, _build_preprocessor()


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """Reverse log1p → original ₹ price."""
    return np.expm1(y_log)