"""
Preprocessing script for the German Credit dataset (or similar tabular CSV).

Produces:
  - data/processed_train.csv
  - data/processed_test.csv
  - config/cf_constraints.yaml  (default example if not present)

Improvements in this version:
 - Normalizes column names to snake_case (lowercase, underscores).
 - Drops stray 'Unnamed:*' index columns automatically.
 - Default target is 'risk' (maps 'good'->1, 'bad'->0).
 - Sets sensible defaults for immutable/sensitive features for this dataset.
 - Writes metadata to artifacts/preprocess_metadata.json
"""

from pathlib import Path
import argparse
import yaml
import json
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------- Configuration defaults ----------
DEFAULT_CONFIG_PATH = Path("config/cf_constraints.yaml")
PROCESSED_TRAIN = Path("data/processed_train.csv")
PROCESSED_TEST = Path("data/processed_test.csv")

DEFAULT_CONFIG = {
    # use lowercase snake_case names (we normalize columns)
    "immutable_features": ["age", "sex"],
    "sensitive_features": [],
    "actionable_features": None,       # None -> all non-immutable, non-sensitive features
    "continuous_features": None,       # will be inferred if None
    "categorical_features": None,      # will be inferred if None
    "feature_ranges": {},              # optional: {"credit_amount": [min, max], ...}
    "notes": "Edit immutable/actionable lists and feature_ranges to reflect domain constraints."
}


# ---------- Utility functions ----------
def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(parents=True, exist_ok=True)


def read_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = pd.read_csv(input_path, index_col=None)
    return df


def write_yaml(path: Path, obj: dict):
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _normalize_col(col: str) -> str:
    """Normalize column names to snake_case lower strings."""
    col = str(col).strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w_]", "", col)
    col = col.lower()
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names and drop unnamed index columns."""
    df = df.copy()
    # Drop columns like Unnamed: 0 that are likely index artifacts
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df = df.rename(columns={c: _normalize_col(c) for c in df.columns})
    return df


def infer_feature_types(df: pd.DataFrame, target_col: str = "risk"):
    # Treat object and categorical dtypes as categorical; numeric as continuous
    features = [c for c in df.columns if c != target_col]
    categorical = [c for c in features if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
    continuous = [c for c in features if c not in categorical]
    return continuous, categorical


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Common fixes: strip whitespace in column names already handled, drop fully empty rows/cols, fill simple NaNs
    df = df.copy()
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    # If small number of NaNs, drop rows; otherwise lightweight imputation
    total = df.shape[0] * df.shape[1]
    missing = df.isna().sum().sum()
    if missing > 0:
        if missing / total < 0.02:
            df = df.dropna(axis=0)
        else:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")
    return df


def one_hot_encode(df: pd.DataFrame, categorical_cols):
    # Returns transformed df and final list of feature columns (excluding target)
    df = df.copy()
    if not categorical_cols:
        return df, [c for c in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    feature_cols = [c for c in df_encoded.columns]
    return df_encoded, feature_cols


# ---------- Main preprocessing pipeline ----------
def preprocess(
    input_csv: str,
    target_col: str = "risk",
    test_size: float = 0.2,
    random_state: int = 42,
    save_processed: bool = True,
):
    """
    Load, clean, encode, and split the dataset.
    Also creates a default config YAML if none exists.
    Returns: (train_df, test_df, metadata_dict)
    """
    ensure_dirs()
    in_path = Path(input_csv)
    df_raw = read_csv(in_path)
    print(f"[preprocess] Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Normalize column names and drop Unnamed cols
    df = normalize_columns(df_raw)
    print(f"[preprocess] Columns after normalization: {list(df.columns)}")

    df = basic_cleaning(df)

    # Check for target column (case-insensitive via normalization)
    if target_col not in df.columns:
        # try common alternatives
        alts = ["risk", "creditability", "credit_risk", "class"]
        found = None
        for alt in alts:
            if alt in df.columns:
                found = alt
                break
        if found:
            print(f"[preprocess] Using alternative target column '{found}' found in dataset")
            target_col = found
        else:
            raise KeyError(f"Target column '{target_col}' not found and no common alternative matched. Columns: {list(df.columns)}")

    # Map common target values to 0/1
    y = df[target_col]
    if y.dtype == object or pd.api.types.is_categorical_dtype(y):
        y_unique = sorted([str(v).strip().lower() for v in y.unique()])
        if set(y_unique) == {"good", "bad"} or set(y_unique) == {"bad", "good"}:
            df[target_col] = df[target_col].str.strip().str.lower().map({"good": 1, "bad": 0}).astype(int)
            print("[preprocess] Mapped 'good'/'bad' to 1/0 for target.")
        else:
            # fallback: factorize with more deterministic ordering (keep highest label as 1)
            codes, uniques = pd.factorize(df[target_col])
            if len(uniques) == 2:
                # map first unique -> 0, second -> 1
                mapping = {uniques[0]: 0, uniques[1]: 1}
                df[target_col] = df[target_col].map(mapping).astype(int)
                print(f"[preprocess] Factorized target with mapping {mapping}")
            else:
                # fallback to median-split (shouldn't happen for German credit)
                df[target_col] = (pd.to_numeric(df[target_col], errors="coerce") > pd.to_numeric(df[target_col], errors="coerce").median()).astype(int)
                print("[preprocess] Binarized target by median split (fallback).")
    else:
        # numeric; ensure binary 0/1 or convert if 1/2
        y_vals = sorted(df[target_col].dropna().unique())
        if set(y_vals) == {1, 2}:
            df[target_col] = df[target_col].map({1: 1, 2: 0}).astype(int)
            print("[preprocess] Mapped numeric target 1/2 to 1/0.")
        elif not set(y_vals).issubset({0, 1}):
            # fallback to median split
            df[target_col] = (df[target_col] > df[target_col].median()).astype(int)
            print("[preprocess] Binarized numeric target by median split (fallback).")

    # Infer feature types (after normalization)
    continuous, categorical = infer_feature_types(df, target_col=target_col)
    print(f"[preprocess] Inferred {len(continuous)} continuous and {len(categorical)} categorical features")

    # Load or create config
    if DEFAULT_CONFIG_PATH.exists():
        cfg = read_yaml(DEFAULT_CONFIG_PATH)
        print(f"[preprocess] Loaded existing config: {DEFAULT_CONFIG_PATH}")
    else:
        cfg = DEFAULT_CONFIG.copy()
        cfg["continuous_features"] = continuous
        cfg["categorical_features"] = categorical
        imm = cfg.get("immutable_features", [])
        sens = cfg.get("sensitive_features", [])
        candidate_actionables = [c for c in continuous + categorical if c not in imm + sens + [target_col]]
        cfg["actionable_features"] = candidate_actionables
        write_yaml(DEFAULT_CONFIG_PATH, cfg)
        print(f"[preprocess] Wrote default config to: {DEFAULT_CONFIG_PATH} (edit before generating CFs if needed)")

    # One-hot encode categorical columns (ensure target isn't encoded)
    cat_cols = cfg.get("categorical_features") or categorical
    cat_cols = [c for c in cat_cols if c != target_col and c in df.columns]

    df_encoded, final_features = one_hot_encode(df, categorical_cols=cat_cols)

    # final_features includes target too only if target was in original columns and not encoded; make sure
    final_feature_cols = [c for c in final_features if c != target_col]

    # Reorder columns: features..., target
    processed_cols = final_feature_cols + [target_col]
    df_processed = df_encoded[processed_cols]

    # Train-test split
    train_df, test_df = train_test_split(
        df_processed, test_size=test_size, random_state=random_state, stratify=df_processed[target_col]
    )

    if save_processed:
        train_df.to_csv(PROCESSED_TRAIN, index=False)
        test_df.to_csv(PROCESSED_TEST, index=False)
        metadata = {
            "target_col": target_col,
            "feature_columns": final_feature_cols,
            "continuous_features": cfg.get("continuous_features"),
            "categorical_features": cfg.get("categorical_features"),
            "immutable_features": cfg.get("immutable_features"),
            "sensitive_features": cfg.get("sensitive_features"),
            "actionable_features": cfg.get("actionable_features"),
            "n_train": train_df.shape[0],
            "n_test": test_df.shape[0],
        }
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/preprocess_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[preprocess] Saved processed train/test to {PROCESSED_TRAIN} / {PROCESSED_TEST}")
        print(f"[preprocess] Saved metadata to artifacts/preprocess_metadata.json")

    summary = {
        "n_rows_raw": df_raw.shape[0],
        "n_rows_processed": df_processed.shape[0],
        "n_features": len(final_feature_cols),
        "feature_sample": final_feature_cols[:10],
        "target_col": target_col,
    }
    print("[preprocess] Summary:", summary)
    return train_df, test_df, metadata


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess dataset for counterfactual explanations project")
    p.add_argument("--input", type=str, default="data/german_credit.csv", help="Path to raw CSV")
    p.add_argument("--target", type=str, default="risk", help="Name of target column (normalized to lowercase)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    p.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(input_csv=args.input, target_col=args.target, test_size=args.test_size, random_state=args.random_state)
