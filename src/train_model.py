"""
Train models for the Counterfactual Explanations project.

Produces:
 - artifacts/model.joblib       (contains best RandomForest model + scaler + metadata)
 - artifacts/logreg.joblib      (optional baseline LogisticRegression pipeline)
 - reports/metrics.json         (aggregated test metrics for models)
 - reports/<model>_report.json  (detailed classification report per model)

Usage:
  python src/train_model.py           # quick train
  python src/train_model.py --tune    # runs GridSearchCV for RandomForest
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")

# Paths
TRAIN_CSV = Path("data/processed_train.csv")
TEST_CSV = Path("data/processed_test.csv")
METADATA_PATH = Path("artifacts/preprocess_metadata.json")
ARTIFACT_MODEL = Path("artifacts/model.joblib")
ARTIFACT_LOGREG = Path("artifacts/logreg.joblib")
REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path("artifacts")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def load_data():
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError("Processed train/test CSVs not found. Run preprocess.py first.")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    # try load metadata
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        feature_cols = meta.get("feature_columns")
        target_col = meta.get("target_col")
    else:
        # infer: last column is target
        feature_cols = [c for c in train.columns[:-1]]
        target_col = train.columns[-1]
    return train, test, feature_cols, target_col


def build_preprocessor(train_df, feature_cols):
    # Decide numeric columns among feature_cols
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    non_numeric = [c for c in feature_cols if c not in numeric_cols]
    # Standard scale numeric columns only; passthrough categorical/one-hot columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough",  # keep other columns in original order
        sparse_threshold=0,
    )
    return preprocessor, numeric_cols, non_numeric


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback if pipeline changes order
            y_proba = None
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    return metrics, report


# ---------- Main training flow ----------
def train_and_save(tune: bool = False, random_state: int = 42):
    train_df, test_df, feature_cols, target_col = load_data()
    print(f"[train] Loaded train: {train_df.shape}, test: {test_df.shape}")
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    preprocessor, numeric_cols, non_numeric = build_preprocessor(train_df, feature_cols)
    print(f"[train] Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"[train] Non-numeric cols ({len(non_numeric)}): {non_numeric}")

    # -------- Logistic Regression baseline pipeline --------
    logreg_pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", LogisticRegression(max_iter=500, random_state=random_state))])
    print("[train] Training LogisticRegression baseline...")
    logreg_pipe.fit(X_train, y_train)
    logreg_metrics, logreg_report = evaluate_model(logreg_pipe, X_test, y_test)
    # save baseline artifact
    joblib.dump({"pipeline": logreg_pipe, "feature_columns": feature_cols, "target_col": target_col}, ARTIFACT_LOGREG)
    with open(REPORTS_DIR / "logreg_report.json", "w") as f:
        json.dump(logreg_report, f, indent=2)
    print(f"[train] LogisticRegression saved to {ARTIFACT_LOGREG}")

    # -------- RandomForest (primary) pipeline --------
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf_pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", rf)])

    if tune:
        print("[train] Running GridSearchCV for RandomForest (small grid)...")
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        }
        grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"[train] GridSearchCV best params: {grid.best_params_}")
        # Save cv results summary
        cv_res = {
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
        }
        with open(REPORTS_DIR / "rf_gridsearch_summary.json", "w") as f:
            json.dump(cv_res, f, indent=2)
    else:
        print("[train] Training RandomForest (no tuning)...")
        best_model = rf_pipe
        best_model.fit(X_train, y_train)

    rf_metrics, rf_report = evaluate_model(best_model, X_test, y_test)
    with open(REPORTS_DIR / "rf_report.json", "w") as f:
        json.dump(rf_report, f, indent=2)

    # Save best RF artifact (store pipeline, feature list and metadata)
    saved_obj = {
        "pipeline": best_model,
        "feature_columns": feature_cols,
        "target_col": target_col,
        "numeric_columns": numeric_cols,
    }
    joblib.dump(saved_obj, ARTIFACT_MODEL)
    print(f"[train] RandomForest artifact saved to {ARTIFACT_MODEL}")

    # Aggregate metrics and write a summary
    summary = {
        "logistic_regression": logreg_metrics,
        "random_forest": rf_metrics,
        "dataset": {"train_rows": int(train_df.shape[0]), "test_rows": int(test_df.shape[0]), "n_features": len(feature_cols)},
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[train] Training complete. Summary metrics written to reports/metrics.json")
    return summary


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train models for Counterfactual Explanations project")
    p.add_argument("--tune", action="store_true", help="Run GridSearchCV for RandomForest")
    p.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(tune=args.tune, random_state=args.random_state)
