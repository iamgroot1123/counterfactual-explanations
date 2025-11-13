"""
Generate counterfactuals using DiCE with constraints and post-filtering.

Usage:
  python src/generate_cf.py --n-instances 8 --total-cfs 5 --method genetic

Outputs:
  artifacts/counterfactuals_raw.json
  artifacts/counterfactuals_summary.csv
"""
from pathlib import Path
import argparse, json, joblib, pandas as pd, numpy as np, yaml, warnings
warnings.filterwarnings("ignore")
import dice_ml
from dice_ml import Dice

TRAIN_CSV = Path("data/processed_train.csv")
TEST_CSV = Path("data/processed_test.csv")
MODEL_ARTIFACT = Path("artifacts/model.joblib")
CFG_PATH = Path("config/cf_constraints.yaml")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def expand_to_encoded(names, all_features):
    expanded = []
    all_lower = [f.lower() for f in all_features]
    for name in names or []:
        name = name.lower()
        if name in all_lower:
            expanded.append(all_features[all_lower.index(name)])
            continue
        matches = [f for f in all_features if f.lower().startswith(name + "_") or f.lower().startswith(name + ".")]
        if matches:
            expanded.extend(matches)
            continue
        contains = [f for f in all_features if name in f.lower()]
        if contains:
            expanded.extend(contains)
            continue
    unique = []
    for f in all_features:
        if f in expanded and f not in unique:
            unique.append(f)
    return unique

def load_artifact(artifact_path):
    if not artifact_path.exists():
        raise FileNotFoundError("Model artifact not found; run training first.")
    saved = joblib.load(artifact_path)
    pipeline = saved.get("pipeline") or saved.get("model") or saved
    feature_columns = saved.get("feature_columns", None)
    target_col = saved.get("target_col", None)
    return pipeline, feature_columns, target_col

def group_onehot_prefixes(feature_cols):
    groups = {}
    for c in feature_cols:
        if "_" in c:
            p = c.split("_")[0]
            groups.setdefault(p, []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    return groups

def check_onehot_integrity(cf_vals, groups):
    for prefix, cols in groups.items():
        if len(cols) <= 1:
            continue
        s = 0.0
        for c in cols:
            v = cf_vals.get(c, 0)
            try:
                s += float(v)
            except Exception:
                pass
        if not (abs(s - 1.0) < 1e-6):
            return False, prefix, s
    return True, None, None

def generate_for_instances(pipeline, train_df, test_df, feature_cols, target_col, cfg, n_instances=5, total_cfs=4, method="genetic"):
    if target_col not in train_df.columns:
        raise KeyError("Target column missing in training data.")
    # ensure no bools
    train_df = train_df.copy()
    bool_cols = train_df.select_dtypes(include=['bool']).columns
    train_df[bool_cols] = train_df[bool_cols].astype(int)

    continuous = feature_cols.copy()
    d = dice_ml.Data(dataframe=train_df[[*feature_cols, target_col]], continuous_features=continuous, outcome_name=target_col)
    m = dice_ml.Model(model=pipeline, backend="sklearn", model_type="classifier")
    exp = Dice(d, m, method=method)

    X_test = test_df[feature_cols].reset_index(drop=True)
    preds = pipeline.predict(X_test)
    neg_idx = np.where(preds == 0)[0]
    candidates = neg_idx[:n_instances].tolist() if len(neg_idx)>0 else list(range(min(n_instances, len(X_test))))

    print(f"[generate] Generating CFs for {len(candidates)} instances (indices: {candidates})")
    all_cf_records = []
    groups = group_onehot_prefixes(feature_cols)

    for i, idx in enumerate(candidates):
        instance = X_test.loc[[idx]].reset_index(drop=True)
        # Build constraints
        immut = cfg.get("immutable_features", [])
        immut_expanded = expand_to_encoded(immut, feature_cols)
        permitted_range = {}
        permitted_list = {}
        def is_binary_val(v):
            try:
                if isinstance(v, (bool, np.bool_)):
                    return True
                iv = int(v)
                return iv in (0,1)
            except Exception:
                return False
        for col in immut_expanded:
            if col not in instance.columns:
                continue
            val = instance[col].iloc[0]
            if is_binary_val(val):
                permitted_list[col] = [int(val)]
            else:
                try:
                    fv = float(val)
                    permitted_range[col] = [fv, fv]
                except Exception:
                    permitted_list[col] = [str(val)]
        # add feature ranges
        if "feature_ranges" in cfg and isinstance(cfg["feature_ranges"], dict):
            for feat, rng in cfg["feature_ranges"].items():
                enc_cols = expand_to_encoded([feat], feature_cols)
                for col in enc_cols:
                    if col in permitted_list: continue
                    if col not in permitted_range:
                        try:
                            r0, r1 = float(rng[0]), float(rng[1])
                            permitted_range[col] = [r0, r1]
                        except Exception:
                            pass
        print(f"[generate] Instance idx={idx} -> permitted_range keys: {list(permitted_range.keys())}, permitted_list keys: {list(permitted_list.keys())}")
        # call DiCE (permitted_list not supported in this version, so only use permitted_range)
        res = exp.generate_counterfactuals(
            instance,
            total_CFs=total_cfs,
            desired_class="opposite",
            permitted_range=permitted_range if permitted_range else None,
        )

        # extract CF dataframes robustly
        cf_df_list = []
        try:
            for cf_example in res.cf_examples_list:
                if hasattr(cf_example, "final_cfs_df"):
                    cf_df_list.append(cf_example.final_cfs_df)
                elif isinstance(cf_example, dict) and "final_cfs_df" in cf_example:
                    cf_df_list.append(cf_example["final_cfs_df"])
            if not cf_df_list and hasattr(res, "final_cfs_df"):
                cf_df_list = [res.final_cfs_df]
        except Exception:
            try:
                cf_df_list = [res.cf_examples_list[0].final_cfs_df]
            except Exception:
                print(f"[generate] Could not parse DiCE results for idx {idx}; skipping.")
                continue

        for j, cf_df in enumerate(cf_df_list):
            for r_idx, row in cf_df.iterrows():
                cf_row = row.reindex(index=feature_cols, fill_value=np.nan)
                # case-insensitive salvage
                if cf_row.isna().any():
                    col_map = {c.lower(): c for c in feature_cols}
                    remapped = {}
                    for k, v in row.items():
                        kl = k.lower()
                        if kl in col_map:
                            remapped[col_map[kl]] = v
                    for c in feature_cols:
                        if c in remapped:
                            cf_row[c] = remapped[c]
                original_series = instance.loc[0, feature_cols]
                cf_series = pd.Series(cf_row.values, index=feature_cols)
                # basic metrics
                try:
                    orig_pred = int(pipeline.predict(original_series.to_frame().T)[0])
                    cf_pred = int(pipeline.predict(cf_series.to_frame().T)[0])
                except Exception:
                    orig_pred = None; cf_pred = None
                l2 = float(np.linalg.norm((original_series.values.astype(float) - cf_series.values.astype(float))))
                sparsity = int(np.sum(original_series.values != cf_series.values))
                record = {
                    "query_idx": int(idx),
                    "cf_id": f"{idx}_cf_{j}_{r_idx}",
                    "cf_values": cf_row.to_dict(),
                    "metrics": {"l2": l2, "sparsity": sparsity, "orig_pred": orig_pred, "cf_pred": cf_pred, "flipped": int(orig_pred!=cf_pred) if orig_pred is not None and cf_pred is not None else None}
                }
                all_cf_records.append(record)

    # POST-FILTER: remove CFs violating immutables or one-hot integrity
    # Build groups and X_test for access
    groups = group_onehot_prefixes(feature_cols)
    X_test = test_df[feature_cols].reset_index(drop=True)
    immut = cfg.get("immutable_features", [])
    immut_expanded_all = expand_to_encoded(immut, feature_cols)

    clean_records = []
    for rec in all_cf_records:
        qidx = rec["query_idx"]
        orig_row = X_test.loc[int(qidx)]
        cf_vals = rec["cf_values"]
        violated = False
        # immutable check
        for im in immut_expanded_all:
            if im in cf_vals:
                try:
                    if float(orig_row.get(im,0)) != float(cf_vals.get(im, orig_row.get(im,0))):
                        violated = True
                        break
                except Exception:
                    if str(orig_row.get(im)) != str(cf_vals.get(im, orig_row.get(im))):
                        violated = True
                        break
        if violated:
            continue
        # one-hot integrity
        ok, bad_prefix, s = check_onehot_integrity(cf_vals, groups)
        if not ok:
            continue
        clean_records.append(rec)

    # Save outputs
    out_json = OUT_DIR / "counterfactuals_raw.json"
    with open(out_json, "w") as f:
        json.dump(clean_records, f, indent=2)
    rows = []
    for rec in clean_records:
        rows.append({"query_idx": rec["query_idx"], "cf_id": rec["cf_id"], **rec["metrics"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "counterfactuals_summary.csv", index=False)
    print(f"[generate] Saved {len(clean_records)} CF records to {out_json}")
    return clean_records

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-instances", type=int, default=5)
    p.add_argument("--total-cfs", type=int, default=4)
    p.add_argument("--method", type=str, default="genetic")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline, feature_cols, target_col = load_artifact(MODEL_ARTIFACT)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c != train_df.columns[-1]]
        target_col = train_df.columns[-1]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    cfg = {}
    if CFG_PATH.exists():
        with open(CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    generate_for_instances(
        pipeline=pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        cfg=cfg,
        n_instances=args.n_instances,
        total_cfs=args.total_cfs,
        method=args.method,
    )
