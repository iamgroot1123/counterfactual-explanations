# src/evaluate_cf.py
"""
Evaluate and filter counterfactuals: normalized L2, weighted cost, KNN plausibility, and fairness stub.

Outputs:
  artifacts/counterfactuals_evaluated.csv
  artifacts/counterfactuals_filtered.csv
  artifacts/fairness_recourse.csv
"""
from pathlib import Path
import json, yaml, math
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

ART = Path("artifacts")
RAW = ART / "counterfactuals_raw.json"
TRAIN = Path("data/processed_train.csv")
TEST = Path("data/processed_test.csv")
CFG = Path("config/cf_constraints.yaml")

# ---- Parameters you can tune ----
PLAUS_K = 5                  # k for KNN plausibility
MAX_COST = 1.0               # keep CFs with cost <= this
MAX_NORM_L2 = 5.0            # keep CFs with normalized L2 <= this
NUMERIC_WEIGHT = 1.0         # weight for numeric portion of cost
CATEGORICAL_WEIGHT = 0.7     # weight per categorical swap
# ---------------------------------

def load_all():
    raw = json.load(open(RAW)) if RAW.exists() else []
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST) if TEST.exists() else None
    cfg = yaml.safe_load(open(CFG)) if CFG.exists() else {}
    return raw, train, test, cfg

def group_onehot_prefixes(feature_cols):
    groups = {}
    for c in feature_cols:
        if "_" in c:
            p = c.split("_")[0]
            groups.setdefault(p, []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    return groups

def knn_plausibility(cf_vec, train_matrix, k=5):
    if len(train_matrix) == 0:
        return float("nan")
    nbr = NearestNeighbors(n_neighbors=min(k, len(train_matrix))).fit(train_matrix)
    dists, _ = nbr.kneighbors([cf_vec])
    return float(dists.mean())

def normalized_l2(orig_vec, cf_vec, std_vec):
    # std_vec should not contain zeros (we replace with 1)
    normed = (cf_vec - orig_vec) / std_vec
    return float(np.linalg.norm(normed))

def compute_weighted_cost(orig_vec, cf_vec, numeric_idx, prefix_groups, feature_idx_map):
    # numeric percent-change cost
    num_cost = 0.0
    for i in numeric_idx:
        denom = max(abs(orig_vec[i]), 1.0)
        num_cost += abs(cf_vec[i] - orig_vec[i]) / denom
    # categorical cost: 1 per original feature that flips category
    cat_cost = 0
    for prefix, cols in prefix_groups.items():
        if len(cols) <= 1:
            continue
        idxs = [feature_idx_map[c] for c in cols]
        orig_choice = int(np.argmax(orig_vec[idxs]))
        cf_choice = int(np.argmax(cf_vec[idxs]))
        if orig_choice != cf_choice:
            cat_cost += 1
    return NUMERIC_WEIGHT * num_cost + CATEGORICAL_WEIGHT * cat_cost

def evaluate():
    raw, train, test, cfg = load_all()
    if not raw:
        print("[evaluate] No counterfactuals_raw.json found. Run generate_cf.py first.")
        return

    feature_cols = [c for c in train.columns if c != train.columns[-1]]
    train_mat = train[feature_cols].values
    std_vec = train[feature_cols].std().replace(0, 1).values  # avoid divide-by-zero
    prefix_groups = group_onehot_prefixes(feature_cols)
    feature_idx_map = {c:i for i,c in enumerate(feature_cols)}
    # numeric_idx: features that are standalone numeric (not part of multi-col categorical group)
    numeric_idx = [feature_idx_map[c] for c in feature_cols if len(prefix_groups.get(c.split('_')[0], [])) == 1]

    rows = []
    filtered_rows = []
    for rec in raw:
        q = rec.get("query_idx")
        cf_vals = rec.get("cf_values", {})
        # find original row from test set if available
        orig_row = None
        if test is not None and q is not None and q < len(test):
            orig_row = test.loc[q, feature_cols].astype(float).values
        # build cf vector aligned to feature_cols
        cf_vec = np.array([float(cf_vals.get(c, 0)) for c in feature_cols])
        # plausibility (KNN distance to train)
        plaus = knn_plausibility(cf_vec, train_mat, k=PLAUS_K)
        # normalized_l2 (if orig exists)
        norm_l2 = None
        cost = None
        if orig_row is not None:
            norm_l2 = normalized_l2(orig_row, cf_vec, std_vec)
            cost = compute_weighted_cost(orig_row, cf_vec, numeric_idx, prefix_groups, feature_idx_map)
        # store
        out = {
            "query_idx": q,
            "cf_id": rec.get("cf_id"),
            "l2": rec.get("metrics", {}).get("l2"),
            "sparsity_encoded": rec.get("metrics", {}).get("sparsity"),
            "flipped": rec.get("metrics", {}).get("flipped"),
            "plausibility": plaus,
            "normalized_l2": norm_l2,
            "cost": cost
        }
        rows.append(out)
        # filtering logic: require flipped==1 and plaus not NaN; then thresholds
        keep = True
        if out["flipped"] is not None and out["flipped"] != 1:
            keep = False
        if math.isnan(plaus):
            # keep but warn (no training data)
            pass
        if cost is not None and cost > MAX_COST:
            keep = False
        if norm_l2 is not None and norm_l2 > MAX_NORM_L2:
            keep = False
        if keep:
            filtered_rows.append(out)

    df_all = pd.DataFrame(rows)
    df_filtered = pd.DataFrame(filtered_rows)

    ART.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(ART / "counterfactuals_evaluated.csv", index=False)
    df_filtered.to_csv(ART / "counterfactuals_filtered.csv", index=False)
    print(f"[evaluate] wrote {len(df_all)} evaluated rows, {len(df_filtered)} filtered rows")

    # Basic fairness: if sensitive_features listed in config, compute counts of CFs per group
    fairness_rows = []
    sens = cfg.get("sensitive_features", []) if isinstance(cfg, dict) else []
    if sens and test is not None:
        # only supports a single sensitive feature for simple summary (encoded may be one-hot)
        sf = sens[0]
        # expand to encoded
        def expand(name):
            all_cols = feature_cols
            res = [c for c in all_cols if c.lower().startswith(name.lower() + "_") or c.lower()==name.lower()]
            return res
        sens_cols = expand(sf)
        if sens_cols:
            # map each filtered CF to the original group's encoded column in test
            groups_stats = {}
            try:
                test_enc = test[feature_cols].reset_index(drop=True)
                for idx, row in df_filtered.iterrows():
                    q = int(row["query_idx"])
                    if q < len(test_enc):
                        # find which sensitive encoded col is 1 in original
                        orig = test_enc.loc[q, sens_cols]
                        for c in sens_cols:
                            if float(orig.get(c,0)) == 1:
                                groups_stats.setdefault(c, []).append(row["cost"])
                                break
                fairness_rows = [{"group": g, "avg_cost": float(np.nanmean(v)), "n": len(v)} for g,v in groups_stats.items()]
            except Exception:
                fairness_rows = [{"note":"could not compute group-wise fairness; check encodings"}]
        else:
            fairness_rows = [{"note":"sensitive feature expansion yielded no columns"}]
    else:
        fairness_rows = [{"note":"no sensitive_features in config or test missing"}]

    pd.DataFrame(fairness_rows).to_csv(ART / "fairness_recourse.csv", index=False)
    print("[evaluate] fairness summary written (see artifacts/fairness_recourse.csv)")

if __name__ == "__main__":
    evaluate()
