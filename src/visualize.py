"""
Create human-readable counterfactual actions from filtered CFs and write top-N slide-ready CSV.

Reads:
  artifacts/counterfactuals_raw.json  (for cf_values)
  artifacts/counterfactuals_filtered.csv  (for filtered CF ids & metrics)
  data/processed_test.csv

Writes:
  artifacts/counterfactuals_readable.csv
  artifacts/top_cfs_for_slides.csv
  artifacts/feature_change_counts.png
"""
from pathlib import Path
import pandas as pd, numpy as np, json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

ART = Path("artifacts")
RAW = ART / "counterfactuals_raw.json"
FILTERED = ART / "counterfactuals_filtered.csv"
TEST = Path("data/processed_test.csv")

READABLE = ART / "counterfactuals_readable.csv"
TOP = ART / "top_cfs_for_slides.csv"
PLOT = ART / "feature_change_counts.png"

def group_prefixes(cols):
    groups = {}
    for c in cols:
        if "_" in c:
            p = c.split("_")[0]
            groups.setdefault(p, []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    return groups

def decode_choice(row, cols):
    for c in cols:
        v = row.get(c,0)
        try:
            if float(v) == 1:
                return c.split("_",1)[1] if "_" in c else str(v)
        except Exception:
            pass
    # fallback: take argmax
    vals = [(c, float(row.get(c,0))) for c in cols]
    vals = sorted(vals, key=lambda x: -x[1])
    return vals[0][0].split("_",1)[1] if "_" in vals[0][0] else str(vals[0][1])

def human_change_string(orig_row, cf_row, prefix_map, feature_cols):
    changes = []
    actions = []
    for orig, cols in prefix_map.items():
        if len(cols) == 1:
            col = cols[0]
            orig_v = orig_row.get(col)
            cf_v = cf_row.get(col, orig_v)
            try:
                if float(orig_v) != float(cf_v):
                    changes.append(orig)
                    actions.append(f"{orig}: {orig_v} → {cf_v}")
            except Exception:
                if str(orig_v) != str(cf_v):
                    changes.append(orig)
                    actions.append(f"{orig}: {orig_v} → {cf_v}")
        else:
            orig_choice = decode_choice(orig_row, cols)
            cf_choice = decode_choice(cf_row, cols)
            if orig_choice != cf_choice:
                changes.append(orig)
                actions.append(f"{orig}: {orig_choice} → {cf_choice}")
    return changes, "; ".join(actions)

def main(top_n=8):
    # load
    raw = json.load(open(RAW)) if RAW.exists() else []
    filtered = pd.read_csv(FILTERED) if FILTERED.exists() else pd.DataFrame()
    test = pd.read_csv(TEST)
    feature_cols = [c for c in test.columns if c != test.columns[-1]]
    prefix_map = group_prefixes(feature_cols)
    test_map = test.reset_index(drop=True).to_dict(orient="index")
    # map filtered CF ids to raw records
    raw_map = {r["cf_id"]: r for r in raw}
    readable_rows = []
    feat_counter = Counter()
    for _, r in filtered.iterrows():
        cfid = r["cf_id"]
        rec = raw_map.get(cfid)
        if not rec:
            continue
        q = int(r["query_idx"])
        if q not in test_map:
            continue
        orig_row = test_map[q]
        cf_vals = rec["cf_values"]
        # ensure cf_vals have all feature cols
        cf_row = {c: cf_vals.get(c, orig_row.get(c,0)) for c in feature_cols}
        changed, action_str = human_change_string(orig_row, cf_row, prefix_map, feature_cols)
        for f in changed:
            feat_counter[f] += 1
        readable_rows.append({
            "query_idx": q,
            "cf_id": cfid,
            "orig_pred": r.get("orig_pred") if "orig_pred" in r else None,
            "cf_pred": r.get("cf_pred") if "cf_pred" in r else None,
            "flipped": r.get("flipped"),
            "l2": r.get("l2"),
            "normalized_l2": r.get("normalized_l2") if "normalized_l2" in r else None,
            "cost": r.get("cost") if "cost" in r else None,
            "human_changed_count": len(changed),
            "changed_features": ",".join(changed),
            "actionable_text": action_str
        })
    # Normalize columns to ensure consistent schema
    expected_cols = ["l2", "normalized_l2", "cost", "sparsity_encoded", "human_changed_count", "flipped", "orig_pred", "cf_pred", "actionable_text"]
    for r in readable_rows:
        for c in expected_cols:
            if c not in r:
                r[c] = None
    readable_df = pd.DataFrame(readable_rows)
    readable_df.to_csv(READABLE, index=False)
    # produce top N
    if readable_df.empty:
        print("[visualize] No readable CFs found after filtering.")
        return
    candidates = readable_df[readable_df["flipped"]==1].copy()
    if candidates.empty:
        candidates = readable_df.copy()
    # sort: fewest human changes, then cost (if present), then normalized_l2
    sort_keys = ["human_changed_count"]
    if "cost" in candidates.columns:
        sort_keys.append("cost")
    if "normalized_l2" in candidates.columns:
        sort_keys.append("normalized_l2")
    candidates = candidates.sort_values(sort_keys, ascending=True)
    top = candidates.head(top_n)
    top.to_csv(TOP, index=False)
    print(f"[visualize] Wrote {len(readable_df)} readable CFs and top {len(top)} to {TOP}")
    # plot
    feat, counts = zip(*feat_counter.most_common()) if feat_counter else ([], [])
    if feat:
        plt.figure(figsize=(8,4))
        plt.bar(feat, counts)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Number of CFs that changed feature")
        plt.tight_layout()
        plt.savefig(PLOT, dpi=150)
        print(f"[visualize] Saved plot to {PLOT}")
    else:
        print("[visualize] no feature changes to plot")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=8)
    args = p.parse_args()
    main(top_n=args.top_n)
