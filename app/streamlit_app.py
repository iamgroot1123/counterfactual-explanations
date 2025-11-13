# app/streamlit_app.py
"""
Streamlit demo for interactive counterfactual explanations.

Features:
 - user enters values for an applicant
 - model predicts (approve/deny)
 - if deny -> generate counterfactuals (DiCE) for that instance with constraints
 - show human-readable actionable suggestions
 - sliders let user apply a chosen CF and immediately re-run the model

Run:
  streamlit run app/streamlit_app.py
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import yaml
import json
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings("ignore")

# ---- Paths (adjust if your repo differs) ----
ART = Path("artifacts")
MODEL_ARTIFACT = ART / "model.joblib"
METADATA = ART / "preprocess_metadata.json"
CFG = Path("config/cf_constraints.yaml")
TRAIN_CSV = Path("data/processed_train.csv")   # for DiCE data object
TEST_CSV = Path("data/processed_test.csv")     # optional

# ---- Helpers (mapping and decode) ----
def load_model_and_metadata():
    if not MODEL_ARTIFACT.exists():
        st.error("Model artifact not found. Run training first (src/train_model.py).")
        st.stop()
    saved = joblib.load(MODEL_ARTIFACT)
    pipeline = saved.get("pipeline") or saved.get("model") or saved
    feature_columns = saved.get("feature_columns")
    target_col = saved.get("target_col")
    return pipeline, feature_columns, target_col

def normalize_feature_cols(cols):
    # ensure columns are strings and in same order
    return [str(c) for c in cols]

def expand_to_encoded(names, all_features):
    # expand original feature names to encoded columns (sex -> sex_male, sex_female)
    expanded = []
    all_lower = [f.lower() for f in all_features]
    for name in names or []:
        name = name.lower()
        if name in all_lower:
            expanded.append(all_features[all_lower.index(name)])
            continue
        matches = [f for f in all_features if f.lower().startswith(name + "_")]
        if matches:
            expanded.extend(matches)
            continue
        contains = [f for f in all_features if name in f.lower()]
        if contains:
            expanded.extend(contains)
    # keep order as in all_features
    unique = []
    for f in all_features:
        if f in expanded and f not in unique:
            unique.append(f)
    return unique

def group_onehot_prefixes(feature_cols):
    groups = {}
    for c in feature_cols:
        if "_" in c:
            p = c.split("_")[0]
            groups.setdefault(p, []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    return groups

def decode_choice(row, cols):
    # row is a dict-like mapping encoded col -> val
    for c in cols:
        try:
            if float(row.get(c, 0)) == 1:
                return c.split("_", 1)[1] if "_" in c else str(row.get(c))
        except Exception:
            pass
    # fallback: max value
    vals = [(c, float(row.get(c, 0) if row.get(c, 0) is not None else 0)) for c in cols]
    vals = sorted(vals, key=lambda x: -x[1])
    if not vals:
        return None
    return vals[0][0].split("_", 1)[1] if "_" in vals[0][0] else str(vals[0][1])

def human_change_string(orig_row, cf_row, prefix_map):
    changes = []
    actions = []
    for orig, cols in prefix_map.items():
        if len(cols) == 1:
            c = cols[0]
            orig_v = orig_row.get(c)
            cf_v = cf_row.get(c, orig_v)
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

def compute_simple_cost(orig, cf, prefix_map):
    num_cost = 0.0; cat_cost = 0
    for pref, cols in prefix_map.items():
        if len(cols) == 1:
            c = cols[0]
            try:
                denom = max(abs(float(orig.get(c,0))), 1.0)
                num_cost += abs(float(cf.get(c, orig.get(c,0))) - float(orig.get(c,0))) / denom
            except Exception:
                pass
        else:
            oc = decode_choice(orig, cols)
            cc = decode_choice(cf, cols)
            if oc != cc:
                cat_cost += 1
    return num_cost + 0.7*cat_cost

def set_col_val_float(template, col, val):
    if col in template.columns:
        try:
            template.at[0, col] = float(val)
        except Exception:
            # fallback: set 0.0 if conversion fails
            template.at[0, col] = 0.0
    return template

# ---- UI: Title and short explanation ----
st.set_page_config(page_title="Counterfactuals Demo — Truth Decoders", layout="wide")
st.title("Counterfactuals Demo — Truth Decoders")
st.markdown(
    "Counterfactual explanations show minimal changes that would flip a model's decision.\n"
    "Enter applicant details below. If the model denies the application, the app will suggest actionable changes."
)

# Initialize session state for realism threshold and CFs
if 'realism' not in st.session_state:
    st.session_state['realism'] = 1.0
if 'readable' not in st.session_state:
    st.session_state['readable'] = []

# ---- Load model & metadata ----
pipeline, feature_columns, target_col = load_model_and_metadata()
feature_columns = normalize_feature_cols(feature_columns)
prefix_map = group_onehot_prefixes(feature_columns)

# Load config and training data for DiCE
cfg = {}
if CFG.exists():
    with open(CFG, "r") as f:
        cfg = yaml.safe_load(f) or {}
if TRAIN_CSV.exists():
    train_df = pd.read_csv(TRAIN_CSV)
    # Ensure numeric dtypes for DiCE compatibility
    for col in feature_columns + [target_col]:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(float)
else:
    st.warning("Processed train CSV not found; DiCE will not run.")
    train_df = None

# ---- Build a simple human input form (matching original features) ----
# NOTE: These are the original features we expect for German Credit: adjust if your dataset differs.
st.sidebar.header("Applicant input")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", options=["male", "female"], index=0)
job = st.sidebar.selectbox("Job (category)", options=[0,1,2,3], index=1)
housing = st.sidebar.selectbox("Housing", options=["own", "rent", "free"], index=0)
saving_accounts = st.sidebar.selectbox("Saving accounts", options=["little","moderate","quite rich","rich","missing"], index=0)
checking_account = st.sidebar.selectbox("Checking account", options=["little","moderate","rich","missing"], index=0)
credit_amount = st.sidebar.number_input("Credit amount", min_value=100, max_value=50000, value=3000)
duration = st.sidebar.number_input("Duration (months)", min_value=1, max_value=120, value=24)
purpose = st.sidebar.selectbox("Purpose", options=[
    "radio/TV","furniture/equipment","car","education","business","domestic appliances","repairs","vacation/others"
], index=2)

# Build a one-row DataFrame aligned with pipeline expected features (encoded)
# Strategy: create a DataFrame using train_df's columns if available, else try best-effort mapping
if train_df is None:
    st.error("Cannot generate CFs because processed training data is missing. Run preprocess.py.")
else:
    # ---------- BEGIN PATCH: robust numeric template creation & debug prints ----------
    # construct a numeric template row with zeros (ensure numeric dtype)
    template = pd.DataFrame([ [0.0]*len(feature_columns) ], columns=feature_columns, dtype=float)

    # set numeric/simple
    template = set_col_val_float(template, "age", age)
    template = set_col_val_float(template, "job", job)
    template = set_col_val_float(template, "credit_amount", credit_amount)
    template = set_col_val_float(template, "duration", duration)

    # set one-hot encoded categories as floats (1.0)
    if f"sex_{sex}" in template.columns:
        template.at[0, f"sex_{sex}"] = 1.0
    else:
        # fallback if sex is single column
        if "sex" in template.columns:
            # try numeric mapping
            template.at[0, "sex"] = 1.0 if str(sex).lower() in ("male","m","1","true") else 0.0

    if f"housing_{housing}" in template.columns:
        template.at[0, f"housing_{housing}"] = 1.0

    sk = saving_accounts
    if f"saving_accounts_{sk}" in template.columns:
        template.at[0, f"saving_accounts_{sk}"] = 1.0

    ck = checking_account
    if f"checking_account_{ck}" in template.columns:
        template.at[0, f"checking_account_{ck}"] = 1.0

    pk = purpose
    if f"purpose_{pk}" in template.columns:
        template.at[0, f"purpose_{pk}"] = 1.0

    # Fill missing numeric columns (non-one-hot) with train median if available
    for c in template.columns:
        # don't overwrite explicitly set one-hot columns with medians (they are floats already but may be 0.0)
        if "_" in c:
            # keep as-is (one-hot)
            continue
        if template.at[0, c] == 0.0 and c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]):
            template.at[0, c] = float(train_df[c].median())

    # Defensive cleanup: force numeric dtype for all columns; convert boolean-like or text to numeric 0/1
    for c in template.columns:
        if not pd.api.types.is_numeric_dtype(template[c]):
            try:
                template[c] = template[c].astype(float)
            except Exception:
                # fallback common string->bool mapping
                template[c] = template[c].apply(lambda x: 1.0 if str(x).lower() in ("true","1","yes") else 0.0)
    # final coercion
    template = template.astype(float)

    # DEBUG: show dtypes and sample values in UI so we can confirm everything is numeric
    st.write("Template dtypes (should be numeric):")
    st.json({k: str(v) for k,v in template.dtypes.to_dict().items()})
    st.write("Template sample row (first 10 cols):")
    st.write(template.iloc[0].to_dict())

    # Now safe to call model predict
    try:
        pred = pipeline.predict(template)[0]
        pred_proba = pipeline.predict_proba(template)[0][1] if hasattr(pipeline, "predict_proba") else None
    except Exception as e:
        st.error(f"Prediction failed (template types): {e}")
        st.stop()
    # ---------- END PATCH ----------

    st.markdown("### Model result")
    if pred == 1:
        st.success("Model predicted: **APPROVE** (good risk)")
        if pred_proba is not None:
            st.write(f"Approval probability: {pred_proba:.3f}")
    else:
        st.error("Model predicted: **DENY** (bad risk)")
        if pred_proba is not None:
            st.write(f"Approval probability: {pred_proba:.3f}")

        # Generate CFs with DiCE for this single instance
        with st.spinner("Generating counterfactual explanations (DiCE)..."):
            try:
                continuous = feature_columns.copy()
                d = dice_ml.Data(
                    dataframe=train_df[[*feature_columns, target_col]],
                    continuous_features=continuous,
                    outcome_name=target_col,
                )
                m = dice_ml.Model(model=pipeline, backend="sklearn", model_type="classifier")
                exp = Dice(d, m, method="genetic")

                # build constraints from cfg (lock immutables)
                permitted_range = {}
                permitted_list = {}
                immut = cfg.get("immutable_features", [])
                immut_expanded = expand_to_encoded(immut, feature_columns)

                # lock immutables to original values
                for col in immut_expanded:
                    if col in template.columns:
                        val = template.at[0, col]

                        # binary / integer one-hot case
                        if str(val) in ("0", "1") or isinstance(val, (int, np.integer)):
                            permitted_list[col] = [int(val)]
                        else:
                            # try to interpret as float; otherwise keep as string
                            try:
                                fv = float(val)
                            except Exception:
                                permitted_list[col] = [str(val)]
                            else:
                                permitted_range[col] = [fv, fv]

                # add feature_ranges from cfg
                if "feature_ranges" in cfg and isinstance(cfg["feature_ranges"], dict):
                    for feat, rng in cfg["feature_ranges"].items():
                        enc_cols = expand_to_encoded([feat], feature_columns)
                        for col in enc_cols:
                            if col not in permitted_range and col not in permitted_list:
                                try:
                                    permitted_range[col] = [float(rng[0]), float(rng[1])]
                                except Exception:
                                    pass

                # Ensure the template is numeric (float) and has no Python bools/strings
                template = template.astype(float)

                # call DiCE
                res = None
                try:
                    res = exp.generate_counterfactuals(template, total_CFs=5, desired_class="opposite",
                                                       permitted_range=permitted_range if permitted_range else None,
                                                       permitted_list=permitted_list if permitted_list else None)
                except TypeError:
                    # DiCE API mismatch: fallback to unconstrained generation and apply post-filtering
                    res = exp.generate_counterfactuals(template, total_CFs=5, desired_class="opposite")

                # parse DiCE result
                cf_dfs = []
                try:
                    for ex in res.cf_examples_list:
                        if hasattr(ex, "final_cfs_df"):
                            cf_dfs.append(ex.final_cfs_df)
                        elif isinstance(ex, dict) and "final_cfs_df" in ex:
                            cf_dfs.append(ex["final_cfs_df"])
                    if not cf_dfs and hasattr(res, "final_cfs_df"):
                        cf_dfs = [res.final_cfs_df]
                except Exception:
                    st.warning("Could not parse DiCE output structure; no CFs shown.")
                    cf_dfs = []

                # convert CFs to readable actions
                readable = []
                X_test_one = template.reset_index(drop=True)  # for original values
                orig_row = X_test_one.loc[0].to_dict()
                # optionally compute simple cost: numeric percent + categorical count

                # collect CFs
                for df_i, cf_df in enumerate(cf_dfs):
                    for r_idx, row in cf_df.iterrows():
                        # row may contain many columns; map to feature_columns
                        cf_row = {c: row.get(c, orig_row.get(c, 0)) for c in feature_columns}
                        # salvage lowercased names
                        for k,v in list(row.items()):
                            kl = k.lower()
                            for c in feature_columns:
                                if kl == c.lower() and pd.isna(cf_row.get(c)):
                                    cf_row[c] = v
                        # post-filter one-hot integrity
                        ok = True
                        for pref, cols in prefix_map.items():
                            if len(cols) <= 1:
                                continue
                            s = 0
                            for c in cols:
                                try:
                                    s += float(cf_row.get(c, 0))
                                except Exception:
                                    pass
                            if not (abs(s - 1.0) < 1e-6):
                                ok = False; break
                        if not ok:
                            continue
                        # human-change string
                        changed, action_str = human_change_string(orig_row, cf_row, prefix_map)
                        cost = compute_simple_cost(orig_row, cf_row, prefix_map)
                        # minimality ranking metric (prefer fewer human changes then lower cost)
                        readable.append({
                            "cf_id": f"u_cf_{df_i}_{r_idx}",
                            "changed": changed,
                            "action": action_str,
                            "cost": cost,
                            "n_changed": len(changed),
                            "cf_row": cf_row
                        })
                if not readable:
                    st.warning("No valid counterfactuals found (after integrity checks).")
                else:
                    # Store readable CFs in session state
                    st.session_state['readable'] = readable
                    # allow user to pick realism threshold to filter
                    st.subheader("Suggested minimal changes")
                    realism = st.slider("Realism threshold (max cost)", min_value=0.0, max_value=5.0, value=st.session_state['realism'], step=0.1, key='realism_slider')
                    st.session_state['realism'] = realism
                    # filter and sort
                    filtered = [r for r in st.session_state['readable'] if r["cost"] <= realism]
                    if not filtered:
                        st.info("No CFs meet the realism threshold. Lower the threshold or try different inputs.")
                    else:
                        # sort by n_changed then cost
                        filtered = sorted(filtered, key=lambda x: (x["n_changed"], x["cost"]))
                        # show top 3 readable bullets
                        for i, rec in enumerate(filtered[:3]):
                            st.markdown(f"**Option {i+1}** — {rec['n_changed']} change(s), score={rec['cost']:.3f}")
                            st.write(rec["action"])
                            if st.button(f"Apply Option {i+1} and re-run prediction", key=f"apply_{i}"):
                                # apply cf_row to template and re-run model
                                new_row = pd.Series(rec["cf_row"])
                                try:
                                    new_pred = pipeline.predict(new_row.to_frame().T)[0]
                                except Exception as e:
                                    st.error(f"Re-prediction failed: {e}")
                                    new_pred = None
                                if new_pred == 1:
                                    st.success("After applying changes: model predicts APPROVE ✅")
                                elif new_pred == 0:
                                    st.error("After applying changes: model still predicts DENY ❌")
                                else:
                                    st.write("Prediction result:", new_pred)
            except Exception as e:
                st.error(f"Counterfactual generation failed: {e}")

if st.sidebar.button("Predict & Suggest CFs"):
    pass  # Button click triggers re-run, but logic is outside
