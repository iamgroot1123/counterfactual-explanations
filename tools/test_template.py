
# tools/test_template.py
import pandas as pd, yaml, json
from pathlib import Path

ART = Path("artifacts")
MODEL_ARTIFACT = ART / "model.joblib"
TRAIN_CSV = Path("data/processed_train.csv")
feature_columns = None
# try to load feature columns from preprocess metadata or model artifact
if (ART / "preprocess_metadata.json").exists():
    feature_columns = json.load(open(ART / "preprocess_metadata.json")).get("feature_columns")
else:
    import joblib
    saved = joblib.load(MODEL_ARTIFACT)
    feature_columns = saved.get("feature_columns")

train_df = pd.read_csv(TRAIN_CSV)
# Example user inputs (adjust to test)
age=30; sex='female'; job=1; housing='own'; saving_accounts='missing'; checking_account='little'
credit_amount=200; duration=24; purpose='car'

template = pd.DataFrame([ [0.0]*len(feature_columns) ], columns=feature_columns, dtype=float)
template.at[0, "age"] = age
template.at[0, "job"] = job
template.at[0, "credit_amount"] = credit_amount
template.at[0, "duration"] = duration
if f"sex_{sex}" in template.columns:
    template.at[0, f"sex_{sex}"] = 1.0
if f"housing_{housing}" in template.columns:
    template.at[0, f"housing_{housing}"] = 1.0
if f"saving_accounts_{saving_accounts}" in template.columns:
    template.at[0, f"saving_accounts_{saving_accounts}"] = 1.0
if f"checking_account_{checking_account}" in template.columns:
    template.at[0, f"checking_account_{checking_account}"] = 1.0
if f"purpose_{purpose}" in template.columns:
    template.at[0, f"purpose_{purpose}"] = 1.0

# final coercion
template = template.astype(float)

print("DTYPES:", template.dtypes.to_dict())
print("SAMPLE:", template.iloc[0].to_dict())
