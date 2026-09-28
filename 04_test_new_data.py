import joblib
import pandas as pd
import numpy as np
import importlib
clean_data_module = importlib.import_module("01_clean_data")
clean_data = clean_data_module.clean_data

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
)

def test_new_data():
    print("=== STEP 2: FRESH-DATA GENERALISATION TEST ===\n")
    
    raw_input = "loan_default_new_raw.csv"
    clean_output = "loan_default_new_clean.csv"

    # 1. Clean the fresh raw dataset using saved winsor bounds
    print(f"Cleaning fresh data from {raw_input}...")
    clean_data(input_csv=raw_input, output_csv=clean_output)

    # 2. Load cleaned fresh data
    df_new = pd.read_csv(clean_output)
    if "loan_id" in df_new.columns:
        df_new = df_new.drop(columns=["loan_id"])

    X_new = df_new.drop(columns=["default"])
    y_new = df_new["default"]

    # 3. Load saved model
    saved_obj = joblib.load("loan_default_model.pkl")
    pipeline = saved_obj["pipeline"]
    threshold = 0.35

    # 4. Predict probabilities
    y_probs = pipeline.predict_proba(X_new)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_new, y_probs)
    pr_auc = average_precision_score(y_new, y_probs)
    prec = precision_score(y_new, y_pred)
    rec = recall_score(y_new, y_pred)
    f1 = f1_score(y_new, y_pred)
    cm = confusion_matrix(y_new, y_pred)

    print("\n--- PERFORMANCE ON FRESH DATA (Threshold = 0.35) ---")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    # Comparison against original training/CV performance (~0.77 ROC-AUC)
    orig_roc_auc = 0.7670
    diff = orig_roc_auc - roc_auc

    print(f"\nOriginal CV ROC-AUC: {orig_roc_auc:.4f}")
    print(f"Fresh Data ROC-AUC : {roc_auc:.4f}")
    print(f"ROC-AUC Drop       : {diff:.4f}")

    if diff > 0.05:
        print("Generalisation Result: WARNING (ROC-AUC drop is > 0.05, possible overfitting or cleaning inconsistency)")
    else:
        print("Generalisation Result: PASS (Model generalises well to unseen fresh dataset)")

if __name__ == "__main__":
    test_new_data()
