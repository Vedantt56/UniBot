import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def threshold_analysis():
    print("=== STEP 4: THRESHOLD TRADE-OFF ANALYSIS ===\n")
    os.makedirs("plots", exist_ok=True)

    # Combine test data (held-out clean dataset + fresh clean dataset)
    df_clean1 = pd.read_csv("loan_default_clean.csv")
    df_clean2 = pd.read_csv("loan_default_new_clean.csv") if os.path.exists("loan_default_new_clean.csv") else None

    if df_clean2 is not None:
        df_combined = pd.concat([df_clean1, df_clean2], ignore_index=True)
    else:
        df_combined = df_clean1

    if "loan_id" in df_combined.columns:
        df_combined = df_combined.drop(columns=["loan_id"])

    X = df_combined.drop(columns=["default"])
    y = df_combined["default"]

    # Load saved model
    saved_obj = joblib.load("loan_default_model.pkl")
    pipeline = saved_obj["pipeline"]

    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    records = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        # Good customers wrongly flagged per real defaulter caught (FP / TP)
        fp_per_tp = fp / tp if tp > 0 else np.nan

        records.append({
            "Threshold": t,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "True_Positives (TP)": tp,
            "False_Positives (FP)": fp,
            "False_Negatives (FN)": fn,
            "True_Negatives (TN)": tn,
            "False_Alarms_Per_Defaulter_Caught": round(fp_per_tp, 2)
        })

    results_df = pd.DataFrame(records)
    print(results_df.to_string(index=False))

    # Save table to CSV
    results_df.to_csv("threshold_analysis.csv", index=False)
    print("\nSaved threshold analysis to threshold_analysis.csv")

    # Plot Precision & Recall vs Threshold
    plt.figure(figsize=(7, 5))
    plt.plot(results_df["Threshold"], results_df["Precision"], marker='o', label='Precision', color='blue')
    plt.plot(results_df["Threshold"], results_df["Recall"], marker='s', label='Recall', color='green')
    plt.plot(results_df["Threshold"], results_df["F1_Score"], marker='^', label='F1 Score', color='orange', linestyle='--')
    plt.axvline(x=0.35, color='red', linestyle=':', label='Chosen Threshold (0.35)')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Trade-off: Precision, Recall, and F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/threshold_tradeoff.png", dpi=300)
    plt.close()
    print("Saved trade-off plot to plots/threshold_tradeoff.png")

    # Recommendation synthesis
    rec_at_035 = results_df.loc[results_df["Threshold"] == 0.35, "Recall"].values[0]
    prec_at_035 = results_df.loc[results_df["Threshold"] == 0.35, "Precision"].values[0]
    fa_per_tp = results_df.loc[results_df["Threshold"] == 0.35, "False_Alarms_Per_Defaulter_Caught"].values[0]

    print("\n--- PLAIN-ENGLISH THRESHOLD RECOMMENDATION ---")
    print(f"At the chosen threshold of 0.35, the model achieves a high Recall of {rec_at_035:.1%} on defaulters.")
    print(f"However, because the target class is heavily imbalanced (~10-11% defaulters), Precision is {prec_at_035:.1%}, meaning")
    print(f"there are approximately {fa_per_tp} false alarms (good customers flagged for review) for every 1 actual defaulter caught.")
    print("In credit risk assessment, a missed defaulter (False Negative) leads to heavy loan principal loss, while a false alarm (False Positive)")
    print("merely triggers standard manual underwriter verification. Thus, threshold 0.35 is recommended for risk-averse lending operations.")

if __name__ == "__main__":
    threshold_analysis()
