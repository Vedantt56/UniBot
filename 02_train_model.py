import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)

def train_and_evaluate():
    print("=== STAGE 2: MODEL TRAINING & EVALUATION ===\n")
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # 1. Load cleaned data and drop loan_id
    df = pd.read_csv("loan_default_clean.csv")
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    X = df.drop(columns=["default"])
    y = df["default"]

    print(f"Dataset Loaded. Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Defaulter ratio in target: {y.mean():.4f}\n")

    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}\n")

    # 2. Stratified train/test split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build sklearn Pipeline with ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 4. Define Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_estimators=100
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_auc = -1
    best_pipeline = None
    results = {}

    # 5. Cross-Validation & Comparison
    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # 5-fold Stratified CV
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"]
        )
        mean_roc_auc = cv_results["test_roc_auc"].mean()
        mean_pr_auc = cv_results["test_average_precision"].mean()
        print(f"[{name}] 5-Fold CV ROC-AUC: {mean_roc_auc:.4f}, PR-AUC: {mean_pr_auc:.4f}")

        # Fit on whole training set
        pipeline.fit(X_train, y_train)
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_probs)
        pr_auc = average_precision_score(y_test, y_probs)

        results[name] = {
            "pipeline": pipeline,
            "y_probs": y_probs,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n---> Best Model based on ROC-AUC: {best_model_name} (ROC-AUC: {best_auc:.4f})\n")

    # 6. Detailed Evaluation of Best Model on Held-Out Test Set
    best_y_probs = results[best_model_name]["y_probs"]
    
    # 7. Tune Decision Threshold to favor recall on defaulters
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_y_probs)
    
    # Select threshold where recall is at least 0.80 (or maximizes F1)
    target_recall = 0.80
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        best_thresh_idx = valid_indices[np.argmax(precisions[valid_indices])]
        chosen_threshold = thresholds[min(best_thresh_idx, len(thresholds)-1)]
    else:
        chosen_threshold = 0.5

    print(f"Default Threshold (0.50) Performance for {best_model_name}:")
    y_pred_def = (best_y_probs >= 0.50).astype(int)
    print(f"  Precision: {precision_score(y_test, y_pred_def):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_def):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred_def):.4f}")

    print(f"\nTuned Decision Threshold ({chosen_threshold:.3f}) Performance (Prioritising Defaulter Recall):")
    y_pred_tuned = (best_y_probs >= chosen_threshold).astype(int)
    print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_tuned):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred_tuned):.4f}")

    # Generate and Save Plots
    # A. ROC Curves
    plt.figure(figsize=(7, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png", dpi=300)
    plt.close()

    # B. Precision-Recall Curves
    plt.figure(figsize=(7, 5))
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_probs"])
        plt.plot(r, p, label=f"{name} (PR-AUC = {res['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/precision_recall_curve.png", dpi=300)
    plt.close()

    # C. Confusion Matrix (Tuned Threshold)
    cm = confusion_matrix(y_test, y_pred_tuned)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Defaulter', 'Defaulter'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix ({best_model_name} @ Threshold {chosen_threshold:.2f})')
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=300)
    plt.close()

    # 8. Top 10 Feature Importances
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    
    # Get feature names after one-hot encoding
    cat_ohe_features = preprocessor_fitted.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_ohe_features

    classifier_fitted = best_pipeline.named_steps["classifier"]
    
    if hasattr(classifier_fitted, "feature_importances_"):
        importances = classifier_fitted.feature_importances_
    elif hasattr(classifier_fitted, "coef_"):
        importances = np.abs(classifier_fitted.coef_[0])
    else:
        importances = np.zeros(len(all_feature_names))

    feat_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n--- TOP 10 MOST IMPORTANT FEATURES ---")
    print(feat_df.head(10).to_string(index=False))

    plt.figure(figsize=(8, 5))
    top_10 = feat_df.head(10).sort_values(by="Importance", ascending=True)
    plt.barh(top_10["Feature"], top_10["Importance"], color="skyblue")
    plt.xlabel("Importance Score")
    plt.title(f"Top 10 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=300)
    plt.close()

    # 9. Save Best Pipeline with joblib
    model_filename = "loan_default_model.pkl"
    joblib.dump({
        "pipeline": best_pipeline,
        "threshold": chosen_threshold,
        "feature_names": all_feature_names
    }, model_filename)
    print(f"\nBest pipeline saved successfully to {model_filename}")

    # 10. Test predict_example function
    print("\n--- TESTING PREDICT_EXAMPLE() FUNCTION ---")
    sample_applicant = {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Bachelor",
        "employment_type": "Salaried",
        "years_employed": 5.0,
        "dependents": 2,
        "city_tier": "Tier-1",
        "home_ownership": "Rent",
        "annual_income": 45000.0,
        "savings_balance": 2000.0,
        "existing_debt": 15000.0,
        "credit_score": 620.0,
        "num_credit_lines": 4,
        "num_late_payments_2y": 2,
        "previous_default": 0,
        "account_age_years": 4.0,
        "loan_purpose": "Personal",
        "loan_amount": 20000.0,
        "loan_term_months": 36,
        "interest_rate": 18.5,
        "dti_ratio": 0.33,
        "application_year": 2023,
        "application_month": 6,
        "income_missing": 0
    }
    
    prob, pred = predict_example(sample_applicant, model_file=model_filename)
    print(f"Sample Applicant Default Probability: {prob:.4f}")
    print(f"Predicted Class (Threshold={chosen_threshold:.2f}): {'DEFAULTER (1)' if pred == 1 else 'NON-DEFAULTER (0)'}")

def predict_example(applicant_dict, model_file="loan_default_model.pkl"):
    """
    Takes one applicant as a dict and returns (default_probability, binary_prediction)
    """
    saved_obj = joblib.load(model_file)
    pipeline = saved_obj["pipeline"]
    threshold = saved_obj["threshold"]
    
    df_single = pd.DataFrame([applicant_dict])
    prob = pipeline.predict_proba(df_single)[0, 1]
    prediction = int(prob >= threshold)
    return prob, prediction

if __name__ == "__main__":
    train_and_evaluate()
