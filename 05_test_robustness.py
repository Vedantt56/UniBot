import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_robustness():
    print("=== STEP 3: ROBUSTNESS TESTS ===\n")
    os.makedirs("plots", exist_ok=True)
    
    saved_obj = joblib.load("loan_default_model.pkl")
    pipeline = saved_obj["pipeline"]
    threshold = 0.35

    # Baseline applicant
    baseline_applicant = {
        "age": 38,
        "gender": "Female",
        "marital_status": "Married",
        "education": "Bachelor",
        "employment_type": "Salaried",
        "years_employed": 8.0,
        "dependents": 1,
        "city_tier": "Tier-2",
        "home_ownership": "Own",
        "annual_income": 600000.0,
        "savings_balance": 150000.0,
        "existing_debt": 80000.0,
        "credit_score": 720.0,
        "num_credit_lines": 4,
        "num_late_payments_2y": 0,
        "previous_default": 0,
        "account_age_years": 6.0,
        "loan_purpose": "Home",
        "loan_amount": 300000.0,
        "loan_term_months": 60,
        "interest_rate": 10.5,
        "dti_ratio": 0.25,
        "application_year": 2023,
        "application_month": 3,
        "income_missing": 0
    }

    # 1. Missing values test
    print("--- 1. MISSING VALUES TEST ---")
    missing_applicant = baseline_applicant.copy()
    missing_applicant["annual_income"] = None
    missing_applicant["credit_score"] = None
    missing_applicant["education"] = None
    missing_applicant["income_missing"] = 1

    try:
        df_miss = pd.DataFrame([missing_applicant])
        prob_miss = pipeline.predict_proba(df_miss)[0, 1]
        print(f"Missing Values Applicant Default Probability: {prob_miss:.4f}")
        print("Missing Values Test: PASS (Pipeline handled NaNs gracefully without crashing)")
    except Exception as e:
        print(f"Missing Values Test: FAIL ({str(e)})")

    # 2. Unseen categories test
    print("\n--- 2. UNSEEN CATEGORIES TEST ---")
    unseen_applicant = baseline_applicant.copy()
    unseen_applicant["loan_purpose"] = "Crypto"

    try:
        df_unseen = pd.DataFrame([unseen_applicant])
        prob_unseen = pipeline.predict_proba(df_unseen)[0, 1]
        print(f"Unseen Category ('Crypto') Applicant Default Probability: {prob_unseen:.4f}")
        print("Unseen Categories Test: PASS (OneHotEncoder ignored unknown category without crashing)")
    except Exception as e:
        print(f"Unseen Categories Test: FAIL ({str(e)})")

    # 3. Monotonic behavior tests & plots
    print("\n--- 3. MONOTONIC BEHAVIOUR TESTS ---")
    
    # A. Vary credit_score from 400 to 850
    scores = np.linspace(400, 850, 46)
    probs_cs = []
    for s in scores:
        app = baseline_applicant.copy()
        app["credit_score"] = s
        probs_cs.append(pipeline.predict_proba(pd.DataFrame([app]))[0, 1])

    plt.figure(figsize=(6, 4))
    plt.plot(scores, probs_cs, marker='o', color='blue')
    plt.xlabel('Credit Score')
    plt.ylabel('Predicted Default Probability')
    plt.title('Monotonic Test: Credit Score vs Default Probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/monotonic_credit_score.png", dpi=300)
    plt.close()

    # B. Vary num_late_payments_2y from 0 to 10
    late_payments = np.arange(0, 11)
    probs_lp = []
    for lp in late_payments:
        app = baseline_applicant.copy()
        app["num_late_payments_2y"] = lp
        probs_lp.append(pipeline.predict_proba(pd.DataFrame([app]))[0, 1])

    plt.figure(figsize=(6, 4))
    plt.plot(late_payments, probs_lp, marker='s', color='red')
    plt.xlabel('Num Late Payments (2y)')
    plt.ylabel('Predicted Default Probability')
    plt.title('Monotonic Test: Late Payments vs Default Probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/monotonic_late_payments.png", dpi=300)
    plt.close()

    # C. Vary dti_ratio from 0.1 to 3.0
    dtis = np.linspace(0.1, 3.0, 30)
    probs_dti = []
    for dti in dtis:
        app = baseline_applicant.copy()
        app["dti_ratio"] = dti
        probs_dti.append(pipeline.predict_proba(pd.DataFrame([app]))[0, 1])

    plt.figure(figsize=(6, 4))
    plt.plot(dtis, probs_dti, marker='^', color='purple')
    plt.xlabel('DTI Ratio')
    plt.ylabel('Predicted Default Probability')
    plt.title('Monotonic Test: DTI Ratio vs Default Probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/monotonic_dti_ratio.png", dpi=300)
    plt.close()

    # D. Vary previous_default (0 vs 1)
    app_prev0 = baseline_applicant.copy()
    app_prev0["previous_default"] = 0
    prob_prev0 = pipeline.predict_proba(pd.DataFrame([app_prev0]))[0, 1]

    app_prev1 = baseline_applicant.copy()
    app_prev1["previous_default"] = 1
    prob_prev1 = pipeline.predict_proba(pd.DataFrame([app_prev1]))[0, 1]

    print(f"Credit Score Trend: {probs_cs[0]:.4f} (at 400) -> {probs_cs[-1]:.4f} (at 850) [Direction: {'Decreasing Risk (Expected)' if probs_cs[0] > probs_cs[-1] else 'Increasing'}]")
    print(f"Late Payments Trend: {probs_lp[0]:.4f} (at 0) -> {probs_lp[-1]:.4f} (at 10) [Direction: {'Increasing Risk (Expected)' if probs_lp[-1] > probs_lp[0] else 'Decreasing'}]")
    print(f"DTI Ratio Trend: {probs_dti[0]:.4f} (at 0.1) -> {probs_dti[-1]:.4f} (at 3.0) [Direction: {'Increasing Risk (Expected)' if probs_dti[-1] > probs_dti[0] else 'Decreasing'}]")
    print(f"Previous Default 0 vs 1: {prob_prev0:.4f} -> {prob_prev1:.4f} [Direction: {'Increasing Risk (Expected)' if prob_prev1 > prob_prev0 else 'Decreasing'}]")
    print("Monotonic Plots saved to plots/ directory.")

    # 4. Extreme values test
    print("\n--- 4. EXTREME VALUES TEST ---")
    extreme_applicant1 = baseline_applicant.copy()
    extreme_applicant1["annual_income"] = 1e8
    
    extreme_applicant2 = baseline_applicant.copy()
    extreme_applicant2["credit_score"] = 300
    extreme_applicant2["age"] = 100

    prob_ext1 = pipeline.predict_proba(pd.DataFrame([extreme_applicant1]))[0, 1]
    prob_ext2 = pipeline.predict_proba(pd.DataFrame([extreme_applicant2]))[0, 1]

    print(f"Extreme Income (100M) Probability : {prob_ext1:.6f}")
    print(f"Extreme Low Score & Age (300, 100) Probability: {prob_ext2:.6f}")

    assert 0.0 <= prob_ext1 <= 1.0, "Extreme test 1 failed: probability out of bounds"
    assert 0.0 <= prob_ext2 <= 1.0, "Extreme test 2 failed: probability out of bounds"
    print("Extreme Values Test: PASS (All outputs remain valid probabilities between 0 and 1)")

if __name__ == "__main__":
    test_robustness()
