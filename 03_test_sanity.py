import joblib
import pandas as pd
import numpy as np

def test_sanity():
    print("=== STEP 1: SINGLE-APPLICANT SANITY TEST ===\n")
    
    # 1. Load model pickle
    saved_obj = joblib.load("loan_default_model.pkl")
    pipeline = saved_obj["pipeline"]
    threshold = 0.35

    # 2. Hand-made Applicants using cleaned feature columns
    # Low-risk applicant
    low_risk_applicant = {
        "age": 42,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Bachelor",
        "employment_type": "Salaried",
        "years_employed": 15.0,
        "dependents": 1,
        "city_tier": "Tier-1",
        "home_ownership": "Own",
        "annual_income": 1500000.0,
        "savings_balance": 900000.0,
        "existing_debt": 100000.0,
        "credit_score": 790.0,
        "num_credit_lines": 3,
        "num_late_payments_2y": 0,
        "previous_default": 0,
        "account_age_years": 10.0,
        "loan_purpose": "Home",
        "loan_amount": 500000.0,
        "loan_term_months": 60,
        "interest_rate": 9.0,
        "dti_ratio": 0.15,
        "application_year": 2023,
        "application_month": 5,
        "income_missing": 0
    }

    # High-risk applicant
    high_risk_applicant = {
        "age": 27,
        "gender": "Male",
        "marital_status": "Single",
        "education": "High School",
        "employment_type": "Unemployed",
        "years_employed": 0.0,
        "dependents": 2,
        "city_tier": "Tier-3",
        "home_ownership": "Rent",
        "annual_income": 150000.0,
        "savings_balance": 5000.0,
        "existing_debt": 400000.0,
        "credit_score": 480.0,
        "num_credit_lines": 8,
        "num_late_payments_2y": 6,
        "previous_default": 1,
        "account_age_years": 2.0,
        "loan_purpose": "Personal",
        "loan_amount": 1200000.0,
        "loan_term_months": 36,
        "interest_rate": 22.0,
        "dti_ratio": 2.5,
        "application_year": 2023,
        "application_month": 5,
        "income_missing": 0
    }

    df_test = pd.DataFrame([low_risk_applicant, high_risk_applicant])
    
    # Calculate probabilities
    probs = pipeline.predict_proba(df_test)[:, 1]
    low_risk_prob = probs[0]
    high_risk_prob = probs[1]

    low_risk_decision = "DEFAULTER (1)" if low_risk_prob >= threshold else "NON-DEFAULTER (0)"
    high_risk_decision = "DEFAULTER (1)" if high_risk_prob >= threshold else "NON-DEFAULTER (0)"

    print(f"Low-Risk Applicant  : Default Probability = {low_risk_prob:.4f} | Decision (>= {threshold}) = {low_risk_decision}")
    print(f"High-Risk Applicant : Default Probability = {high_risk_prob:.4f} | Decision (>= {threshold}) = {high_risk_decision}")

    # Assertion
    assert high_risk_prob > low_risk_prob, f"Assertion Failed: High-risk prob ({high_risk_prob:.4f}) <= Low-risk prob ({low_risk_prob:.4f})"
    print("\nSanity Check Result: PASS (High-risk applicant has significantly higher probability of default than low-risk applicant)")

if __name__ == "__main__":
    test_sanity()
