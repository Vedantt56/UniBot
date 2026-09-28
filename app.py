import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Bank Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Disclaimer banner
st.warning("⚠️ **Disclaimer:** Trained on synthetic data. For learning purposes only. Not for real lending decisions.")

st.title("🏦 Bank Loan Default Risk Assessment")
st.markdown("Enter applicant details below to evaluate credit default risk using the pre-trained Machine Learning Pipeline.")

# Load model pipeline
@st.cache_resource
def load_model():
    saved_obj = joblib.load("loan_default_model.pkl")
    return saved_obj

model_data = load_model()
pipeline = model_data["pipeline"]
tuned_threshold = model_data.get("threshold", 0.35)

st.sidebar.header("Navigation & Settings")
st.sidebar.info(f"**Decision Threshold:** `{tuned_threshold:.2f}`\n\nThreshold is tuned to prioritise high recall on potential defaulters.")

# Create input layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner", "Unemployed", "Retired"])
    years_employed = st.number_input("Years Employed", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)

with col2:
    st.subheader("💰 Financial Status")
    annual_income = st.number_input("Annual Income ($)", min_value=1.0, max_value=10000000.0, value=600000.0, step=10000.0)
    savings_balance = st.number_input("Savings Balance ($)", min_value=0.0, max_value=5000000.0, value=150000.0, step=5000.0)
    existing_debt = st.number_input("Existing Debt ($)", min_value=0.0, max_value=5000000.0, value=80000.0, step=5000.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=720)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=30, value=4)
    num_late_payments_2y = st.number_input("Late Payments (Past 2 Years)", min_value=0, max_value=20, value=0)
    previous_default = st.radio("Previous Default History", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    account_age_years = st.number_input("Account Age (Years)", min_value=0.0, max_value=50.0, value=6.0, step=0.5)

with col3:
    st.subheader("📋 Loan Request & Context")
    city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"])
    home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage", "Other"])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Personal", "Education", "Business", "Medical", "Debt Consolidation"])
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=2000000.0, value=300000.0, step=10000.0)
    loan_term_months = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 84, 120, 180, 240], index=4)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=10.5, step=0.25)
    
    # Auto-compute DTI ratio or override
    auto_dti = round(existing_debt / annual_income, 4) if annual_income > 0 else 0.0
    dti_ratio = st.number_input("DTI Ratio (Existing Debt / Income)", min_value=0.0, max_value=10.0, value=auto_dti, step=0.01)
    
    application_year = st.number_input("Application Year", min_value=2018, max_value=2026, value=2023)
    application_month = st.slider("Application Month", min_value=1, max_value=12, value=5)

st.markdown("---")

if st.button("🔍 Predict Default Risk", use_container_width=True):
    applicant_dict = {
        "age": float(age),
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "years_employed": float(years_employed),
        "dependents": int(dependents),
        "city_tier": city_tier,
        "home_ownership": home_ownership,
        "annual_income": float(annual_income),
        "savings_balance": float(savings_balance),
        "existing_debt": float(existing_debt),
        "credit_score": float(credit_score),
        "num_credit_lines": int(num_credit_lines),
        "num_late_payments_2y": int(num_late_payments_2y),
        "previous_default": int(previous_default),
        "account_age_years": float(account_age_years),
        "loan_purpose": loan_purpose,
        "loan_amount": float(loan_amount),
        "loan_term_months": int(loan_term_months),
        "interest_rate": float(interest_rate),
        "dti_ratio": float(dti_ratio),
        "application_year": int(application_year),
        "application_month": int(application_month),
        "income_missing": 0 if annual_income > 0 else 1
    }

    df_single = pd.DataFrame([applicant_dict])
    prob = pipeline.predict_proba(df_single)[0, 1]

    # Categorize Risk
    # Low Risk: < 0.20
    # Medium Risk: 0.20 - 0.35
    # High Risk: >= 0.35
    if prob < 0.20:
        risk_label = "Low Risk"
        color = "green"
        recommendation = "Approved (Low probability of default)"
    elif prob < tuned_threshold:
        risk_label = "Medium Risk"
        color = "orange"
        recommendation = "Manual Review Suggested (Moderate default risk)"
    else:
        risk_label = "High Risk"
        color = "red"
        recommendation = "Flagged / High Risk of Default (Exceeds decision threshold)"

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Assessment Result")
        st.metric(label="Predicted Default Probability", value=f"{prob:.2%}")
        st.markdown(f"**Risk Level:** :{color}[**{risk_label}**]")
        st.markdown(f"**Decision (Threshold {tuned_threshold}):** {recommendation}")

    with res_col2:
        st.subheader("Top Risk Factors")
        try:
            preprocessor_fitted = pipeline.named_steps["preprocessor"]
            classifier_fitted = pipeline.named_steps["classifier"]

            num_cols = df_single.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = df_single.select_dtypes(include=["object", "category"]).columns.tolist()

            cat_ohe_features = preprocessor_fitted.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols).tolist()
            all_feature_names = num_cols + cat_ohe_features

            if hasattr(classifier_fitted, "coef_"):
                importances = np.abs(classifier_fitted.coef_[0])
            elif hasattr(classifier_fitted, "feature_importances_"):
                importances = classifier_fitted.feature_importances_
            else:
                importances = np.zeros(len(all_feature_names))

            feat_df = pd.DataFrame({
                "Feature": all_feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(5)

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color='steelblue')
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 5 Contributing Model Features")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info("Feature importance summary not available.")
