import pandas as pd
import numpy as np
import re
import json
import os

WINSOR_BOUNDS_FILE = "winsor_bounds.json"

def clean_data(input_csv="loan_default_dataset.csv", output_csv="loan_default_clean.csv", bounds_file=WINSOR_BOUNDS_FILE):
    print(f"=== DATA CLEANING ({input_csv} -> {output_csv}) ===\n")
    
    # 1. Load CSV
    df = pd.read_csv(input_csv)
    
    print("--- BEFORE CLEANING ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    print("-" * 50 + "\n")

    initial_rows = len(df)

    # 2. Clean Target First if target column exists
    if 'default' in df.columns:
        def parse_target(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            if s in ["1", "1.0", "yes", "y", "true"]:
                return 1
            elif s in ["0", "0.0", "no", "n", "false"]:
                return 0
            try:
                f = float(s)
                if f == 1.0:
                    return 1
                elif f == 0.0:
                    return 0
            except:
                pass
            return np.nan

        df['default'] = df['default'].apply(parse_target)
        df = df.dropna(subset=['default'])
        df['default'] = df['default'].astype(int)

    # 3. Numeric Columns conversion
    num_cols = [
        'age', 'years_employed', 'dependents', 'annual_income', 'savings_balance',
        'existing_debt', 'credit_score', 'num_credit_lines', 'num_late_payments_2y',
        'previous_default', 'account_age_years', 'loan_amount', 'loan_term_months',
        'interest_rate', 'dti_ratio'
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['N/A', 'n/a', '?', 'NaN', 'nan', 'null', 'None', ''], np.nan)
            
            def sanitize_num(x):
                if pd.isna(x):
                    return np.nan
                x_str = str(x).strip()
                if x_str.lower() == 'thirty':
                    return 30.0
                clean = re.sub(r'[Rs\.\s,]', '', x_str) if 'Rs' in x_str or ',' in x_str else x_str
                return clean

            df[col] = df[col].apply(sanitize_num)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Standardise Categorical Columns
    if 'gender' in df.columns:
        def std_gender(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            if s in ['m', 'male']:
                return 'Male'
            elif s in ['f', 'female']:
                return 'Female'
            elif s in ['other', 'o']:
                return 'Other'
            return np.nan
        df['gender'] = df['gender'].apply(std_gender)

    if 'employment_type' in df.columns:
        def std_employment(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower().replace('_', ' ').replace('-', ' ')
            if 'salaried' in s or s == 'salary':
                return 'Salaried'
            elif 'self' in s or s == 'se':
                return 'Self-Employed'
            elif 'business' in s or 'biz' in s or s == 'owner':
                return 'Business Owner'
            elif 'unemployed' in s:
                return 'Unemployed'
            elif 'retired' in s:
                return 'Retired'
            return np.nan
        df['employment_type'] = df['employment_type'].apply(std_employment)

    if 'city_tier' in df.columns:
        def std_city_tier(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower().replace('_', '').replace('-', '').replace(' ', '')
            if s in ['tier1', 't1', '1']:
                return 'Tier-1'
            elif s in ['tier2', 't2', '2']:
                return 'Tier-2'
            elif s in ['tier3', 't3', '3']:
                return 'Tier-3'
            return np.nan
        df['city_tier'] = df['city_tier'].apply(std_city_tier)

    if 'loan_purpose' in df.columns:
        def std_loan_purpose(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower().replace('_', ' ').replace('-', ' ')
            if 'home' in s:
                return 'Home'
            elif 'auto' in s or 'car' in s:
                return 'Auto'
            elif 'personal' in s:
                return 'Personal'
            elif 'edu' in s:
                return 'Education'
            elif 'biz' in s or 'business' in s:
                return 'Business'
            elif 'med' in s:
                return 'Medical'
            elif 'debt' in s or 'consolidation' in s:
                return 'Debt Consolidation'
            return np.nan
        df['loan_purpose'] = df['loan_purpose'].apply(std_loan_purpose)

    if 'marital_status' in df.columns:
        df['marital_status'] = df['marital_status'].astype(str).str.strip().str.title()
        df['marital_status'] = df['marital_status'].replace(['Nan', 'None', 'N/A', '?'], np.nan)
    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].astype(str).str.strip().str.title()
        df['home_ownership'] = df['home_ownership'].replace(['Nan', 'None', 'N/A', '?'], np.nan)
    if 'education' in df.columns:
        df['education'] = df['education'].astype(str).str.strip().str.title()
        df['education'] = df['education'].replace(['Nan', 'None', 'N/A', '?'], np.nan)

    # 5. Parse application_date robustly
    if 'application_date' in df.columns:
        def parse_date(val):
            if pd.isna(val):
                return pd.NaT
            try:
                dt = pd.to_datetime(val, errors='coerce', dayfirst=False)
                if pd.isna(dt):
                    dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
                if pd.isna(dt):
                    return pd.NaT
                if dt.year < 2018 or dt > pd.to_datetime('2026-09-28'):
                    return pd.NaT
                return dt
            except:
                return pd.NaT

        df['parsed_date'] = df['application_date'].apply(parse_date)
        df['application_year'] = df['parsed_date'].dt.year
        df['application_month'] = df['parsed_date'].dt.month
        df = df.drop(columns=['application_date', 'parsed_date'])

    # 6. Set impossible values to NaN
    valid_bounds = {
        'age': (18, 100),
        'credit_score': (300, 900),
        'annual_income': (0.001, np.inf),
        'loan_amount': (0.001, np.inf),
        'dependents': (0, 10),
        'previous_default': (0, 1),
        'dti_ratio': (0, 10),
        'interest_rate': (1, 40),
        'years_employed': (0, 80),
        'savings_balance': (0, np.inf),
        'existing_debt': (0, np.inf),
        'num_credit_lines': (0, 50),
        'num_late_payments_2y': (0, 50),
        'account_age_years': (0, 80)
    }
    valid_terms = [12, 24, 36, 48, 60, 84, 120, 180, 240]

    for col, (low, high) in valid_bounds.items():
        if col in df.columns:
            df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    if 'loan_term_months' in df.columns:
        df.loc[~df['loan_term_months'].isin(valid_terms), 'loan_term_months'] = np.nan

    # 7. Fix logical inconsistencies
    if 'years_employed' in df.columns and 'age' in df.columns:
        mask_emp = (df['years_employed'] > (df['age'] - 14))
        df.loc[mask_emp, 'years_employed'] = np.nan

    if 'employment_type' in df.columns and 'years_employed' in df.columns:
        mask_unemp = (df['employment_type'] == 'Unemployed') & (df['years_employed'] > 0)
        df.loc[mask_unemp, 'years_employed'] = 0

    if 'account_age_years' in df.columns and 'age' in df.columns:
        mask_acc = (df['account_age_years'] > (df['age'] - 14))
        df.loc[mask_acc, 'account_age_years'] = np.nan

    if 'dti_ratio' in df.columns and 'existing_debt' in df.columns and 'annual_income' in df.columns:
        recalc_dti = (df['existing_debt'] / df['annual_income'])
        df['dti_ratio'] = df['dti_ratio'].fillna(recalc_dti)

    # 8. Remove exact duplicates and handle repeated loan_ids
    df = df.drop_duplicates()
    if 'loan_id' in df.columns:
        df['non_null_count'] = df.notnull().sum(axis=1)
        df = df.sort_values(by=['loan_id', 'non_null_count'], ascending=[True, False])
        df = df.drop_duplicates(subset=['loan_id'], keep='first')
        df = df.drop(columns=['non_null_count'])

    # 9. Handle outliers by winsorising (clip at 1st and 99th percentile)
    money_cols = ['annual_income', 'savings_balance', 'existing_debt', 'loan_amount']
    
    if os.path.exists(bounds_file):
        # Load saved bounds from training data to avoid leakage
        with open(bounds_file, "r") as f:
            winsor_dict = json.load(f)
    else:
        # Compute bounds and save them if training bounds file does not exist
        winsor_dict = {}
        for col in money_cols:
            if col in df.columns and df[col].notnull().any():
                winsor_dict[col] = {
                    "low": float(df[col].quantile(0.01)),
                    "high": float(df[col].quantile(0.99))
                }
        with open(bounds_file, "w") as f:
            json.dump(winsor_dict, f, indent=4)

    for col in money_cols:
        if col in df.columns and col in winsor_dict:
            q_low = winsor_dict[col]["low"]
            q_high = winsor_dict[col]["high"]
            df[col] = df[col].clip(lower=q_low, upper=q_high)

    # 10. Add missing-indicator column for annual_income
    if 'annual_income' in df.columns:
        df['income_missing'] = df['annual_income'].isnull().astype(int)

    # 11. Save cleaned file
    df.to_csv(output_csv, index=False)
    print(f"\nCleaned dataset saved successfully to {output_csv}.")

    print("\n--- CLEANING REPORT ---")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Total rows dropped: {initial_rows - len(df)}")
    print("\nRemaining Missing Values Count Per Column:")
    print(df.isnull().sum())
    print("\nFinal Data Shape:", df.shape)
    return df

if __name__ == "__main__":
    clean_data("loan_default_dataset.csv", "loan_default_clean.csv")
