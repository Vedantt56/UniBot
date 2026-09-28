"""
Synthetic bank loan-default dataset generator (intentionally messy).

Usage:  python generate_loan_default_data.py [n_rows] [seed]
Output: loan_default_dataset.csv

Injected data-quality problems (for practising cleaning / robust modelling):
  - Missing values (random + income missing more often for self-employed)
  - Numeric outliers (extreme income, loan amount, debt)
  - Impossible values (negative age, age > 100, credit score out of range, ...)
  - Logical inconsistencies (years_employed > age - 14, dti not matching debt/income)
  - Inconsistent categorical labels ("Male", "male", "M", " MALE ")
  - Numbers stored as text ("N/A", "?", "45,000")
  - Future / malformed dates
  - Exact duplicate rows
"""
import sys
import numpy as np
import pandas as pd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
rng = np.random.default_rng(seed)

# ---------------------------------------------------------------- clean base
age = np.clip(rng.normal(38, 11, n), 21, 70).round().astype(int)
gender = rng.choice(["Male", "Female", "Other"], n, p=[0.56, 0.42, 0.02])
marital = rng.choice(["Single", "Married", "Divorced", "Widowed"], n, p=[0.32, 0.53, 0.11, 0.04])
education = rng.choice(["High School", "Graduate", "Postgraduate", "Doctorate"], n, p=[0.28, 0.47, 0.22, 0.03])
employment = rng.choice(["Salaried", "Self-Employed", "Business Owner", "Unemployed", "Retired"], n,
                        p=[0.55, 0.20, 0.10, 0.08, 0.07])
dependents = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.28, 0.24, 0.26, 0.14, 0.06, 0.02])
city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.38, 0.37, 0.25])
home = rng.choice(["Rent", "Own", "Mortgage", "Family"], n, p=[0.30, 0.22, 0.28, 0.20])

years_emp = np.clip(age - 21 - rng.exponential(4, n), 0, None).round(1)
years_emp[employment == "Unemployed"] = 0
years_emp[employment == "Retired"] = np.clip(age[employment == "Retired"] - 25, 0, 40)

edu_mult = pd.Series(education).map({"High School": 0.7, "Graduate": 1.0, "Postgraduate": 1.35, "Doctorate": 1.6}).values
tier_mult = pd.Series(city_tier).map({"Tier-1": 1.3, "Tier-2": 1.0, "Tier-3": 0.8}).values
income = rng.lognormal(np.log(600_000), 0.55, n) * edu_mult * tier_mult * (1 + years_emp / 40)
income[employment == "Unemployed"] *= 0.15
income[employment == "Retired"] *= 0.45
income = income.round(-2)

loan_purpose = rng.choice(["Home", "Auto", "Personal", "Education", "Business", "Medical", "Debt Consolidation"], n,
                          p=[0.20, 0.18, 0.22, 0.12, 0.10, 0.06, 0.12])
loan_amount = (income * rng.uniform(0.3, 3.0, n)).round(-3)
loan_amount[loan_purpose == "Home"] *= 2.5
loan_amount = np.clip(loan_amount, 25_000, 15_000_000).round(-3)
term = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n, p=[0.05, 0.10, 0.17, 0.13, 0.20, 0.10, 0.10, 0.08, 0.07])

home_mask = loan_purpose == "Home"
term[home_mask] = rng.choice([120, 180, 240], home_mask.sum())
credit_score = np.clip(rng.normal(690, 85, n), 300, 900).round().astype(int)
late_pay = np.clip(rng.poisson(np.clip((720 - credit_score) / 90, 0.05, None)), 0, 12)
prev_default = (rng.random(n) < np.clip((650 - credit_score) / 500 + 0.03, 0.01, 0.4)).astype(int)
credit_lines = np.clip(rng.poisson(4, n), 0, 20)
existing_debt = (income * rng.uniform(0, 1.2, n) * (credit_lines / 5)).round(-2)
interest = np.clip(16 - (credit_score - 300) / 600 * 8 + rng.normal(0, 1, n) + (term > 120) * -1.5, 6, 28).round(2)
savings = (income * rng.exponential(0.35, n)).round(-2)
acct_age = np.clip(rng.exponential(6, n), 0.1, 40).round(1)
dti = ((existing_debt * 0.12 + loan_amount / (term / 12) * 1.1) / np.clip(income, 1, None)).round(3)

app_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D")

# ------------------------------------------------ target (learnable signal)
z = (-3.0
     + 0.015 * (650 - credit_score) / 10 * 6 / 6
     + 2.2 * np.clip(dti - 0.4, 0, None)
     + 0.35 * late_pay
     + 1.1 * prev_default
     + 0.06 * (interest - 12)
     + 0.9 * (employment == "Unemployed")
     - 0.25 * (employment == "Salaried")
     + 0.15 * dependents
     - 0.8 * np.log1p(savings / np.clip(income, 1, None))
     - 0.02 * years_emp
     + 0.25 * (loan_purpose == "Business")
     + rng.normal(0, 0.6, n))
p_default = 1 / (1 + np.exp(-z))
default = (rng.random(n) < p_default).astype(int)

df = pd.DataFrame({
    "loan_id": [f"LN{100000 + i}" for i in range(n)],
    "application_date": app_date.strftime("%Y-%m-%d"),
    "age": age, "gender": gender, "marital_status": marital, "education": education,
    "employment_type": employment, "years_employed": years_emp, "dependents": dependents,
    "city_tier": city_tier, "home_ownership": home,
    "annual_income": income, "savings_balance": savings, "existing_debt": existing_debt,
    "credit_score": credit_score, "num_credit_lines": credit_lines,
    "num_late_payments_2y": late_pay, "previous_default": prev_default,
    "account_age_years": acct_age,
    "loan_purpose": loan_purpose, "loan_amount": loan_amount, "loan_term_months": term,
    "interest_rate": interest, "dti_ratio": dti,
    "default": default,
})

# ------------------------------------------------------- inject messiness
def pick(frac):
    return rng.choice(n, int(n * frac), replace=False)

# 1) Missing values (random)
for col, frac in {"age": 0.02, "education": 0.04, "years_employed": 0.05, "credit_score": 0.06,
                  "savings_balance": 0.07, "num_credit_lines": 0.03, "marital_status": 0.03,
                  "interest_rate": 0.02, "home_ownership": 0.04, "account_age_years": 0.05,
                  "num_late_payments_2y": 0.04}.items():
    df.loc[pick(frac), col] = np.nan

# 2) Missing values (informative / MNAR): income hidden more for self-employed & defaulters
mask = ((df.employment_type.isin(["Self-Employed", "Business Owner"])) & (rng.random(n) < 0.20)) | \
       ((df.default == 1) & (rng.random(n) < 0.07)) | (rng.random(n) < 0.02)
df.loc[mask, "annual_income"] = np.nan

# 3) Outliers (plausible but extreme)
idx = pick(0.012); df.loc[idx, "annual_income"] = df.loc[idx, "annual_income"] * rng.uniform(8, 40, len(idx))
idx = pick(0.010); df.loc[idx, "loan_amount"] = df.loc[idx, "loan_amount"] * rng.uniform(10, 60, len(idx))
idx = pick(0.010); df.loc[idx, "existing_debt"] = df.loc[idx, "existing_debt"] * rng.uniform(15, 50, len(idx))
idx = pick(0.008); df.loc[idx, "savings_balance"] = rng.uniform(5e7, 5e8, len(idx)).round(-3)
idx = pick(0.008); df.loc[idx, "num_credit_lines"] = rng.integers(35, 90, len(idx))
idx = pick(0.008); df.loc[idx, "num_late_payments_2y"] = rng.integers(25, 60, len(idx))
idx = pick(0.006); df.loc[idx, "interest_rate"] = rng.uniform(45, 99, len(idx)).round(2)

# 4) Impossible / anomalous values
idx = pick(0.006); df.loc[idx, "age"] = rng.choice([-5, -1, 0, 2, 7, 15, 17, 120, 150, 999], len(idx))
idx = pick(0.008); df.loc[idx, "credit_score"] = rng.choice([-1, 0, 100, 1000, 1200, 9999], len(idx))
idx = pick(0.006); df.loc[idx, "annual_income"] = rng.choice([-50000, -1, 0], len(idx))
idx = pick(0.005); df.loc[idx, "loan_amount"] = rng.choice([-100000, 0, 1], len(idx))
idx = pick(0.005); df.loc[idx, "loan_term_months"] = rng.choice([0, -12, 7, 999], len(idx))
idx = pick(0.005); df.loc[idx, "dependents"] = rng.choice([-2, 15, 30], len(idx))
idx = pick(0.006); df.loc[idx, "previous_default"] = rng.choice([2, 5, -1], len(idx))
idx = pick(0.006); df.loc[idx, "dti_ratio"] = rng.choice([-0.4, 25.0, 99.9], len(idx))

# 5) Logical inconsistencies
idx = pick(0.012)
df.loc[idx, "years_employed"] = (df.loc[idx, "age"].fillna(30) + rng.integers(1, 15, len(idx))).round(1)
idx = pick(0.010)  # unemployed but reporting long employment / high income
idx = idx[:len(idx)]
df.loc[idx, "employment_type"] = "Unemployed"
df.loc[idx, "years_employed"] = rng.uniform(5, 25, len(idx)).round(1)
idx = pick(0.015)  # DTI not consistent with debt / income
df.loc[idx, "dti_ratio"] = (df.loc[idx, "dti_ratio"] * rng.uniform(0.1, 0.4, len(idx))).round(3)
idx = pick(0.008)  # account older than customer
df.loc[idx, "account_age_years"] = (df.loc[idx, "age"].fillna(30) + rng.uniform(1, 30, len(idx))).round(1)

# 6) Inconsistent categorical labels
gender_variants = {"Male": ["male", "M", " MALE ", "m", "Male "], "Female": ["female", "F", "FEMALE", "f", " Female"],
                   "Other": ["other", "O", "Non-binary"]}
idx = pick(0.10)
df["gender"] = df["gender"].astype(object)
for i in idx:
    g = df.at[i, "gender"]
    if g in gender_variants:
        df.at[i, "gender"] = rng.choice(gender_variants[g])
emp_variants = {"Self-Employed": ["self employed", "SELF-EMPLOYED", "Self_Employed", "selfemployed"],
                "Salaried": ["salaried", "SALARIED", "Salary", "Employed"],
                "Business Owner": ["business owner", "Owner", "BUSINESS OWNER"],
                "Unemployed": ["unemployed", "None", "UNEMPLOYED"], "Retired": ["retired", "Pensioner"]}
for i in pick(0.08):
    e = df.at[i, "employment_type"]
    if e in emp_variants:
        df.at[i, "employment_type"] = rng.choice(emp_variants[e])
for i in pick(0.05):
    df.at[i, "city_tier"] = rng.choice(["tier 1", "T1", "Tier1", "tier-2", "T3", "TIER-3"])
for i in pick(0.05):
    df.at[i, "loan_purpose"] = rng.choice(["home loan", "AUTO", "personal", "Edu", "biz", "Debt consolidation", "misc"])
df["default"] = df["default"].astype(object)
for i in pick(0.01):
    df.at[i, "default"] = rng.choice(["Yes", "No", "Y", "N", "1.0", "0.0"])

# 7) Numbers stored as text
for col, junk in {"annual_income": ["N/A", "?", "unknown", "--"], "credit_score": ["NA", "not available", "?"],
                  "loan_amount": ["TBD", "N/A"], "age": ["thirty", "NA"]}.items():
    df[col] = df[col].astype(object)
    for i in pick(0.006):
        df.at[i, col] = rng.choice(junk)
for i in pick(0.01):
    v = df.at[i, "annual_income"]
    if isinstance(v, (int, float, np.number)) and not pd.isna(v):
        df.at[i, "annual_income"] = f"{int(v):,}"
for i in pick(0.01):
    v = df.at[i, "loan_amount"]
    if isinstance(v, (int, float, np.number)) and not pd.isna(v):
        df.at[i, "loan_amount"] = f"Rs. {int(v)}"

# 8) Bad dates
for i in pick(0.01):
    df.at[i, "application_date"] = rng.choice(["2031-04-15", "2099-01-01", "1900-01-01", "15/03/2021", "2021-13-45", "", "N/A"])
for i in pick(0.02):
    d = df.at[i, "application_date"]
    try:
        df.at[i, "application_date"] = pd.to_datetime(d, format="%Y-%m-%d").strftime("%d-%m-%Y")
    except Exception:
        pass

# 9) Exact duplicate rows + a few duplicate IDs with conflicting values
dups = df.sample(int(n * 0.015), random_state=seed)
df = pd.concat([df, dups], ignore_index=True)
conf = df.sample(int(n * 0.004), random_state=seed + 1).copy()
conf["credit_score"] = pd.to_numeric(conf["credit_score"], errors="coerce") + rng.integers(-40, 40, len(conf))
df = pd.concat([df, conf], ignore_index=True)

df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
df.to_csv("loan_default_dataset.csv", index=False)
print(df.shape)
print("Missing per column:\n", df.isna().sum())
print("Default rate (numeric rows):", pd.to_numeric(df["default"], errors="coerce").mean().round(3))
