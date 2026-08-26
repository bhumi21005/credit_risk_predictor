import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model.pkl')

st.set_page_config(
    page_title="AI Risk Manager | Credit Default Predictor",
    page_icon="📊",
    layout="centered"
)

# --- Custom styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #2E3440;
        padding-bottom: 6px;
    }
    div.stButton > button {
        width: 100%;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="main-title">📊 AI Credit Risk Manager</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the likelihood of loan default using an XGBoost-based credit risk model</p>', unsafe_allow_html=True)

# --- Input Section ---
st.markdown('<p class="section-header">Applicant Profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=3500)
    debt_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=2.0, value=0.30, step=0.05)
    utilization = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=2.0, value=0.30, step=0.05)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=0)

with col2:
    open_credit_lines = st.number_input("Open Credit Lines / Loans", min_value=0, value=5)
    real_estate_loans = st.number_input("Real Estate Loans", min_value=0, value=1)
    late_30_59 = st.number_input("Late Payments (30–59 days)", min_value=0, value=0)
    late_60_89 = st.number_input("Late Payments (60–89 days)", min_value=0, value=0)
    late_90 = st.number_input("Late Payments (90+ days)", min_value=0, value=0)

st.write("")
predict_clicked = st.button("Calculate Risk Score", type="primary")

# --- Prediction ---
if predict_clicked:
    total_past_due = late_30_59 + late_60_89 + late_90
    has_dependents = 1 if num_dependents > 0 else 0
    income_per_dependent = monthly_income / (num_dependents + 1)

    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': num_dependents,
        'TotalPastDue': total_past_due,
        'HasDependents': has_dependents,
        'IncomePerDependent': income_per_dependent
    }])

    risk_probability = model.predict_proba(input_data)[0][1]
    risk_percent = risk_probability * 100

    st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

    if risk_percent < 10:
        st.success(f"**Low Risk** — Estimated default probability: {risk_percent:.1f}%")
    elif risk_percent < 30:
        st.warning(f"**Medium Risk** — Estimated default probability: {risk_percent:.1f}%")
    else:
        st.error(f"**High Risk** — Estimated default probability: {risk_percent:.1f}%")

    st.progress(float(min(risk_probability, 1.0)))

    with st.expander("What influences this score?"):
        st.write(
            "This model was trained on historical credit data and weighs factors such as "
            "past payment delinquency, credit utilization, and applicant age most heavily. "
            "A higher number of past late payments and high credit utilization are the "
            "strongest indicators of future default risk."
        )

st.markdown("---")
st.caption("Built with XGBoost · Trained on the Give Me Some Credit dataset · For research and educational demonstration purposes only.")