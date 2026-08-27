import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load('model.pkl')

OPTIMAL_THRESHOLD = 0.55  # cost-optimized threshold (minimizes false-negative + false-positive business cost)

REQUIRED_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

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
st.markdown('<p class="subtitle">Predict the likelihood of loan default using a cost-optimized XGBoost model</p>', unsafe_allow_html=True)


def engineer_features(df):
    """Apply the same feature engineering used during training."""
    df = df.copy()
    df['TotalPastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse']
        + df['NumberOfTime60-89DaysPastDueNotWorse']
        + df['NumberOfTimes90DaysLate']
    )
    df['HasDependents'] = (df['NumberOfDependents'] > 0).astype(int)
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    return df


def risk_label(risk_percent):
    if risk_percent < 10:
        return "Low"
    elif risk_percent < 30:
        return "Medium"
    else:
        return "High"


tab1, tab2 = st.tabs(["Single Applicant", "Batch Scoring (CSV)"])

# ============== TAB 1: SINGLE APPLICANT ==============
with tab1:
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

    if predict_clicked:
        raw = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': utilization,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': monthly_income,
            'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
            'NumberOfTimes90DaysLate': late_90,
            'NumberRealEstateLoansOrLines': real_estate_loans,
            'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
            'NumberOfDependents': num_dependents
        }])

        input_data = engineer_features(raw)
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_percent = risk_probability * 100

        st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

        label = risk_label(risk_percent)
        if label == "Low":
            st.success(f"**Low Risk** — Estimated default probability: {risk_percent:.1f}%")
        elif label == "Medium":
            st.warning(f"**Medium Risk** — Estimated default probability: {risk_percent:.1f}%")
        else:
            st.error(f"**High Risk** — Estimated default probability: {risk_percent:.1f}%")

        st.progress(float(min(risk_probability, 1.0)))

        decision = "Reject / Flag for review" if risk_probability >= OPTIMAL_THRESHOLD else "Approve"
        st.metric("Lending Decision (at cost-optimized threshold)", decision)

        with st.expander("What influences this score?"):
            st.write(
                "This model was trained on historical credit data and weighs factors such as "
                "past payment delinquency, credit utilization, and applicant age most heavily. "
                "A higher number of past late payments and high credit utilization are the "
                "strongest indicators of future default risk. The approve/reject decision uses "
                "a threshold of 0.55, chosen by minimizing estimated business cost across false "
                "negatives (missed defaulters) and false positives (wrongly rejected applicants)."
            )

# ============== TAB 2: BATCH SCORING ==============
with tab2:
    st.markdown('<p class="section-header">Batch Applicant Scoring</p>', unsafe_allow_html=True)
    st.write(
        "Upload a CSV of multiple applicants to score them all at once. "
        "The file must contain these columns:"
    )
    st.code(", ".join(REQUIRED_COLUMNS), language=None)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                batch_input = engineer_features(batch_df[REQUIRED_COLUMNS])
                probs = model.predict_proba(batch_input)[:, 1]

                results = batch_df.copy()
                results['Risk_Percent'] = (probs * 100).round(1)
                results['Risk_Level'] = results['Risk_Percent'].apply(risk_label)
                results['Decision'] = ["Reject / Flag for review" if p >= OPTIMAL_THRESHOLD else "Approve" for p in probs]

                st.success(f"Scored {len(results)} applicants.")

                st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True)

                csv_download = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results as CSV",
                    data=csv_download,
                    file_name="scored_applicants.csv",
                    mime="text/csv"
                )

                st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                results['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High']).plot(
                    kind='bar', ax=ax, color=['#22c55e', '#eab308', '#ef4444']
                )
                ax.set_xlabel("Risk Level")
                ax.set_ylabel("Number of Applicants")
                ax.set_title("Applicant Risk Distribution")
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info("No file uploaded yet. Upload a CSV to see batch results.")

st.markdown("---")
st.caption("Built with XGBoost · Trained on the Give Me Some Credit dataset · For research and educational demonstration purposes only.")