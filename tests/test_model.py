import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import engineer_features

model = joblib.load('model.pkl')

REQUIRED_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]


def make_sample_applicant(**overrides):
    """Helper: builds a baseline applicant row, with optional field overrides."""
    base = {
        'RevolvingUtilizationOfUnsecuredLines': 0.3,
        'age': 35,
        'NumberOfTime30-59DaysPastDueNotWorse': 0,
        'DebtRatio': 0.3,
        'MonthlyIncome': 3500,
        'NumberOfOpenCreditLinesAndLoans': 5,
        'NumberOfTimes90DaysLate': 0,
        'NumberRealEstateLoansOrLines': 1,
        'NumberOfTime60-89DaysPastDueNotWorse': 0,
        'NumberOfDependents': 0
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_model_output_is_valid_probability():
    """Model predictions must be valid probabilities between 0 and 1."""
    applicant = engineer_features(make_sample_applicant())
    prob = model.predict_proba(applicant)[0][1]
    assert 0.0 <= prob <= 1.0


def test_feature_engineering_total_past_due():
    """TotalPastDue should correctly sum all three late-payment columns."""
    applicant = make_sample_applicant(
        **{'NumberOfTime30-59DaysPastDueNotWorse': 2, 'NumberOfTime60-89DaysPastDueNotWorse': 1, 'NumberOfTimes90DaysLate': 3}
    )
    result = engineer_features(applicant)
    assert result['TotalPastDue'].iloc[0] == 6


def test_feature_engineering_has_dependents_flag():
    """HasDependents should be 1 when dependents > 0, else 0."""
    with_deps = engineer_features(make_sample_applicant(NumberOfDependents=2))
    without_deps = engineer_features(make_sample_applicant(NumberOfDependents=0))
    assert with_deps['HasDependents'].iloc[0] == 1
    assert without_deps['HasDependents'].iloc[0] == 0


def test_higher_past_due_increases_risk():
    """A clean applicant should score lower risk than one with heavy late-payment history."""
    clean_applicant = engineer_features(make_sample_applicant())
    risky_applicant = engineer_features(make_sample_applicant(
        **{'NumberOfTime30-59DaysPastDueNotWorse': 5, 'NumberOfTimes90DaysLate': 3}
    ))

    clean_risk = model.predict_proba(clean_applicant)[0][1]
    risky_risk = model.predict_proba(risky_applicant)[0][1]

    assert risky_risk > clean_risk


def test_required_columns_present_after_engineering():
    """Feature engineering should preserve all original required columns."""
    applicant = engineer_features(make_sample_applicant())
    for col in REQUIRED_COLUMNS:
        assert col in applicant.columns