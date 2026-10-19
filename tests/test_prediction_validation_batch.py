import numpy as np
import pandas as pd
import pytest

from src.validation import ValidationError, validate_student_record

OK = {"G1": 12, "G2": 13, "subject": "Mathematics"}


# ---------------- validation rules
@pytest.mark.parametrize("bad", [
    {}, {"G1": 12, "G2": 13}, {"G1": 50, "G2": 10, "subject": "Mathematics"},
    {"G1": -1, "G2": 10, "subject": "Mathematics"}, {"G1": "abc", "G2": 10, "subject": "Mathematics"},
    {"G1": float("nan"), "G2": 10, "subject": "Mathematics"}, {**OK, "absences": -10}, {**OK, "absences": 500},
    {**OK, "studytime": 9}, {**OK, "failures": 1.5}, {**OK, "sex": "Z"}, {**OK, "subject": "Physics"},
    {**OK, "age": 200},
])
def test_invalid_records_are_rejected(predictor, bad):
    with pytest.raises(ValidationError):
        predictor.predict_single(bad)


def test_all_errors_are_reported_together():
    with pytest.raises(ValidationError) as exc:
        validate_student_record({"G1": 99, "G2": "x"}, {})
    assert len(exc.value.errors) >= 3


def test_defaults_are_reported_not_hidden(predictor):
    r = predictor.predict_single(OK)
    assert "absences" in r["defaulted_fields"] and r["inputs_complete"] is False
    assert predictor.defaults["absences"] == predictor.meta["defaults"]["absences"]


# ---------------- prediction behaviour
def test_prediction_output_contract(predictor):
    r = predictor.predict_single({**OK, "absences": 3, "studytime": 2, "failures": 0})
    assert 0 <= r["predicted_score_pct"] <= 100
    assert r["prediction_range"]["low"] <= r["predicted_score_pct"] <= r["prediction_range"]["high"]
    assert sum(r["support_probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert r["support_level"] in {"High Support", "Medium Support", "Low Support"}
    assert "estimate" in r["disclaimer"].lower() or "not a confirmed" in r["disclaimer"].lower()


def test_higher_grades_give_higher_predictions(predictor):
    low = predictor.predict_single({"G1": 6, "G2": 6, "subject": "Mathematics"})["predicted_score_pct"]
    high = predictor.predict_single({"G1": 17, "G2": 18, "subject": "Mathematics"})["predicted_score_pct"]
    assert high > low + 30


def test_prediction_is_deterministic(predictor):
    a = predictor.predict_single({**OK, "absences": 4})
    b = predictor.predict_single({**OK, "absences": 4})
    assert a["predicted_score_pct"] == b["predicted_score_pct"]


def test_extreme_valid_inputs_stay_in_range(predictor):
    for g in (0, 20):
        r = predictor.predict_single({"G1": g, "G2": g, "subject": "Portuguese", "absences": 93})
        assert 0 <= r["predicted_score_pct"] <= 100


def test_tier_disagreement_is_flagged_consistently(predictor):
    r = predictor.predict_single(OK)
    assert r["tiers_agree"] == (r["support_level"] == r["score_implied_tier"])


# ---------------- batch
def test_batch_keeps_all_rows_including_duplicates(predictor, test_df):
    df = test_df.drop(columns=["final_score_pct", "support_level", "student_group"]).head(6)
    batch = pd.concat([df, df.iloc[:3]], ignore_index=True)
    out = predictor.predict_batch(batch)
    assert len(out) == 9 and out["Validation_Status"].str.startswith("OK").all()


def test_batch_flags_invalid_rows_without_dropping(predictor):
    df = pd.DataFrame([{**OK}, {**OK, "G1": 99}, {**OK, "G2": 5}])
    out = predictor.predict_batch(df)
    assert len(out) == 3
    assert out.loc[1, "Validation_Status"].startswith("INVALID") and np.isnan(out.loc[1, "Predicted_Score_Pct"])
    assert out.loc[0, "Predicted_Score_Pct"] > 0


def test_batch_missing_required_columns_and_empty(predictor):
    with pytest.raises(ValidationError):
        predictor.predict_batch(pd.DataFrame({"G1": [1]}))
    with pytest.raises(ValidationError):
        predictor.predict_batch(pd.DataFrame())


def test_batch_matches_single(predictor):
    rec = {**OK, "absences": 5}
    single = predictor.predict_single(rec)["predicted_score_pct"]
    batch = predictor.predict_batch(pd.DataFrame([rec]))["Predicted_Score_Pct"].iloc[0]
    assert single == batch
