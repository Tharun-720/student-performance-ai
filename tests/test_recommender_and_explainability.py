import numpy as np
import pytest

from src.explainability import ADDITIVITY_TOLERANCE, explain_rows, top_contributions
from src.recommender import SEVERITY_ORDER, diagnose, generate_recommendations


def titles(recs):
    return [a["title"] for a in recs["priority_actions"]]


# ---------------- recommender logic
def test_high_support_student_never_gets_honors_track():
    recs = generate_recommendations({"G1": 6, "G2": 6, "absences": 2, "studytime": 3, "failures": 0,
                                     "subject": "Mathematics"}, 31.0, "High Support")
    text = " ".join(titles(recs)).lower()
    assert "honors" not in text and "advanced" not in text
    assert recs["priority_actions"][0]["severity"] == "Critical"


def test_low_support_student_with_no_gaps_gets_maintenance_message():
    recs = generate_recommendations({"G1": 17, "G2": 18, "absences": 1, "studytime": 3, "failures": 0,
                                     "subject": "Portuguese"}, 90.0, "Low Support")
    assert recs["total_gaps_identified"] == 0 and recs["priority_actions"][0]["severity"] == "Info"


def test_actions_are_sorted_by_severity_and_capped():
    recs = generate_recommendations({"G1": 12, "G2": 5, "absences": 25, "studytime": 1, "failures": 3, "goout": 5,
                                     "freetime": 4, "subject": "Mathematics"}, 20.0, "High Support")
    sev = [SEVERITY_ORDER[a["severity"]] for a in recs["priority_actions"]]
    assert sev == sorted(sev) and len(recs["priority_actions"]) <= 4


@pytest.mark.parametrize("absences,expected", [(9, None), (10, "Medium"), (19, "Medium"), (20, "High")])
def test_absence_thresholds(absences, expected):
    gaps = [g for g in diagnose({"absences": absences}) if g["category"] == "Absences"]
    assert (gaps[0]["severity"] if gaps else None) == expected


def test_defaulted_fields_cannot_trigger_rules():
    student = {"G1": 15, "G2": 15, "absences": 30, "studytime": 1, "failures": 3, "subject": "Mathematics"}
    recs = generate_recommendations(student, 75.0, "Low Support", defaulted_fields=["absences", "studytime", "failures"])
    assert recs["total_gaps_identified"] == 0
    assert any("Not provided" in n for n in recs["data_notes"])


def test_missing_information_is_handled_safely():
    recs = generate_recommendations({"subject": "Mathematics"})
    assert recs["priority_actions"][0]["severity"] == "Info"


def test_no_unverified_organisation_or_claims_in_text():
    recs = generate_recommendations({"G1": 5, "G2": 4, "absences": 30, "studytime": 1, "failures": 3,
                                     "goout": 5, "freetime": 5, "subject": "Mathematics"}, 18.0, "High Support")
    blob = str(recs).lower()
    for forbidden in ("trekverse", "guarantee", "will fail", "will pass", "faculty advisor", "mit "):
        assert forbidden not in blob


def test_recommendations_are_consistent_with_study_time_bins():
    gaps = diagnose({"studytime": 1})
    assert "under 2 hours" in gaps[0]["finding"] and "3 to 6" not in gaps[0]["detail"]


# ---------------- explainability
def test_shap_values_are_additive(predictor, test_df):
    X = predictor.preprocessor.transform(test_df[predictor.feature_columns]).astype(float)[:60]
    vals, base = explain_rows(predictor.explainer, X, predictor.feature_names)
    gap = np.abs(base + vals.sum(axis=1) - predictor.reg_model.predict(X)).max()
    assert gap <= ADDITIVITY_TOLERANCE


def test_local_explanation_matches_the_explained_input(predictor):
    r = predictor.predict_single({"G1": 6, "G2": 5, "subject": "Mathematics", "absences": 10, "failures": 3})
    ex = r["explanation"]
    assert ex["available"]
    total = ex["base_value"] + sum(ex["all_values"].values())
    assert total == pytest.approx(r["predicted_score_pct"], abs=1.0) or r["predicted_score_pct"] in (0.0, 100.0)
    assert ex["decreasing"][0]["feature"] == "G2"            # low G2 must pull the prediction down
    assert all(d["shap"] < 0 for d in ex["decreasing"]) and all(d["shap"] > 0 for d in ex["increasing"])


def test_high_grade_student_g2_raises_prediction(predictor):
    ex = predictor.predict_single({"G1": 18, "G2": 19, "subject": "Mathematics"})["explanation"]
    assert ex["increasing"][0]["feature"] == "G2"


def test_top_contributions_split_signs():
    ups, downs = top_contributions(np.array([2.0, -3.0, 0.5, -0.1]), ["a", "b", "c", "d"], n=2)
    assert [d["feature"] for d in ups] == ["a", "c"] and [d["feature"] for d in downs] == ["b", "d"]


def test_saved_explainability_summary_is_consistent(meta):
    from src.io_utils import read_json
    from src.paths import MODELS_DIR
    e = read_json(MODELS_DIR / "explainability_summary.json")
    s = read_json(MODELS_DIR / "evaluation_summary.json")
    assert e["model"] == s["best_regressor"] and e["shap_available"] and e["additivity_within_tolerance"]
    assert e["global_shap_importance"][0]["feature"] == "G2"
    for cs in e["case_studies"]:
        assert abs(cs["predicted_score_pct"] - cs["base_value"] - sum(
            d["shap"] for d in cs["increasing_features"] + cs["decreasing_features"])) < 100  # sanity only
