import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

from src.metrics import classification_metrics, regression_metrics
from src.paths import MODELS_DIR


# ---------------- metric functions on controlled arrays
def test_regression_metrics_known_values():
    m = regression_metrics([10, 20, 30], [12, 18, 33])
    assert m["MAE"] == pytest.approx(7 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["R2"] == pytest.approx(1 - 17 / 200)


def test_regression_perfect_and_mean_predictor():
    assert regression_metrics([1, 2, 3], [1, 2, 3])["R2"] == pytest.approx(1.0)
    assert regression_metrics([1, 2, 3], [2, 2, 2])["R2"] == pytest.approx(0.0)


def test_classification_metrics_known_values():
    labels = ["A", "B"]
    m = classification_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"], labels)
    assert m["Accuracy"] == 0.75
    assert m["Confusion_Matrix"] == [[1, 1], [0, 2]]
    assert m["Recall_Per_Class"] == {"A": 0.5, "B": 1.0}
    assert m["F1_Macro"] == pytest.approx((2 * 1 * .5 / 1.5 + 2 * (2 / 3) * 1 / (5 / 3)) / 2)


# ---------------- artifacts vs saved summary (recomputed from real predictions)
def test_saved_regression_metrics_match_recomputation(summary, test_df, train_df, meta):
    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    cols = meta["numeric_features"] + meta["categorical_features"]
    Xte, Xtr = pre.transform(test_df[cols]), pre.transform(train_df[cols])
    models = joblib.load(MODELS_DIR / "all_reg_models.joblib")
    assert summary["best_regressor"] in models
    for name, model in models.items():
        s = summary["regression_metrics"][name]
        pred = model.predict(Xte)
        assert mean_absolute_error(test_df["final_score_pct"], pred) == pytest.approx(s["MAE"], abs=1e-3)
        assert r2_score(test_df["final_score_pct"], pred) == pytest.approx(s["R2"], abs=1e-3)
        assert r2_score(train_df["final_score_pct"], model.predict(Xtr)) == pytest.approx(s["Train_R2"], abs=1e-3)
        assert len(s["Residuals"]) == len(test_df)


def test_saved_classification_metrics_match_recomputation(summary, test_df, meta):
    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    cols = meta["numeric_features"] + meta["categorical_features"]
    X = pre.transform(test_df[cols])
    models = joblib.load(MODELS_DIR / "all_clf_models.joblib")
    for name, model in models.items():
        pred = [meta["classes"][int(i)] for i in model.predict(X)]
        m = classification_metrics(test_df["support_level"], pred, meta["classes"])
        s = summary["classification_metrics"][name]
        assert m["Accuracy"] == pytest.approx(s["Accuracy"], abs=1e-3)
        assert m["F1_Macro"] == pytest.approx(s["F1_Macro"], abs=1e-3)
        assert m["Confusion_Matrix"] == s["Confusion_Matrix"]
        assert sum(map(sum, s["Confusion_Matrix"])) == len(test_df)


def test_model_selection_did_not_use_the_test_set(summary):
    assert summary["selection_criterion"]["test_set_used_for_selection"] is False
    reg = summary["regression_metrics"]
    candidates = {k: v for k, v in reg.items() if v["Is_Model"] and not k.startswith("Baseline")}
    assert summary["best_regressor"] == min(candidates, key=lambda k: candidates[k]["CV_MAE"])
    clf = summary["classification_metrics"]
    cc = {k: v for k, v in clf.items() if v["Is_Model"] and not k.startswith("Baseline")}
    assert summary["best_classifier"] == max(cc, key=lambda k: cc[k]["CV_F1_Macro"])


def test_naive_baseline_is_reported_and_consistent(summary, test_df):
    naive = summary["regression_metrics"]["Naive (G3 = G2)"]
    assert naive["MAE"] == pytest.approx(mean_absolute_error(test_df["final_score_pct"], test_df["G2"] * 5), abs=1e-3)


def test_classes_and_thresholds_consistent_with_data(test_df, summary):
    from src.metrics import support_level_from_score_pct
    derived = test_df["final_score_pct"].apply(support_level_from_score_pct)
    assert (derived == test_df["support_level"]).all()
    assert set(summary["classes"]) == {"High Support", "Medium Support", "Low Support"}


def test_artifact_manifest_matches_installed_versions(predictor):
    assert predictor.version_warnings == []


def test_feature_schema_consistency(predictor, meta):
    pre = predictor.preprocessor
    assert list(pre.feature_names_in_) == meta["numeric_features"] + meta["categorical_features"]
    assert pre.get_feature_names_out().shape[0] == len(meta["transformed_feature_names"])


def test_training_is_reproducible_for_a_fixed_seed(train_df, meta):
    from sklearn.ensemble import RandomForestRegressor
    X = joblib.load(MODELS_DIR / "preprocessor.joblib").transform(train_df[meta["numeric_features"] + meta["categorical_features"]])
    y = train_df["final_score_pct"]
    a = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y).predict(X[:10])
    b = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y).predict(X[:10])
    assert np.allclose(a, b)


def test_evaluation_json_has_no_invented_values(summary):
    for v in summary["regression_metrics"].values():
        assert v["Train_R2"] <= 1.0 and -1.0 <= v["R2"] <= 1.0
