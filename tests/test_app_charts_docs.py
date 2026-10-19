import json
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.eda_visualizations import (get_interactive_absences_scatter, get_interactive_cluster_radar,
                                    get_interactive_grade_dist, get_interactive_studytime_failures,
                                    get_interactive_trajectory)
from src.paths import DOCS_DIR, IMAGES_DIR, MODELS_DIR, PROJECT_ROOT
from src.io_utils import read_json

APP = str(PROJECT_ROOT / "app" / "main.py")


# ---------------- chart data
def test_chart_functions_return_figures_with_data():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "student_cleaned.csv")
    for fn in (get_interactive_grade_dist, get_interactive_absences_scatter, get_interactive_trajectory,
               get_interactive_studytime_failures):
        fig = fn(df)
        assert len(fig.data) > 0


def test_chart_functions_accept_empty_and_single_group_frames():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "student_cleaned.csv")
    for fn in (get_interactive_grade_dist, get_interactive_absences_scatter, get_interactive_trajectory,
               get_interactive_studytime_failures):
        fn(df.iloc[0:0])
        fn(df[df["support_level"] == "High Support"])


def test_cluster_radar_uses_saved_profiles():
    s = read_json(MODELS_DIR / "evaluation_summary.json")
    fig = get_interactive_cluster_radar(s)
    assert len(fig.data) == s["clustering"]["k"]


def test_static_figures_exist():
    for name in ("06_regression_benchmark.png", "07_confusion_matrix.png", "08_kmeans_clusters_pca.png",
                 "09_residual_analysis.png"):
        assert (IMAGES_DIR / name).stat().st_size > 5000


def test_cluster_names_follow_centroids():
    s = read_json(MODELS_DIR / "evaluation_summary.json")["clustering"]
    for cid, prof in s["profiles"].items():
        name = s["archetype_names"][cid].lower()
        z = prof["centroid_zscores"]
        if "high study time" in name:
            assert z["studytime"] > 0
        if "low study time" in name:
            assert z["studytime"] < 0
        if "more past failures" in name:
            assert z["failures"] > 0


# ---------------- Streamlit app
def _run():
    at = AppTest.from_file(APP, default_timeout=120)
    return at.run()


def test_app_starts_without_exception():
    at = _run()
    assert not at.exception


def test_every_page_renders():
    at = _run()
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_prediction_requires_button_press():
    at = _run()
    at.sidebar.radio[0].set_value("🔮 Student Assessment & Predictor").run()
    assert "latest_result" not in at.session_state
    at.button[0].click()  # first plain button is a persona quick-fill, not the form submit
    at.run()
    assert "latest_result" not in at.session_state
    submit = [b for b in at.button if "Run prediction" in b.label][0]
    submit.click().run()
    assert "latest_result" in at.session_state and not at.exception


def test_recommendation_and_explainability_pages_use_the_prediction():
    at = _run()
    at.sidebar.radio[0].set_value("🔮 Student Assessment & Predictor").run()
    [b for b in at.button if "Run prediction" in b.label][0].click().run()
    for page in ("🧠 Explainability (SHAP)", "🎯 Recommendations"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception


def test_eda_page_with_empty_filters_shows_warning_not_error():
    at = _run()
    at.sidebar.radio[0].set_value("📊 Exploratory Data Analysis").run()
    at.multiselect[0].set_value([]).run()
    assert not at.exception and any("No records" in w.value for w in at.warning)


# ---------------- documentation consistency with saved results
def _num_variants(x, nd):
    return {f"{x:.{nd}f}"}


def test_documents_quote_numbers_that_exist_in_saved_results():
    s = read_json(MODELS_DIR / "evaluation_summary.json")
    br = s["best_regressor"]
    r = s["regression_metrics"][br]
    for doc in ("README.md", "docs/FINAL_TECHNICAL_REPORT.md"):
        text = (PROJECT_ROOT / doc).read_text(encoding="utf-8")
        assert f"{r['MAE']:.2f}" in text and f"{r['R2']:.3f}" in text, doc
        assert br in text
    forbidden = ["0.958", "R² = 0.909", "8.42", "57.24"]
    for doc in PROJECT_ROOT.glob("**/*.md"):
        if ".venv" in doc.parts:
            continue
        t = doc.read_text(encoding="utf-8")
        if doc.name in ("ISSUE_AND_CORRECTION_REPORT.md", "VIVA_PREPARATION_GUIDE.md"):
            continue  # these quote the old wrong values on purpose (to explain the correction)
        for f in forbidden:
            assert f not in t, f"{f} found in {doc.name}"


def test_documents_do_not_make_unsupported_claims():
    banned = ["FERPA", "GDPR compliant", "enterprise-grade", "industry-grade", "100% pass", "SHAP waterfall",
              "Silhouette Verified", "balanced XGBoost"]
    for doc in list((PROJECT_ROOT).glob("*.md")) + list(DOCS_DIR.glob("*.md")):
        if doc.name in ("ISSUE_AND_CORRECTION_REPORT.md", "VIVA_PREPARATION_GUIDE.md"):
            continue
        t = doc.read_text(encoding="utf-8")
        for b in banned:
            assert b.lower() not in t.lower(), f"'{b}' in {doc.name}"
