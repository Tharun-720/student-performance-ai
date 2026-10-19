"""Streamlit dashboard: Student Performance Prediction and Personalized Learning Recommendation System."""
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.eda_visualizations import (TIER_COLORS, TIER_ORDER, get_interactive_absences_scatter,
                                    get_interactive_cluster_radar, get_interactive_grade_dist,
                                    get_interactive_studytime_failures, get_interactive_trajectory)
from src.paths import DATA_PROCESSED, IMAGES_DIR, MODELS_DIR
from src.predict import StudentPredictor
from src.validation import ValidationError

st.set_page_config(page_title="Student Performance Prediction System", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.note {background:#f1f5f9;border-left:4px solid #64748b;padding:.7rem 1rem;border-radius:4px;color:#1e293b;font-size:.92rem;}
</style>""", unsafe_allow_html=True)

NAV = ["🏠 Dashboard", "📊 Exploratory Data Analysis", "🔮 Student Assessment & Predictor",
       "🧠 Explainability (SHAP)", "🎯 Recommendations", "🏆 Model Benchmark",
       "👥 Student Clusters (exploratory)", "📁 Batch Scoring"]
DISCLAIMER = ("Predictions are statistical estimates from historical data, not confirmed outcomes. "
              "They should support a teacher's judgement, never replace it.")


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading models ...")
def load_predictor():
    return StudentPredictor()


@st.cache_data
def load_json(name):
    with open(MODELS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA_PROCESSED / name)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


try:
    predictor = load_predictor()
    summary = load_json("evaluation_summary.json")
    meta = load_json("feature_metadata.json")
    explain = load_json("explainability_summary.json")
    sens = load_json("sensitivity_analysis.json")
    df_all = load_csv("student_cleaned.csv")
except FileNotFoundError as exc:
    st.error("Model or data files are missing. Run `python -m src.run_pipeline` from the project folder.")
    st.caption(f"Details: {exc}")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 Student Performance AI")
    st.caption("Prediction & learning-support recommendations")
    page = st.radio("Navigation", NAV, key="nav")
    st.divider()
    best_r, best_c = summary["best_regressor"], summary["best_classifier"]
    st.markdown(f"**Regressor:** {best_r}  \n**Classifier:** {best_c}  \n**Clustering:** K-Means (exploratory)")
    st.caption("Evaluation: grouped 5-fold CV on the training portion for model selection; one held-out "
               f"test split ({summary['evaluation_split']['test_rows']} records) for reporting.")
    for w in predictor.version_warnings:
        st.warning(w)
    st.divider()
    st.caption("Author: Tharun · Roll no.: 232G1A3249  \nCollege: [College name] · Guide: [Guide name]")

reg_best = summary["regression_metrics"][best_r]
clf_best = summary["classification_metrics"][best_c]
naive = summary["regression_metrics"]["Naive (G3 = G2)"]

# ================================================================== 1. DASHBOARD
if page == NAV[0]:
    st.title("Student Performance Prediction & Personalized Learning Recommendation")
    st.caption("Decision-support prototype built on the public UCI Student Performance dataset (two Portuguese secondary schools).")
    c = st.columns(4)
    c[0].metric("Records", f"{len(df_all):,}", help="Student-subject records (Maths + Portuguese).")
    c[1].metric("Distinct students (est.)", f"{meta['distinct_student_fingerprints']:,}",
                help="Estimated from matching attributes; the dataset has no student id.")
    c[2].metric("Test MAE (score points)", f"{reg_best['MAE']:.2f}")
    c[3].metric("Test macro-F1 (tier)", f"{clf_best['F1_Macro']:.3f}")
    st.subheader("How the system works")
    st.markdown("""
1. **Data**: Maths and Portuguese records are stacked; features are engineered (grade trend, estimated attendance, engagement index ...).
2. **Split**: a student-aware split keeps one student's records on one side only.
3. **Models**: a regressor predicts the final score (0-100); a classifier predicts the support tier; K-Means groups similar students (exploratory).
4. **Explanation**: SHAP values show which inputs raised or lowered a prediction.
5. **Recommendations**: transparent rules combine the inputs with the predicted tier.
""")
    st.subheader("Support tiers (from the final grade G3, 0-20)")
    st.table(pd.DataFrame({"Tier": TIER_ORDER, "Rule": ["G3 below 10", "10 ≤ G3 < 14", "G3 of 14 or more"]}))
    with st.expander("⚠️ Limitations and responsible use", expanded=True):
        st.markdown(f"""
- **Mostly a grade-continuation model.** G2 (second-period grade) carries almost all the signal. The simple rule "final = G2" has a test MAE of **{naive['MAE']:.2f}** versus **{reg_best['MAE']:.2f}** for the selected model. Without G1/G2 the model's R² drops to about **{sens['without_G1_G2_trend']['R2']:.2f}**.
- **Attendance is an estimate.** The dataset records a count of absences only; "estimated attendance" assumes 93 sessions, which is not a recorded figure.
- **Data artifact.** All {sens['g3_zero_records']['n_total']} records with a final grade of 0 have 0 absences, which weakens the absences signal.
- **Small, old, regional data** (Portuguese secondary schools); results may not transfer to other settings.
- **Not a verdict.** {DISCLAIMER}
""")

# ================================================================== 2. EDA
elif page == NAV[1]:
    st.title("Exploratory Data Analysis")
    f1, f2 = st.columns(2)
    subj = f1.multiselect("Subject", sorted(df_all["subject"].unique()), default=sorted(df_all["subject"].unique()))
    tiers = f2.multiselect("Actual final tier", TIER_ORDER, default=TIER_ORDER)
    fdf = df_all[df_all["subject"].isin(subj) & df_all["support_level"].isin(tiers)]
    if fdf.empty:
        st.warning("No records match the selected filters. Select at least one subject and one tier.")
    else:
        st.caption(f"{len(fdf):,} records selected")
        a, b = st.columns(2)
        a.plotly_chart(get_interactive_grade_dist(fdf), width="stretch")
        b.plotly_chart(get_interactive_absences_scatter(fdf), width="stretch")
        a.plotly_chart(get_interactive_trajectory(fdf), width="stretch")
        b.plotly_chart(get_interactive_studytime_failures(fdf), width="stretch")
        cols = ["final_score_pct", "G1", "G2", "studytime", "failures", "absences", "assessment_trend", "engagement_consistency"]
        st.plotly_chart(px.imshow(fdf[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Correlation matrix (Pearson)"), width="stretch")
        note("Note: every record with a final score of 0 has 0 absences, so the absences/score relationship is distorted.")
        with st.expander("Data preview"):
            st.dataframe(fdf.head(200), width="stretch")

# ================================================================== 3. PREDICTOR
elif page == NAV[2]:
    st.title("Student Assessment & Predictor")
    note(DISCLAIMER)
    PERSONAS = {"Typical": dict(G1=11, G2=11, studytime=2, failures=0, absences=4, goout=3, freetime=3),
                "Needs support": dict(G1=7, G2=6, studytime=1, failures=2, absences=12, goout=4, freetime=4),
                "Strong": dict(G1=16, G2=17, studytime=3, failures=0, absences=1, goout=2, freetime=3)}
    for k, v in dict(subject="Mathematics", higher="yes", **PERSONAS["Typical"]).items():
        st.session_state.setdefault(f"in_{k}", v)

    def apply_persona(name):
        for k, v in PERSONAS[name].items():
            st.session_state[f"in_{k}"] = v

    st.write("Quick-fill examples (illustrative inputs, not real students):")
    pc = st.columns(3)
    for col, name in zip(pc, PERSONAS):
        col.button(name, on_click=apply_persona, args=(name,), width="stretch")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        subject = c1.selectbox("Subject", ["Mathematics", "Portuguese"], key="in_subject")
        g1 = c1.slider("G1: first-period grade (0-20)", 0, 20, key="in_G1")
        g2 = c1.slider("G2: second-period grade (0-20)", 0, 20, key="in_G2")
        absences = c2.slider("Absences recorded (count)", 0, 93, key="in_absences")
        failures = c2.selectbox("Past class failures", [0, 1, 2, 3], key="in_failures")
        studytime = c2.select_slider("Weekly study time", [1, 2, 3, 4], key="in_studytime",
                                     format_func=lambda x: {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}[x])
        goout = c3.slider("Going out with friends (1-5)", 1, 5, key="in_goout")
        freetime = c3.slider("Free time after school (1-5)", 1, 5, key="in_freetime")
        higher = c3.selectbox("Wants higher education", ["yes", "no"], key="in_higher")
        submitted = st.form_submit_button("Run prediction", type="primary", width="stretch")
    st.caption("All other attributes (sex, age, school, family details ...) are not asked here; they are set to the "
               "training-data median/mode and listed in the result. Changing them moved test predictions by under 0.5 points on average (largest single-student change about 3.6 points).")

    if submitted:
        record = dict(subject=subject, G1=g1, G2=g2, absences=absences, failures=failures, studytime=studytime,
                      goout=goout, freetime=freetime, higher=higher)
        try:
            st.session_state["latest_result"] = predictor.predict_single(record)
            st.session_state["latest_input"] = record
        except ValidationError as exc:
            st.session_state.pop("latest_result", None)
            for e in exc.errors:
                st.error(e)

    res = st.session_state.get("latest_result")
    if res is None:
        st.info("Set the inputs and press **Run prediction**.")
    else:
        m = st.columns(3)
        m[0].metric("Predicted final score", f"{res['predicted_score_pct']:.1f}%", f"≈ grade {res['predicted_grade_20']}/20", delta_color="off")
        m[1].metric("Predicted support tier", res["support_level"])
        pr = res["prediction_range"]
        m[2].metric("Approx. 95% range", f"{pr['low']:.0f}% – {pr['high']:.0f}%")
        if not res["tiers_agree"]:
            st.warning(f"The two models disagree: the classifier says **{res['support_level']}**, while the predicted "
                       f"score implies **{res['score_implied_tier']}**. Treat this case as borderline.")
        probs = pd.DataFrame({"Tier": list(res["support_probabilities"]), "Probability": list(res["support_probabilities"].values())})
        st.plotly_chart(px.bar(probs, x="Tier", y="Probability", color="Tier", color_discrete_map=TIER_COLORS,
                               range_y=[0, 1], title="Classifier probability per tier"), width="stretch")
        st.caption(f"Prediction range: {pr['label']}. Student cluster (exploratory): {res['cluster_name']}.")
        if res["defaulted_fields"]:
            with st.expander(f"{len(res['defaulted_fields'])} inputs were not asked and use training medians/modes"):
                st.write(", ".join(res["defaulted_fields"]))
        note(res["disclaimer"])

# ================================================================== 4. EXPLAINABILITY
elif page == NAV[3]:
    st.title("Explainability (SHAP)")
    st.markdown("**SHAP values** are additive contributions to one prediction, in points of predicted score (0-100). "
                "They describe what the *model* uses, not what *causes* a student's result.")
    t1, t2, t3 = st.tabs(["Global importance", "Local explanation (last prediction)", "Case studies"])
    with t1:
        gs = pd.DataFrame(explain["global_shap_importance"])
        st.plotly_chart(px.bar(gs.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
                               title=f"Mean |SHAP| over {explain['n_rows_explained']} test records ({explain['model']})"),
                        width="stretch")
        pi = pd.Series(explain["global_permutation_importance"]).head(10).rename("Drop in test R² when shuffled")
        ti = pd.Series(explain.get("global_tree_importance", {})).head(10).rename("Tree importance")
        st.write("Three different measures (they need not agree):")
        st.dataframe(pd.concat([pi, ti], axis=1), width="stretch")
        note("G2 dominates every method. G1, absences and estimated attendance follow, with much smaller effects. "
             "Part of that signal is a data artifact: every record with a final grade of 0 has 0 absences, so the model "
             "can learn a spurious link. Read it as what the model uses, not as a cause.")
    with t2:
        res = st.session_state.get("latest_result")
        if res is None:
            st.info("Run a prediction first (Student Assessment & Predictor).")
        elif not res["explanation"]["available"]:
            st.warning("SHAP is not available for the selected model type.")
        else:
            ex = res["explanation"]
            st.write(f"Base value (average prediction): **{ex['base_value']:.1f}%** → this student: **{res['predicted_score_pct']:.1f}%**")
            rows = [(d["feature"], d["shap"]) for d in ex["increasing"]] + [(d["feature"], d["shap"]) for d in ex["decreasing"]]
            ed = pd.DataFrame(rows, columns=["feature", "shap"]).sort_values("shap")
            fig = go.Figure(go.Bar(x=ed["shap"], y=ed["feature"], orientation="h",
                                   marker_color=["#d62728" if v < 0 else "#2ca02c" for v in ed["shap"]]))
            fig.update_layout(title="Largest SHAP contributions for this student (points)", xaxis_title="Change in predicted score")
            st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            c1.success("**Raise the predicted score:** " + (", ".join(f"{d['feature']} ({d['shap']:+.1f})" for d in ex["increasing"]) or "none"))
            c2.warning("**Lower the predicted score:** " + (", ".join(f"{d['feature']} ({d['shap']:+.1f})" for d in ex["decreasing"]) or "none"))
    with t3:
        st.caption("Real test-set records chosen by a fixed rule (not hand-picked): lowest score in High Support, "
                   "median of Medium Support, highest score in Low Support.")
        for cs in explain["case_studies"]:
            with st.expander(f"{cs['title']}: {cs['subject']}, actual {cs['actual_score_pct']}% → predicted {cs['predicted_score_pct']}%"):
                st.write(f"G1 = {cs['G1']:g}, G2 = {cs['G2']:g}, absences = {cs['absences']:g}, failures = {cs['failures']}, study time = {cs['studytime']}")
                st.write("Increase prediction: " + ", ".join(f"{d['feature']} ({d['shap']:+.1f})" for d in cs["increasing_features"]))
                st.write("Decrease prediction: " + ", ".join(f"{d['feature']} ({d['shap']:+.1f})" for d in cs["decreasing_features"]))
                if cs["note"]:
                    st.info(cs["note"])

# ================================================================== 5. RECOMMENDATIONS
elif page == NAV[4]:
    st.title("Recommendations")
    res = st.session_state.get("latest_result")
    if res is None:
        st.info("Run a prediction first (Student Assessment & Predictor). Recommendations use the inputs and the predicted tier.")
    else:
        recs = res["recommendations"]
        st.write(f"Predicted tier: **{res['support_level']}** · {recs['total_gaps_identified']} gap(s) identified")
        for a in recs["priority_actions"]:
            icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵", "Info": "⚪"}[a["severity"]]
            with st.expander(f"{icon} {a['severity']}: {a['title']}", expanded=a["severity"] in ("Critical", "High")):
                st.write(f"**Why:** {a['finding']}")
                st.write(f"**Suggested step:** {a['detail']}")
        st.subheader("Generic weekly study template")
        st.caption("The same starting template for every student; adapt it together with the student.")
        st.dataframe(pd.DataFrame(recs["weekly_template"]), width="stretch", hide_index=True)
        st.subheader("Example resources")
        for r in recs["resources"]:
            st.markdown(f"- [{r['name']}]({r['url']}): {r['use']}" if r["url"] else f"- {r['name']}: {r['use']}")
        for n in recs["data_notes"]:
            note(n)
        lines = ["ACADEMIC SUPPORT SUMMARY (decision support, not a verdict)", "=" * 56,
                 f"Predicted score: {res['predicted_score_pct']}%  ({res['prediction_range']['low']}-{res['prediction_range']['high']}% approx. range)",
                 f"Predicted tier: {res['support_level']}", ""]
        lines += [f"[{a['severity']}] {a['title']}: {a['finding']}" for a in recs["priority_actions"]]
        lines += ["", *recs["data_notes"]]
        st.download_button("Download summary (.txt)", "\n".join(lines), file_name="support_summary.txt", width="stretch")

# ================================================================== 6. BENCHMARK
elif page == NAV[5]:
    st.title("Model Benchmark")
    note(f"Models were selected by grouped 5-fold cross-validation on the training portion "
         f"({summary['evaluation_split']['train_rows']} records). The held-out test split "
         f"({summary['evaluation_split']['test_rows']} records) was used once for the numbers below. "
         "With so few test records, small differences between models are within noise.")
    rm = summary["regression_metrics"]
    rdf = pd.DataFrame([{"Model": k, "CV MAE": v["CV_MAE"], "Test MAE": v["MAE"], "Test RMSE": v["RMSE"],
                         "Test R²": v["R2"], "Train R²": v["Train_R2"]} for k, v in rm.items()])
    st.subheader("Regression (final score, 0-100)")
    st.dataframe(rdf, width="stretch", hide_index=True)
    st.write(f"Selected: **{best_r}**. The no-learning rule *final = G2* has test MAE {naive['MAE']:.2f} "
             f"(selected model: {reg_best['MAE']:.2f}).")
    cm_ = summary["classification_metrics"]
    cdf = pd.DataFrame([{"Model": k, "CV macro-F1": v["CV_F1_Macro"], "Test accuracy": v["Accuracy"],
                         "Test macro-F1": v["F1_Macro"], "ROC-AUC": v["ROC_AUC"]} for k, v in cm_.items()])
    st.subheader("Classification (support tier)")
    st.dataframe(cdf, width="stretch", hide_index=True)
    rc = clf_best["Recall_Per_Class"]
    st.write("Recall per tier for the selected classifier: " + ", ".join(f"{k} {v:.2f}" for k, v in rc.items()))
    a, b = st.columns(2)
    a.image(str(IMAGES_DIR / "07_confusion_matrix.png"), caption="Confusion matrix (test split)", width="stretch")
    b.image(str(IMAGES_DIR / "09_residual_analysis.png"), caption="Residuals of the selected regressor (all test records)", width="stretch")
    st.subheader("Sensitivity checks")
    rs = sens["random_vs_group_split"]
    st.write(f"Across 5 different student-aware splits the default Random Forest has mean R² {rs['group_split_5_seeds']['R2']['mean']:.3f} ± "
             f"{rs['group_split_5_seeds']['R2']['sd']:.3f} (student-aware) and test MAE {rs['group_split_5_seeds']['MAE']['mean']:.2f}; "
             "single splits vary noticeably, so avoid over-reading one number.")

# ================================================================== 7. CLUSTERS
elif page == NAV[6]:
    st.title("Student Clusters (exploratory)")
    cl = summary["clustering"]
    note(f"K-Means with k = {cl['k']} on {len(cl['features'])} behavioural features. Silhouette = "
         f"{cl['silhouette_selected_k']} (k=2: {cl['k_metrics']['2']}); values below ~0.25 mean weak structure. "
         "The clusters are descriptive groupings, not validated student types. Names are generated from each centroid.")
    st.plotly_chart(get_interactive_cluster_radar(summary), width="stretch")
    prof = pd.DataFrame(cl["profiles"]).T
    prof.index = [cl["archetype_names"][i] for i in prof.index]
    st.dataframe(prof.drop(columns=["centroid_zscores"]), width="stretch")
    st.image(str(IMAGES_DIR / "08_kmeans_clusters_pca.png"), caption="2-D PCA projection of the clusters", width="stretch")

# ================================================================== 8. BATCH
else:
    st.title("Batch Scoring")
    st.write("Upload a CSV with at least the columns **G1, G2, subject**. Other known columns are used if present; "
             "missing ones use training medians/modes. Rows are never dropped; invalid rows are flagged.")
    sample = load_csv("test.csv").drop(columns=["final_score_pct", "support_level", "student_group"]).head(100)
    st.download_button("Download sample CSV (100 rows from the test split)", sample.to_csv(index=False),
                       file_name="sample_students.csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as exc:
            st.error(f"Could not read the file as CSV: {exc}")
            data = None
        if data is not None:
            st.caption(f"{len(data):,} rows, {len(data.columns)} columns")
            if st.button("Run batch prediction", type="primary"):
                try:
                    st.session_state["batch_result"] = predictor.predict_batch(data)
                except ValidationError as exc:
                    st.session_state.pop("batch_result", None)
                    for e in exc.errors:
                        st.error(e)
    out = st.session_state.get("batch_result")
    if out is not None:
        bad = int(out["Validation_Status"].str.startswith("INVALID").sum())
        c = st.columns(3)
        c[0].metric("Rows", len(out)); c[1].metric("Invalid rows", bad)
        c[2].metric("High Support predicted", int((out["Support_Level"] == "High Support").sum()))
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("Download results (.csv)", out.to_csv(index=False), file_name="batch_predictions.csv")

st.divider()
st.caption("Educational prototype · UCI Student Performance dataset (CC BY 4.0) · " + DISCLAIMER)
