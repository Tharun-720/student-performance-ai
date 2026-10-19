# Technical Report

**Project:** Student Performance Prediction and Personalized Learning Recommendation System
**Author:** Tharun · **Roll no.:** 232G1A3249 · **College:** [College name] · **Guide:** [Guide name]

## 1. Abstract

This project builds a decision-support prototype on the public UCI *Student Performance* dataset. It predicts a
student's final score and a three-level support tier from earlier grades and background attributes, explains
predictions with SHAP, groups students with K-Means (exploratory) and turns the inputs and the predicted tier into
rule-based support suggestions, all in a Streamlit dashboard. The evaluation is deliberately conservative: a
student-aware split, model selection by cross-validation on the training portion only, and comparison with a
no-learning baseline.

## 2. Headline results

<!-- RESULTS:START -->
| Task | Selected model | Test result (held-out split) | No-learning reference |
|---|---|---|---|
| Final score (0-100) | Gradient Boosting (Tuned) | MAE **4.54**, RMSE 8.26, R² **0.795** | "final = G2": MAE 3.85, R² 0.802 |
| Support tier | Random Forest (Tuned) | accuracy **0.870**, macro-F1 **0.871** | tier from G2: accuracy 0.861, macro-F1 0.861 |

Evaluation split: 836 training and 208 test records; a student's records never appear on both sides. Models were selected by 5-fold grouped cross-validation on the training portion; the test split was used once. Without G1/G2 the same model family reaches only R² 0.21 (MAE 11.7).
<!-- RESULTS:END -->

Honest reading: the second-period grade (G2) carries almost all the predictive power. The trained models add a
small amount over the rule *final = G2* on some metrics and not on others; the differences are within the noise of a
~200-record test set. The project's value lies in the end-to-end, explainable and tested pipeline, not in beating that
rule by a wide margin.

## 3. Dataset

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, *Student Performance* (Cortez & Silva, 2008), CC BY 4.0 |
| Files | `student-mat.csv` (395 rows), `student-por.csv` (649 rows), stacked with a `subject` column |
| Records | 1,044; **about 662 distinct students** (382 students appear in both files; the dataset has no student id) |
| Missing values / exact duplicates | 0 / 0 |
| Target | `G3` final grade (0-20); `final_score_pct = G3 × 5`; support tier from G3 |
| Context | Two Portuguese secondary schools; students aged 15-22 |

Support tiers: **High Support** G3 < 10; **Medium Support** 10 ≤ G3 < 14; **Low Support** G3 ≥ 14
(class counts in the raw data: Medium 520, Low 294, High 230). The thresholds are a project choice, not an official standard.

## 4. Methodology

### 4.1 Features
Inputs are the 30 survey/school attributes plus G1 and G2 (see `DATA_DICTIONARY.md`). **G3 is never an input**
(target leakage). Derived features: `assessment_trend` (G2 − G1), `estimated_attendance_pct`,
`study_to_social_ratio`, `parent_edu_avg`, `past_failure_flag`, `engagement_consistency` (a weighted index of study time,
estimated attendance and failures; a project-defined index, not a validated psychological measure).

**Attendance assumption.** The raw data holds `absences` (a count). The UCI documentation says "numeric: from 0 to 93",
which is the *range* of the column. The project uses 93 as an *assumed* number of sessions to derive
`estimated_attendance_pct = 100 − absences/93 × 100` (clipped to 0-100). This is an **estimate**, not recorded
attendance. The dashboard therefore shows counts of absences and labels the derived value "estimated".

### 4.2 Train/test split (student-aware)
Students enrolled in both courses appear twice. A purely random split can put one record in training and the other
in test. A "same student" fingerprint (the 13 attributes used by the dataset authors' merge script) is used as a group id, and a stratified group split keeps each
student on one side (`StratifiedGroupKFold`, first of 5 folds, seed 42). The fingerprint is a heuristic.
Averaged over 5 seeds the old random split and the student-aware split gave similar scores (see
`MODEL_EVALUATION_REPORT.md`); the student-aware split is still the methodologically correct choice.

### 4.3 Preprocessing and leakage control
Standard scaling and one-hot encoding, the absence cap (99th percentile) and the default values for optional inputs are
all fitted on the **training portion only**. Hyper-parameter tuning and model selection use grouped, tier-stratified
5-fold cross-validation with the preprocessor *inside* each fold. The test split is evaluated once.

### 4.4 Models
Regression: mean baseline, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, plus tuned Gradient Boosting
and Random Forest (small grids). Classification: majority baseline, Logistic Regression, Decision Tree, Random Forest
(class-weighted), Gradient Boosting, XGBoost, tuned Random Forest. A no-learning reference ("final = G2", and the tier
computed from G2) is always reported. The tuned models' CV score is the best over a grid and therefore slightly optimistic.
Fixed seeds make runs repeatable.

### 4.5 Clustering (exploratory)
K-Means on eight behavioural features after standard scaling, fitted on the training portion. k = 4 was chosen for
interpretability; silhouette scores are low (about 0.19-0.26 for k = 2-6), i.e. weak structure. Cluster names are generated
from each centroid's most distinctive features, so they cannot contradict the data. Clusters are descriptive groupings, not validated types.

### 4.6 Explainability
`shap.TreeExplainer` on the selected regressor. SHAP values are exactly additive (base value + sum of contributions =
model output; maximum gap reported in `EXPLAINABILITY_REPORT.md`). Three different measures are shown and never mixed:
SHAP values, permutation importance (drop in test R²), and tree importance. SHAP describes what the **model** relies
on; it does not establish causes. The live-student panel uses the same TreeExplainer on the exact input shown.

### 4.7 Prediction range
The "approximate 95% range" adds the 2.5% and 97.5% quantiles of the out-of-fold residuals of the selected model to the
prediction. It assumes future errors resemble past errors and is not a guarantee.

### 4.8 Recommendation rules
Rules use only inputs the user supplied (defaulted fields cannot trigger a rule) and the predicted tier. Actions are
sorted by severity (Critical, High, Medium, Low, Info) and at most four are shown.

| Rule | Condition | Severity |
|---|---|---|
| Predicted outcome | predicted tier = High Support | Critical |
| Predicted outcome | predicted tier = Medium Support | Medium |
| Current attainment | G2 < 10 | Critical |
| Current attainment | 10 ≤ G2 < 12 | Medium |
| Grade trend | G2 − G1 ≤ −3 / ≤ −1 | High / Medium |
| Absences | ≥ 20 / ≥ 10 | High / Medium |
| Study habits | study time = 1 (< 2 h) / = 2 and tier is High or Medium | High / Low |
| Academic history | failures ≥ 2 / = 1 | High / Medium |
| Time balance | going out ≥ 4 and study time ≤ 2 | Low |
| No gap, tier Low Support | none | Info: maintain habits |
| No gap, other tier | none | Info: review with teacher |

The thresholds are project choices informed by common-sense cut-offs, not validated clinical or pedagogical standards.
The weekly timetable is a generic template, identical for every student.

## 5. Software design

`src/paths.py` resolves every path from the project root, so the app and tests work from any folder (and on Windows).
`src/validation.py` rejects invalid input (type, range, allowed categories, required G1/G2/subject) with all
problems listed at once; optional inputs fall back to training medians/modes and are **listed in the result**.
Batch scoring never drops or reorders rows; invalid rows are flagged. Saved models record library versions in
`models/model_manifest.json`; the app warns if the installed versions differ.

## 6. Testing
See `TEST_RESULTS.md` for the actual run output. The suite covers data loading and validation, tier thresholds,
feature engineering on small controlled data, student-aware splitting, input validation, metric functions with
known answers, re-computation of every saved metric from the saved models, model-selection rules, batch behaviour,
recommender logic, SHAP additivity, charts with empty data, Streamlit startup and every page, and documentation
consistency with the JSON results.

## 7. Limitations
1. Predictions depend on the quality and relevance of the data; this dataset is small, old and regional.
2. G2 dominates; without G1/G2 performance is much lower, so the system is not an early-warning tool for students with no grade history.
3. Attendance is estimated, not recorded (section 4.1).
4. All 53 records with final grade 0 have 0 absences (probably exam non-attendance), which distorts the absences signal.
5. The two models can disagree on borderline students; the dashboard flags this.
6. The test set is about 200 records, so small differences between models are not meaningful.
7. Sensitive attributes are model inputs; no formal fairness audit was done.
8. Rule thresholds and cluster count are project choices.
9. Not validated on any institution's real students; not suitable for high-stakes decisions.

## 8. Future work
Collect real attendance totals; add repeated cross-validation with confidence intervals; fairness analysis by group;
calibrate classifier probabilities; evaluate on data from another institution.

## 9. Reference
P. Cortez and A. Silva (2008). *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J.
Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12.
