# Student Performance Prediction and Personalized Learning Recommendation System

A B.Tech project that predicts a student's final score and support tier from earlier grades and background
information, explains each prediction with SHAP, and suggests learning-support actions through transparent rules.
It is a **decision-support prototype**, not a verdict on any student.

| | |
|---|---|
| Author | Tharun |
| Roll number | 232G1A3249 |
| College / Department | [College name], [Department] |
| Project guide | [Guide name] |
| Organisation (as supplied) | Trekverse Edutech - role: Project Manager *(not independently verified; edit as needed)* |

## What it does

- **Data**: UCI *Student Performance* dataset (Cortez & Silva, 2008; two Portuguese secondary schools; CC BY 4.0):
  395 Maths + 649 Portuguese records = 1,044 records, about 662 distinct students (382 students are in both files).
- **Regression**: predicts the final score (0-100, i.e. final grade G3 x 5).
- **Classification**: predicts a support tier - *High Support* (G3 < 10), *Medium Support* (10 to <14), *Low Support* (14+).
- **Clustering (exploratory)**: K-Means groups students by behavioural features.
- **Explainability**: SHAP values (global ranking, per-student explanation, three case studies from the test set).
- **Recommendations**: rule-based, using the inputs and the predicted tier; rules are documented in the technical report.
- **Dashboard**: 8 Streamlit pages including single-student prediction and batch CSV scoring with validation.

## Results

<!-- RESULTS:START -->
| Task | Selected model | Test result (held-out split) | No-learning reference |
|---|---|---|---|
| Final score (0-100) | Gradient Boosting (Tuned) | MAE **4.54**, RMSE 8.26, R² **0.795** | "final = G2": MAE 3.85, R² 0.802 |
| Support tier | Random Forest (Tuned) | accuracy **0.870**, macro-F1 **0.871** | tier from G2: accuracy 0.861, macro-F1 0.861 |

Evaluation split: 836 training and 208 test records; a student's records never appear on both sides. Models were selected by 5-fold grouped cross-validation on the training portion; the test split was used once. Without G1/G2 the same model family reaches only R² 0.21 (MAE 11.7).
<!-- RESULTS:END -->

Full tables: [`docs/MODEL_EVALUATION_REPORT.md`](docs/MODEL_EVALUATION_REPORT.md),
[`docs/EXPLAINABILITY_REPORT.md`](docs/EXPLAINABILITY_REPORT.md). These files are generated from the saved JSON results.

## Limitations (please read)

1. **Mostly a "G2 predicts G3" model.** The second-period grade carries almost all of the signal. A rule that simply
   predicts *final = G2* is competitive with the trained models (see the table above). Without G1/G2 the models are weak,
   so this is **not** an early-warning system for students with no grade history.
2. **Attendance is an estimate.** The dataset contains only a count of absences. "Estimated attendance" assumes 93
   sessions, a number taken from the dataset's documentation of the *range* of absences; it is not a recorded total.
3. **Data artifact.** All 53 records with final grade 0 have 0 absences (probably students who did not sit the exam),
   which weakens the absences signal. See `docs/MODEL_EVALUATION_REPORT.md`.
4. **Small, old, regional data**: results may not transfer to other schools or countries.
5. **Predictions are estimates**, not facts; they depend on the quality and relevance of the data.
6. **Sensitive attributes** (sex, age, parental education, ...) are among the model inputs. Changing them moved test
   predictions by under 0.5 points on average (largest single-student change about 3.6 points), but the model has not been audited for fairness.
7. **Clusters are descriptive.** Silhouette scores are low (weak structure).
8. Recommendation links are examples and were not checked automatically.

## Setup on Windows (PowerShell or Command Prompt)

Tested here on Linux with Python 3.12; your machine may use a different Python version (only 3.12 was verified; other versions are untested).
Check your own run and note the result in `docs/TEST_RESULTS.md`.

```bat
python --version
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app/main.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first, or use Command Prompt.

The shipped models were built with Python 3.13 / scikit-learn 1.9.1 / xgboost 3.4.1 (see `models/model_manifest.json`); older versions
may show `InconsistentVersionWarning`. If the app warns that the saved models were created with other library versions, rebuild them for your environment
(takes about two minutes; no internet needed). Note: results can differ slightly between library versions (in my lower-bound
test the tuned model chosen by cross-validation was a different one), so after rebuilding, the tables in `docs/` are regenerated
automatically and are the ones to quote:

```bat
python -m src.run_pipeline
```

## Project structure

```
app/main.py                 Streamlit dashboard (entry point)
src/
  paths.py                  project-root based paths (works from any folder)
  data_loader.py            load + validate raw data
  data_preprocessing.py     features, student-aware split, preprocessing
  validation.py             input validation for single / batch prediction
  train_models.py           CV-based selection, evaluation, clustering
  explainability.py         SHAP + permutation importance
  predict.py                inference, explanations, recommendations
  recommender.py            documented rule engine
  sensitivity.py            leakage / G3=0 / no-G1-G2 analyses
  eda_visualizations.py     static + interactive figures
  generate_reports.py       writes result tables from the JSON files
  run_pipeline.py           rebuilds everything
tests/                      pytest suite
data/raw, data/processed    UCI files, generated train/test CSVs
models/                     saved models and JSON results
docs/                       reports, data dictionary, viva guide, checklists, figures
```

## Documentation

`docs/FINAL_TECHNICAL_REPORT.md` · `docs/MODEL_EVALUATION_REPORT.md` · `docs/EXPLAINABILITY_REPORT.md` ·
`docs/DATA_DICTIONARY.md` · `docs/TEST_RESULTS.md` · `docs/ISSUE_AND_CORRECTION_REPORT.md` ·
`docs/VIVA_PREPARATION_GUIDE.md` · `docs/SCREENSHOT_CHECKLIST.md` · `docs/SUBMISSION_CHECKLIST.md`

## Data source and licence

Dataset: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008
(UCI Machine Learning Repository, CC BY 4.0). The project's own source-code licence is not yet chosen
(add a `LICENSE` file if you publish it).
