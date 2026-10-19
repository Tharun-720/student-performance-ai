# Issue and Correction Report

Audit of the original `student-performance-ai` project. Status: **Fixed** (code/data changed and verified by tests),
**Documented** (cannot be fixed with the available data; now stated clearly), **Open** (needs your decision or action).
IDs match `AUDIT_CHECKLIST_PHASE1.md`. This report intentionally quotes the *old, wrong* values.

## Critical
| ID | Problem found | Root cause | Correction | Status |
|---|---|---|---|---|
| C1 | `EXPLAINABILITY_REPORT.md` and viva Q6 quoted SHAP values (G2 = 8.42, G1 = 3.89, base 57.24) and three case-study profiles that did not exist in the saved results (e.g. case 1 "24 absences, 2 failures" while the real record had 0 and 0) | Numbers and narratives were written by hand, not taken from the pipeline output | Explainability re-run; report regenerated from `explainability_summary.json`; case studies are real test rows chosen by a fixed rule; viva guide rewritten | Fixed |
| C2 | Technical report listed Train R² (e.g. Random Forest 0.958) that did not match the saved results (0.928) for all six models | Hand-typed values. Recomputing from the saved model reproduced the JSON, so the pipeline was right and the document wrong | All result tables generated from JSON; a test fails if documents contain the old values | Fixed |
| C3 | Dashboard "SHAP waterfall" for a live student was `standardized value × feature importance × 10`, not SHAP; it called 10 absences "protective" | Approximation labelled as SHAP | Real `shap.TreeExplainer` on the exact input; neutral wording ("raises / lowers the predicted score"); additivity tested | Fixed |

## High
| ID | Problem | Correction | Status |
|---|---|---|---|
| H1 | The same students were in both train and test (55% of test rows had the student's other-subject record in train) | Student-aware group split. **Correction to my first estimate:** averaged over 5 seeds a random split scored about the same as the student-aware split (see evaluation report); the earlier headline R² of 0.909 came largely from one favourable split, not systematic leakage | Fixed |
| H2 | Best models were chosen using test-set scores | Selection by grouped 5-fold CV on training only; test used once; tested | Fixed |
| H3 | Recommender gave "Honors & Advanced Problem-Solving Track" to a student predicted High Support | Rules now use predicted tier, current grade and trend; severity-sorted; tested | Fixed |
| H4 | No input validation: empty input scored 55.8%; G1 = 50 and absences = −10 accepted | `src/validation.py`; required fields; ranges; categories; defaults reported | Fixed |
| H5 | "93 total sessions" treated as fact; attendance % shown as real | Relabelled "estimated"; absences shown as counts; rules use absences; documented | Fixed (estimate kept, labelled) |
| H6 | All 53 records with G3 = 0 have 0 absences, distorting the absences signal | Not silently altered. Sensitivity analysis added; documented in README, report, dashboard, viva | Documented |
| H7 | Cluster names hard-coded and contradicted the data ("High Absenteeism" cluster had 94% attendance); "silhouette verified" claim | Names generated from centroids; silhouette stated honestly (weak) | Fixed |
| H8 | "72% improvement over baseline" compared with the mean predictor | Added no-learning baselines (final = G2); honest reading in all documents. Result: the baselines are competitive | Fixed |

## Medium
| ID | Problem | Status |
|---|---|---|
| M1 | Working-directory-dependent paths (tests failed from another folder) | Fixed (`src/paths.py`, tested) |
| M2 | Prediction ran on page load; form button unused | Fixed (tested) |
| M3 | Batch results disappeared when downloading | Fixed (stored in session state) |
| M4 | Inference dropped duplicate rows (9 in, 6 out) and recomputed data-dependent cleaning | Fixed (no row dropping; training-time statistics; tested) |
| M5 | Hard-coded σ = 5.8 and "95% CI" label | Fixed (range from out-of-fold residual quantiles, labelled "approximate 95% prediction range") |
| M6 | XGBoost pickle version warning; unpinned packages | Mitigated: version manifest and in-app warning; `python -m src.run_pipeline` rebuilds artifacts; lower bounds tested (two wrong bounds found and fixed). Verify on your machine |
| M7 | pandas deprecation warning in `select_dtypes` | Fixed |
| M8 | 18 uses of deprecated `use_container_width` | Fixed (`width="stretch"`; requires streamlit ≥ 1.50) |
| M9 | Residual chart used 50 of 209 residuals | Fixed (all residuals) |
| M10 | Classifier/regressor disagreement not shown | Fixed (flagged in the UI); per-tier recall reported |
| M11 | Unsupported claims (FERPA/GDPR, "enterprise-grade", "100% pass rate", "balanced XGBoost", "Silhouette Verified", "Stratified K-Fold evaluation", "all 1,044 students") | Removed or corrected; a test scans documents |
| M12 | `TEST_RESULTS.md` overclaimed and exposed a local path/user name | Rewritten from real runs |
| M13 | Thin, partly vacuous tests (PII check asserted a hard-coded value) | New suite; PII check is real |
| M14 | University-style advice, invented "remedial pathway", wrong study-time text | Rewritten for secondary-school students; no invented programmes |
| M15 | Unverified affiliation / GitHub URL; no author placeholders | URL and unsupported claims removed; affiliation kept and marked unverified; placeholders added | Open: **you must fill in your details** |
| M16 | Early-warning limitation not stated | Stated everywhere; measured in the sensitivity analysis | Documented |

## Low (all addressed unless noted)
Fixed demographics in the form are disclosed (effect measured: under 0.5 points on average, largest single change about 3.6 points) · cap/imputation now training-only · CV with preprocessor inside folds ·
SHAP additivity checked with a tolerance (gap reported) · `.gitignore` extended, caches not shipped · README accuracy · Google Fonts / icons8 dependencies removed ·
"Section N" spec wording removed · radar chart axes explained · notebook generator updated · `data/raw/student.zip` and `.student.zip_old` kept (original UCI archives).
Still open: no `LICENSE` file for your own code; recommendation URLs are not checked automatically.

## Honest summary of how the results changed
| Quantity | Original claim | Now |
|---|---|---|
| Regression test R² / MAE | 0.909 / 3.95 (single random split, test used for selection) | see `MODEL_EVALUATION_REPORT.md` (student-aware split, CV selection) |
| Random Forest Train R² | 0.958 in the report | recomputed from the model in the evaluation report |
| Baseline | "72% better than baseline" (mean predictor) | compared with *final = G2*, which is competitive |
| Clusters | named personas | data-derived descriptions, weak structure |


## Second review (after the first correction round)
| Problem found | Correction |
|---|---|
| The "same student" fingerprint used 25 columns, giving 672 groups; the dataset authors' merge script uses 13 and gives 662. About 10 students were split into two groups, and 3 students (6 test rows) could sit on both sides. | Fingerprint now uses the 13 official attributes (662 groups). Whole pipeline re-run; all result tables, figures and documents regenerated. Selected regressor and all test numbers changed with the new split. |
| Dashboard said absences/age have "very small effects" although absences ranks #2 in SHAP. | Wording corrected; the G3 = 0 / 0-absences artifact is named as part of the reason. |
| Benchmark page called the student-aware runs "random splits". | Wording corrected. |
| Dead code in `train_models.py`; unused `None` placeholder and empty comment in `sensitivity.py`. | Removed. |
| Recommender would crash if a High Support tier was passed without a score. | Guarded. |
| README claimed Python 3.10-3.13 "expected to work". | Now says only 3.12 was verified. |
| Viva answer 7 and "672 students" text no longer matched the new results. | Updated to 662 and to the new comparison with the "final = G2" rule. |
Still open: tuning and model selection share the same CV folds, so CV scores of tuned models are slightly optimistic (stated in the evaluation summary).

## Third review (2026-10-17, independent re-check)
| Problem found | Correction |
|---|---|
| Recommendation text said "predicts a final score of about 53% (below the 50% pass mark)" when the *classifier* said High Support but the *regressor* predicted 53%. Tiers disagreed on 22 of 208 test records, and 14 records had a score under 50% but a "Medium Support" tier whose text said "above the pass mark". | Wording now refers to the classifier's tier and shows the predicted score without claiming it is below/above the pass mark. |
| "Changing sensitive attributes moved test predictions by under 1 point" had no source file and was not true for single students (re-measured on the 208 test records: mean change under 0.5 points; largest single change about 3.6 points, for sex). | Claim reworded in the app, README, viva guide and this report. |
Re-checked and found correct: data preparation reproduces the shipped CSVs; saved regressor/classifier reproduce the reported test metrics; the 95% prediction range covered 94.2% of test records (96.0% excluding G3 = 0).
Not re-run here (no internet for xgboost/shap/pytest/streamlit): the full 88-test suite, the Streamlit app, and `python -m src.run_pipeline`. Run them on your machine.
| Notebook `eda_analysis.ipynb` still showed the old output "distinct student fingerprints: 672" (re-running every code cell gave 662; all other saved text outputs matched). | Saved output corrected to 662. |
| Cluster sizes in `MODEL_EVALUATION_REPORT.md` were labelled "students", but they count training *records* (91 + 183 + 167 + 395 = 836). | Generator wording changed to "training records"; report regenerated. |
| Viva answer 15 said the model "barely reacts to absences"; absences is 3rd by SHAP (mean abs SHAP 1.1 vs 11.3 for G2). The dashboard SHAP note said absences "rank next" after G2, skipping G1 (2nd by SHAP and permutation importance). | Both texts corrected. |
