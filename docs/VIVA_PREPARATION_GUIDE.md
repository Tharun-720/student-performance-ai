# Viva Preparation Guide

Short, honest answers. For exact numbers always quote `docs/MODEL_EVALUATION_REPORT.md`, not memory.

**1. What problem does the project solve?**
It estimates a student's final score and support tier from earlier grades and background data, explains why, and
suggests support actions, so a teacher can look at borderline students earlier. It supports teachers; it does not decide anything.

**2. Which dataset? Is it real student data?**
The public UCI *Student Performance* dataset (two Portuguese secondary schools, 2008). 1,044 records, about 662 distinct students
because 382 students appear in both the Maths and Portuguese files. It is not data from my college or from any company.

**3. Why G3 × 5 and these three tiers?**
G3 is the final grade out of 20, so × 5 gives a percentage. Tiers: below 10 (below the pass mark), 10 to <14, and 14 or more. These
cut-offs are my design choice; a different school could use different ones.

**4. Why can't G3 be an input? What is leakage?**
G3 *is* the target. Using it, or anything computed from it, would leak the answer. I also fit scaling, encoding and the absence cap on
the training data only, so no information from the test set reaches the model.

**5. Why a student-aware split?**
The same student can appear twice (once per subject). A random split could put one record in training and the other in test. I group
records by a "same student" fingerprint so a student is only on one side. (The fingerprint is a heuristic; the dataset has no student id.)
Averaged over several seeds the scores were similar to a random split, but the student-aware split is the correct design.

**6. How did you choose the best model? Did you use the test set?**
No. Models were compared by 5-fold grouped cross-validation on the training portion; the test split was used once to report. The earlier
version of this project selected models using test scores; I corrected that.

**7. Which model is best and by how much?**
Look at the results table. Important point: a trivial rule, *final = G2*, is competitive. The selected regressor is not better than that rule on MAE or R², and the selected classifier is only about one percentage point more accurate. The honest message is that G2 carries nearly all the information; differences between models are within the noise of about 200 test records.

**8. The report used to say 0.958 and the JSON said 0.928. What happened?**
The training R² values in the old report were typed in by hand and did not come from any run (all six differed from the saved results; I recomputed
the JSON value from the saved model and it matched). Now result tables in the documents are generated from the JSON and a test checks this.

**9. What is the difference between training R² and test R²?**
Training R² measures fit on data the model saw; test R² measures performance on unseen students. A big gap means over-fitting.

**10. What are MAE, RMSE, R²?**
MAE: average absolute error in score points. RMSE: like MAE but penalises large errors more. R²: share of variance explained (1 is perfect, 0 equals predicting the mean).

**11. Why accuracy is not enough for the tier classifier?**
Missing a student who needs High Support is the costly error, so per-class recall and the confusion matrix matter. I report macro-F1 and per-tier recall.

**12. What is SHAP? Is the dashboard explanation real SHAP?**
SHAP gives each feature an additive contribution to one prediction (base value + contributions = model output). Yes, the dashboard uses `shap.TreeExplainer` on the same
model and input. (An earlier version used a feature-importance × z-score approximation labelled as SHAP; I replaced it.) SHAP shows what the model uses, not what causes the result.

**13. Permutation importance vs tree importance vs SHAP?**
Tree importance: how much the trees used a feature in training. Permutation importance: how much test R² drops when a feature is shuffled. SHAP: per-prediction contribution. They answer different questions and can rank features differently.

**14. Is attendance real?**
No. The data only has a *count* of absences. "Estimated attendance" assumes 93 sessions (the top of the documented absences range), so it is an estimate, labelled as such.

**15. What is odd about absences in this dataset?**
All 53 records with final grade 0 have 0 absences (likely students who did not take the exam). That distorts the absences signal: absences still ranks 3rd by SHAP (far behind G2 and G1), and part of that effect is this artifact rather than a real attendance effect.

**16. What does clustering show? Are the clusters real?**
K-Means groups students by behaviour features. Silhouette scores are low, so the structure is weak; k = 4 was chosen for interpretability. Cluster names are generated from the centroids; I do not claim validated student types.

**17. How are recommendations produced?**
By documented rules (technical report section 4.8) that use the supplied inputs and the predicted tier. Fields the user did not provide are ignored by the rules. Output is ordered by severity.

**18. What if inputs are invalid or missing?**
G1, G2 and subject are required. Values are checked for type, range and allowed categories; errors are listed. Optional fields use training medians/modes and are shown in the result. Batch uploads flag bad rows instead of dropping them.

**19. What are the main limitations?**
Small, old, regional dataset; G2 dominates; estimated attendance; the zero-grade artifact; weak clusters; no fairness audit; tiers and rule thresholds are my choices; not validated on real students.

**20. Does it predict a student's future?**
No. It estimates the final grade given past data and gives an approximate range. A prediction is not a confirmed outcome and must not be used alone for decisions about a student.

**21. Ethics: sex, age and family attributes are inputs. Is that fair?**
They are in the dataset and used by the model; changing them moved test predictions by under 0.5 points on average (largest single-student change about 3.6 points), but I have not done a formal fairness audit. In practice such attributes should be excluded or audited before any real use.

**22. How would you improve it?**
Real attendance totals, data from more institutions, repeated cross-validation with confidence intervals, fairness analysis, calibrated probabilities, and a user study with teachers.

**23. How do I reproduce your results?**
`python -m src.run_pipeline` rebuilds data, models, SHAP, figures and the report tables from the raw files with fixed seeds; `python -m pytest -q` runs the tests.
