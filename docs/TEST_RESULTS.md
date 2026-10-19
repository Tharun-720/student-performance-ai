# Test Results

*Generated from a real `python -m pytest` run (JUnit output). Re-run the tests and update this file if you change the code.*

> **Status note (read first).** The 88/88 result below was recorded on 2026-10-16. Several source files
> (`train_models.py`, `sensitivity.py`, `data_preprocessing.py`, `recommender.py`, `app/main.py`) and all saved
> models were modified or regenerated on 2026-10-17, after that run. On 2026-10-17 a partial re-check (no
> pytest/xgboost/shap available) confirmed: all `.py` files compile; data preparation reproduces the shipped
> `train.csv`/`test.csv`/`student_cleaned.csv` byte for byte; `generate_reports` reproduces the shipped docs;
> the saved regressor (MAE 4.54, R2 0.795) and classifier (accuracy 0.870, macro-F1 0.871) reproduce the
> reported test metrics. The full test suite has **not** been re-run since the 2026-10-17 changes - run
> `python -m pytest -q` and record the result in the Windows section below.

## Run in the audit environment (Linux, not your Windows laptop)

- Date: 2026-10-16
- Python 3.12.3 on Linux
- Packages: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.53.0, streamlit 1.66.0, plotly 7.1.0, pytest 9.1.1
- Command: `python -m pytest -q -W default` (warnings were displayed, not suppressed)

| Tests discovered | Passed | Failed | Errors | Skipped | Warnings shown |
|---|---|---|---|---|---|
| 88 | 88 | 0 | 0 | 0 | 0 |

## By file

| File | Passed | Failed | What it covers |
|---|---|---|---|
| `test_app_charts_docs.py` | 12 | 0 | Charts, Streamlit app startup/pages/button behaviour, documentation-vs-JSON consistency |
| `test_data_and_preprocessing.py` | 24 | 0 | Data loading/validation, tier thresholds, feature engineering (controlled data), cleaning, student-aware split, paths |
| `test_models_and_metrics.py` | 12 | 0 | Metric functions (known answers), recomputing every saved metric from saved models, selection rules, schema, reproducibility |
| `test_prediction_validation_batch.py` | 24 | 0 | Input validation, prediction contract, batch behaviour |
| `test_recommender_and_explainability.py` | 16 | 0 | Recommendation rules, SHAP additivity and local explanation correctness |

## Second environment: declared lower bounds (Linux, Python 3.12)

pandas 2.2.3, numpy 2.0.2, scikit-learn 1.6.1, xgboost 2.1.4, shap 0.46.0, streamlit 1.50.0, plotly 5.20.0, matplotlib 3.8.4, pytest 8.0.2.
Artifacts were rebuilt there with `python -m src.run_pipeline`, then **88 passed, 0 failed**, with 4 warnings that come from old
matplotlib/shap versions (not project code). This run found two wrong bounds in the first draft of `requirements.txt`
(xgboost 2.0 crashes with scikit-learn 1.6; numpy 1.26 conflicts with current pyarrow); both were corrected. The selected models differed
from the latest-library run (different cross-validation winner), which is why result tables must always be regenerated, not copied.

## Also verified manually in the audit environment

- `streamlit run app/main.py --server.headless true` started; health endpoint returned `ok`; HTTP 200 (before the rewrite of the app, re-checked after: see the final report).
- `python -m src.run_pipeline` rebuilt every artifact from the raw data in a fresh virtual environment from `requirements.txt`; results were identical to the first run.

## Not verified here

- Windows, and Python 3.13/3.14 (the original author's machine used Python 3.14.5).
- Browser rendering of the dashboard (the Streamlit tests simulate the app but do not draw pixels).
- Recommendation links (not fetched).

## Your Windows run (fill in)

```
python --version        -> [paste]
python -m pytest -q     -> [paste the last 3 lines]
```

## Known limits of the test suite

- Tests of saved results re-compute metrics from the saved models; they do not re-train the models (use `python -m src.run_pipeline` for that).
- Documentation tests check that key numbers and banned claims are consistent with the JSON; they do not prove the prose is correct.
- No test can show that predictions are *right* for real students; they check correctness of the code and consistency of reported numbers.
