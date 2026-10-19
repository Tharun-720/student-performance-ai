# Dashboard Screenshots to Capture (manually)

I cannot take screenshots for you. Run `streamlit run app/main.py`, then capture these (save in `docs/screenshots/`, which does not exist yet):

1. **Dashboard**: KPI row and the "Limitations and responsible use" expander open.
2. **EDA**: all four charts with both filters selected; then one screenshot with a single subject selected (shows filters work).
3. **EDA, empty filters**: the "No records match" warning.
4. **Predictor**: the form with the "Needs support" quick-fill applied, before pressing the button (shows no prediction yet).
5. **Predictor result**: score, tier, approximate range, probability chart (after pressing **Run prediction**).
6. **Predictor, invalid case**: not possible with sliders; instead screenshot the batch page with an invalid row flagged.
7. **Explainability, Global**: SHAP bar chart and the three-measure table.
8. **Explainability, Local**: the SHAP bar chart for the last prediction with the raise/lower lists.
9. **Explainability, Case studies**: one expanded case.
10. **Recommendations**: priority actions expanded, weekly template, notes.
11. **Model Benchmark**: both tables, confusion matrix and residual plot.
12. **Clusters**: radar chart and the profile table with the weak-structure note.
13. **Batch scoring**: after uploading `sample_students.csv` and pressing **Run batch prediction**.
14. **Terminal**: `python -m pytest -q` showing the pass count on *your* machine.
15. **Terminal**: `streamlit run app/main.py` start-up lines.
