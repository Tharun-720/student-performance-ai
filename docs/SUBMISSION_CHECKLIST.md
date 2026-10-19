# Final Submission Checklist

## Before you submit
- [ ] Replace every placeholder: `[Student name]`, `[Roll number]`, `[College name]`, `[Department]`, `[Guide name]` (README, technical report, proposal, dashboard sidebar in `app/main.py`).
- [ ] Decide about the "Trekverse Edutech - Project Manager" line: keep only if it is true and permitted.
- [ ] Create a fresh virtual environment on your laptop and run the commands in the README; fix anything that differs.
- [ ] Run `python -m src.run_pipeline` once on your machine (rebuilds models for your library versions), then `python -m pytest -q`.
- [ ] Paste your own test output into `docs/TEST_RESULTS.md` (section "Your Windows run").
- [ ] Capture the screenshots in `docs/SCREENSHOT_CHECKLIST.md`.
- [ ] Open the links in the Recommendations page and remove any that do not work.
- [ ] Re-read `docs/FINAL_TECHNICAL_REPORT.md` section 7 (limitations) and be ready to explain each point.

## Report (PDF/Word, if your college requires it)
- [ ] Export from the markdown reports yourself; I have not produced a PDF or Word file.
- [ ] Include: abstract, dataset, method, results (copy tables from `MODEL_EVALUATION_REPORT.md`), SHAP figures, limitations, references.
- [ ] Include figures from `docs/images/` and your own screenshots.

## Source code package
- [ ] Zip the project **without** `.venv/`, `__pycache__/`, `.pytest_cache/`.
- [ ] Keep `models/`, `data/`, `docs/`, `tests/`, `app/`, `src/`, `requirements.txt`, `pytest.ini`, `README.md`.
- [ ] No passwords, keys or personal data inside the files.

## Presentation
- [ ] Slides: problem, dataset, pipeline, student-aware split, results with the *final = G2* comparison, SHAP, recommendations, limitations, future work.
- [ ] Say clearly that predictions are estimates and not decisions about students.

## Demonstration (see the README / final answer for the step-by-step script)
- [ ] Laptop charged; app started once beforehand; no internet needed.
- [ ] A backup of screenshots in case the demo fails.
