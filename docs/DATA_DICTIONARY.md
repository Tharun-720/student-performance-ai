# Data Dictionary

Source: UCI *Student Performance* dataset (`data/raw/student.txt` has the original attribute list). Two files, Maths and
Portuguese, 33 original columns each; this project adds `subject`. **1,044 records; about 662 distinct students**
(382 students are in both files). No missing values and no exact duplicates in the raw files.

## Original attributes

| Column | Type / values | Meaning |
|---|---|---|
| school | GP, MS | Student's school (Gabriel Pereira / Mousinho da Silveira) |
| sex | F, M | Sex |
| age | 15-22 | Age in years |
| address | U, R | Urban / rural home |
| famsize | LE3, GT3 | Family size ≤ 3 / > 3 |
| Pstatus | T, A | Parents living together / apart |
| Medu, Fedu | 0-4 | Mother's / father's education (0 none, 1 primary, 2 5th-9th grade, 3 secondary, 4 higher) |
| Mjob, Fjob | teacher, health, services, at_home, other | Parent's job |
| reason | home, reputation, course, other | Reason for choosing the school |
| guardian | mother, father, other | Guardian |
| traveltime | 1-4 | Home-to-school time (1 < 15 min ... 4 > 1 h) |
| studytime | 1-4 | Weekly study time (1 < 2 h, 2: 2-5 h, 3: 5-10 h, 4 > 10 h) |
| failures | 0-3 | Number of past class failures (UCI describes the coding as "n if 1 ≤ n < 3, else 4"; observed values are 0-3, interpret 3 as "3 or more" with caution) |
| schoolsup, famsup, paid, activities, nursery, higher, internet, romantic | yes, no | School/family support, extra paid classes in the subject, extra-curriculars, attended nursery, wants higher education, internet at home, in a relationship |
| famrel, freetime, goout, Dalc, Walc, health | 1-5 | Family relationship quality, free time, going out, weekday / weekend alcohol use, health (self-rated; 1 = very low, 5 = very high) |
| **absences** | integer, 0-75 observed (UCI: "from 0 to 93") | **Count of school absences.** This is a count, not an attendance percentage. |
| G1, G2 | 0-20 | First / second period grade |
| G3 | 0-20 | Final grade (**target; never a model input**) |

## Added or derived columns

| Column | Definition | Notes |
|---|---|---|
| subject | Mathematics / Portuguese | Which file the record came from |
| final_score_pct | G3 × 5 | Regression target (0-100) |
| support_level | High Support (G3 < 10), Medium Support (10 ≤ G3 < 14), Low Support (G3 ≥ 14) | Classification target |
| assessment_trend | G2 − G1 | Grade momentum |
| **estimated_attendance_pct** | clip(100 − absences / 93 × 100, 0, 100) | **ESTIMATE.** 93 is the upper end of the documented range of `absences`, used as an *assumed* number of sessions. It is not recorded attendance. |
| study_to_social_ratio | studytime / (goout + freetime + 0.1) | Study vs leisure balance |
| parent_edu_avg | (Medu + Fedu) / 2 | |
| past_failure_flag | 1 if failures > 0 | |
| engagement_consistency | 40 × studytime/4 + 40 × estimated_attendance/100 + 20 × (1 − failures/3) | Project-defined index (0-100), not a validated measure |
| student_group | integer id | Same-student fingerprint from the 13 matching attributes used by the dataset authors' merge script, used only for splitting; a heuristic |

`absences` is additionally capped at the training-data 99th percentile (stored in `models/feature_metadata.json`) before modelling.

## Data quality notes
- All 53 records with G3 = 0 have `absences` = 0 (see `docs/MODEL_EVALUATION_REPORT.md`).
- The dataset has no student identifier; quasi-identifiers (school, age, sex, family attributes) are present, so treat it as sensitive if combined with other data.
- Generated files: `data/processed/student_cleaned.csv`, `train.csv`, `test.csv` (re-created by `python -m src.run_pipeline`).
