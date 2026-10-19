import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data, validate_data
from src.data_preprocessing import (ASSUMED_TOTAL_SESSIONS, apply_cleaning, engineer_features,
                                    fit_cleaning_params, get_feature_lists, group_split, student_fingerprint)
from src.metrics import support_level_from_grade, support_level_from_score_pct
from src.paths import PROJECT_ROOT


# ---------------- data loading / validation
def test_raw_data_loads_expected_shape():
    df = load_raw_data()
    assert len(df) == 1044 and set(df["subject"]) == {"Mathematics", "Portuguese"}
    assert (df["subject"] == "Mathematics").sum() == 395


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path)


def test_validate_empty_dataset_raises():
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame())


def test_validate_missing_columns_raises(small_raw):
    with pytest.raises(ValueError):
        validate_data(small_raw.drop(columns=["G1"]))


def test_validate_detects_duplicates_missing_and_bad_grades(small_raw):
    bad = pd.concat([small_raw, small_raw.iloc[[0]]], ignore_index=True)
    bad.loc[1, "absences"] = np.nan
    bad.loc[2, "G2"] = 25
    rep = validate_data(bad)
    assert rep["duplicate_rows"] == 1 and rep["total_missing"] == 1 and rep["grade_ranges_valid"] is False


def test_validate_pii_detection_is_real(small_raw):
    assert validate_data(small_raw)["pii_detected"] is False
    assert validate_data(small_raw.assign(student_name="x"))["pii_detected"] is True


# ---------------- tier thresholds
@pytest.mark.parametrize("grade,tier", [(0, "High Support"), (9, "High Support"), (9.99, "High Support"),
                                        (10, "Medium Support"), (13, "Medium Support"), (13.99, "Medium Support"),
                                        (14, "Low Support"), (20, "Low Support")])
def test_support_thresholds(grade, tier):
    assert support_level_from_grade(grade) == tier
    assert support_level_from_score_pct(grade * 5) == tier


# ---------------- feature engineering with known values
def test_engineered_values(small_raw):
    out = engineer_features(small_raw)
    assert out["final_score_pct"].tolist() == [45.0, 50.0, 70.0, 65.0]
    assert out["support_level"].tolist() == ["High Support", "Medium Support", "Low Support", "Medium Support"]
    assert out["assessment_trend"].tolist() == [0, 1, -1, -1]
    assert out.loc[0, "estimated_attendance_pct"] == 100.0
    assert out.loc[1, "estimated_attendance_pct"] == 0.0          # 93 absences with assumed 93 sessions
    assert out.loc[3, "estimated_attendance_pct"] == 0.0          # clipped, never negative
    assert out.loc[2, "estimated_attendance_pct"] == pytest.approx(100 - 9 / ASSUMED_TOTAL_SESSIONS * 100)
    assert out["parent_edu_avg"].tolist() == [3.0, 2.0, 2.0, 2.0]
    assert out["past_failure_flag"].tolist() == [0, 0, 1, 1]
    assert out["engagement_consistency"].between(0, 100).all()


def test_engineer_without_g3_has_no_targets(small_raw):
    out = engineer_features(small_raw.drop(columns=["G3"]))
    assert "final_score_pct" not in out and "support_level" not in out


def test_no_division_by_zero_in_ratio(small_raw):
    small_raw["goout"], small_raw["freetime"] = 0, 0
    assert np.isfinite(engineer_features(small_raw)["study_to_social_ratio"]).all()


def test_g3_is_never_a_model_input():
    num, cat = get_feature_lists()
    assert "G3" not in num + cat and "final_score_pct" not in num + cat and "support_level" not in num + cat


# ---------------- cleaning
def test_cleaning_uses_train_statistics_and_never_drops_rows(small_raw):
    train = engineer_features(small_raw.iloc[:3])
    params = fit_cleaning_params(train)
    new = engineer_features(pd.concat([small_raw.iloc[[0]]] * 3, ignore_index=True))   # 3 identical rows
    new.loc[0, "absences"] = 500
    cleaned = apply_cleaning(new, params)
    assert len(cleaned) == 3                                   # duplicates are NOT dropped at inference
    assert cleaned.loc[0, "absences"] == pytest.approx(params["absences_cap"])


def test_cleaning_fills_missing_from_params(small_raw):
    params = fit_cleaning_params(engineer_features(small_raw))
    df = engineer_features(small_raw)
    df.loc[0, "freetime"] = np.nan
    assert apply_cleaning(df, params)["freetime"].notna().all()


# ---------------- split
def test_split_has_no_student_in_both_sides(train_df, test_df, meta):
    assert not set(train_df["student_group"]) & set(test_df["student_group"])
    assert meta["split"]["groups_in_both_splits"] == 0
    assert len(train_df) + len(test_df) == 1044


def test_split_is_reproducible():
    df = engineer_features(load_raw_data())
    g = student_fingerprint(df)
    a = group_split(df, g)
    b = group_split(df, g)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_processed_files_match_metadata_schema(train_df, test_df, meta):
    cols = meta["numeric_features"] + meta["categorical_features"]
    for frame in (train_df, test_df):
        assert set(cols).issubset(frame.columns)
        assert frame[cols].isnull().sum().sum() == 0


# ---------------- paths
def test_modules_work_from_another_working_directory(tmp_path):
    code = ("from src.predict import StudentPredictor; "
            "p=StudentPredictor(); r=p.predict_single({'G1':12,'G2':12,'subject':'Mathematics'}); print(r['support_level'])")
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
