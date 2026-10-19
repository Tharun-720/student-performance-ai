"""Feature engineering, student-aware train/test split and preprocessing pipeline."""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):  # allow `python src/data_preprocessing.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_data, validate_data
from src.metrics import support_level_from_grade
from src.paths import DATA_PROCESSED, DATA_RAW, MODELS_DIR

RANDOM_STATE = 42
# UCI describes `absences` as "numeric: from 0 to 93". That is the range of the
# column, NOT a count of sessions. Using 93 as a denominator is an ASSUMPTION, so
# the derived value is only an *estimate* (see docs/DATA_DICTIONARY.md).
ASSUMED_TOTAL_SESSIONS = 93

# The dataset has no student id. The dataset authors' own merge script (data/raw/student-merge.R)
# identifies students enrolled in both courses by these 13 attributes (382 matches), so the same
# key is used here to keep a student's two records on the same side of the split.
# This is still a heuristic, not ground truth.
STUDENT_FINGERPRINT_COLS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "nursery", "internet",
]

RAW_NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
               "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
ENGINEERED_NUMERIC = ["assessment_trend", "estimated_attendance_pct", "study_to_social_ratio",
                      "parent_edu_avg", "past_failure_flag", "engagement_consistency"]
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
               "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
               "romantic", "subject"]


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Numeric and categorical model inputs. G3 is never an input (target leakage)."""
    return RAW_NUMERIC + ENGINEERED_NUMERIC, list(CATEGORICAL)


def student_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Integer group id per student fingerprint (same id => probably the same student)."""
    key = df[STUDENT_FINGERPRINT_COLS].astype(str).agg("|".join, axis=1)
    return pd.Series(pd.factorize(key)[0], index=df.index, name="student_group")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features row by row (no statistics are learned from the data here).

    Targets (``final_score_pct`` and ``support_level``) are only added when G3 is present.
    """
    df = df.copy()
    if "G3" in df.columns:
        df["final_score_pct"] = df["G3"] * 5.0
        df["support_level"] = df["G3"].apply(support_level_from_grade)
    df["assessment_trend"] = df["G2"] - df["G1"]
    df["estimated_attendance_pct"] = np.clip(
        100.0 - (df["absences"] / ASSUMED_TOTAL_SESSIONS * 100.0), 0.0, 100.0)
    df["study_to_social_ratio"] = df["studytime"] / (df["goout"] + df["freetime"] + 0.1)
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2.0
    df["past_failure_flag"] = (df["failures"] > 0).astype(int)
    norm_study = (df["studytime"] / 4.0) * 40.0
    norm_att = (df["estimated_attendance_pct"] / 100.0) * 40.0
    fail_score = np.clip(1.0 - (df["failures"] / 3.0), 0.0, 1.0) * 20.0
    df["engagement_consistency"] = norm_study + norm_att + fail_score
    return df


def fit_cleaning_params(train_df: pd.DataFrame) -> Dict:
    """Learn cleaning statistics from the TRAINING data only."""
    numeric_cols = [c for c in train_df.columns if is_numeric_dtype(train_df[c])]
    other_cols = [c for c in train_df.columns if c not in numeric_cols]
    medians = {c: float(train_df[c].median()) for c in numeric_cols}
    modes = {c: train_df[c].mode().iloc[0] for c in other_cols if train_df[c].notna().any()}
    return {
        "absences_cap": float(train_df["absences"].quantile(0.99)),
        "medians": medians,
        "modes": {k: str(v) for k, v in modes.items()},
    }


def apply_cleaning(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Apply learned cleaning parameters. Never drops rows (safe for inference)."""
    df = df.copy()
    for col, median in params["medians"].items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(median)
    for col, mode in params["modes"].items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(mode)
    df["absences"] = np.clip(df["absences"], 0, params["absences_cap"])
    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Standardise numeric columns, one-hot encode categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), list(numeric_features)),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             list(categorical_features)),
        ],
        remainder="drop",
    )


def compute_defaults(train_df: pd.DataFrame) -> Dict:
    """Training-set median (numeric) / mode (categorical) used for OPTIONAL missing inputs."""
    defaults = {}
    for col in RAW_NUMERIC:
        if col in ("G1", "G2"):
            continue  # never defaulted
        defaults[col] = int(round(float(train_df[col].median())))
    for col in CATEGORICAL:
        if col == "subject":
            continue  # never defaulted
        defaults[col] = str(train_df[col].mode().iloc[0])
    return defaults


def group_split(df: pd.DataFrame, groups: pd.Series, n_splits: int = 5,
                random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """~20% test split that keeps each student fingerprint on one side, stratified by tier."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["support_level"], groups))
    return train_idx, test_idx


def prepare_and_save_data(raw_data_dir=DATA_RAW, processed_dir=DATA_PROCESSED, models_dir=MODELS_DIR):
    """Load, engineer, split (student-aware), clean with train-only statistics, save artifacts."""
    processed_dir, models_dir = Path(processed_dir), Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(raw_data_dir)
    val_report = validate_data(df_raw)
    n_duplicates = val_report["duplicate_rows"]
    df = df_raw.drop_duplicates().reset_index(drop=True)
    df = engineer_features(df)
    df["student_group"] = student_fingerprint(df)

    train_idx, test_idx = group_split(df, df["student_group"])
    train_raw, test_raw = df.iloc[train_idx], df.iloc[test_idx]

    params = fit_cleaning_params(train_raw)           # train only
    train_df = apply_cleaning(train_raw, params)
    test_df = apply_cleaning(test_raw, params)
    cleaned = apply_cleaning(df, params)
    cleaned.to_csv(processed_dir / "student_cleaned.csv", index=False)

    num_cols, cat_cols = get_feature_lists()
    feature_cols = num_cols + cat_cols
    preprocessor = build_preprocessor(num_cols, cat_cols).fit(train_df[feature_cols])  # train only
    ohe_cols = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    transformed_names = list(num_cols) + list(ohe_cols)

    keep = feature_cols + ["student_group", "final_score_pct", "support_level"]
    train_df[keep].to_csv(processed_dir / "train.csv", index=False)
    test_df[keep].to_csv(processed_dir / "test.csv", index=False)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    overlap = set(train_df["student_group"]) & set(test_df["student_group"])
    metadata = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "transformed_feature_names": transformed_names,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "target_regression": "final_score_pct",
        "target_classification": "support_level",
        "classes": sorted(df["support_level"].unique().tolist()),
        "validation_report": val_report,
        "duplicates_removed": int(n_duplicates),
        "distinct_student_fingerprints": int(df["student_group"].nunique()),
        "split": {
            "method": "StratifiedGroupKFold(n_splits=5, shuffle=True) first fold; groups = student fingerprint",
            "random_state": RANDOM_STATE,
            "groups_in_both_splits": len(overlap),
        },
        "absences_cap": params["absences_cap"],
        "assumed_total_sessions_for_attendance_estimate": ASSUMED_TOTAL_SESSIONS,
        "defaults": compute_defaults(train_df),
    }
    with open(models_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Rows: {len(df)} ({metadata['distinct_student_fingerprints']} distinct student fingerprints)")
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | fingerprints in both splits: {len(overlap)}")
    print(f"Transformed features: {len(transformed_names)}")
    return train_df, test_df, metadata


if __name__ == "__main__":
    prepare_and_save_data()
