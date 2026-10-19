"""Data loading and validation for the UCI Student Performance dataset."""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.paths import DATA_RAW

REQUIRED_COLUMNS = ["school", "sex", "age", "absences", "G1", "G2", "G3"]
# Column names that would indicate personal identifiers.
PII_LIKE_NAMES = ("name", "email", "phone", "mobile", "ssn", "aadhaar", "aadhar",
                  "passport", "roll", "student_id", "address_line", "dob", "birth")


def load_raw_data(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and stack the Mathematics and Portuguese files, adding a ``subject`` column.

    Note: 382 students appear in both files (see data/raw/student.txt); the
    stacked frame therefore has more rows than distinct students.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_RAW
    mat_path = data_dir / "student-mat.csv"
    por_path = data_dir / "student-por.csv"
    if not mat_path.exists() or not por_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. Expected student-mat.csv and student-por.csv.")
    df_mat = pd.read_csv(mat_path, sep=";")
    df_mat["subject"] = "Mathematics"
    df_por = pd.read_csv(por_path, sep=";")
    df_por["subject"] = "Portuguese"
    return pd.concat([df_mat, df_por], ignore_index=True)


def validate_data(df: pd.DataFrame) -> dict:
    """Return a validation report. Raises ValueError for an empty frame or missing columns."""
    if df is None or len(df) == 0:
        raise ValueError("Dataset is empty.")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    pii_columns = [c for c in df.columns if any(tok in str(c).lower() for tok in PII_LIKE_NAMES)]
    grades_ok = all(df[g].between(0, 20).all() for g in ("G1", "G2", "G3"))
    return {
        "total_records": int(len(df)),
        "total_features": int(len(df.columns)),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "grade_ranges_valid": bool(grades_ok),
        "pii_like_columns": pii_columns,
        "pii_detected": bool(pii_columns),
    }


if __name__ == "__main__":
    data = load_raw_data()
    report = validate_data(data)
    print(f"Records: {report['total_records']}, columns: {report['total_features']}")
    print(f"Missing: {report['total_missing']}, duplicates: {report['duplicate_rows']}")
    print(f"Grades within 0-20: {report['grade_ranges_valid']}; PII-like columns: {report['pii_like_columns']}")
