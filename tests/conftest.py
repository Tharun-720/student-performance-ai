import json
import warnings

import pandas as pd
import pytest

from src.paths import DATA_PROCESSED, MODELS_DIR
from src.io_utils import read_json


@pytest.fixture(scope="session")
def predictor():
    from src.predict import StudentPredictor
    return StudentPredictor()


@pytest.fixture(scope="session")
def summary():
    return read_json(MODELS_DIR / "evaluation_summary.json")


@pytest.fixture(scope="session")
def meta():
    return read_json(MODELS_DIR / "feature_metadata.json")


@pytest.fixture(scope="session")
def train_df():
    return pd.read_csv(DATA_PROCESSED / "train.csv")


@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(DATA_PROCESSED / "test.csv")


@pytest.fixture
def small_raw():
    """Tiny controlled raw-style frame with known values."""
    return pd.DataFrame({
        "school": ["GP", "GP", "MS", "MS"], "sex": ["F", "M", "F", "M"], "age": [16, 17, 18, 16],
        "address": ["U", "R", "U", "R"], "famsize": ["GT3", "LE3", "GT3", "LE3"], "Pstatus": ["T"] * 4,
        "Medu": [4, 2, 0, 3], "Fedu": [2, 2, 4, 1], "Mjob": ["other"] * 4, "Fjob": ["other"] * 4,
        "reason": ["course"] * 4, "guardian": ["mother"] * 4, "traveltime": [1, 2, 1, 1],
        "studytime": [2, 4, 1, 3], "failures": [0, 0, 3, 1], "schoolsup": ["no"] * 4, "famsup": ["yes"] * 4,
        "paid": ["no"] * 4, "activities": ["no"] * 4, "nursery": ["yes"] * 4, "higher": ["yes"] * 4,
        "internet": ["yes"] * 4, "romantic": ["no"] * 4, "famrel": [4] * 4, "freetime": [3, 2, 4, 3],
        "goout": [3, 1, 5, 2], "Dalc": [1] * 4, "Walc": [1] * 4, "health": [4] * 4,
        "absences": [0, 93, 9, 186], "G1": [10, 18, 5, 14], "G2": [10, 19, 4, 13], "G3": [9, 10, 14, 13],
        "subject": ["Mathematics", "Portuguese", "Mathematics", "Portuguese"],
    })
