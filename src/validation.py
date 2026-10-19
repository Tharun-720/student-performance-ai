"""Input validation for single-student and batch inference.

Ranges and allowed values follow the UCI "Student Performance" attribute
description shipped in ``data/raw/student.txt``.
"""
import math
from typing import Any, Dict, List, Tuple

# Inclusive ranges. Whole-number fields must be integers.
NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
    "studytime": (1, 4), "failures": (0, 3), "famrel": (1, 5), "freetime": (1, 5),
    "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 93), "G1": (0, 20), "G2": (0, 20),
}
WHOLE_NUMBER_FIELDS = set(NUMERIC_RANGES) - {"G1", "G2"}

ALLOWED_CATEGORIES: Dict[str, set] = {
    "school": {"GP", "MS"}, "sex": {"F", "M"}, "address": {"U", "R"},
    "famsize": {"LE3", "GT3"}, "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": {"yes", "no"}, "famsup": {"yes", "no"}, "paid": {"yes", "no"},
    "activities": {"yes", "no"}, "nursery": {"yes", "no"}, "higher": {"yes", "no"},
    "internet": {"yes", "no"}, "romantic": {"yes", "no"},
    "subject": {"Mathematics", "Portuguese"},
}

# Without these the prediction is meaningless, so they are never defaulted.
REQUIRED_FIELDS = ["G1", "G2", "subject"]

INPUT_FIELDS = list(NUMERIC_RANGES) + list(ALLOWED_CATEGORIES)


class ValidationError(ValueError):
    """Raised when an input record is invalid. ``errors`` lists every problem found."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def validate_student_record(record: Dict[str, Any], defaults: Dict[str, Any]
                            ) -> Tuple[Dict[str, Any], List[str]]:
    """Validate one student record.

    Returns ``(clean_record, defaulted_fields)``. Optional fields that are
    missing are filled from ``defaults`` (training-set medians/modes) and
    reported in ``defaulted_fields``. Raises ``ValidationError`` listing every
    problem if a required field is missing or any value is invalid.
    """
    errors: List[str] = []
    clean: Dict[str, Any] = {}
    defaulted: List[str] = []

    for field in INPUT_FIELDS:
        value = record.get(field)
        if _is_missing(value):
            if field in REQUIRED_FIELDS:
                errors.append(f"'{field}' is required")
            elif field in defaults:
                clean[field] = defaults[field]
                defaulted.append(field)
            else:
                errors.append(f"'{field}' is missing and no default is available")
            continue

        if field in NUMERIC_RANGES:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a number (got {value!r})")
                continue
            if math.isnan(number) or math.isinf(number):
                errors.append(f"'{field}' must be a finite number")
                continue
            low, high = NUMERIC_RANGES[field]
            if not low <= number <= high:
                errors.append(f"'{field}' must be between {low} and {high} (got {number:g})")
                continue
            if field in WHOLE_NUMBER_FIELDS:
                if number != int(number):
                    errors.append(f"'{field}' must be a whole number (got {number:g})")
                    continue
                number = int(number)
            clean[field] = number
        else:
            text = str(value).strip()
            if text not in ALLOWED_CATEGORIES[field]:
                allowed = ", ".join(sorted(ALLOWED_CATEGORIES[field]))
                errors.append(f"'{field}' must be one of [{allowed}] (got {text!r})")
                continue
            clean[field] = text

    if errors:
        raise ValidationError(errors)
    return clean, defaulted
