"""Inference: validated single and batch prediction with explanations and recommendations."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import apply_cleaning, engineer_features
from src.explainability import explain_rows, make_tree_explainer, top_contributions
from src.metrics import support_level_from_score_pct
from src.paths import MODELS_DIR
from src.io_utils import read_json
from src.recommender import generate_recommendations
from src.validation import (INPUT_FIELDS, REQUIRED_FIELDS, ValidationError, validate_student_record)


class StudentPredictor:
    """Loads the saved artifacts once and serves predictions.

    Predictions are statistical estimates, not facts about a student's future.
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        d = self.models_dir
        for name in ("preprocessor.joblib", "best_regressor.joblib", "best_classifier.joblib",
                     "kmeans_model.joblib", "cluster_scaler.joblib", "feature_metadata.json",
                     "evaluation_summary.json"):
            if not (d / name).exists():
                raise FileNotFoundError(
                    f"Missing model artifact '{name}' in {d}. Run `python -m src.run_pipeline` to create it.")
        self.preprocessor = joblib.load(d / "preprocessor.joblib")
        self.reg_model = joblib.load(d / "best_regressor.joblib")
        self.clf_model = joblib.load(d / "best_classifier.joblib")
        self.kmeans = joblib.load(d / "kmeans_model.joblib")
        self.cluster_scaler = joblib.load(d / "cluster_scaler.joblib")
        self.meta = read_json(d / "feature_metadata.json")
        self.summary = read_json(d / "evaluation_summary.json")
        self.numeric_features = self.meta["numeric_features"]
        self.categorical_features = self.meta["categorical_features"]
        self.feature_columns = self.numeric_features + self.categorical_features
        self.feature_names = self.meta["transformed_feature_names"]
        self.classes = self.meta["classes"]
        self.defaults = self.meta["defaults"]
        self.cleaning_params = {"absences_cap": self.meta["absences_cap"], "medians": {}, "modes": {}}
        self.interval = self.summary["prediction_interval"]
        self.cluster_names = self.summary["clustering"]["archetype_names"]
        self.cluster_features = self.summary["clustering"]["features"]
        self.explainer = make_tree_explainer(self.reg_model)
        self.version_warnings = self._check_versions()

    def _check_versions(self) -> List[str]:
        path = self.models_dir / "model_manifest.json"
        if not path.exists():
            return ["model_manifest.json not found; library versions of the saved models are unknown."]
        import sklearn
        import xgboost
        manifest = read_json(path)
        warnings = []
        for lib, now in (("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__)):
            if manifest.get(lib) != now:
                warnings.append(f"Models were saved with {lib} {manifest.get(lib)} but {now} is installed. "
                                f"Run `python -m src.run_pipeline` to rebuild them for this environment.")
        return warnings

    # ------------------------------------------------------------------ helpers
    def _frame(self, clean: Dict[str, Any]) -> pd.DataFrame:
        df = engineer_features(pd.DataFrame([clean]))
        df = apply_cleaning(df, self.cleaning_params)
        return df[self.feature_columns]

    def _cluster(self, df: pd.DataFrame) -> int:
        return int(self.kmeans.predict(self.cluster_scaler.transform(df[self.cluster_features]))[0])

    def _decode(self, raw_pred) -> str:
        v = raw_pred
        return self.classes[int(v)] if not isinstance(v, str) else v

    # ------------------------------------------------------------------ single
    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and predict one student. Raises ValidationError for bad input."""
        clean, defaulted = validate_student_record(record, self.defaults)
        full = pd.concat([engineer_features(pd.DataFrame([clean]))], axis=1)
        full = apply_cleaning(full, self.cleaning_params)
        X = self.preprocessor.transform(full[self.feature_columns])

        score = float(np.clip(self.reg_model.predict(X)[0], 0.0, 100.0))
        proba = self.clf_model.predict_proba(X)[0]
        clf_tier = self._decode(self.clf_model.predict(X)[0])
        score_tier = support_level_from_score_pct(score)
        lo = float(np.clip(score + self.interval["q025"], 0, 100))
        hi = float(np.clip(score + self.interval["q975"], 0, 100))

        explanation = {"available": False, "base_value": None, "increasing": [], "decreasing": [],
                       "method": "SHAP TreeExplainer"}
        if self.explainer is not None:
            vals, base = explain_rows(self.explainer, X, self.feature_names)
            ups, downs = top_contributions(vals[0], self.feature_names, n=5)
            explanation.update({"available": True, "base_value": round(base, 2),
                                "increasing": ups, "decreasing": downs,
                                "all_values": {n: float(v) for n, v in zip(self.feature_names, vals[0])}})

        cluster = self._cluster(full)
        recs = generate_recommendations(clean, score, clf_tier, defaulted)
        att = float(full["estimated_attendance_pct"].iloc[0])
        return {
            "predicted_score_pct": round(score, 1),
            "predicted_grade_20": round(score / 5.0, 1),
            "prediction_range": {"low": round(lo, 1), "high": round(hi, 1),
                                 "label": "approximate 95% prediction range (from cross-validated residuals)"},
            "support_level": clf_tier,
            "support_probabilities": {c: round(float(p), 3) for c, p in zip(self.classes, proba)},
            "score_implied_tier": score_tier,
            "tiers_agree": clf_tier == score_tier,
            "cluster": cluster, "cluster_name": self.cluster_names.get(str(cluster), f"Cluster {cluster}"),
            "explanation": explanation, "recommendations": recs,
            "defaulted_fields": defaulted, "inputs_complete": not defaulted,
            "student_summary": {"absences": clean["absences"], "estimated_attendance_pct": round(att, 1),
                                "G1": clean["G1"], "G2": clean["G2"],
                                "assessment_trend": clean["G2"] - clean["G1"]},
            "disclaimer": ("Statistical estimate based on past data. It is not a confirmed outcome and should "
                           "support, not replace, a teacher's judgement."),
        }

    # ------------------------------------------------------------------ batch
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict every row. Rows are never dropped or reordered.

        Invalid rows get ``Validation_Status`` = the error text and empty predictions.
        Raises ValidationError if required columns are missing or the frame is empty.
        """
        if df is None or len(df) == 0:
            raise ValidationError(["The uploaded file contains no rows."])
        missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
        if missing:
            raise ValidationError([f"Required column(s) missing: {', '.join(missing)}"])

        rows = []
        for i, record in enumerate(df.to_dict(orient="records")):
            base = {"Row": i + 1}
            try:
                r = self.predict_single(record)
                base.update({
                    "Validation_Status": "OK" if r["inputs_complete"] else
                    "OK (defaults used for: " + ", ".join(r["defaulted_fields"]) + ")",
                    "Predicted_Score_Pct": r["predicted_score_pct"],
                    "Range_Low": r["prediction_range"]["low"], "Range_High": r["prediction_range"]["high"],
                    "Support_Level": r["support_level"], "Score_Implied_Tier": r["score_implied_tier"],
                    "Cluster": r["cluster_name"]})
            except ValidationError as exc:
                base.update({"Validation_Status": "INVALID: " + "; ".join(exc.errors),
                             "Predicted_Score_Pct": np.nan, "Range_Low": np.nan, "Range_High": np.nan,
                             "Support_Level": None, "Score_Implied_Tier": None, "Cluster": None})
            rows.append(base)
        return pd.concat([pd.DataFrame(rows).set_index(df.index)], axis=1)
