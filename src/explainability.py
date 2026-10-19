"""Global and local explanations using genuine SHAP values (TreeExplainer).

Three different things are reported and kept apart:
  * tree impurity importance  (``feature_importances_``): how much a model used a feature in training
  * permutation importance     : drop in test R2 when a feature is shuffled
  * SHAP values                : additive per-prediction contributions, in points of predicted score (0-100)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import support_level_from_score_pct
from src.paths import DATA_PROCESSED, MODELS_DIR
from src.io_utils import read_json

ADDITIVITY_TOLERANCE = 1.0  # points of predicted score; reported, not hidden


def make_tree_explainer(model):
    """Return a SHAP TreeExplainer, or None if the model is not a supported tree ensemble."""
    try:
        return shap.TreeExplainer(model)
    except Exception:  # model type not supported by TreeExplainer
        return None


def explain_rows(explainer, X, feature_names):
    """SHAP values for transformed rows. Returns (values, base_value)."""
    values = np.asarray(explainer.shap_values(X, check_additivity=False))
    base = float(np.ravel(explainer.expected_value)[0])
    return values, base


def additivity_gap(explainer, model, X, feature_names):
    """Max |base + sum(SHAP) - prediction| over the rows (points)."""
    values, base = explain_rows(explainer, X, feature_names)
    return float(np.max(np.abs(base + values.sum(axis=1) - model.predict(X))))


def top_contributions(values_row, feature_names, n=5):
    """Split a SHAP row into features that raise vs lower this prediction."""
    order = np.argsort(-np.abs(values_row))
    ups = [{"feature": feature_names[i], "shap": round(float(values_row[i]), 3)}
           for i in order if values_row[i] > 0][:n]
    downs = [{"feature": feature_names[i], "shap": round(float(values_row[i]), 3)}
             for i in order if values_row[i] < 0][:n]
    return ups, downs


def main():
    meta = read_json(MODELS_DIR / "feature_metadata.json")
    summary = read_json(MODELS_DIR / "evaluation_summary.json")
    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "best_regressor.joblib")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    cols = meta["numeric_features"] + meta["categorical_features"]
    names = meta["transformed_feature_names"]
    X = pre.transform(test[cols])
    y = test["final_score_pct"].values
    best_name = summary["best_regressor"]

    out = {"model": best_name, "n_rows_explained": int(len(test)),
           "method_notes": {
               "tree_importance": "feature_importances_ of the fitted model (training-time impurity reduction)",
               "permutation_importance": "mean drop in TEST R2 over 10 shuffles",
               "shap": "shap.TreeExplainer (tree_path_dependent); values in points of predicted score (0-100)"}}
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
        out["global_tree_importance"] = {k: round(float(v), 4) for k, v in imp.head(15).items()}
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="r2")
    pser = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False)
    out["global_permutation_importance"] = {k: round(float(v), 4) for k, v in pser.head(15).items()}
    out["permutation_baseline_r2"] = round(float(model.score(X, y)), 4)

    explainer = make_tree_explainer(model)
    if explainer is None:
        out["shap_available"] = False
        out["case_studies"] = []
    else:
        vals, base = explain_rows(explainer, X, names)
        gap = float(np.max(np.abs(base + vals.sum(axis=1) - model.predict(X))))
        out.update({"shap_available": True, "base_value": round(base, 3),
                    "base_value_definition": "mean model prediction over the training data (TreeExplainer expected_value)",
                    "additivity_max_gap_points": round(gap, 4),
                    "additivity_within_tolerance": bool(gap <= ADDITIVITY_TOLERANCE)})
        mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=names).sort_values(ascending=False)
        out["global_shap_importance"] = [{"feature": k, "mean_abs_shap": round(float(v), 3)}
                                         for k, v in mean_abs.head(15).items()]
        # Case studies: real test rows chosen by actual outcome tier (rule stated in 'selection').
        preds = model.predict(X)
        tiers = np.array([support_level_from_score_pct(v) for v in y])
        picks = []
        for title, tier, how in [("Lowest actual score among High Support", "High Support", "min"),
                                 ("Typical Medium Support student", "Medium Support", "median"),
                                 ("Highest actual score among Low Support", "Low Support", "max")]:
            idx_all = np.where(tiers == tier)[0]
            order = idx_all[np.argsort(y[idx_all])]
            idx = order[0] if how == "min" else order[-1] if how == "max" else order[len(order) // 2]
            picks.append((title, int(idx), how))
        cases = []
        for title, i, how in picks:
            row = test.iloc[i]
            ups, downs = top_contributions(vals[i], names)
            cases.append({
                "title": title, "selection": f"{how} actual score within tier",
                "test_row_index": i, "subject": row["subject"], "G1": float(row["G1"]), "G2": float(row["G2"]),
                "absences": float(row["absences"]), "failures": int(row["failures"]),
                "studytime": int(row["studytime"]),
                "actual_score_pct": round(float(y[i]), 1), "predicted_score_pct": round(float(preds[i]), 1),
                "actual_tier": str(tiers[i]), "base_value": round(base, 3),
                "increasing_features": ups, "decreasing_features": downs,
                "note": ("Actual final grade is 0; the dataset cannot tell whether the student failed the exam or "
                         "did not sit it.") if y[i] == 0 else "",
            })
        out["case_studies"] = cases

    with open(MODELS_DIR / "explainability_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Explained", best_name, "on", len(test), "rows | SHAP:", out["shap_available"],
          "| additivity max gap:", out.get("additivity_max_gap_points"))
    print("Top SHAP:", [d["feature"] for d in out.get("global_shap_importance", [])[:5]])


if __name__ == "__main__":
    main()
