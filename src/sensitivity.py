"""Sensitivity analyses that back the limitations stated in the documentation.

Outputs models/sensitivity_analysis.json. Uses fixed hyper-parameters (the
default Random Forest regressor and XGBoost classifier) for speed, so numbers
are comparable across scenarios but not identical to the tuned final models.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import RANDOM_STATE, build_preprocessor, get_feature_lists, group_split
from src.metrics import classification_metrics, regression_metrics
from src.paths import DATA_PROCESSED, MODELS_DIR
from src.io_utils import read_json


def fit_eval(df, tr, te, drop=(), classes=None):
    num, cat = get_feature_lists()
    num = [c for c in num if c not in drop]
    pre = build_preprocessor(num, cat)
    Xtr = pre.fit_transform(df.iloc[tr][num + cat])
    Xte = pre.transform(df.iloc[te][num + cat])
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=4,
                               random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, df.iloc[tr]["final_score_pct"])
    reg = regression_metrics(df.iloc[te]["final_score_pct"], rf.predict(Xte))
    idx = {c: i for i, c in enumerate(classes)}
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE,
                        eval_metric="mlogloss", n_jobs=1).fit(Xtr, df.iloc[tr]["support_level"].map(idx))
    pred = [classes[int(i)] for i in xgb.predict(Xte)]
    cm = classification_metrics(df.iloc[te]["support_level"], pred, classes)
    return {"R2": reg["R2"], "MAE": reg["MAE"], "F1_Macro": cm["F1_Macro"], "Accuracy": cm["Accuracy"]}


def summarise(runs):
    keys = runs[0].keys()
    return {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                "sd": round(float(np.std([r[k] for r in runs], ddof=1)), 4),
                "min": round(float(np.min([r[k] for r in runs])), 4),
                "max": round(float(np.max([r[k] for r in runs])), 4)} for k in keys} | {"n_runs": len(runs)}


def main():
    df = pd.read_csv(DATA_PROCESSED / "student_cleaned.csv")
    meta = read_json(MODELS_DIR / "feature_metadata.json")
    classes = meta["classes"]
    groups = df["student_group"]
    out = {"note": "Fixed hyper-parameters; 5 repeated splits where 'runs' are mentioned."}

    # A. Old-style random split vs student-aware split (same models, same seeds).
    rnd, grp = [], []
    for seed in range(5):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(sss.split(df, df["support_level"]))
        rnd.append(fit_eval(df, tr, te, classes=classes))
        tr, te = group_split(df, groups, random_state=seed)
        grp.append(fit_eval(df, tr, te, classes=classes))
    out["random_vs_group_split"] = {"random_split_5_seeds": summarise(rnd), "group_split_5_seeds": summarise(grp)}
    # How many test rows have the same student in train under a random split (seed 42)?
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(sss.split(df, df["support_level"]))
    shared = np.isin(groups.iloc[te].values, groups.iloc[tr].values).mean()
    out["random_split_test_rows_with_same_student_in_train_pct"] = round(float(shared) * 100, 1)

    # B. Project (group) split: effect of the G3 = 0 records.
    tr, te = group_split(df, groups)
    base = fit_eval(df, tr, te, classes=classes)
    nz = df["final_score_pct"] > 0
    tr_nz = np.array([i for i in tr if nz.iloc[i]])
    te_nz = np.array([i for i in te if nz.iloc[i]])
    out["g3_zero_records"] = {
        "n_total": int((~nz).sum()),
        "n_with_zero_absences": int(((~nz) & (df["absences"] == 0)).sum()),
        "pct_zero_g3_among_zero_absence_students": round(float((~nz)[df["absences"] == 0].mean() * 100), 1),
        "pct_zero_g3_among_students_with_absences": round(float((~nz)[df["absences"] > 0].mean() * 100), 1),
        "corr_absences_G3_all": round(float(df["absences"].corr(df["G3"])), 3),
        "corr_absences_G3_excluding_zero_G3": round(float(df.loc[nz, "absences"].corr(df.loc[nz, "G3"])), 3),
        "metrics_all_rows": base,
        "metrics_train_and_test_without_G3_zero": fit_eval(df, tr_nz, te_nz, classes=classes),
    }
    # C. Early-warning ablation: remove G1, G2 and the trend.
    out["without_G1_G2_trend"] = fit_eval(df, tr, te, drop=("G1", "G2", "assessment_trend"), classes=classes)
    out["with_all_features_reference"] = base
    with open(MODELS_DIR / "sensitivity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    r, g = out["random_vs_group_split"]["random_split_5_seeds"], out["random_vs_group_split"]["group_split_5_seeds"]
    print("random split  R2 %.3f MAE %.2f F1 %.3f | group split R2 %.3f MAE %.2f F1 %.3f" % (
        r["R2"]["mean"], r["MAE"]["mean"], r["F1_Macro"]["mean"], g["R2"]["mean"], g["MAE"]["mean"], g["F1_Macro"]["mean"]))
    print("same-student test rows under random split: %.1f%%" % out["random_split_test_rows_with_same_student_in_train_pct"])
    print("G3=0:", {k: v for k, v in out["g3_zero_records"].items() if not k.startswith("metrics")})
    print("no G3=0 rows:", out["g3_zero_records"]["metrics_train_and_test_without_G3_zero"])
    print("without G1/G2:", out["without_G1_G2_trend"])


if __name__ == "__main__":
    main()
