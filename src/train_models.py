"""Train, select and evaluate regression, classification and clustering models.

Model selection uses grouped cross-validation on the TRAINING portion only.
The held-out test split is evaluated once, after selection.
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import RANDOM_STATE, build_preprocessor, get_feature_lists
from src.metrics import (classification_metrics, regression_metrics, support_level_from_grade)
from src.paths import DATA_PROCESSED, MODELS_DIR
from src.io_utils import read_json

CV_FOLDS = 5
CLUSTER_K = 4  # chosen for interpretability; silhouette values are reported honestly
CLUSTER_FEATURES = ["studytime", "failures", "goout", "freetime", "Dalc", "Walc",
                    "estimated_attendance_pct", "engagement_consistency"]
CLUSTER_DESCRIPTORS = {
    "studytime": ("high study time", "low study time"),
    "failures": ("more past failures", "fewer past failures"),
    "goout": ("goes out often", "goes out rarely"),
    "freetime": ("more free time", "less free time"),
    "Dalc": ("higher weekday alcohol", "lower weekday alcohol"),
    "Walc": ("higher weekend alcohol", "lower weekend alcohol"),
    "estimated_attendance_pct": ("higher est. attendance", "lower est. attendance"),
    "engagement_consistency": ("higher engagement index", "lower engagement index"),
}


def _round(x, nd=4):
    return None if x is None else round(float(x), nd)


def make_cv_splits(X, y_tier, groups):
    """Grouped, tier-stratified CV folds (a student's records never span train/validation)."""
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, y_tier, groups))


def _pipe(num, cat, model):
    return Pipeline([("pre", build_preprocessor(num, cat)), ("model", model)])


def _tune(pipe, grid, X, y, splits, scoring):
    """Grid search with the preprocessor inside the CV loop (no validation-fold leakage)."""
    gs = GridSearchCV(pipe, grid, cv=splits, scoring=scoring, refit=False, n_jobs=1)
    gs.fit(X, y)
    return {k.replace("model__", ""): v for k, v in gs.best_params_.items()}


def regression_candidates(num, cat, X, y, splits):
    cands = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=4,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                                                       random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                random_state=RANDOM_STATE, n_jobs=1),
    }
    gb_params = _tune(_pipe(num, cat, GradientBoostingRegressor(random_state=RANDOM_STATE)),
                      {"model__n_estimators": [100, 150], "model__learning_rate": [0.03, 0.08],
                       "model__max_depth": [2, 3]}, X, y, splits, "neg_mean_absolute_error")
    cands["Gradient Boosting (Tuned)"] = GradientBoostingRegressor(random_state=RANDOM_STATE, **gb_params)
    rf_params = _tune(_pipe(num, cat, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
                      {"model__n_estimators": [150, 300], "model__max_depth": [6, 8],
                       "model__min_samples_leaf": [2, 4]}, X, y, splits, "neg_mean_absolute_error")
    cands["Random Forest (Tuned)"] = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    return cands, {"Gradient Boosting (Tuned)": gb_params, "Random Forest (Tuned)": rf_params}


def classification_candidates(num, cat, X, y, splits):
    cands = {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4,
                                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                                        random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE,
                                 eval_metric="mlogloss", n_jobs=1),
    }
    rf_params = _tune(_pipe(num, cat, RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                                             n_jobs=-1)),
                      {"model__n_estimators": [150, 300], "model__max_depth": [6, 8],
                       "model__min_samples_leaf": [2, 4]}, X, y, splits, "f1_macro")
    cands["Random Forest (Tuned)"] = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                                            n_jobs=-1, **rf_params)
    return cands, {"Random Forest (Tuned)": rf_params}


def describe_cluster(center_z: np.ndarray, features, top_n=2, threshold=0.5) -> str:
    """Factual description from the cluster centroid (z-scores). No invented personas."""
    order = np.argsort(-np.abs(center_z))
    parts = []
    for i in order[:top_n]:
        if abs(center_z[i]) >= threshold:
            high, low = CLUSTER_DESCRIPTORS[features[i]]
            parts.append(high if center_z[i] > 0 else low)
    return ", ".join(parts) if parts else "near-average profile"


def run_clustering(train_df):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_df[CLUSTER_FEATURES])
    k_metrics = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
        k_metrics[str(k)] = _round(silhouette_score(Xs, labels))
    km = KMeans(n_clusters=CLUSTER_K, random_state=RANDOM_STATE, n_init=10).fit(Xs)
    d = train_df.copy()
    d["cluster"] = km.labels_
    names, profiles = {}, {}
    for c in range(CLUSTER_K):
        sub = d[d["cluster"] == c]
        names[str(c)] = f"Cluster {c}: " + describe_cluster(km.cluster_centers_[c], CLUSTER_FEATURES)
        profiles[str(c)] = {
            "size": int(len(sub)), "pct": _round(len(sub) / len(d) * 100, 1),
            "mean_score_pct": _round(sub["final_score_pct"].mean(), 2),
            "mean_grade_G3": _round(sub["final_score_pct"].mean() / 5, 2),
            "mean_failures": _round(sub["failures"].mean(), 2),
            "mean_studytime": _round(sub["studytime"].mean(), 2),
            "mean_estimated_attendance_pct": _round(sub["estimated_attendance_pct"].mean(), 1),
            "mean_engagement": _round(sub["engagement_consistency"].mean(), 1),
            "mean_goout": _round(sub["goout"].mean(), 2),
            "centroid_zscores": {f: _round(v, 2) for f, v in zip(CLUSTER_FEATURES, km.cluster_centers_[c])},
        }
    joblib.dump(km, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")
    return {
        "features": CLUSTER_FEATURES, "k": CLUSTER_K, "k_metrics": k_metrics,
        "silhouette_selected_k": k_metrics[str(CLUSTER_K)], "archetype_names": names, "profiles": profiles,
        "note": ("Exploratory only. Silhouette values below ~0.25 indicate weak cluster structure; k=4 was chosen "
                 "for interpretability, not because it maximises silhouette."),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    meta = read_json(MODELS_DIR / "feature_metadata.json")
    num, cat = meta["numeric_features"], meta["categorical_features"]
    cols = num + cat
    classes = meta["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}

    Xtr_raw, Xte_raw = train[cols], test[cols]
    ytr, yte = train["final_score_pct"].values, test["final_score_pct"].values
    ytr_c = train["support_level"].map(cls_idx).values
    yte_lab = test["support_level"].values
    splits = make_cv_splits(Xtr_raw, train["support_level"], train["student_group"])

    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    Xtr, Xte = pre.transform(Xtr_raw), pre.transform(Xte_raw)

    # ---------------- Regression ----------------
    print("Regression: grouped CV + tuning ...")
    reg_c, reg_params = regression_candidates(num, cat, Xtr_raw, ytr, splits)
    reg_metrics, fitted_reg, oof_store = {}, {}, {}
    for name, est in reg_c.items():
        oof = cross_val_predict(_pipe(num, cat, clone(est)), Xtr_raw, ytr, cv=splits)
        oof_store[name] = oof
        model = clone(est).fit(Xtr, ytr)
        fitted_reg[name] = model
        pte, ptr = model.predict(Xte), model.predict(Xtr)
        m = regression_metrics(yte, pte)
        res = yte - pte
        reg_metrics[name] = {
            "MAE": _round(m["MAE"]), "RMSE": _round(m["RMSE"]), "R2": _round(m["R2"]),
            "Train_R2": _round(r2_score(ytr, ptr)),
            "CV_MAE": _round(np.mean(np.abs(ytr - oof))), "CV_R2": _round(r2_score(ytr, oof)),
            "Residual_Mean": _round(res.mean()), "Residual_Std": _round(res.std(ddof=1)),
            "Residuals": [round(float(r), 3) for r in res], "Is_Model": True,
            "Tuned_Params": reg_params.get(name),
        }
    # Naive rule: final grade = second-period grade (no learning involved).
    naive_te = test["G2"].values * 5.0
    naive_tr = train["G2"].values * 5.0
    m = regression_metrics(yte, naive_te)
    reg_metrics["Naive (G3 = G2)"] = {
        "MAE": _round(m["MAE"]), "RMSE": _round(m["RMSE"]), "R2": _round(m["R2"]),
        "Train_R2": _round(r2_score(ytr, naive_tr)),
        "CV_MAE": _round(np.mean(np.abs(ytr - naive_tr))), "CV_R2": _round(r2_score(ytr, naive_tr)),
        "Residual_Mean": _round((yte - naive_te).mean()), "Residual_Std": _round((yte - naive_te).std(ddof=1)),
        "Residuals": [round(float(r), 3) for r in (yte - naive_te)], "Is_Model": False, "Tuned_Params": None,
    }
    selectable = [k for k, v in reg_metrics.items() if v["Is_Model"] and not k.startswith("Baseline")]
    best_reg_name = min(selectable, key=lambda k: reg_metrics[k]["CV_MAE"])
    oof_res = ytr - oof_store[best_reg_name]
    interval = {
        "source": f"out-of-fold residuals of '{best_reg_name}' on the training portion (grouped CV)",
        "residual_sd": _round(oof_res.std(ddof=1)),
        "q025": _round(np.quantile(oof_res, 0.025)), "q975": _round(np.quantile(oof_res, 0.975)),
    }

    # ---------------- Classification ----------------
    print("Classification: grouped CV + tuning ...")
    clf_c, clf_params = classification_candidates(num, cat, Xtr_raw, ytr_c, splits)
    clf_metrics, fitted_clf = {}, {}
    from sklearn.metrics import f1_score
    for name, est in clf_c.items():
        oof = cross_val_predict(_pipe(num, cat, clone(est)), Xtr_raw, ytr_c, cv=splits)
        model = clone(est).fit(Xtr, ytr_c)
        fitted_clf[name] = model
        pred = np.array([classes[int(i)] for i in model.predict(Xte)])
        proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        cm = classification_metrics(yte_lab, pred, classes, proba)
        cm["CV_F1_Macro"] = _round(f1_score(ytr_c, oof, average="macro"))
        cm["Is_Model"] = True
        cm["Tuned_Params"] = clf_params.get(name)
        clf_metrics[name] = cm
    # Naive tier from G2 using the same thresholds.
    naive_pred = np.array([support_level_from_grade(g) for g in test["G2"].values])
    naive_tr_pred = np.array([support_level_from_grade(g) for g in train["G2"].values])
    nm = classification_metrics(yte_lab, naive_pred, classes, None)
    nm["CV_F1_Macro"] = _round(f1_score(train["support_level"], naive_tr_pred, average="macro"))
    nm["Is_Model"], nm["Tuned_Params"] = False, None
    clf_metrics["Naive (tier from G2)"] = nm
    sel_c = [k for k, v in clf_metrics.items() if v["Is_Model"] and not k.startswith("Baseline")]
    best_clf_name = max(sel_c, key=lambda k: clf_metrics[k]["CV_F1_Macro"])
    for v in clf_metrics.values():
        for key in ("Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC"):
            v[key] = _round(v[key])
        v["Recall_Per_Class"] = {k: _round(x) for k, x in v["Recall_Per_Class"].items()}

    # ---------------- Clustering ----------------
    print("Clustering ...")
    clustering = run_clustering(train)

    # ---------------- Save ----------------
    joblib.dump(fitted_reg, MODELS_DIR / "all_reg_models.joblib")
    joblib.dump(fitted_clf, MODELS_DIR / "all_clf_models.joblib")
    joblib.dump(fitted_reg[best_reg_name], MODELS_DIR / "best_regressor.joblib")
    joblib.dump(fitted_clf[best_clf_name], MODELS_DIR / "best_classifier.joblib")

    summary = {
        "best_regressor": best_reg_name, "best_classifier": best_clf_name, "classes": classes,
        "selection_criterion": {
            "regression": "lowest grouped-CV MAE on the training portion (tuned models' CV score is the best "
                          "over a small grid and therefore slightly optimistic)",
            "classification": "highest grouped-CV macro-F1 on the training portion",
            "test_set_used_for_selection": False,
        },
        "evaluation_split": {"train_rows": int(len(train)), "test_rows": int(len(test)),
                             "cv_folds": CV_FOLDS, "split": meta["split"]},
        "regression_metrics": reg_metrics, "classification_metrics": clf_metrics,
        "prediction_interval": interval, "clustering": clustering,
    }
    with open(MODELS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "random_state": RANDOM_STATE,
    }
    with open(MODELS_DIR / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Best regressor : {best_reg_name} | CV MAE {reg_metrics[best_reg_name]['CV_MAE']} | "
          f"TEST MAE {reg_metrics[best_reg_name]['MAE']} R2 {reg_metrics[best_reg_name]['R2']}")
    print(f"Best classifier: {best_clf_name} | CV F1 {clf_metrics[best_clf_name]['CV_F1_Macro']} | "
          f"TEST acc {clf_metrics[best_clf_name]['Accuracy']} F1 {clf_metrics[best_clf_name]['F1_Macro']}")
    print("Naive G3=G2    :", {k: reg_metrics['Naive (G3 = G2)'][k] for k in ('MAE', 'R2')})
    print("Silhouette k=2..6:", clustering["k_metrics"])
    for n in clustering["archetype_names"].values():
        print("  ", n)


if __name__ == "__main__":
    main()
