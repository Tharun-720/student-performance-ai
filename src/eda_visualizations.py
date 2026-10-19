"""Static (saved) and interactive (Plotly) figures. All data comes from the pipeline outputs."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import DATA_PROCESSED, IMAGES_DIR, MODELS_DIR
from src.io_utils import read_json

TIER_ORDER = ["High Support", "Medium Support", "Low Support"]
TIER_COLORS = {"High Support": "#d62728", "Medium Support": "#ff9f1c", "Low Support": "#2ca02c"}


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=160)
    plt.close(fig)


def generate_static_visualizations(output_dir=IMAGES_DIR):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA_PROCESSED / "student_cleaned.csv")
    summary = read_json(MODELS_DIR / "evaluation_summary.json")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["final_score_pct"], bins=21, kde=True, ax=ax, color="#4c72b0")
    ax.axvline(50, color="red", ls="--", label="Pass mark (50% = grade 10/20)")
    ax.set(title="Distribution of final scores (records, not unique students)", xlabel="Final score (%)")
    ax.legend(); _save(fig, out, "01_grade_distribution.png")

    cols = ["final_score_pct", "G1", "G2", "studytime", "failures", "absences", "estimated_attendance_pct",
            "assessment_trend", "engagement_consistency"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (Pearson)"); _save(fig, out, "02_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="absences", y="final_score_pct", hue="support_level", hue_order=TIER_ORDER,
                    palette=TIER_COLORS, alpha=0.6, ax=ax)
    ax.set(title="Recorded absences vs final score\n(note: every record with a final score of 0 has 0 absences)",
           xlabel="Absences (count)", ylabel="Final score (%)"); _save(fig, out, "03_absences_vs_score.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby("support_level")[["G1", "G2", "G3"]].mean().reindex(TIER_ORDER)
    for tier, row in means.iterrows():
        ax.plot(["G1", "G2", "G3"], row.values, marker="o", label=tier, color=TIER_COLORS[tier])
    ax.set(title="Mean grade per period by actual final tier", ylabel="Grade (0-20)"); ax.legend()
    _save(fig, out, "04_grade_trajectory.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="support_level", y="failures", hue="studytime", order=TIER_ORDER, errorbar=None, ax=ax)
    ax.set(title="Mean past failures by tier and study-time category", ylabel="Mean failures")
    ax.legend(title="Study time (1 = lowest)"); _save(fig, out, "05_studytime_failures_tier.png")

    reg = summary["regression_metrics"]
    names = list(reg.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, [reg[n]["MAE"] for n in names], color=["#999" if not reg[n]["Is_Model"] or n.startswith("Baseline") else "#4c72b0" for n in names])
    axes[0].set(title="Test MAE (lower is better)", xlabel="Points of score (0-100)")
    axes[1].barh(names, [reg[n]["R2"] for n in names], color="#55a868"); axes[1].set(title="Test R2")
    _save(fig, out, "06_regression_benchmark.png")

    best_c = summary["best_classifier"]; cm = np.array(summary["classification_metrics"][best_c]["Confusion_Matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=summary["classes"], yticklabels=summary["classes"], ax=ax)
    ax.set(title=f"Confusion matrix: {best_c}\n(rows = actual, columns = predicted; test split)", xlabel="Predicted", ylabel="Actual")
    _save(fig, out, "07_confusion_matrix.png")

    km = joblib.load(MODELS_DIR / "kmeans_model.joblib"); sc = joblib.load(MODELS_DIR / "cluster_scaler.joblib")
    feats = summary["clustering"]["features"]
    Xs = sc.transform(df[feats]); labels = km.predict(Xs)
    pcs = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=[summary["clustering"]["archetype_names"][str(l)] for l in labels],
                    palette="Set2", alpha=0.7, ax=ax)
    ax.set(title=f"K-Means clusters in 2-D PCA space (exploratory; silhouette {summary['clustering']['silhouette_selected_k']})")
    ax.legend(fontsize=7); _save(fig, out, "08_kmeans_clusters_pca.png")

    best_r = summary["best_regressor"]; res = np.array(reg[best_r]["Residuals"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(res, bins=25, kde=True, ax=axes[0], color="#c44e52")
    axes[0].axvline(0, color="k", ls="--"); axes[0].set(title=f"Test residuals (actual - predicted), n={len(res)}", xlabel="Points")
    axes[1].scatter(test["final_score_pct"], res, alpha=0.6); axes[1].axhline(0, color="k", ls="--")
    axes[1].set(title="Residual vs actual score", xlabel="Actual score (%)", ylabel="Residual")
    _save(fig, out, "09_residual_analysis.png")
    print("Saved 9 figures to", out)


# ---------------------------------------------------------------- interactive (Plotly)
def get_interactive_grade_dist(df):
    return px.histogram(df, x="final_score_pct", color="support_level", nbins=21, barmode="overlay", opacity=0.7,
                        category_orders={"support_level": TIER_ORDER}, color_discrete_map=TIER_COLORS,
                        title="Final score distribution", labels={"final_score_pct": "Final score (%)"})


def get_interactive_absences_scatter(df):
    return px.scatter(df, x="absences", y="final_score_pct", color="support_level", opacity=0.6,
                      category_orders={"support_level": TIER_ORDER}, color_discrete_map=TIER_COLORS,
                      title="Recorded absences vs final score",
                      labels={"absences": "Absences (count)", "final_score_pct": "Final score (%)"})


def get_interactive_trajectory(df):
    means = df.groupby("support_level")[["G1", "G2", "G3"]].mean().reindex(TIER_ORDER).dropna(how="all")
    fig = go.Figure()
    for tier, row in means.iterrows():
        fig.add_trace(go.Scatter(x=["G1", "G2", "G3"], y=row.values, mode="lines+markers", name=tier,
                                 line=dict(color=TIER_COLORS[tier])))
    fig.update_layout(title="Mean grade per period by actual final tier", yaxis_title="Grade (0-20)")
    return fig


def get_interactive_studytime_failures(df):
    g = df.groupby(["support_level", "studytime"], as_index=False)["failures"].mean()
    g["studytime"] = g["studytime"].astype(str)
    return px.bar(g, x="support_level", y="failures", color="studytime", barmode="group",
                  category_orders={"support_level": TIER_ORDER},
                  title="Mean past failures by tier and study-time category",
                  labels={"failures": "Mean failures", "studytime": "Study time (1 = lowest)"})


def get_interactive_cluster_radar(eval_summary):
    """Radar of cluster means, each axis scaled 0-100 over its possible range (see axis names)."""
    fig = go.Figure()
    axes = {"Study time (1-4)": ("mean_studytime", 1, 4), "Engagement index (0-100)": ("mean_engagement", 0, 100),
            "Est. attendance (%)": ("mean_estimated_attendance_pct", 0, 100),
            "Going out (1-5)": ("mean_goout", 1, 5), "Failures (0-3)": ("mean_failures", 0, 3)}
    for cid, prof in eval_summary["clustering"]["profiles"].items():
        vals = [(prof[key] - lo) / (hi - lo) * 100 for key, lo, hi in axes.values()]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(axes) + [list(axes)[0]], fill="toself",
                                      name=eval_summary["clustering"]["archetype_names"][cid]))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                      title="Cluster profiles (each axis rescaled to its possible range)")
    return fig


if __name__ == "__main__":
    generate_static_visualizations()
