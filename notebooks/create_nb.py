"""Generate notebooks/eda_analysis.ipynb. Run from anywhere: python notebooks/create_nb.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


_n = [0]


def _id():
    _n[0] += 1
    return f"cell{_n[0]:03d}"


def md(text):
    return {"id": _id(), "cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"id": _id(), "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md("# Exploratory Data Analysis\nUCI Student Performance dataset (Maths 395 + Portuguese 649 records). "
       "Run this notebook from the `notebooks/` folder or the project root."),
    code("import sys\nfrom pathlib import Path\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
         "sys.path.insert(0, str(ROOT))\n"
         "import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n"
         "from src.data_loader import load_raw_data, validate_data\n"
         "from src.data_preprocessing import engineer_features, student_fingerprint\n"
         "sns.set_theme(style='whitegrid')"),
    md("## 1. Loading and validation"),
    code("raw = load_raw_data()\nrep = validate_data(raw)\n"
         "print('records:', rep['total_records'], '| missing:', rep['total_missing'], '| duplicates:', rep['duplicate_rows'])\n"
         "print('grades within 0-20:', rep['grade_ranges_valid'], '| PII-like columns:', rep['pii_like_columns'])\nraw.head(3)"),
    md("## 2. Records are not distinct students\nThe dataset has no student id; students enrolled in both courses appear twice."),
    code("df = engineer_features(raw)\ndf['student_group'] = student_fingerprint(df)\n"
         "print('records:', len(df), '| distinct student fingerprints:', df['student_group'].nunique())"),
    md("## 3. Absences and the final grade\nEvery record with G3 = 0 has 0 absences, which distorts the relationship."),
    code("z = df[df.G3 == 0]\nprint('G3 == 0 records:', len(z), '| of which absences == 0:', (z.absences == 0).sum())\n"
         "print('corr(absences, G3) all rows:', round(df.absences.corr(df.G3), 3))\n"
         "nz = df[df.G3 > 0]\nprint('corr(absences, G3) excluding G3 == 0:', round(nz.absences.corr(nz.G3), 3))"),
    md("## 4. Correlations"),
    code("cols = ['final_score_pct','G1','G2','studytime','failures','absences','assessment_trend']\n"
         "plt.figure(figsize=(8,6)); sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0); plt.show()"),
    md("## 5. How much does G2 explain?\nA rule that predicts final = G2 needs no model."),
    code("from sklearn.metrics import mean_absolute_error, r2_score\n"
         "print('MAE of final = G2 (score points):', round(mean_absolute_error(df.final_score_pct, df.G2*5), 2))\n"
         "print('R2  of final = G2:', round(r2_score(df.final_score_pct, df.G2*5), 3))"),
]
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
out = HERE / "eda_analysis.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out)
