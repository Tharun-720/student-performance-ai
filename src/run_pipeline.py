"""Rebuild every artifact in order: data -> models -> explanations -> figures -> reports.

Usage (from anywhere):  python -m src.run_pipeline
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (data_preprocessing, eda_visualizations, explainability, generate_reports, sensitivity,
                 train_models)


def main():
    steps = [("1/6 data preparation", data_preprocessing.prepare_and_save_data),
             ("2/6 model training and evaluation", train_models.main),
             ("3/6 explainability (SHAP)", explainability.main),
             ("4/6 sensitivity analyses", sensitivity.main),
             ("5/6 figures", eda_visualizations.generate_static_visualizations),
             ("6/6 data-driven reports", generate_reports.main)]
    for title, fn in steps:
        print(f"\n=== {title} ===")
        fn()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
