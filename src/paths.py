"""Project-root based paths.

All modules resolve files relative to the project root (the folder that
contains ``src/``), never relative to the current working directory. This
makes the app, tests and scripts work no matter where they are launched from,
on Windows as well as Linux/macOS.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = DOCS_DIR / "images"
