"""Shared constants and small helper functions used across the project."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python and NumPy RNGs (scikit-learn objects receive the seed explicitly)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.joblib"


def ensure_dirs() -> None:
    """Create data/ and models/ if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
GSE_ID = "GSE7621"
GSE_URL = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GSE_ID}"
GSE_TITLE = "Expression data of substantia nigra from postmortem human brain of Parkinson's disease patients"
GSE_PLATFORM = "GPL570 - Affymetrix Human Genome U133 Plus 2.0 Array"
EXPECTED_COUNTS = {"Control": 9, "Parkinson's Disease": 16}

CONTROL = 0
PARKINSON = 1
CLASS_NAMES = {CONTROL: "Control", PARKINSON: "Parkinson's Disease"}
CLASS_COLORS = {"Control": "#2E86AB", "Parkinson's Disease": "#D1495B"}

FEATURE_COUNT_OPTIONS = [25, 50, 100, 250, 500]
CV_FOLD_OPTIONS = [3, 5]

# --------------------------------------------------------------------------- #
# Wording that must appear in the application
# --------------------------------------------------------------------------- #
DISCLAIMER_TITLE = "Research/Educational Use Only"
DISCLAIMER_TEXT = (
    "This application demonstrates computational identification of gene-expression patterns "
    "associated with Parkinson's disease using publicly available research data. It is not a "
    "medical diagnostic tool and should not be used to diagnose, predict, or make treatment "
    "decisions for individual patients. The identified genes represent statistical and "
    "machine-learning associations within the analyzed dataset and do not establish biological "
    "causality."
)
CONSENSUS_LABEL = "Project-specific computational consensus score (not biologically validated)."


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def minmax(values: Iterable[float]) -> pd.Series:
    """Min-max normalise to [0, 1]. A constant vector maps to all zeros."""
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """Serialise a DataFrame for st.download_button."""
    return df.to_csv(index=index).encode("utf-8")


def label_series_to_names(y: pd.Series) -> pd.Series:
    """Map 0/1 labels to readable class names."""
    return y.map(CLASS_NAMES)


def format_p(p: float) -> str:
    """Compact p-value formatting for display."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


class ProjectError(Exception):
    """Base class for errors whose message is safe to show to end users."""
