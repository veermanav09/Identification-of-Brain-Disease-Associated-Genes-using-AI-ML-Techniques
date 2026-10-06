"""Differential gene expression: Parkinson's disease vs control.

Method
------
For every gene we run Welch's two-sample t-test (`scipy.stats.ttest_ind`, equal_var=False) on log2
expression values and correct the p-values for multiple testing with the Benjamini-Hochberg
false-discovery-rate procedure (`statsmodels.stats.multitest.multipletests`).

Because the input is already on a log2 scale:
    log2 fold change = mean(disease) - mean(control)
    fold change      = 2 ** log2 fold change

Assumptions and limitations (also shown in the app / README)
-----------------------------------------------------------
* Samples are treated as independent; post-mortem brains may differ in age, sex, post-mortem
  interval, RNA quality or cell-type composition, none of which are modelled here.
* The t-test assumes approximately normal values within each group. With n = 9 vs 16 the test has
  limited power and is sensitive to outliers; dedicated microarray methods such as limma's moderated
  t-statistic borrow information across genes and are generally preferred in publications.
* Results describe associations in this dataset only and do not establish causality.
* The differential-expression table is computed on ALL samples and is descriptive. It is not part of
  the cross-validated performance estimate (feature selection inside the ML pipeline is redone within
  each training fold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .utils import CONTROL, PARKINSON, ProjectError

DE_COLUMNS = [
    "Gene", "Control_Mean", "Disease_Mean", "Fold_Change", "Log2_Fold_Change",
    "P_Value", "Adjusted_P_Value",
]


def run_differential_expression(X: pd.DataFrame, y: pd.Series, min_group_size: int = 2) -> pd.DataFrame:
    """Compute per-gene statistics.

    Parameters
    ----------
    X : samples x genes (log2 scale)
    y : Series of 0/1 labels aligned to X.index

    Returns
    -------
    DataFrame with columns Gene, Control_Mean, Disease_Mean, Fold_Change, Log2_Fold_Change,
    P_Value, Adjusted_P_Value, Regulation - sorted by adjusted p-value, then |log2FC|.
    """
    y = y.reindex(X.index)
    ctrl = X.loc[y == CONTROL]
    dis = X.loc[y == PARKINSON]
    if len(ctrl) < min_group_size or len(dis) < min_group_size:
        raise ProjectError(
            f"Need at least {min_group_size} samples per group for differential expression "
            f"(found {len(ctrl)} control and {len(dis)} Parkinson's)."
        )

    t_stat, p_val = stats.ttest_ind(
        dis.to_numpy(dtype=float), ctrl.to_numpy(dtype=float), axis=0, equal_var=False, nan_policy="omit"
    )
    p_val = np.asarray(p_val, dtype=float)
    p_val = np.where(np.isfinite(p_val), p_val, 1.0)  # zero-variance genes -> not significant

    adj = np.ones_like(p_val)
    adj[:] = multipletests(p_val, method="fdr_bh")[1]

    ctrl_mean = ctrl.mean(axis=0, skipna=True).to_numpy()
    dis_mean = dis.mean(axis=0, skipna=True).to_numpy()
    log2fc = dis_mean - ctrl_mean

    de = pd.DataFrame({
        "Gene": X.columns.astype(str),
        "Control_Mean": ctrl_mean,
        "Disease_Mean": dis_mean,
        "Fold_Change": np.power(2.0, log2fc),
        "Log2_Fold_Change": log2fc,
        "P_Value": p_val,
        "Adjusted_P_Value": adj,
    })
    de["Abs_Log2FC"] = de["Log2_Fold_Change"].abs()
    de = de.sort_values(["Adjusted_P_Value", "Abs_Log2FC"], ascending=[True, False]).reset_index(drop=True)
    de = de.drop(columns="Abs_Log2FC")
    return de


def add_regulation_label(de: pd.DataFrame, max_adj_p: float, min_abs_log2fc: float) -> pd.DataFrame:
    """Add a 'Regulation' column: Up in PD / Down in PD / Not significant."""
    out = de.copy()
    significant = (out["Adjusted_P_Value"] <= max_adj_p) & (out["Log2_Fold_Change"].abs() >= min_abs_log2fc)
    out["Regulation"] = np.where(
        significant & (out["Log2_Fold_Change"] > 0), "Up in PD",
        np.where(significant & (out["Log2_Fold_Change"] < 0), "Down in PD", "Not significant"),
    )
    return out


def filter_de_results(
    de: pd.DataFrame, max_adj_p: float = 0.05, min_abs_log2fc: float = 0.5, top_n: int = 100
) -> pd.DataFrame:
    """Apply user thresholds and return at most `top_n` significant genes."""
    mask = (de["Adjusted_P_Value"] <= max_adj_p) & (de["Log2_Fold_Change"].abs() >= min_abs_log2fc)
    return de.loc[mask].head(int(top_n)).reset_index(drop=True)


def top_regulated(de: pd.DataFrame, direction: str, n: int = 10, max_adj_p: float = 0.05,
                  min_abs_log2fc: float = 0.0) -> pd.DataFrame:
    """Top up- ('up') or down- ('down') regulated genes among those passing the thresholds."""
    sig = de[(de["Adjusted_P_Value"] <= max_adj_p) & (de["Log2_Fold_Change"].abs() >= min_abs_log2fc)]
    if direction == "up":
        sig = sig[sig["Log2_Fold_Change"] > 0].sort_values("Log2_Fold_Change", ascending=False)
    elif direction == "down":
        sig = sig[sig["Log2_Fold_Change"] < 0].sort_values("Log2_Fold_Change", ascending=True)
    else:
        raise ValueError("direction must be 'up' or 'down'")
    return sig.head(n).reset_index(drop=True)


def gene_statistics(de: pd.DataFrame, gene: str) -> dict | None:
    """Return the DE row for one gene (case-insensitive) or None."""
    hit = de[de["Gene"].str.upper() == gene.strip().upper()]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()
