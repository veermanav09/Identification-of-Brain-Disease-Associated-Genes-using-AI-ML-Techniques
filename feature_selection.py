"""Feature reduction for the high-dimensional gene-expression problem.

A microarray has ~20,000 genes but this study has only 25 samples (p >> n). Training a model on every
gene would overfit almost immediately, so we reduce the feature space in two label-independent /
label-dependent steps that are placed INSIDE the scikit-learn Pipeline. Consequently, during
cross-validation both steps are re-fitted on each training fold and never see the held-out samples
(no data leakage).

    1. VarianceFilter      unsupervised: drop the lowest-variance genes (mostly noise)
    2. TopKSelector        supervised: keep the K genes with the highest ANOVA F-statistic
                           (for two classes F = t^2 of the pooled-variance t-test)

Both are small scikit-learn-compatible transformers so that selected genes can always be recovered.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted

DEFAULT_VARIANCE_KEEP_FRACTION = 0.5


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Keep the top `keep_fraction` of features ranked by variance (computed on the training data)."""

    def __init__(self, keep_fraction: float = DEFAULT_VARIANCE_KEEP_FRACTION):
        self.keep_fraction = keep_fraction

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        variances = np.nanvar(X, axis=0)
        variances = np.where(np.isfinite(variances), variances, -np.inf)
        n_keep = max(1, int(np.ceil(self.keep_fraction * X.shape[1])))
        order = np.argsort(-variances, kind="stable")[:n_keep]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order] = True
        self.support_ = mask
        self.variances_ = variances
        self.n_features_in_ = X.shape[1]
        return self

    def get_support(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X[:, self.support_]


class TopKSelector(BaseEstimator, TransformerMixin):
    """Select the K features with the largest ANOVA F-statistic (capped at the number available)."""

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        scores, pvalues = f_classif(X, y)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        k = int(min(self.k, X.shape[1]))
        order = np.argsort(-scores, kind="stable")[:k]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order] = True
        self.support_ = mask
        self.scores_ = scores
        self.pvalues_ = pvalues
        self.n_features_in_ = X.shape[1]
        return self

    def get_support(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X[:, self.support_]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def filter_low_variance(X: pd.DataFrame, keep_fraction: float = DEFAULT_VARIANCE_KEEP_FRACTION) -> pd.DataFrame:
    """Unsupervised variance filter for descriptive analyses (EDA / DE display)."""
    vf = VarianceFilter(keep_fraction).fit(X.to_numpy())
    return X.loc[:, vf.get_support()]


def rank_genes_by_f_test(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Univariate ANOVA-F ranking of all genes (descriptive)."""
    scores, p = f_classif(X.to_numpy(dtype=float), y.reindex(X.index).to_numpy())
    out = pd.DataFrame({"Gene": X.columns, "F_Score": scores, "F_P_Value": p})
    return out.sort_values("F_Score", ascending=False).reset_index(drop=True)


def get_selected_genes(fitted_pipeline, input_genes: Sequence[str]) -> List[str]:
    """Recover the gene names kept by a fitted pipeline containing 'variance' and 'select' steps.

    The SimpleImputer step keeps column order and count (all-missing genes are removed beforehand),
    so masks can be composed directly.
    """
    genes = np.asarray(list(input_genes))
    named = fitted_pipeline.named_steps
    if "variance" in named:
        genes = genes[named["variance"].get_support()]
    if "select" in named:
        genes = genes[named["select"].get_support()]
    return genes.tolist()
