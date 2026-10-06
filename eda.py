"""Exploratory data analysis: overview statistics and visualisations (Plotly + seaborn)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .feature_selection import filter_low_variance
from .preprocessing import impute_median
from .utils import CLASS_COLORS, CLASS_NAMES, RANDOM_STATE


def _names(y: pd.Series) -> pd.Series:
    return y.map(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def dataset_overview(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    counts = _names(y).value_counts()
    return {
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_disease": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "class_distribution": counts.to_dict(),
        "missing_values": int(X.isna().sum().sum()),
        "value_min": float(np.nanmin(X.to_numpy())),
        "value_max": float(np.nanmax(X.to_numpy())),
        "value_mean": float(np.nanmean(X.to_numpy())),
    }


def expression_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """Summary of per-gene values (mean, std, ...)."""
    stats = X.stack().describe(percentiles=[0.25, 0.5, 0.75]).to_frame("All expression values")
    return stats


def class_distribution_fig(y: pd.Series) -> go.Figure:
    counts = _names(y).value_counts().reset_index()
    counts.columns = ["Class", "Samples"]
    fig = px.bar(
        counts, x="Class", y="Samples", color="Class", text="Samples",
        color_discrete_map=CLASS_COLORS, title="Class distribution",
    )
    fig.update_layout(showlegend=False, yaxis_title="Number of samples")
    return fig


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def expression_distribution_fig(X: pd.DataFrame, y: pd.Series, max_genes: int = 2000) -> go.Figure:
    """Per-sample distribution of log2 expression (box plot), coloured by class."""
    rng = np.random.default_rng(RANDOM_STATE)
    cols = X.columns
    if len(cols) > max_genes:
        cols = rng.choice(cols, size=max_genes, replace=False)
    long = X[cols].stack().rename("Expression").reset_index()
    long.columns = ["Sample", "Gene", "Expression"]
    long["Class"] = long["Sample"].map(_names(y))
    order = _names(y).sort_values().index.tolist()  # groups samples of the same class together
    fig = px.box(
        long, x="Sample", y="Expression", color="Class", color_discrete_map=CLASS_COLORS,
        category_orders={"Sample": order}, title=f"Gene-expression distribution per sample ({len(cols)} random genes)",
        points=False,
    )
    fig.update_layout(xaxis_tickangle=-60, xaxis_title="")
    return fig


def expression_histogram_fig(X: pd.DataFrame, y: pd.Series, max_genes: int = 2000) -> go.Figure:
    """Overlaid histograms of gene means for the two classes."""
    means = X.groupby(_names(y)).mean().T
    if len(means) > max_genes:
        means = means.sample(max_genes, random_state=RANDOM_STATE)
    long = means.melt(var_name="Class", value_name="Mean expression")
    fig = px.histogram(
        long, x="Mean expression", color="Class", nbins=60, barmode="overlay", opacity=0.6,
        color_discrete_map=CLASS_COLORS, title="Distribution of gene mean expression by class",
    )
    return fig


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def compute_pca(X: pd.DataFrame, n_components: int = 3, keep_fraction: float = 0.5) -> Tuple[pd.DataFrame, np.ndarray]:
    """PCA on standardised, variance-filtered genes. Descriptive only (uses all samples)."""
    Xf = impute_median(filter_low_variance(X, keep_fraction))
    n_components = int(min(n_components, Xf.shape[0], Xf.shape[1]))
    Z = StandardScaler().fit_transform(Xf.to_numpy())
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)
    df = pd.DataFrame(coords, index=X.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    return df, pca.explained_variance_ratio_


def pca_fig(coords: pd.DataFrame, explained: np.ndarray, y: pd.Series, three_d: bool = False) -> go.Figure:
    df = coords.copy()
    df["Class"] = _names(y).reindex(df.index)
    df["Sample"] = df.index
    labels = {c: f"{c} ({explained[i] * 100:.1f}%)" for i, c in enumerate(coords.columns)}
    if three_d and coords.shape[1] >= 3:
        fig = px.scatter_3d(
            df, x="PC1", y="PC2", z="PC3", color="Class", hover_name="Sample",
            color_discrete_map=CLASS_COLORS, labels=labels, title="PCA of gene expression",
        )
    else:
        fig = px.scatter(
            df, x="PC1", y="PC2", color="Class", hover_name="Sample",
            color_discrete_map=CLASS_COLORS, labels=labels, title="PCA of gene expression",
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    return fig


# --------------------------------------------------------------------------- #
# Correlation / clustering
# --------------------------------------------------------------------------- #
def top_variable_genes(X: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    var = X.var(axis=0, skipna=True).sort_values(ascending=False).head(n)
    return pd.DataFrame({"Gene": var.index, "Variance": var.values})


def top_variable_genes_fig(X: pd.DataFrame, n: int = 25) -> go.Figure:
    tv = top_variable_genes(X, n).iloc[::-1]
    fig = px.bar(tv, x="Variance", y="Gene", orientation="h", title=f"Top {n} most variable genes")
    fig.update_layout(height=max(400, 20 * n + 120))
    return fig


def correlation_matrix(X: pd.DataFrame, top_n_genes: int = 1000) -> pd.DataFrame:
    """Sample-by-sample Pearson correlation using the most variable genes."""
    genes = X.var(axis=0).sort_values(ascending=False).head(top_n_genes).index
    return impute_median(X[genes]).T.corr()


def correlation_heatmap_fig(corr: pd.DataFrame, y: pd.Series) -> go.Figure:
    order = _names(y).reindex(corr.index).sort_values().index
    c = corr.loc[order, order]
    labels = [f"{'PD' if y[i] == 1 else 'Ctrl'} | {i}" for i in c.index]
    fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=labels, y=labels, colorscale="RdBu_r", zmid=float(c.to_numpy().mean()),
                               colorbar=dict(title="r")))
    fig.update_layout(title="Sample-sample correlation (top variable genes; samples grouped by class)",
                      height=650, yaxis=dict(autorange="reversed"))
    return fig


def sample_clustermap(X: pd.DataFrame, y: pd.Series, top_n_genes: int = 500):
    """Hierarchically clustered sample-sample correlation heatmap (seaborn clustermap).

    Returns a matplotlib Figure. Row/column colour bars show the class of each sample.
    """
    import seaborn as sns

    corr = correlation_matrix(X, top_n_genes)
    palette = {CLASS_NAMES[0]: CLASS_COLORS["Control"], CLASS_NAMES[1]: CLASS_COLORS["Parkinson's Disease"]}
    colors = _names(y).reindex(corr.index).map(palette)
    grid = sns.clustermap(
        corr, method="average", metric="euclidean", cmap="vlag",
        row_colors=colors, col_colors=colors, figsize=(8, 8),
        xticklabels=False, yticklabels=False, cbar_kws={"label": "Pearson r"},
    )
    grid.fig.suptitle("Hierarchical clustering of samples (blue = control, red = Parkinson's)", y=1.02)
    return grid.fig
