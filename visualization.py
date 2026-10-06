"""Interactive Plotly visualisations for differential expression and model results."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from .utils import CLASS_COLORS, CLASS_NAMES

REGULATION_COLORS = {"Up in PD": "#D1495B", "Down in PD": "#2E86AB", "Not significant": "#B8B8B8"}


# --------------------------------------------------------------------------- #
# Differential expression
# --------------------------------------------------------------------------- #
def volcano_plot(de: pd.DataFrame, max_adj_p: float = 0.05, min_abs_log2fc: float = 0.5,
                 label_top: int = 10) -> go.Figure:
    """Volcano plot: x = log2 fold change, y = -log10(adjusted p). Significant genes are coloured."""
    df = de.copy()
    df["neg_log10_adj_p"] = -np.log10(df["Adjusted_P_Value"].clip(lower=1e-300))
    sig = (df["Adjusted_P_Value"] <= max_adj_p) & (df["Log2_Fold_Change"].abs() >= min_abs_log2fc)
    df["Regulation"] = np.where(
        sig & (df["Log2_Fold_Change"] > 0), "Up in PD",
        np.where(sig & (df["Log2_Fold_Change"] < 0), "Down in PD", "Not significant"),
    )
    fig = px.scatter(
        df, x="Log2_Fold_Change", y="neg_log10_adj_p", color="Regulation",
        color_discrete_map=REGULATION_COLORS, hover_name="Gene",
        hover_data={"Log2_Fold_Change": ":.3f", "Adjusted_P_Value": ":.2e", "neg_log10_adj_p": False},
        labels={"Log2_Fold_Change": "log2 fold change (PD vs control)", "neg_log10_adj_p": "-log10(adjusted p-value)"},
        title="Volcano plot", render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.add_hline(y=-np.log10(max_adj_p), line_dash="dash", line_color="grey")
    fig.add_vline(x=min_abs_log2fc, line_dash="dash", line_color="grey")
    fig.add_vline(x=-min_abs_log2fc, line_dash="dash", line_color="grey")

    top = df[sig].sort_values("neg_log10_adj_p", ascending=False).head(label_top)
    if not top.empty:
        fig.add_trace(go.Scatter(
            x=top["Log2_Fold_Change"], y=top["neg_log10_adj_p"], mode="text", text=top["Gene"],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))
    return fig


def top_genes_bar(table: pd.DataFrame, value_col: str, n: int = 20, title: str = "Top genes",
                  color_col: Optional[str] = None) -> go.Figure:
    """Horizontal bar chart of the top-n genes by `value_col`."""
    top = table.head(n).iloc[::-1]
    if color_col and color_col in top:
        fig = px.bar(top, x=value_col, y="Gene", orientation="h", color=color_col,
                     color_continuous_scale="RdBu_r", color_continuous_midpoint=0, title=title)
    else:
        fig = px.bar(top, x=value_col, y="Gene", orientation="h", title=title)
    fig.update_layout(height=max(400, 22 * len(top) + 140), yaxis_title="")
    return fig


def gene_boxplot(X: pd.DataFrame, y: pd.Series, gene: str) -> go.Figure:
    """Control vs Parkinson's expression for one gene (box + individual points)."""
    df = pd.DataFrame({
        "Expression": X[gene].to_numpy(),
        "Class": y.reindex(X.index).map(CLASS_NAMES).to_numpy(),
        "Sample": X.index,
    })
    fig = px.box(
        df, x="Class", y="Expression", color="Class", points="all", hover_name="Sample",
        color_discrete_map=CLASS_COLORS, category_orders={"Class": [CLASS_NAMES[0], CLASS_NAMES[1]]},
        title=f"{gene} expression (log2 scale)",
    )
    fig.update_layout(showlegend=False)
    return fig


def expression_heatmap(X: pd.DataFrame, y: pd.Series, genes: List[str]) -> go.Figure:
    """Gene-wise z-scored heatmap of selected genes; samples ordered by class."""
    genes = [g for g in genes if g in X.columns]
    order = y.reindex(X.index).sort_values().index  # controls first
    sub = X.loc[order, genes]
    z = (sub - sub.mean(axis=0)) / sub.std(axis=0).replace(0, 1)
    labels = [f"{'PD' if y[i] == 1 else 'Ctrl'} | {i}" for i in z.index]
    fig = go.Figure(go.Heatmap(
        z=z.T.to_numpy(), x=labels, y=genes, colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="z-score"),
    ))
    fig.update_layout(title="Expression of top genes across samples (z-score per gene)",
                      height=max(450, 18 * len(genes) + 200), yaxis=dict(autorange="reversed"), xaxis_tickangle=-60)
    return fig


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
def confusion_matrix_fig(y_true, y_pred, title: str = "Confusion matrix (pooled out-of-fold)") -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    names = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig = px.imshow(
        cm, x=[f"Predicted {n}" for n in names], y=[f"Actual {n}" for n in names],
        text_auto=True, color_continuous_scale="Blues", title=title,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def roc_curve_fig(oof_by_model: Dict[str, pd.DataFrame]) -> go.Figure:
    """ROC curves from pooled out-of-fold scores, one line per model."""
    fig = go.Figure()
    for name, oof in oof_by_model.items():
        fpr, tpr, _ = roc_curve(oof["True"], oof["Score_PD"])
        auc = roc_auc_score(oof["True"], oof["Score_PD"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc:.2f})", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="ROC curve (pooled out-of-fold predictions)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=480)
    return fig


def model_comparison_fig(cv_summary: pd.DataFrame,
                         metrics: Optional[List[str]] = None) -> go.Figure:
    """Grouped bars of mean cross-validated metrics with std error bars."""
    metrics = metrics or ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Specificity"]
    fig = go.Figure()
    for _, row in cv_summary.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"], x=[m.replace("_", "-") for m in metrics],
            y=[row[f"{m}_Mean"] for m in metrics],
            error_y=dict(type="data", array=[row[f"{m}_Std"] for m in metrics]),
        ))
    fig.update_layout(barmode="group", title="Cross-validated model comparison (mean +/- SD across folds)",
                      yaxis=dict(range=[0, 1.05], title="Score"), height=480)
    return fig


def shap_sample_fig(shap_values: pd.DataFrame, sample: str, top_n: int = 15) -> go.Figure:
    """Bar chart of SHAP values for one training sample."""
    row = shap_values.loc[sample]
    top = row.reindex(row.abs().sort_values(ascending=False).head(top_n).index).iloc[::-1]
    colors = ["#D1495B" if v > 0 else "#2E86AB" for v in top.values]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=colors))
    fig.update_layout(title=f"SHAP explanation for {sample} (red -> toward Parkinson's class)",
                      xaxis_title="SHAP value", height=max(380, 24 * len(top) + 140))
    return fig


def contribution_bar(contrib: pd.DataFrame, title: str = "Top contributing genes for this sample") -> go.Figure:
    df = contrib.iloc[::-1]
    colors = ["#D1495B" if v > 0 else "#2E86AB" for v in df["Contribution"]]
    fig = go.Figure(go.Bar(x=df["Contribution"], y=df["Gene"], orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title="Contribution (red = toward Parkinson's class)",
                      height=max(380, 24 * len(df) + 140))
    return fig


def probability_gauge(p_pd: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_pd * 100,
        number={"suffix": "%"}, title={"text": "Model probability: Parkinson's class"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#D1495B"},
               "steps": [{"range": [0, 50], "color": "#DCEBF3"}, {"range": [50, 100], "color": "#F6DADD"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10))
    return fig
