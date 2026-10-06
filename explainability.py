"""Explainable AI: which genes drive the classification?

Four complementary views are produced, then combined into one consensus ranking:

A. Logistic-regression coefficients   (signed, model-specific)
B. Random-forest impurity importance  (model-specific)
C. Permutation importance             (model-agnostic; sklearn.inspection.permutation_importance)
D. SHAP values                        (TreeExplainer for RF, LinearExplainer for linear models)

All importances describe how the *model* uses genes in this dataset. They are computational
associations, not evidence that a gene causes Parkinson's disease.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sk_permutation_importance

from .utils import CONSENSUS_LABEL, RANDOM_STATE, minmax


class ShapUnavailable(Exception):
    """Raised when SHAP cannot be used (not installed, version problem, unsupported model)."""


# --------------------------------------------------------------------------- #
# A. Logistic regression / linear SVM coefficients
# --------------------------------------------------------------------------- #
def logistic_regression_importance(pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    """Coefficients of a linear classifier on standardised genes.

    Positive coefficient -> higher expression pushes the prediction toward the Parkinson's class.
    Negative coefficient -> higher expression pushes the prediction toward the control class.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise ValueError("This classifier has no linear coefficients.")
    coef = np.ravel(clf.coef_)
    df = pd.DataFrame({"Gene": list(feature_names), "Coefficient": coef})
    df["Abs_Importance"] = df["Coefficient"].abs()
    df["Direction"] = np.where(df["Coefficient"] > 0, "Toward Parkinson's class", "Toward control class")
    return df.sort_values("Abs_Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# B. Random forest importance
# --------------------------------------------------------------------------- #
def random_forest_importance(pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError("This classifier has no feature_importances_.")
    df = pd.DataFrame({"Gene": list(feature_names), "Importance": clf.feature_importances_})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# C. Permutation importance
# --------------------------------------------------------------------------- #
def compute_permutation_importance(
    clf, X_transformed: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    n_repeats: int = 10, scoring: str = "neg_brier_score",
) -> pd.DataFrame:
    """Mean and standard deviation of the loss in model quality when one gene is shuffled.

    Scoring uses the (negative) Brier score, i.e. the squared error of the predicted probability, so
    importance = increase in Brier score after shuffling the gene. A probability-based score is used
    on purpose: on only 25 samples the fitted models separate the training data perfectly, so ROC-AUC
    or accuracy would not change when a single gene is shuffled and every importance would be zero.

    Computed on the fitted classifier and the (scaled) training samples. With so few samples a separate
    hold-out set is not available, so this is an in-sample importance: it shows which genes the fitted
    model relies on, not how well it generalises (that is what cross-validation measures).
    """
    result = sk_permutation_importance(
        clf, X_transformed, y, scoring=scoring, n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    df = pd.DataFrame({
        "Gene": list(feature_names),
        "Mean_Importance": result.importances_mean,
        "Std_Importance": result.importances_std,
    })
    return df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# D. SHAP
# --------------------------------------------------------------------------- #
def _positive_class_values(shap_values) -> np.ndarray:
    """Normalise the different shapes returned by SHAP versions to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def compute_shap(final_model: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """SHAP values (positive class) for the deployed model on the training samples.

    Raises ShapUnavailable with a readable message if anything goes wrong, so the caller can fall
    back to permutation importance.
    """
    try:
        import shap
    except Exception as exc:
        raise ShapUnavailable(f"SHAP is not available in this environment ({type(exc).__name__}).") from None

    name = final_model["model_name"]
    pipeline = final_model["pipeline"]
    features = list(final_model["selected_features"])
    clf = pipeline.named_steps["clf"]

    try:
        Xt = pipeline[:-1].transform(X[features])
        if name == "Random Forest":
            explainer = shap.TreeExplainer(clf)
            values = explainer.shap_values(Xt, check_additivity=False)
        elif hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer((np.ravel(clf.coef_), np.ravel(clf.intercept_)), Xt)
            values = explainer.shap_values(Xt)
        else:
            raise ShapUnavailable(f"SHAP is not implemented for {name}.")
        values = _positive_class_values(values)
        if values.shape != Xt.shape:
            raise ShapUnavailable("Unexpected SHAP output shape.")
    except ShapUnavailable:
        raise
    except Exception as exc:
        raise ShapUnavailable(f"SHAP could not be computed ({type(exc).__name__}: {exc}).") from None

    shap_df = pd.DataFrame(values, index=X.index, columns=features)
    scaled_df = pd.DataFrame(Xt, index=X.index, columns=features)
    importance = pd.DataFrame({
        "Gene": features,
        "Mean_Abs_SHAP": np.abs(values).mean(axis=0),
        "Mean_SHAP": values.mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    return {
        "model_name": name, "importance": importance,
        "shap_values": shap_df, "scaled_data": scaled_df,
        "labels": y.reindex(X.index).to_numpy(),
    }


def shap_summary_figure(shap_result: Dict[str, Any], max_display: int = 20):
    """Matplotlib SHAP beeswarm summary plot; returns a Figure."""
    import matplotlib.pyplot as plt
    import shap

    plt.close("all")
    shap.summary_plot(
        shap_result["shap_values"].to_numpy(),
        shap_result["scaled_data"],
        feature_names=list(shap_result["shap_values"].columns),
        max_display=max_display, show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(8, max(4, 0.32 * max_display + 1.5))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Explain one new sample (Prediction page)
# --------------------------------------------------------------------------- #
def explain_single_sample(bundle: Dict[str, Any], sample_row: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Per-gene contribution of one sample to the model's prediction.

    * Linear models (Logistic Regression, linear SVM): exact contribution = coefficient x standardised value.
    * Random forest: SHAP TreeExplainer when available, otherwise an occlusion analysis (change in
      predicted probability when a gene is reset to the training mean).
    Positive contributions push toward the Parkinson's class.
    """
    pipeline = bundle["model"]
    features = list(bundle["selected_features"])
    clf = pipeline.named_steps["clf"]
    x_df = pd.DataFrame([sample_row.reindex(features).to_numpy(dtype=float)], columns=features)
    scaled = pipeline[:-1].transform(x_df)  # 1 x k

    method = ""
    if hasattr(clf, "coef_"):
        contribution = np.ravel(clf.coef_) * scaled[0]
        method = "coefficient x standardised expression"
    else:
        contribution = None
        try:
            import shap
            values = shap.TreeExplainer(clf).shap_values(scaled, check_additivity=False)
            contribution = _positive_class_values(values)[0]
            method = "SHAP (TreeExplainer)"
        except Exception:
            contribution = None
        if contribution is None:
            base = clf.predict_proba(scaled)[0, 1]
            occluded = np.repeat(scaled, len(features), axis=0)
            occluded[np.arange(len(features)), np.arange(len(features))] = 0.0  # 0 = training mean
            contribution = base - clf.predict_proba(occluded)[:, 1]
            method = "occlusion (change in P(Parkinson's) when gene set to training mean)"

    df = pd.DataFrame({
        "Gene": features,
        "Expression": x_df.iloc[0].to_numpy(),
        "Standardised_Expression": scaled[0],
        "Contribution": contribution,
    })
    df["Direction"] = np.where(df["Contribution"] > 0, "Toward Parkinson's class", "Toward control class")
    df["Method"] = method
    df["_abs"] = df["Contribution"].abs()
    return df.sort_values("_abs", ascending=False).drop(columns="_abs").head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Consensus ranking
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {"DE": 0.30, "RF": 0.25, "Permutation": 0.25, "LR": 0.20}


def compute_consensus_ranking(
    de_results: pd.DataFrame,
    rf_importance: Optional[pd.DataFrame] = None,
    lr_importance: Optional[pd.DataFrame] = None,
    permutation_importance: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Combine differential expression and model importances into one ranking.

    Each component is min-max normalised to [0, 1] across genes:
        DE score          = 0.5 * norm(-log10 adjusted p) + 0.5 * norm(|log2 FC|)
        RF score          = norm(random-forest importance)
        Permutation score = norm(mean permutation importance, negatives set to 0)
        LR score          = norm(|logistic-regression coefficient|)
    Genes that were not selected as model features receive 0 for the model-based components.
    Weights: 0.30 DE / 0.25 RF / 0.25 permutation / 0.20 LR (renormalised if a component is absent).

    This is a project-specific heuristic and is NOT biologically validated.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    df = de_results[["Gene", "Log2_Fold_Change", "Adjusted_P_Value"]].copy()

    sig = -np.log10(df["Adjusted_P_Value"].clip(lower=1e-300))
    df["DE_Score"] = 0.5 * minmax(sig.to_numpy()).to_numpy() + 0.5 * minmax(df["Log2_Fold_Change"].abs().to_numpy()).to_numpy()

    def attach(source: Optional[pd.DataFrame], column: str, out: str, transform=lambda s: s) -> bool:
        if source is None or source.empty:
            df[out + "_Raw"] = np.nan
            df[out + "_Score"] = 0.0
            return False
        lookup = source.set_index("Gene")[column]
        raw = df["Gene"].map(lookup).fillna(0.0)
        df[out + "_Raw"] = df["Gene"].map(lookup)
        df[out + "_Score"] = minmax(transform(raw).to_numpy()).to_numpy()
        return True

    available = {
        "DE": True,
        "RF": attach(rf_importance, "Importance", "RF"),
        "Permutation": attach(permutation_importance, "Mean_Importance", "Perm", lambda s: s.clip(lower=0)),
        "LR": attach(lr_importance, "Abs_Importance", "LR"),
    }
    active = {k: w for k, w in weights.items() if available.get(k)}
    total = sum(active.values())
    norm_w = {k: w / total for k, w in active.items()}

    df["Consensus_Score"] = (
        norm_w.get("DE", 0) * df["DE_Score"]
        + norm_w.get("RF", 0) * df["RF_Score"]
        + norm_w.get("Permutation", 0) * df["Perm_Score"]
        + norm_w.get("LR", 0) * df["LR_Score"]
    )
    df = df.sort_values(["Consensus_Score", "DE_Score"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    df = df.rename(columns={
        "RF_Raw": "RF_Importance", "Perm_Raw": "Permutation_Importance", "LR_Raw": "LR_Importance",
    })
    ordered = [
        "Rank", "Gene", "Log2_Fold_Change", "Adjusted_P_Value", "RF_Importance",
        "Permutation_Importance", "LR_Importance", "Consensus_Score",
        "DE_Score", "RF_Score", "Perm_Score", "LR_Score",
    ]
    df = df[ordered]

    used = ", ".join(f"{k} {w:.2f}" for k, w in norm_w.items())
    note = f"{CONSENSUS_LABEL} Weights used: {used}."
    return df, note


# --------------------------------------------------------------------------- #
# Helpers for the Gene Importance page
# --------------------------------------------------------------------------- #
def importance_table_for(method: str, results: Dict[str, Any], de: pd.DataFrame, model_name: Optional[str] = None
                         ) -> Tuple[pd.DataFrame, str]:
    """Return a tidy [Rank, Gene, Importance, Log2FC, Adjusted P] table for one method.

    Falls back to permutation importance when SHAP is unavailable. The second return value is a
    human-readable note about what was shown.
    """
    imp = results["importance"]
    primary = results["primary_model"]
    note = ""
    if method == "Logistic Regression":
        src = imp.get("Logistic Regression")
        if src is None:
            src = imp.get("SVM")
            note = "Logistic Regression was not trained; showing linear-SVM coefficients instead." if src is not None else ""
        if src is None:
            raise KeyError("Logistic Regression was not trained in this run.")
        table = src.rename(columns={"Abs_Importance": "Importance"})[["Gene", "Importance", "Coefficient", "Direction"]]
    elif method == "Random Forest":
        if "Random Forest" not in imp:
            raise KeyError("Random Forest was not trained in this run.")
        table = imp["Random Forest"][["Gene", "Importance"]]
    elif method == "SHAP":
        shap_res = imp.get("SHAP")
        if shap_res is None:
            note = (results.get("shap_message") or "SHAP is unavailable.") + " Showing permutation importance instead."
            return importance_table_for("Permutation Importance", results, de, model_name)[0], note
        table = shap_res["importance"].rename(columns={"Mean_Abs_SHAP": "Importance"})[["Gene", "Importance", "Mean_SHAP"]]
        note = f"Mean |SHAP| for the {shap_res['model_name']} model."
    elif method == "Permutation Importance":
        chosen = model_name if model_name in imp["Permutation"] else primary
        table = imp["Permutation"][chosen].rename(columns={"Mean_Importance": "Importance"})[
            ["Gene", "Importance", "Std_Importance"]]
        note = f"Mean increase in Brier score (probability error) when the gene is shuffled ({chosen}, in-sample)."
    elif method == "Consensus":
        cons = results["consensus"]
        table = cons.rename(columns={"Consensus_Score": "Importance"})[["Gene", "Importance"]]
        note = results.get("consensus_note", CONSENSUS_LABEL)
    else:
        raise KeyError(f"Unknown method {method}")

    table = table.merge(de[["Gene", "Log2_Fold_Change", "Adjusted_P_Value"]], on="Gene", how="left")
    table = table.sort_values("Importance", ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table, note
