"""Machine-learning models, cross-validation, persistence and the training orchestrator.

Leakage-safe design
-------------------
Every model is wrapped in a scikit-learn Pipeline:

    SimpleImputer(median)  ->  VarianceFilter  ->  TopKSelector(K genes)  ->  StandardScaler  ->  classifier

In cross-validation the *whole* pipeline is rebuilt and fitted on each training fold, so median
imputation, variance filtering, gene selection and scaling never see the held-out samples.

Deployment model
----------------
After evaluation, the gene selection is run once on all samples to obtain the final K genes. A compact
"deployment pipeline" (imputer -> scaler -> classifier) is then trained on those K genes only. It is
functionally the tail of the full pipeline, but it only needs the K selected genes at prediction time,
which is what makes CSV upload practical.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .feature_selection import (
    DEFAULT_VARIANCE_KEEP_FRACTION, TopKSelector, VarianceFilter, get_selected_genes,
)
from .utils import (
    CLASS_NAMES, MODEL_BUNDLE_PATH, PARKINSON, RANDOM_STATE, TRAINING_RESULTS_PATH,
    ProjectError, ensure_dirs, get_logger, set_global_seed,
)

logger = get_logger(__name__)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "SVM"]
COMPARE_ALL = "Compare All"
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Specificity"]


class ModelLoadError(ProjectError):
    """Raised when a saved model file is missing, corrupted or incompatible."""


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
def get_classifier(name: str):
    """Return an unfitted classifier with sensible defaults for tiny, high-dimensional data."""
    if name == "Logistic Regression":
        # Default L2 regularisation with a fairly strong C=0.1 keeps coefficients stable when p >> n.
        return LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=1,  # n_jobs=1: tiny data, parallel overhead dominates
        )
    if name == "SVM":
        # Linear kernel: the standard choice for p >> n and it exposes coefficients for explanation.
        return SVC(
            kernel="linear", C=1.0, probability=True, class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(
    model_name: str,
    n_features: int = 50,
    variance_keep_fraction: float = DEFAULT_VARIANCE_KEEP_FRACTION,
    with_selection: bool = True,
) -> Pipeline:
    """Create the (unfitted) modelling pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_selection:
        steps += [
            ("variance", VarianceFilter(keep_fraction=variance_keep_fraction)),
            ("select", TopKSelector(k=n_features)),
        ]
    steps += [("scaler", StandardScaler()), ("clf", get_classifier(model_name))]
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    """Accuracy, precision, recall (sensitivity), F1, ROC-AUC and specificity."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": auc,
        "Specificity": specificity,
    }


def cross_validate_model(
    X: pd.DataFrame, y: pd.Series, model_name: str, n_features: int = 50, n_splits: int = 5
) -> Dict[str, Any]:
    """Stratified K-fold CV with feature selection nested inside every training fold."""
    y = y.reindex(X.index)
    smallest_class = int(y.value_counts().min())
    if smallest_class < 2:
        raise ProjectError("Each class needs at least two samples for cross-validation.")
    n_splits = int(min(n_splits, smallest_class))
    n_features = int(min(n_features, X.shape[1]))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: List[Dict[str, Any]] = []
    oof_parts: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        pipe = build_pipeline(model_name, n_features)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_test = y.iloc[test_idx]
        score = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = pipe.predict(X.iloc[test_idx])
        fold_rows.append({"Model": model_name, "Fold": fold, **compute_metrics(y_test, pred, score)})
        oof_parts.append(pd.DataFrame({
            "Sample": X.index[test_idx], "Fold": fold, "True": y_test.to_numpy(),
            "Predicted": pred, "Score_PD": score,
        }))

    fold_df = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    summary: Dict[str, Any] = {"Model": model_name, "N_Features": n_features, "CV_Folds": n_splits}
    for metric in METRIC_COLUMNS:
        summary[f"{metric}_Mean"] = float(np.nanmean(fold_df[metric]))
        summary[f"{metric}_Std"] = float(np.nanstd(fold_df[metric], ddof=1)) if len(fold_df) > 1 else 0.0

    # Pooled out-of-fold predictions (each sample predicted exactly once by a model that never saw it).
    pooled = compute_metrics(oof["True"], oof["Predicted"], oof["Score_PD"])
    summary.update({f"Pooled_{k}": float(v) for k, v in pooled.items()})
    return {"fold_metrics": fold_df, "oof": oof, "summary": summary}


# --------------------------------------------------------------------------- #
# Final model
# --------------------------------------------------------------------------- #
def train_final_model(X: pd.DataFrame, y: pd.Series, model_name: str, n_features: int = 50) -> Dict[str, Any]:
    """Select genes on all data, then fit the compact deployment pipeline on those genes."""
    y = y.reindex(X.index)
    n_features = int(min(n_features, X.shape[1]))

    # Only the selection steps (imputer -> variance -> top-K) are needed to find the genes.
    selector_pipeline = build_pipeline(model_name, n_features)[:-2]
    selector_pipeline.fit(X, y)
    selected = get_selected_genes(selector_pipeline, X.columns)

    deploy = build_pipeline(model_name, n_features, with_selection=False)
    deploy.fit(X[selected], y)
    return {"model_name": model_name, "pipeline": deploy, "selected_features": selected}


def make_bundle(final: Dict[str, Any], config: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble everything needed to reproduce predictions (spec: model, scaler, features, config, metadata)."""
    pipeline = final["pipeline"]
    return {
        "model_name": final["model_name"],
        "model": pipeline,
        "scaler": pipeline.named_steps["scaler"],
        "selected_features": list(final["selected_features"]),
        "feature_selection_config": {
            "variance_keep_fraction": DEFAULT_VARIANCE_KEEP_FRACTION,
            "selector": "ANOVA F-test (top-K)",
            "n_features": len(final["selected_features"]),
        },
        "metadata": {
            **metadata,
            "config": config,
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "disclaimer": "Research/educational model; not a clinical diagnostic system.",
        },
    }


def save_model_bundle(bundle: Dict[str, Any], path: Path = MODEL_BUNDLE_PATH) -> Path:
    ensure_dirs()
    joblib.dump(bundle, path, compress=3)
    return path


def load_model_bundle(path: Path = MODEL_BUNDLE_PATH) -> Dict[str, Any]:
    """Load and validate a saved bundle. Raises ModelLoadError with a user-safe message."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError("No trained model found. Open the sidebar and click 'Train / Retrain Models'.")
    try:
        bundle = joblib.load(path)
    except Exception:
        raise ModelLoadError(
            "The saved model file is corrupted or was created with an incompatible library version. "
            "Please click 'Train / Retrain Models' to rebuild it."
        ) from None
    required = {"model", "scaler", "selected_features", "feature_selection_config", "metadata", "model_name"}
    if not isinstance(bundle, dict) or not required <= set(bundle):
        raise ModelLoadError("The saved model file has an unexpected structure. Please retrain the models.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ModelLoadError("The saved model cannot produce probabilities. Please retrain the models.")
    return bundle


def predict_samples(bundle: Dict[str, Any], aligned: pd.DataFrame) -> pd.DataFrame:
    """Predict for samples x selected_genes (log2 scale). Column order is enforced from the bundle."""
    features = bundle["selected_features"]
    missing = [g for g in features if g not in aligned.columns]
    if missing:
        raise ProjectError(f"Input is missing {len(missing)} model genes (e.g. {missing[:5]}).")
    data = aligned[features]
    if data.shape[1] != len(features):
        raise ProjectError("Incompatible data dimensions for the trained model.")
    try:
        proba = bundle["model"].predict_proba(data)
    except Exception:
        raise ProjectError(
            "The model could not score this data (incompatible dimensions or invalid values)."
        ) from None
    p_pd = proba[:, 1]
    return pd.DataFrame({
        "Sample": aligned.index.astype(str),
        "Probability_Parkinsons": p_pd,
        "Probability_Control": 1.0 - p_pd,
        "Predicted_Class": np.where(p_pd >= 0.5, CLASS_NAMES[PARKINSON], CLASS_NAMES[0]),
    })


# --------------------------------------------------------------------------- #
# Results persistence
# --------------------------------------------------------------------------- #
def save_training_results(results: Dict[str, Any], path: Path = TRAINING_RESULTS_PATH) -> None:
    ensure_dirs()
    joblib.dump(results, path, compress=3)


def load_training_results(path: Path = TRAINING_RESULTS_PATH) -> Optional[Dict[str, Any]]:
    """Return saved results, or None if absent/corrupted (never raises)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        results = joblib.load(path)
        if isinstance(results, dict) and {"cv_summary", "final", "importance", "consensus"} <= set(results):
            return results
    except Exception as exc:
        logger.warning("Could not load saved training results: %s", exc)
    return None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    de_results: pd.DataFrame,
    model_choice: str = COMPARE_ALL,
    n_features: int = 50,
    n_splits: int = 5,
    permutation_repeats: int = 10,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Cross-validate, fit final models, compute gene importance and save everything.

    Returns the results dictionary consumed by the Streamlit pages.
    """
    from . import explainability as xai  # local import avoids a circular dependency

    set_global_seed(RANDOM_STATE)
    if model_choice != COMPARE_ALL and model_choice not in MODEL_NAMES:
        raise ProjectError(f"Unknown model choice '{model_choice}'.")
    models = MODEL_NAMES if model_choice == COMPARE_ALL else [model_choice]
    y = y.reindex(X.index)
    n_features = int(min(n_features, X.shape[1]))

    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(min(max(fraction, 0.0), 1.0), message)

    # Permutation importance costs (features x repeats) model evaluations; use fewer repeats for large K.
    repeats = int(permutation_repeats if n_features <= 100 else max(3, permutation_repeats // 2))
    total_steps = len(models) * 2 + 2
    step = 0
    cv_rows, fold_frames, oof = [], [], {}
    final: Dict[str, Dict[str, Any]] = {}
    perm_importance: Dict[str, pd.DataFrame] = {}

    for name in models:
        report(step / total_steps, f"Cross-validating {name}...")
        cv = cross_validate_model(X, y, name, n_features, n_splits)
        cv_rows.append(cv["summary"])
        fold_frames.append(cv["fold_metrics"])
        oof[name] = cv["oof"]
        step += 1

        report(step / total_steps, f"Fitting final {name} and computing permutation importance...")
        model = train_final_model(X, y, name, n_features)
        final[name] = model
        Xs = X[model["selected_features"]]
        transformed = model["pipeline"][:-1].transform(Xs)
        perm_importance[name] = xai.compute_permutation_importance(
            model["pipeline"].named_steps["clf"], transformed, y.to_numpy(),
            model["selected_features"], n_repeats=repeats,
        )
        step += 1

    cv_summary = pd.DataFrame(cv_rows)
    primary = (
        cv_summary.sort_values(["ROC_AUC_Mean", "F1_Mean"], ascending=False, na_position="last")
        .iloc[0]["Model"]
    )

    report(step / total_steps, "Computing feature importance...")
    importance: Dict[str, Any] = {"Permutation": perm_importance}
    if "Logistic Regression" in final:
        m = final["Logistic Regression"]
        importance["Logistic Regression"] = xai.logistic_regression_importance(m["pipeline"], m["selected_features"])
    if "Random Forest" in final:
        m = final["Random Forest"]
        importance["Random Forest"] = xai.random_forest_importance(m["pipeline"], m["selected_features"])
    if "SVM" in final:
        m = final["SVM"]
        importance["SVM"] = xai.logistic_regression_importance(m["pipeline"], m["selected_features"])

    report((step + 0.5) / total_steps, "Computing SHAP values (if available)...")
    shap_result, shap_message = None, ""
    try:
        shap_result = xai.compute_shap(final[primary], X, y)
    except xai.ShapUnavailable as exc:
        shap_message = str(exc)
    importance["SHAP"] = shap_result
    step += 1

    report((step + 0.5) / total_steps, "Building consensus ranking...")
    consensus, consensus_note = xai.compute_consensus_ranking(
        de_results,
        rf_importance=importance.get("Random Forest"),
        lr_importance=importance.get("Logistic Regression"),
        permutation_importance=perm_importance.get(primary),
    )

    config = {
        "model_choice": model_choice, "n_features": n_features, "cv_folds": int(cv_summary["CV_Folds"].iloc[0]),
        "permutation_repeats": repeats, "random_state": RANDOM_STATE,
    }
    results = {
        "config": {**config, "trained_at": dt.datetime.now().isoformat(timespec="seconds")},
        "models": models,
        "primary_model": primary,
        "cv_summary": cv_summary,
        "cv_folds": pd.concat(fold_frames, ignore_index=True),
        "oof": oof,
        "final": final,
        "importance": importance,
        "shap_message": shap_message,
        "consensus": consensus,
        "consensus_note": consensus_note,
        "input_shape": tuple(X.shape),
    }

    if save:
        metadata = {
            **(dataset_metadata or {}),
            "n_training_samples": int(X.shape[0]),
            "cv_summary": cv_summary[cv_summary["Model"] == primary].iloc[0].to_dict(),
        }
        bundle = make_bundle(final[primary], config, metadata)
        save_model_bundle(bundle)
        save_training_results(results)

    report(1.0, "Done")
    return results


def get_model_choices() -> List[str]:
    return MODEL_NAMES + [COMPARE_ALL]
