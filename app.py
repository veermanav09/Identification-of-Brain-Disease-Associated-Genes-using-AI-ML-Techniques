"""Streamlit dashboard: Identification of Brain Disease-Associated Genes using AI/ML Techniques.

Run with:   streamlit run app.py

Research / educational demonstration only. NOT a clinical diagnostic system.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src import (
    data_loader, differential_expression as dea, eda, explainability as xai, ml_models,
    preprocessing as pp, visualization as viz,
)
from src.utils import (
    CLASS_NAMES, CONSENSUS_LABEL, CV_FOLD_OPTIONS, DISCLAIMER_TEXT, DISCLAIMER_TITLE,
    FEATURE_COUNT_OPTIONS, GSE_ID, GSE_PLATFORM, GSE_TITLE, GSE_URL, MODEL_BUNDLE_PATH,
    TRAINING_RESULTS_PATH, ProjectError, ensure_dirs, format_p, get_logger, set_global_seed,
    to_csv_bytes,
)

logger = get_logger("app")
set_global_seed()
ensure_dirs()

st.set_page_config(
    page_title="Brain Disease Gene AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded",
)

PAGES = [
    "Home", "Dataset Explorer", "Exploratory Analysis", "Differential Gene Analysis",
    "Machine Learning", "Gene Importance", "Gene Explorer", "Prediction",
]

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.flow-wrap {display:flex; flex-direction:column; align-items:center; gap:0; margin: 0.5rem 0 1rem 0;}
.flow-box {width: min(460px, 95%); text-align:center; padding: 10px 14px; border-radius: 10px;
           border: 1px solid rgba(46,134,171,0.55); background: rgba(46,134,171,0.12);
           font-weight: 600;}
.flow-box small {display:block; font-weight:400; opacity:0.75;}
.flow-arrow {font-size: 22px; line-height: 26px; opacity: 0.7;}
</style>
"""


# --------------------------------------------------------------------------- #
# Display helpers (compatible with old and new Streamlit APIs)
# --------------------------------------------------------------------------- #
def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def show_plot(fig, **kwargs) -> None:
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def disclaimer_banner(full: bool = False) -> None:
    if full:
        st.warning(f"**{DISCLAIMER_TITLE}**\n\n{DISCLAIMER_TEXT}", icon="⚠️")
    else:
        st.warning(
            f"**{DISCLAIMER_TITLE}** - not a medical diagnostic tool. Genes shown are computational "
            "associations within one public dataset and do not establish biological causality.",
            icon="⚠️",
        )


# --------------------------------------------------------------------------- #
# Cached data / model access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=f"Loading {GSE_ID} (downloaded from NCBI GEO on first run, then cached)...")
def load_dataset() -> Dict[str, Any]:
    """Download (once), cache and preprocess GSE7621. Cached across reruns and sessions."""
    raw = data_loader.load_raw_dataset()
    processed = pp.build_processed_dataset(raw)
    processed["gse_title"] = raw.get("title", "")
    processed["platform_id"] = raw.get("platform_id", "")
    return processed


@st.cache_data(show_spinner="Running differential expression...")
def compute_de(_X: pd.DataFrame, _y: pd.Series, fingerprint: str) -> pd.DataFrame:
    return dea.run_differential_expression(_X, _y)


@st.cache_data(show_spinner=False)
def compute_pca_cached(_X: pd.DataFrame, fingerprint: str):
    return eda.compute_pca(_X, n_components=3)


@st.cache_data(show_spinner=False)
def compute_corr_cached(_X: pd.DataFrame, fingerprint: str):
    return eda.correlation_matrix(_X, top_n_genes=1000)


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(mtime: float):
    """Load the saved deployable model once per file version (mtime is the cache key)."""
    return ml_models.load_model_bundle(MODEL_BUNDLE_PATH)


@st.cache_resource(show_spinner=False)
def _load_results_cached(mtime: float):
    return ml_models.load_training_results(TRAINING_RESULTS_PATH)


def get_bundle():
    """Return (bundle, error_message). Never raises."""
    if not MODEL_BUNDLE_PATH.exists():
        return None, None
    try:
        return _load_bundle_cached(MODEL_BUNDLE_PATH.stat().st_mtime), None
    except ml_models.ModelLoadError as exc:
        return None, str(exc)
    except Exception:
        logger.exception("Unexpected error while loading the model bundle")
        return None, "The saved model could not be loaded. Please click 'Train / Retrain Models'."


def get_training(X: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Training results for this session (memory) or from disk; None if unavailable/incompatible."""
    results = st.session_state.get("training")
    if results is None and TRAINING_RESULTS_PATH.exists():
        results = _load_results_cached(TRAINING_RESULTS_PATH.stat().st_mtime)
        if results is not None:
            st.session_state["training"] = results
    if results is not None and tuple(results.get("input_shape", ())) != tuple(X.shape):
        return None  # trained on different data
    return results


def fingerprint(X: pd.DataFrame) -> str:
    return f"{X.shape}-{float(np.nansum(X.to_numpy(dtype=float))):.4f}"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_training(ds: Dict[str, Any], de: pd.DataFrame, model_choice: str, n_features: int, folds: int) -> None:
    X, y = ds["X"], ds["y"]
    meta = {
        "dataset": GSE_ID, "platform": GSE_PLATFORM, "n_input_genes": int(X.shape[1]),
        "log2_transform_applied": bool(ds["report"]["log2_transform_applied"]),
        "input_scale": "log2 expression (GPL570 series matrix values)",
    }
    progress = st.progress(0.0, text="Starting...")

    def callback(fraction: float, message: str) -> None:
        progress.progress(fraction, text=message)

    try:
        results = ml_models.train_and_evaluate(
            X, y, de, model_choice=model_choice, n_features=n_features, n_splits=folds,
            dataset_metadata=meta, progress_callback=callback, save=True,
        )
    except ProjectError as exc:
        progress.empty()
        st.error(str(exc))
        return
    except Exception:
        progress.empty()
        logger.exception("Training failed")
        st.error("Training failed unexpectedly. Try fewer features or another model, then retry.")
        return
    progress.empty()
    st.session_state["training"] = results
    _load_bundle_cached.clear()
    _load_results_cached.clear()
    st.success(
        f"Training complete. Best model by cross-validated ROC-AUC: **{results['primary_model']}**. "
        "Results were saved to the models/ folder."
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def flow_diagram() -> None:
    steps = [
        ("GEO Gene Expression Dataset", f"{GSE_ID} - {GSE_PLATFORM.split(' - ')[0]}"),
        ("Data Cleaning", "labels from sample metadata"),
        ("Probe \u2192 Gene Mapping", "drop unmapped, aggregate duplicate probes"),
        ("Normalization / Transformation", "log2 only if needed"),
        ("Feature Filtering", "low-variance genes removed"),
        ("Differential Expression", "Welch t-test + Benjamini-Hochberg FDR"),
        ("Feature Selection", "top-K genes inside each CV fold"),
        ("Machine Learning Models", "Logistic Regression / Random Forest / SVM"),
        ("Model Evaluation", "stratified cross-validation"),
        ("Explainable AI", "coefficients, RF importance, permutation, SHAP"),
        ("Gene Importance Ranking", "project-specific consensus score"),
        ("Interactive Gene Explorer", "search, plots, upload and predict"),
    ]
    parts = ['<div class="flow-wrap">']
    for i, (title, sub) in enumerate(steps):
        parts.append(f'<div class="flow-box">{title}<small>{sub}</small></div>')
        if i < len(steps) - 1:
            parts.append('<div class="flow-arrow">\u2193</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def page_home(ds: Dict[str, Any]) -> None:
    st.title("Identification of Brain Disease\u2013Associated Genes using AI/ML Techniques")
    st.subheader("AI/ML-based analysis of gene-expression patterns associated with Parkinson's disease")
    disclaimer_banner(full=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Project objective")
        st.markdown(
            "Gene-expression microarrays measure tens of thousands of genes in only a handful of "
            "samples, which makes it hard to tell which genes are genuinely associated with a disease. "
            "This project applies a leakage-safe machine-learning workflow to public "
            "substantia-nigra data to:\n"
            "1. load, clean and map probes to genes;\n"
            "2. find genes that differ between Parkinson's disease and controls;\n"
            "3. train and compare Logistic Regression, Random Forest and SVM classifiers;\n"
            "4. rank predictive genes with explainable-AI methods (coefficients, feature importance, "
            "permutation importance, SHAP) and a consensus score;\n"
            "5. explore genes interactively and score uploaded expression data."
        )
        st.markdown("### Dataset")
        st.markdown(
            f"**[{GSE_ID}]({GSE_URL})** - {GSE_TITLE}.\n\n"
            "- Organism: *Homo sapiens*\n- Tissue: substantia nigra (post-mortem brain)\n"
            f"- Platform: {GSE_PLATFORM}\n"
            f"- Samples in this run: **{ds['X'].shape[0]}** "
            f"({int((ds['y'] == 1).sum())} Parkinson's disease, {int((ds['y'] == 0).sum())} control)\n"
            f"- Genes after probe mapping and filtering: **{ds['X'].shape[1]:,}**"
        )
    with right:
        st.markdown("### Methodology")
        flow_diagram()


def page_dataset(ds: Dict[str, Any], X_desc: pd.DataFrame) -> None:
    st.header("Dataset Explorer")
    y, meta, report = ds["y"], ds["metadata"], ds["report"]
    overview = eda.dataset_overview(X_desc, y)

    st.markdown(
        f"**[{GSE_ID}]({GSE_URL})** - {GSE_TITLE}. Platform: {GSE_PLATFORM}. "
        "Data come from the processed GEO series (probe-level values), mapped to gene symbols here."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", overview["n_samples"])
    c2.metric("Genes", f"{overview['n_genes']:,}")
    c3.metric("Parkinson's samples", overview["n_disease"])
    c4.metric("Control samples", overview["n_control"])

    for message in ds.get("warnings", []):
        st.warning(message)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_plot(eda.class_distribution_fig(y))
    with col_b:
        st.markdown("#### Sample metadata")
        cols = [c for c in ["title", "source_name", "characteristics", "Class"] if c in meta.columns]
        show_df(meta[cols].reset_index())

    st.markdown("#### Expression statistics")
    s1, s2 = st.columns(2)
    with s1:
        st.caption("All expression values (log2 scale)")
        st.dataframe(eda.expression_statistics(X_desc))
    with s2:
        st.caption("Per-sample quality control")
        show_df(ds["qc"].reset_index().round(3))

    with st.expander("Preprocessing report (probe-to-gene mapping, transformation, missing values)"):
        st.markdown(report["note"])
        st.markdown(f"**Transformation:** {report['transformation_note']}")
        st.json({k: v for k, v in report.items() if k not in {"note", "transformation_note"}})


def page_eda(ds: Dict[str, Any], X_desc: pd.DataFrame) -> None:
    st.header("Exploratory Analysis")
    y = ds["y"]
    fp = fingerprint(X_desc)
    tab_pca, tab_corr, tab_dist, tab_cluster, tab_var = st.tabs(
        ["PCA", "Correlation heatmap", "Expression distributions", "Sample clustering", "Top variable genes"]
    )
    with tab_pca:
        coords, explained = compute_pca_cached(X_desc, fp)
        three_d = st.toggle("3-D view", value=False)
        show_plot(eda.pca_fig(coords, explained, y, three_d=three_d))
        st.caption(
            "PCA on standardised, variance-filtered genes. If Parkinson's and control samples separate, "
            "expression carries disease-related signal; overlap suggests the differences are subtle."
        )
    with tab_corr:
        show_plot(eda.correlation_heatmap_fig(compute_corr_cached(X_desc, fp), y))
    with tab_dist:
        show_plot(eda.expression_distribution_fig(X_desc, y))
        show_plot(eda.expression_histogram_fig(X_desc, y))
    with tab_cluster:
        try:
            fig = eda.sample_clustermap(X_desc, y)
            st.pyplot(fig)
        except Exception:
            logger.exception("clustermap failed")
            st.warning("The clustering heatmap could not be drawn in this environment.")
    with tab_var:
        n = st.slider("Number of genes", 10, 100, 25, step=5)
        show_plot(eda.top_variable_genes_fig(X_desc, n))


def page_de(ds: Dict[str, Any], X_desc: pd.DataFrame, de: pd.DataFrame) -> None:
    st.header("Differential Gene Analysis")
    st.caption(
        "Welch's t-test per gene on log2 expression, Benjamini-Hochberg FDR correction. "
        "Computed on all samples; descriptive, not part of the cross-validated performance estimate."
    )
    c1, c2, c3 = st.columns(3)
    max_adj_p = c1.select_slider(
        "Maximum adjusted p-value", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25], value=0.05)
    min_lfc = c2.slider("Minimum |log2 fold change|", 0.0, 3.0, 0.5, 0.1)
    top_n = c3.slider("Number of genes to display", 10, 500, 100, 10)

    sig = dea.filter_de_results(de, max_adj_p, min_lfc, top_n=10 ** 9)
    n_up = int((sig["Log2_Fold_Change"] > 0).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genes tested", f"{len(de):,}")
    m2.metric("Significant genes", f"{len(sig):,}")
    m3.metric("Up in Parkinson's", n_up)
    m4.metric("Down in Parkinson's", len(sig) - n_up)
    if len(sig) == 0:
        st.info("No genes pass these thresholds. Try a larger adjusted p-value or a smaller fold-change cutoff.")

    show_plot(viz.volcano_plot(de, max_adj_p, min_lfc))

    st.markdown("#### Significant genes")
    table = sig.head(top_n)
    show_df(table)
    st.download_button(
        "Download DE Results", to_csv_bytes(de), file_name="differential_expression_results.csv",
        mime="text/csv",
    )

    up = dea.top_regulated(de, "up", 10, max_adj_p, min_lfc)
    down = dea.top_regulated(de, "down", 10, max_adj_p, min_lfc)
    a, b = st.columns(2)
    cols = ["Gene", "Log2_Fold_Change", "Adjusted_P_Value"]
    with a:
        st.markdown("#### Top upregulated genes (higher in PD)")
        show_df(up[cols])
    with b:
        st.markdown("#### Top downregulated genes (lower in PD)")
        show_df(down[cols])

    with st.expander("Statistical assumptions and limitations"):
        st.markdown(
            "- Samples are assumed independent; age, sex, post-mortem interval, RNA quality and "
            "cell-type composition are not modelled.\n"
            "- The t-test assumes roughly normal values; with 9 vs 16 samples power is limited. "
            "Moderated tests (e.g. limma) are usually preferred for publication-grade microarray analysis.\n"
            "- Statistical significance indicates an expression difference in this dataset, not causality."
        )


def page_ml(ds: Dict[str, Any], results: Optional[Dict[str, Any]], model_choice: str, n_features: int,
            folds: int) -> None:
    st.header("Machine Learning")
    st.caption(
        "Pipeline per fold: median imputation \u2192 variance filter \u2192 top-K gene selection \u2192 "
        "standardisation \u2192 classifier. Selection is re-done inside every training fold, so held-out "
        "samples never influence which genes are chosen."
    )
    if results is None:
        st.info("No trained models yet. Choose settings in the sidebar and click **Train / Retrain Models**.")
        return

    cfg = results["config"]
    st.markdown(
        f"Trained with **{cfg['n_features']}** genes, **{cfg['cv_folds']}**-fold stratified cross-validation "
        f"({', '.join(results['models'])}); best model by ROC-AUC: **{results['primary_model']}**. "
        f"Trained at {cfg.get('trained_at', 'n/a')}."
    )
    if (cfg["n_features"], cfg["model_choice"], cfg["cv_folds"]) != (n_features, model_choice, folds):
        st.caption("Sidebar settings differ from the trained results shown here; click Train / Retrain Models to apply them.")

    summary = results["cv_summary"]
    display = pd.DataFrame({"Model": summary["Model"]})
    for metric in ml_models.METRIC_COLUMNS:
        display[metric.replace("_", "-")] = [
            f"{m:.3f} \u00b1 {s:.3f}" for m, s in zip(summary[f"{metric}_Mean"], summary[f"{metric}_Std"])
        ]
    st.markdown("#### Cross-validation results (mean \u00b1 SD across folds)")
    show_df(display)

    st.download_button(
        "Download Model Performance", to_csv_bytes(summary), file_name="model_performance.csv", mime="text/csv",
    )
    show_plot(viz.model_comparison_fig(summary))

    st.markdown("#### Confusion matrix and ROC curve")
    left, right = st.columns(2)
    with left:
        chosen = st.selectbox("Model", results["models"], index=results["models"].index(results["primary_model"]))
        oof = results["oof"][chosen]
        show_plot(viz.confusion_matrix_fig(oof["True"], oof["Predicted"], f"Confusion matrix - {chosen}"))
        row = summary[summary["Model"] == chosen].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{row['Accuracy_Mean']:.2f}")
        b.metric("F1", f"{row['F1_Mean']:.2f}")
        c.metric("ROC-AUC", f"{row['ROC_AUC_Mean']:.2f}")
    with right:
        show_plot(viz.roc_curve_fig(results["oof"]))

    with st.expander("Per-fold metrics"):
        show_df(results["cv_folds"].round(3))
    st.info(
        "Cross-validated scores describe how well the model separated these 25 post-mortem samples. "
        "High values do not imply clinical effectiveness; there is no independent validation cohort.",
        icon="ℹ️",
    )


def page_importance(ds: Dict[str, Any], X_desc: pd.DataFrame, de: pd.DataFrame,
                    results: Optional[Dict[str, Any]]) -> None:
    st.header("Gene Importance")
    if results is None:
        st.info("Train the models first (sidebar \u2192 Train / Retrain Models).")
        return

    methods = ["Consensus", "Logistic Regression", "Random Forest", "Permutation Importance", "SHAP"]
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Importance method", methods)
    top_n = c2.selectbox("Genes to show", [10, 20, 50], index=1)
    model_name = None
    if method == "Permutation Importance":
        options = list(results["importance"]["Permutation"].keys())
        model_name = c3.selectbox("Model", options, index=options.index(results["primary_model"]))

    try:
        table, note = xai.importance_table_for(method, results, de, model_name)
    except KeyError as exc:
        st.warning(f"{exc.args[0]} Train with 'Compare All' to enable every method.")
        return

    if method == "Consensus":
        st.info(f"**{CONSENSUS_LABEL}** {results.get('consensus_note', '')}")
    if note:
        st.caption(note)

    show_plot(viz.top_genes_bar(table, "Importance", top_n, f"Top {top_n} genes - {method}"))
    st.markdown("#### Top predictive genes")
    show_df(table.head(top_n))

    # Downloadable list uses the consensus ranking.
    consensus = results["consensus"]
    st.download_button(
        "Download Top Associated Genes", to_csv_bytes(consensus.head(max(top_n, 50))),
        file_name="top_associated_genes.csv", mime="text/csv",
    )

    st.markdown("#### Expression of top genes across samples")
    n_heat = st.slider("Genes in heatmap", 5, 50, min(20, top_n), key="heat_n")
    show_plot(viz.expression_heatmap(X_desc, ds["y"], table["Gene"].head(n_heat).tolist()))

    if method == "SHAP" and results["importance"].get("SHAP") is not None:
        shap_res = results["importance"]["SHAP"]
        st.markdown("#### SHAP summary plot")
        try:
            st.pyplot(xai.shap_summary_figure(shap_res))
        except Exception:
            logger.exception("SHAP plot failed")
            st.warning("The SHAP summary plot could not be drawn; the table above is still valid.")
        st.markdown("#### Individual sample explanation")
        sample = st.selectbox("Training sample", list(shap_res["shap_values"].index))
        show_plot(viz.shap_sample_fig(shap_res["shap_values"], sample))
    elif method == "SHAP":
        st.warning(results.get("shap_message") or "SHAP is unavailable; permutation importance is shown instead.")


def page_gene_explorer(ds: Dict[str, Any], X_desc: pd.DataFrame, de: pd.DataFrame,
                       results: Optional[Dict[str, Any]]) -> None:
    st.header("Gene Explorer")
    default = "SNCA"
    if results is not None:
        default = str(results["consensus"].iloc[0]["Gene"])
    gene_input = st.text_input("Enter a gene symbol (e.g. SNCA, TH, LRRK2)", value=default, key="gene_query")
    gene = gene_input.strip().upper()
    if not gene:
        st.info("Type a gene symbol to explore it.")
        return

    if gene not in X_desc.columns:
        st.error(f"'{gene}' is not present in this dataset after probe mapping and filtering.")
        close = difflib.get_close_matches(gene, list(X_desc.columns), n=5, cutoff=0.6)
        if close:
            st.info("Did you mean: " + ", ".join(close))
        return

    stats = dea.gene_statistics(de, gene)
    st.success(f"{gene} is present in the dataset.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean (control)", f"{stats['Control_Mean']:.2f}")
    c2.metric("Mean (Parkinson's)", f"{stats['Disease_Mean']:.2f}")
    c3.metric("log2 fold change", f"{stats['Log2_Fold_Change']:+.2f}")
    c4.metric("Adjusted p-value", format_p(stats["Adjusted_P_Value"]))
    rank_text, in_model = "n/a", False
    if results is not None:
        cons = results["consensus"]
        hit = cons[cons["Gene"] == gene]
        if not hit.empty:
            r = int(hit.iloc[0]["Rank"])
            rank_text = f"#{r} of {len(cons):,}"
            selected = set(results["final"][results["primary_model"]]["selected_features"])
            in_model = gene in selected
    c5.metric("Consensus rank", rank_text)

    left, right = st.columns([3, 2])
    with left:
        show_plot(viz.gene_boxplot(X_desc, ds["y"], gene))
    with right:
        st.markdown("#### Machine-learning importance")
        if results is None:
            st.info("Train the models to see model-derived importance for this gene.")
        else:
            rows = []
            imp = results["importance"]
            for name in ["Logistic Regression", "Random Forest"]:
                if name in imp:
                    col = "Abs_Importance" if name == "Logistic Regression" else "Importance"
                    m = imp[name][imp[name]["Gene"] == gene]
                    rows.append({"Method": name, "Importance": float(m.iloc[0][col]) if not m.empty else 0.0})
            perm = imp["Permutation"].get(results["primary_model"])
            if perm is not None:
                m = perm[perm["Gene"] == gene]
                rows.append({"Method": f"Permutation ({results['primary_model']})",
                             "Importance": float(m.iloc[0]["Mean_Importance"]) if not m.empty else 0.0})
            if not hit.empty:
                rows.append({"Method": "Consensus score", "Importance": float(hit.iloc[0]["Consensus_Score"])})
            show_df(pd.DataFrame(rows))
            st.caption("0 means the gene was not among the model's selected features.")

    # Interpretation - cautious wording by design.
    lfc = stats["Log2_Fold_Change"]
    direction = "higher" if lfc > 0 else "lower"
    significant = stats["Adjusted_P_Value"] <= 0.05
    sig_text = (
        ", significant at FDR \u2264 0.05" if significant else ", not significant after multiple-testing correction"
    )
    text = (
        f"This gene shows {direction} average expression in the Parkinson's disease group in this dataset "
        f"(log2 fold change {lfc:+.2f}; adjusted p = {format_p(stats['Adjusted_P_Value'])}{sig_text})."
    )
    if results is not None and not hit.empty:
        if in_model and int(hit.iloc[0]["Rank"]) <= 50:
            text += f" It was ranked highly (consensus rank {rank_text}) and was among the genes selected by the machine-learning model."
        elif in_model:
            text += f" It was among the model-selected genes but ranked lower overall (consensus rank {rank_text})."
        else:
            text += " It was not among the genes selected as model features for the current settings."
    text += (" This represents a computational association and does not establish that the gene causes "
             "Parkinson's disease.")
    st.markdown("#### Interpretation")
    st.info(text)
    if gene in ds["gene_info"].index:
        st.caption(f"Annotation: {ds['gene_info'].get(gene, '')}")


def page_prediction(ds: Dict[str, Any], X_desc: pd.DataFrame, results: Optional[Dict[str, Any]]) -> None:
    st.header("Prediction")
    st.warning(
        "Research/educational demonstration. The output is a model probability for this dataset's "
        "patterns, not a diagnosis, and must not be used for any medical decision.", icon="⚠️",
    )
    bundle, error = get_bundle()
    if error:
        st.error(error)
        return
    if bundle is None:
        st.info("No trained model found. Click **Train / Retrain Models** in the sidebar first.")
        return

    features = bundle["selected_features"]
    cv = bundle["metadata"].get("cv_summary", {})
    st.markdown(
        f"Loaded model: **{bundle['model_name']}** using **{len(features)}** genes "
        f"(cross-validated ROC-AUC {cv.get('ROC_AUC_Mean', float('nan')):.2f}, "
        f"saved {bundle['metadata'].get('saved_at', 'n/a')})."
    )

    with st.expander("Accepted upload formats and example files", expanded=False):
        st.markdown(
            "CSV, TSV or XLSX with **HGNC gene symbols** and **log2-scale** expression values "
            "(raw linear intensities are log2-transformed automatically). The model needs these genes: "
            f"{len(features)} selected features (at most 20% may be absent; missing genes are imputed with "
            "training medians).\n\n"
            "```\nGene,Expression\nGENE1,5.23\nGENE2,8.31\n```\nor\n```\nGene,Sample_1,Sample_2,...\nGENE1,5.2,5.9\nGENE2,8.3,7.7\n```\n"
            "Values must come from the same platform and normalisation as GSE7621 (Affymetrix GPL570); "
            "other platforms or normalisations will give unreliable results."
        )
        sample_for_example = st.selectbox("Sample for example file", list(X_desc.index), key="example_sample")
        single = pd.DataFrame({"Gene": features, "Expression": X_desc.loc[sample_for_example, features].to_numpy()})
        multi = X_desc[features].T
        multi.index.name = "Gene"
        multi = multi.reset_index()
        d1, d2 = st.columns(2)
        d1.download_button("Download example (single sample)", to_csv_bytes(single),
                           file_name="example_single_sample.csv", mime="text/csv")
        d2.download_button("Download example (all GSE7621 samples)", to_csv_bytes(multi),
                           file_name="example_multi_sample.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload gene-expression file", type=["csv", "tsv", "txt", "xlsx"])
    aligned: Optional[pd.DataFrame] = None
    notes: list = []
    source_label = ""
    if uploaded is not None:
        try:
            parsed, messages = pp.parse_uploaded_expression(uploaded.getvalue(), uploaded.name)
            aligned, missing, align_notes = pp.align_upload_to_model(parsed, features)
            aligned, was_logged = pp.apply_upload_transform(aligned)
            notes = messages + align_notes
            if was_logged:
                notes.append("Values looked like linear intensities and were log2(x+1)-transformed.")
            source_label = uploaded.name
        except pp.UploadValidationError as exc:
            st.error(str(exc))
            return
        except Exception:
            logger.exception("Unexpected upload error")
            st.error("The file could not be processed. Please check its format against the examples above.")
            return
    else:
        st.caption("Or test the pipeline on a sample from GSE7621 (note: these samples were used for training).")
        demo = st.selectbox(
            "GSE7621 sample", ["-"] + [f"{i} (actual: {CLASS_NAMES[int(ds['y'][i])]})" for i in X_desc.index])
        if demo != "-":
            sample_id = demo.split(" ")[0]
            aligned = X_desc.loc[[sample_id], features]
            source_label = f"{sample_id} (part of the training data - not an independent test)"
            notes.append("This sample was part of the training set, so the result is not independent validation.")

    if aligned is None:
        return

    for message in notes:
        st.info(message)
    try:
        preds = ml_models.predict_samples(bundle, aligned)
    except ProjectError as exc:
        st.error(str(exc))
        return

    st.success(f"Scored {len(preds)} sample(s) from {source_label}.")
    show_df(preds.round(4))
    st.download_button("Download predictions", to_csv_bytes(preds), file_name="predictions.csv", mime="text/csv")

    sample = preds["Sample"].iloc[0]
    if len(preds) > 1:
        sample = st.selectbox("Explain sample", preds["Sample"].tolist())
    row = preds[preds["Sample"] == sample].iloc[0]
    g1, g2 = st.columns([1, 2])
    with g1:
        show_plot(viz.probability_gauge(float(row["Probability_Parkinsons"])))
        st.caption("Model output for this dataset's patterns; not a clinical result.")
    with g2:
        try:
            contrib = xai.explain_single_sample(bundle, aligned.loc[sample], top_n=15)
            show_plot(viz.contribution_bar(contrib, f"Top contributing genes ({contrib['Method'].iloc[0]})"))
            show_df(contrib.drop(columns="Method").round(4))
        except Exception:
            logger.exception("Explanation failed")
            st.warning("Per-gene contributions could not be computed for this sample.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # -- data ---------------------------------------------------------------
    try:
        ds = load_dataset()
    except ProjectError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        logger.exception("Dataset loading failed")
        st.error(
            "The dataset could not be loaded. Check your internet connection and reload the page. "
            f"You can also place '{GSE_ID}_family.soft.gz' in the data/ folder."
        )
        st.stop()

    X, y = ds["X"], ds["y"]
    X_desc = pp.impute_median(X)  # descriptive analyses only; the ML pipeline imputes inside CV
    fp = fingerprint(X_desc)
    de = compute_de(X_desc, y, fp)

    # -- sidebar ------------------------------------------------------------
    st.sidebar.title("\U0001f9e0 Brain Disease Gene AI")
    st.sidebar.caption("Parkinson's disease \u00b7 GSE7621 \u00b7 research/educational")
    page = st.sidebar.radio("Navigate", PAGES, key="page")
    st.sidebar.divider()
    st.sidebar.subheader("Model settings")
    model_choice = st.sidebar.selectbox("Model", ml_models.get_model_choices(), index=3)
    n_features = st.sidebar.select_slider("Number of features (genes)", options=FEATURE_COUNT_OPTIONS, value=50)
    folds = st.sidebar.radio("Cross-validation folds", CV_FOLD_OPTIONS, index=1, horizontal=True)
    train_clicked = st.sidebar.button("Train / Retrain Models", type="primary")
    st.sidebar.caption("Models are trained only when you click this button and are then saved to models/.")
    st.sidebar.divider()
    st.sidebar.caption(f"**{DISCLAIMER_TITLE}.** Not a diagnostic tool.")

    if page != "Home":
        disclaimer_banner()

    training_area = st.container()
    if train_clicked:
        with training_area:
            run_training(ds, de, model_choice, n_features, folds)

    results = get_training(X)

    # -- pages --------------------------------------------------------------
    try:
        if page == "Home":
            page_home(ds)
        elif page == "Dataset Explorer":
            page_dataset(ds, X_desc)
        elif page == "Exploratory Analysis":
            page_eda(ds, X_desc)
        elif page == "Differential Gene Analysis":
            page_de(ds, X_desc, de)
        elif page == "Machine Learning":
            page_ml(ds, results, model_choice, n_features, folds)
        elif page == "Gene Importance":
            page_importance(ds, X_desc, de, results)
        elif page == "Gene Explorer":
            page_gene_explorer(ds, X_desc, de, results)
        elif page == "Prediction":
            page_prediction(ds, X_desc, results)
    except ProjectError as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Unhandled error on page %s", page)
        st.error("Something went wrong while rendering this page. Please try again or choose another page.")


main()
