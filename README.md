# Identification of Brain Disease–Associated Genes using AI/ML Techniques

> **Research/Educational Use Only.** This application demonstrates computational identification of
> gene-expression patterns associated with Parkinson's disease using publicly available research data.
> It is not a medical diagnostic tool and should not be used to diagnose, predict, or make treatment
> decisions for individual patients. The identified genes represent statistical and machine-learning
> associations within the analyzed dataset and do not establish biological causality.

## Abstract

Parkinson's disease (PD) is a neurodegenerative disorder in which dopamine-producing neurons of the
substantia nigra are progressively lost. Gene-expression microarrays allow thousands of genes to be
measured in post-mortem brain tissue, but turning those measurements into a short list of
disease-associated genes is a statistical and machine-learning problem. This project builds an
end-to-end, explainable ML workflow on the public NCBI GEO dataset **GSE7621**: it downloads and cleans the
data, maps probes to genes, tests every gene for differential expression, trains Logistic Regression,
Random Forest and SVM classifiers with leakage-safe cross-validation, explains the models with
coefficients, feature importance, permutation importance and SHAP, combines everything into a
project-specific consensus gene ranking, and exposes it all in an interactive Streamlit dashboard.

## Problem statement

A microarray measures roughly 20,000 genes, yet GSE7621 contains only 25 samples (p ≫ n). In this regime:

* a classifier can memorise the training samples with almost any set of genes (overfitting);
* thousands of simultaneous statistical tests produce many false positives unless corrected;
* feature selection done on the full dataset *before* cross-validation leaks information and inflates
  accuracy — a very common mistake;
* a gene being "predictive" in one dataset says little about causality.

The project is designed around these pitfalls: selection happens **inside** each cross-validation fold,
p-values are FDR-corrected, several independent importance methods are compared, and all wording avoids causal claims.

## Objectives

1. Load and preprocess public gene-expression data (GSE7621) reproducibly.
2. Identify genes whose expression differs between Parkinson's disease and control samples.
3. Train and compare multiple ML classifiers with stratified cross-validation.
4. Identify the most predictive genes with explainable-AI techniques.
5. Visualise important genes and disease/control expression patterns.
6. Let a user upload compatible expression data and obtain a model probability with per-gene contributions.
7. Provide a Streamlit interface that runs locally and on Streamlit Community Cloud without a GPU.
8. State clearly that this is a research/educational model, **not** a clinical diagnostic system.

## Dataset

| Item | Value |
|---|---|
| Accession | **GSE7621** – <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7621> |
| Source | NCBI Gene Expression Omnibus (GEO) |
| Organism | *Homo sapiens* |
| Tissue | Substantia nigra, post-mortem human brain |
| Disease | Parkinson's disease |
| Technology | Affymetrix microarray |
| Platform | GPL570 – Affymetrix Human Genome U133 Plus 2.0 Array |
| Samples | 25 (16 Parkinson's disease, 9 control) |

The app downloads the official GEO record through `GEOparse` (with a direct HTTPS fallback), caches the parsed
result in `data/`, and never uses unofficial copies of the data.

## Technologies

Python 3.10+ · Pandas · NumPy · SciPy · statsmodels · scikit-learn · Streamlit · Plotly · Matplotlib ·
Seaborn · GEOparse · SHAP · Joblib · Requests · openpyxl

## Project structure

```
brain-disease-gene-ai/
├── app.py                      # Streamlit dashboard (8 pages)
├── requirements.txt
├── README.md
├── .gitignore
├── data/                       # downloaded/cached GEO data (git-ignored)
├── models/                     # saved model bundle + training results (git-ignored)
├── notebooks/
│   └── exploratory_analysis.ipynb
└── src/
    ├── data_loader.py          # GEO download + on-disk cache
    ├── preprocessing.py        # labels, probe→gene mapping, transform, upload validation
    ├── eda.py                  # overview, PCA, correlation, clustering, distributions
    ├── differential_expression.py  # Welch t-test + Benjamini–Hochberg FDR
    ├── feature_selection.py    # VarianceFilter, TopKSelector (leakage-safe transformers)
    ├── ml_models.py            # pipelines, CV, persistence, training orchestrator
    ├── explainability.py       # coefficients, RF importance, permutation, SHAP, consensus
    ├── visualization.py        # volcano, bars, box plots, heatmaps, ROC, confusion matrix
    └── utils.py                # constants, paths, helpers, disclaimer text
```

## Methodology

1. **Data acquisition** – `GEOparse` downloads the series/platform record; the parsed expression matrix,
   sample metadata and probe annotation are cached to `data/`. Streamlit's `@st.cache_data` stops reruns
   from touching disk or network.
2. **Sample identification** – labels (0 = Control, 1 = Parkinson's) are parsed from GEO sample metadata
   (characteristics, title, source name) — never from row positions. The app warns if the counts differ from
   the published 16/9 composition.
3. **Probe → gene mapping** – probes are mapped to gene symbols using the GPL570 annotation embedded in the
   record. Unmapped probes (including `AFFX` controls) are removed; when an annotation lists several genes
   (`A /// B`) the first symbol is used; multiple probes per gene are collapsed with the **median**. The mapping
   report is displayed in the *Dataset Explorer*.
4. **Quality control** – per-sample median, IQR and correlation with the mean profile.
5. **Transformation** – values are log2(x+1)-transformed *only* if they look like linear intensities (the GEO2R
   heuristic). Already-logged data are left untouched.
6. **Missing values** – genes with >20 % missing values are dropped; remaining gaps are median-imputed inside
   the ML pipeline (fitted on training folds only).
7. **Low-variance filtering** – the lowest-variance 50 % of genes are discarded inside the pipeline.
8. **Differential expression** – Welch's t-test per gene, log2 fold change = mean(PD) − mean(control) on the log2
   scale, Benjamini–Hochberg FDR. Genes are ranked by adjusted p-value, then |log2FC|.
9. **Feature selection** – the top *K* genes (25/50/100/250/500) by ANOVA F-statistic, **refitted within every
   training fold**.
10. **Scaling** – `StandardScaler` inside the same pipeline.
11. **Modelling & evaluation** – stratified K-fold cross-validation (3 or 5 folds). Metrics: accuracy,
    precision, recall, F1, ROC-AUC, specificity (mean ± SD across folds) plus a pooled out-of-fold confusion
    matrix and ROC curve.
12. **Explainability** – four importance methods and a consensus score (below).
13. **Persistence** – after training, the model, scaler, selected genes, feature-selection configuration and
    metadata are saved with `joblib` to `models/`. The app loads them instead of retraining.

### Leakage-safe pipeline

```
SimpleImputer(median) → VarianceFilter → TopKSelector(K) → StandardScaler → classifier
```

The whole pipeline is rebuilt and fitted on each training fold. Held-out samples never influence imputation,
variance filtering, gene selection or scaling. As a sanity check, shuffling the labels drives cross-validated
ROC-AUC to ≈ 0.5, as it should.

For deployment, gene selection is run once on all samples and a compact pipeline
(`imputer → scaler → classifier`) is fitted on the K selected genes. It needs only those K genes at prediction
time, which is what makes CSV upload practical.

### Machine-learning models

* **Logistic Regression** – linear model with L2 regularisation (C = 0.1), balanced class weights. Its
  coefficients are directly interpretable: positive → pushes toward the Parkinson's class, negative → toward control.
* **Random Forest** – 200 trees, balanced class weights. Captures non-linear effects and interactions and is
  robust to noisy features; provides impurity-based importance and exact SHAP values (TreeExplainer).
* **SVM** – `SVC(kernel="linear", probability=True)`. Maximum-margin classifiers work well when features
  outnumber samples; the linear kernel exposes coefficients for explanation.

XGBoost is intentionally not included (optional in the brief) to keep installation reliable.

### Explainable AI

| Method | What it measures |
|---|---|
| Logistic-regression coefficients | Signed weight of each standardised gene (`model.coef_`) |
| Random-forest importance | Mean impurity decrease contributed by each gene (`feature_importances_`) |
| Permutation importance | Increase in Brier score (probability error) when one gene is shuffled (`sklearn.inspection.permutation_importance`); mean and SD over repeats |
| SHAP | Per-sample, per-gene additive contributions (TreeExplainer for RF, LinearExplainer for linear models); falls back to permutation importance if SHAP is unavailable |

*Why Brier score?* Fitted models separate 25 training samples perfectly, so shuffling one gene never changes
accuracy or AUC and every importance would be 0. A probability-based score is sensitive to the shuffle. Permutation
importance is computed in-sample (there is no spare hold-out data), so it shows which genes the *fitted model
uses*, whereas cross-validation measures generalisation.

### Consensus ranking

Each component is min-max normalised to [0, 1] across genes, then combined:

```
DE score  = 0.5 · norm(−log10 adjusted p) + 0.5 · norm(|log2 fold change|)

Consensus Score = 0.30 · DE score
                + 0.25 · Random-Forest importance
                + 0.25 · Permutation importance
                + 0.20 · Logistic-Regression |coefficient|
```

Genes not selected as model features score 0 on the model-based components. If a component is unavailable
(e.g. only one model was trained) the remaining weights are re-normalised. This is a **project-specific
computational consensus score and is not biologically validated**.

## Streamlit pages

| Page | Contents |
|---|---|
| Home | Objective, dataset, methodology diagram, disclaimer |
| Dataset Explorer | Sample/gene counts, class distribution, sample metadata, statistics, QC, preprocessing report |
| Exploratory Analysis | PCA (2-D/3-D), correlation heatmap, expression distributions, hierarchical clustering, top variable genes |
| Differential Gene Analysis | Adjusted-p / fold-change / gene-count controls, volcano plot, significant genes, top up/down genes, **Download DE Results** |
| Machine Learning | Sidebar model / features (25–500) / folds (3–5); CV table, comparison chart, confusion matrix, ROC; **Download Model Performance** |
| Gene Importance | Consensus / Logistic Regression / Random Forest / Permutation / SHAP rankings, bar chart, heatmap, SHAP summary and per-sample plots; **Download Top Associated Genes** |
| Gene Explorer | Search any gene: presence, group means, log2FC, adjusted p, ML importance, box plot, cautious interpretation |
| Prediction | Upload CSV/TSV/XLSX → validation → preprocessing → probability → top contributing genes |

Models are trained **only** when you click **Train / Retrain Models** in the sidebar, and are then saved and
re-used. Changing pages or widgets never retrains.

### Upload formats

```
Gene,Expression            Gene,Sample_1,Sample_2,...
GENE1,5.23                 GENE1,5.23,5.91
GENE2,8.31                 GENE2,8.31,7.70
```

Requirements: HGNC gene symbols; expression on the same platform/normalisation as GSE7621 (Affymetrix GPL570).
Linear-scale values are log2-transformed automatically. At most 20 % of the model's genes may be absent
(they are imputed with training medians, with a warning). Malformed, empty, non-numeric or incompatible files
produce a clear message — never a stack trace. The Prediction page offers example files you can download.

## Running locally

```bash
git clone <repository>
cd brain-disease-gene-ai
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+ is required; no GPU is needed. The first start downloads GSE7621 (needs internet) and caches it in
`data/`. If the download is blocked, download `GSE7621_family.soft.gz` manually from
<https://ftp.ncbi.nlm.nih.gov/geo/series/GSE7nnn/GSE7621/soft/> into `data/` and restart the app.

Then open **Machine Learning** (or any page), pick settings in the sidebar and click **Train / Retrain Models**.
Training takes roughly 10–30 s for 'Compare All' depending on the number of features and your CPU.

## Uploading the project to GitHub

```bash
cd brain-disease-gene-ai
git init
git add .
git commit -m "Brain disease gene AI: GSE7621 Parkinson's analysis"
git branch -M main
git remote add origin https://github.com/<your-username>/brain-disease-gene-ai.git
git push -u origin main
```

(Create the empty repository on github.com first. `data/` and `models/` contents are git-ignored; they are recreated automatically.)

## Deploying on Streamlit Community Cloud

1. Push the project to a **public** GitHub repository (as above) with `app.py` and `requirements.txt` at the repository root.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Select the repository, branch `main`, and main file path `app.py`.
5. (Optional) Open **Advanced settings** and choose Python 3.11 or 3.12.
6. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app; the first load downloads GSE7621 into the container.
7. Open the app and click **Train / Retrain Models**. Community Cloud storage is ephemeral, so after the app sleeps or restarts you may need to train again (takes seconds).

## Limitations

* **Small sample size** – 25 samples; cross-validated metrics have wide uncertainty.
* **High-dimensional feature space** – ~20,000 genes for 25 samples; selection is unstable and different gene
  lists can perform similarly.
* **Microarray-specific limitations** – limited dynamic range, probe cross-hybridisation, annotation that
  changes over time, platform/normalisation-specific values (uploaded data must match).
* **Post-mortem tissue** – RNA quality, post-mortem interval, agonal state, age, sex, medication and
  cell-type composition (neuron loss changes the cellular mix of the tissue) can all drive expression differences.
* **Dataset-specific associations** – results describe GSE7621 only.
* **No independent clinical validation** – no external cohort was used.
* **Computational associations do not prove causality**, and the consensus score is a heuristic.
* **The model is not a clinical diagnostic system** and must not be used for medical decisions.
* The t-test is simple; moderated tests (limma) are usually preferred in published microarray studies.

## Viva explanation (concise)

**Why this dataset?** GSE7621 is a well-known, public, official GEO dataset of Parkinson's vs control
substantia nigra — the brain region most affected by the disease — on a standard Affymetrix platform. Its small size makes the
p ≫ n problem tangible, which is exactly what makes it a good ML teaching example.

**What is gene expression?** The amount of RNA produced from a gene; a microarray measures this for ~20,000 genes at once
as fluorescence intensities. Higher expression means the gene is more active in that tissue.

**What does preprocessing do?** It labels samples from metadata, maps probes to genes, collapses duplicates, logs the data
if needed (to stabilise variance and make fold changes symmetric), and handles missing values so models receive a clean
samples × genes table.

**Why is feature selection necessary?** With 20,000 features and 25 samples a model can fit noise perfectly. Selecting K informative genes
reduces overfitting and makes results interpretable. It must happen *inside* cross-validation, otherwise test samples
influence which genes are picked and accuracy is inflated.

**Why several ML algorithms?** Different inductive biases give a robustness check: if linear, tree-based and margin-based models agree on a
gene, the signal is less likely to be an artefact of one algorithm.

**Logistic Regression** – simple, regularised, and its signed coefficients directly say which genes push toward each class.
**Random Forest** – handles non-linear effects and noise, gives importance scores and exact SHAP values.
**SVM** – maximum-margin classifier that performs well when features outnumber samples.

**What does PCA do?** It rotates the data into uncorrelated components ordered by variance, letting us see in 2-D/3-D whether
disease and control samples separate and spot outliers or batch effects.

**What is differential expression?** A per-gene statistical comparison of mean expression between groups; the log2 fold change
gives direction and size (+1 means 2× higher in PD).

**p-value and FDR?** The p-value is the probability of seeing a difference at least this large if there were truly none. When testing
20,000 genes, ~1,000 would be "significant" at 0.05 by chance, so the Benjamini–Hochberg procedure controls the expected
*false-discovery rate* among the genes called significant (adjusted p / q-value).

**Permutation importance and SHAP?** Permutation importance shuffles one gene and measures how much the model gets worse — a
large loss means the model relies on that gene. SHAP assigns each gene a fair-share contribution (from game theory) to each
individual prediction, which can be averaged for global importance.

**How are final genes ranked?** By the consensus score: normalised DE significance and fold change plus RF, permutation and logistic-regression importance with weights 0.30/0.25/0.25/0.20 (heuristic, not validated).

**Limitations?** See above: tiny sample, p ≫ n, microarray and post-mortem caveats, one dataset, no external validation, association ≠ causation, not a diagnostic tool.

## Optional extension: Alzheimer's disease (GSE33000)

The pipeline is modular: `data_loader.load_raw_dataset(gse_id=...)` accepts any GEO accession, and label parsing lives in
`preprocessing.classify_text`. Adding GSE33000 (prefrontal cortex; Alzheimer's, Huntington's and controls) would require
extending the label parser for the new classes and choosing a two-class comparison. It is deliberately **not** implemented
so that the GSE7621 workflow stays stable.

## Reproducibility

`RANDOM_STATE = 42` is used for cross-validation splits, models, permutation importance and sampling. Results are
reproducible given the same library versions.

## License and data use

Educational project. GSE7621 is public data from NCBI GEO — please cite GEO and the original study listed on the dataset's GEO page when using the results.
