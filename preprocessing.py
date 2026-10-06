"""Data preparation for gene-expression analysis.

Stages implemented here
-----------------------
A. Sample identification   labels are parsed from GEO sample metadata (never from row positions)
B. Probe -> gene mapping   probes are mapped to HGNC-style symbols; unmapped probes are dropped and
                           multiple probes per gene are aggregated (median by default)
C. Missing values          genes with excessive missingness are removed; remaining gaps are imputed
                           with the median *inside the ML pipeline* (fit on training data only)
D. Transformation          log2(x + 1) is applied ONLY if the data still look like linear intensities
E. Scaling                 StandardScaler lives inside the scikit-learn Pipeline (see ml_models.py)

Everything here is unsupervised (it never looks at the class labels) except label parsing, so it
cannot leak label information into model evaluation.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import CLASS_NAMES, CONTROL, EXPECTED_COUNTS, PARKINSON, ProjectError, get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# A. Sample identification
# --------------------------------------------------------------------------- #
_CONTROL_RE = re.compile(
    r"\b(controls?|normal|healthy|non[-\s]?parkinson\w*|non[-\s]?pd|unaffected|ctrl)\b", re.IGNORECASE
)
_NON_PD_RE = re.compile(r"non[-\s]?(parkinson\w*|pd)\b", re.IGNORECASE)
_PD_RE = re.compile(r"(parkinson|\bpd\b)", re.IGNORECASE)


def classify_text(text: str) -> Optional[int]:
    """Return 0 (control), 1 (Parkinson's) or None if the text is empty or ambiguous."""
    if not text or not str(text).strip():
        return None
    text = str(text)
    has_control = bool(_CONTROL_RE.search(text))
    # "non-Parkinson" must not count as Parkinson evidence.
    has_pd = bool(_PD_RE.search(_NON_PD_RE.sub(" ", text)))
    if has_control and not has_pd:
        return CONTROL
    if has_pd and not has_control:
        return PARKINSON
    return None  # both present (ambiguous) or neither present


def extract_labels(metadata: pd.DataFrame) -> pd.Series:
    """Derive 0/1 labels from GEO sample metadata.

    Fields are tried in order of reliability; the first field that gives an unambiguous
    answer for a sample wins. Raises ProjectError if any sample cannot be labelled.
    """
    field_priority = [c for c in metadata.columns if c.startswith("char_")]
    field_priority += ["characteristics", "title", "source_name", "description"]
    field_priority = [c for c in field_priority if c in metadata.columns]

    labels: Dict[str, Optional[int]] = {}
    for sample_id, row in metadata.iterrows():
        label = None
        for field in field_priority:
            label = classify_text(str(row[field]))
            if label is not None:
                break
        labels[sample_id] = label

    y = pd.Series(labels, name="label", dtype="object")
    unlabeled = y[y.isna()].index.tolist()
    if unlabeled:
        raise ProjectError(
            "Could not determine Parkinson's/control status from the GEO sample metadata for: "
            f"{unlabeled[:5]}{'...' if len(unlabeled) > 5 else ''}. "
            f"Metadata columns inspected: {field_priority}"
        )
    y = y.astype(int)
    if y.nunique() < 2:
        raise ProjectError("Only one class was found in the sample metadata; cannot train a classifier.")
    return y


def check_expected_counts(y: pd.Series) -> List[str]:
    """Return warnings if class counts differ from the published GSE7621 composition."""
    counts = y.map(CLASS_NAMES).value_counts().to_dict()
    warnings = []
    for name, expected in EXPECTED_COUNTS.items():
        if counts.get(name, 0) != expected:
            warnings.append(
                f"Expected {expected} '{name}' samples for GSE7621 but found {counts.get(name, 0)}."
            )
    return warnings


# --------------------------------------------------------------------------- #
# B. Probe -> gene mapping
# --------------------------------------------------------------------------- #
_INVALID_SYMBOLS = {"", "---", "NAN", "NONE", "NA", "N/A", "NULL"}


def clean_gene_symbol(raw: Any) -> Optional[str]:
    """Extract one usable gene symbol from an annotation string.

    Affymetrix annotations may list several genes ("DDR1 /// MIR4640"). We keep the first symbol,
    a common and documented simplification that avoids duplicating a probe across genes.
    """
    if raw is None or raw is pd.NA or (isinstance(raw, float) and np.isnan(raw)):
        return None
    symbol = re.split(r"\s*(?:///|//|;|\|)\s*", str(raw).strip())[0].strip()
    if symbol.upper() in _INVALID_SYMBOLS:
        return None
    return symbol.upper()


def map_probes_to_genes(
    expression: pd.DataFrame, annotation: pd.DataFrame, aggregation: str = "median"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Map probes to genes and aggregate duplicates.

    Parameters
    ----------
    expression : probes x samples
    annotation : indexed by probe id, with a 'gene_symbol_raw' column
    aggregation: 'median' or 'mean'

    Returns
    -------
    gene_expression : genes x samples
    report          : dictionary documenting the mapping (shown in the app and README)
    """
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")

    symbols = annotation["gene_symbol_raw"].map(clean_gene_symbol).reindex(expression.index)
    n_probes = len(expression)
    mapped_mask = symbols.notna()
    n_unmapped = int((~mapped_mask).sum())

    mapped = expression.loc[mapped_mask].copy()
    mapped.index = symbols[mapped_mask].values
    n_unique_before = mapped.index.nunique()
    n_dup_genes = int((mapped.index.value_counts() > 1).sum())

    grouped = mapped.groupby(level=0)
    gene_expression = grouped.median() if aggregation == "median" else grouped.mean()
    gene_expression = gene_expression.sort_index()

    report = {
        "probes_total": int(n_probes),
        "probes_unmapped_removed": n_unmapped,
        "probes_mapped": int(mapped_mask.sum()),
        "unique_genes": int(n_unique_before),
        "genes_with_multiple_probes": n_dup_genes,
        "aggregation": aggregation,
        "note": (
            "Probe IDs were mapped to gene symbols using the GPL570 annotation embedded in the GEO "
            "record. Probes without a gene symbol (including AFFX control probes) were removed. When "
            "an annotation listed several genes, the first symbol was used. Multiple probes for the "
            f"same gene were collapsed with the {aggregation}."
        ),
    }
    return gene_expression, report


# --------------------------------------------------------------------------- #
# C/D. Missing values, transformation, normalisation
# --------------------------------------------------------------------------- #
def needs_log_transform(values: np.ndarray) -> bool:
    """Heuristic used by GEO2R: are the values still on a linear intensity scale?"""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    q = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
    return bool((q[4] > 100) or ((q[5] - q[0]) > 50 and q[1] > 0))


def maybe_log_transform(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Apply log2(x + 1) only when the data look un-transformed. Returns (data, was_transformed)."""
    if needs_log_transform(df.to_numpy()):
        return np.log2(df.clip(lower=0) + 1.0), True
    return df, False


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Optional quantile normalisation across samples (columns). Off by default."""
    ranks = df.rank(method="first", axis=0).astype(int) - 1
    sorted_means = np.sort(df.to_numpy(), axis=0).mean(axis=1)
    out = df.copy()
    for col in df.columns:
        out[col] = sorted_means[ranks[col].to_numpy()]
    return out


def filter_missing_genes(X: pd.DataFrame, max_missing_fraction: float = 0.2) -> Tuple[pd.DataFrame, int]:
    """Drop genes (columns) with more than `max_missing_fraction` missing values."""
    frac = X.isna().mean(axis=0)
    keep = frac <= max_missing_fraction
    return X.loc[:, keep], int((~keep).sum())


def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute per gene. Used only for descriptive analyses (EDA / DE).

    The predictive pipeline has its own SimpleImputer fitted on training folds only.
    """
    if not X.isna().any().any():
        return X
    return X.fillna(X.median(axis=0))


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def quality_control(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Simple per-sample QC: median, IQR and correlation with the average profile."""
    Xi = impute_median(X)
    mean_profile = Xi.mean(axis=0)
    corr = Xi.apply(lambda row: np.corrcoef(row.to_numpy(), mean_profile.to_numpy())[0, 1], axis=1)
    table = pd.DataFrame({
        "Class": y.map(CLASS_NAMES),
        "Median_Expression": Xi.median(axis=1),
        "IQR": Xi.quantile(0.75, axis=1) - Xi.quantile(0.25, axis=1),
        "Correlation_With_Mean_Profile": corr,
    })
    outliers = table.index[table["Correlation_With_Mean_Profile"] < 0.90].tolist()
    return {"per_sample": table, "low_correlation_samples": outliers}


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def build_processed_dataset(
    raw: Dict[str, Any],
    aggregation: str = "median",
    max_missing_fraction: float = 0.2,
    apply_quantile_normalization: bool = False,
) -> Dict[str, Any]:
    """Convert the raw GEO dictionary into the samples x genes matrix used for ML.

    Returns a dictionary:
        X          DataFrame samples x genes (log2 scale; NaNs possible but rare)
        y          Series 0/1 aligned to X.index
        metadata   sample metadata with a 'Class' column
        gene_info  DataFrame gene -> representative annotation title
        report     dict describing every processing step
        warnings   list of user-visible warnings
    """
    expression = raw["expression"]
    metadata = raw["metadata"].copy()
    annotation = raw["annotation"]

    y = extract_labels(metadata).reindex(expression.columns)
    warnings = check_expected_counts(y)

    gene_expr, mapping_report = map_probes_to_genes(expression, annotation, aggregation)

    gene_expr, log_applied = maybe_log_transform(gene_expr)
    if apply_quantile_normalization:
        gene_expr = quantile_normalize(gene_expr)

    X = gene_expr.T  # samples x genes
    X.index.name = "sample_id"
    X.columns.name = "gene"
    X, n_dropped_missing = filter_missing_genes(X, max_missing_fraction)
    n_missing_left = int(X.isna().sum().sum())

    # Drop constant genes (zero variance): they carry no information and break t-tests.
    variances = X.var(axis=0, skipna=True)
    constant = variances.index[(variances.fillna(0) <= 1e-12)]
    X = X.drop(columns=constant)

    metadata["Class"] = y.map(CLASS_NAMES)
    metadata["label"] = y

    gene_titles = (
        annotation.assign(_sym=annotation["gene_symbol_raw"].map(clean_gene_symbol))
        .dropna(subset=["_sym"])
        .drop_duplicates("_sym")
        .set_index("_sym")["gene_title"]
    )

    qc = quality_control(X, y)

    report = {
        **mapping_report,
        "log2_transform_applied": log_applied,
        "quantile_normalization_applied": apply_quantile_normalization,
        "genes_removed_for_missingness": n_dropped_missing,
        "missing_values_remaining": n_missing_left,
        "constant_genes_removed": int(len(constant)),
        "final_samples": int(X.shape[0]),
        "final_genes": int(X.shape[1]),
        "qc_low_correlation_samples": qc["low_correlation_samples"],
        "transformation_note": (
            "Values were log2(x+1)-transformed because they appeared to be on a linear scale."
            if log_applied
            else "Values already appear to be on a log2 scale; no transformation was applied."
        ),
    }

    return {
        "X": X,
        "y": y,
        "metadata": metadata,
        "gene_info": gene_titles,
        "qc": qc["per_sample"],
        "report": report,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Uploaded data (prediction page)
# --------------------------------------------------------------------------- #
class UploadValidationError(ProjectError):
    """Raised for problems with a user-uploaded file; the message is safe to display."""


_GENE_COLUMN_NAMES = {"gene", "genes", "gene_symbol", "genesymbol", "symbol", "gene symbol", "gene_name", "id"}


def parse_uploaded_expression(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """Parse and validate an uploaded expression table.

    Accepted layouts (CSV, TSV, or XLSX):
        Gene,Expression                 one sample
        Gene,Sample_1,Sample_2,...      several samples

    Returns
    -------
    df       : genes x samples (numeric, unique upper-case gene index)
    messages : informational notes (duplicates merged, non-numeric coerced...)
    """
    if not file_bytes:
        raise UploadValidationError("The uploaded file is empty.")

    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(file_bytes))
        else:
            text = file_bytes.decode("utf-8-sig", errors="strict")
            header = text.splitlines()[0] if text.strip() else ""
            sep = "\t" if "\t" in header else ";" if (";" in header and "," not in header) else ","
            raw = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        raise UploadValidationError(
            "The file could not be read as CSV/TSV/XLSX. Please check that it is a plain table with a "
            "header row (e.g. 'Gene,Expression')."
        ) from None

    if raw.shape[1] < 2:
        raise UploadValidationError(
            "The file needs at least two columns: a gene column and one or more expression columns."
        )
    if raw.shape[0] == 0:
        raise UploadValidationError("The file has a header but no data rows.")

    messages: List[str] = []
    gene_col = next((c for c in raw.columns if str(c).strip().lower() in _GENE_COLUMN_NAMES), None)
    if gene_col is None:
        gene_col = raw.columns[0]
        messages.append(
            f"No column named 'Gene' was found; the first column ('{gene_col}') was used as the gene identifier."
        )

    genes = raw[gene_col].astype(str).str.strip().str.upper()
    values = raw.drop(columns=[gene_col]).copy()
    seen: Dict[str, int] = {}
    unique_names = []
    for c in values.columns:  # sample names must be unique so that each sample can be addressed
        name = str(c).strip()
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    values.columns = unique_names

    numeric = values.apply(pd.to_numeric, errors="coerce")
    n_bad = int(numeric.isna().sum().sum() - values.isna().sum().sum())
    if n_bad > 0:
        messages.append(f"{n_bad} non-numeric expression values were treated as missing.")
    if numeric.notna().sum().sum() == 0:
        raise UploadValidationError("No numeric expression values were found in the file.")

    numeric.index = genes.values
    numeric = numeric.loc[~numeric.index.isin(list(_INVALID_SYMBOLS))]
    if numeric.empty:
        raise UploadValidationError("No valid gene identifiers were found in the gene column.")

    if numeric.index.has_duplicates:
        n_dup = int(numeric.index.duplicated().sum())
        numeric = numeric.groupby(level=0).mean()
        messages.append(f"{n_dup} duplicated gene rows were averaged.")

    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        numeric = numeric.drop(columns=all_missing)
        messages.append(f"Columns with no numeric values were ignored: {all_missing}")
    if numeric.shape[1] == 0:
        raise UploadValidationError("No usable expression column remained after validation.")

    return numeric, messages


def align_upload_to_model(
    upload: pd.DataFrame, required_genes: List[str], max_missing_fraction: float = 0.2
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Select and order the genes the model needs.

    Returns (samples x required_genes, missing_genes, notes). Missing genes are inserted as NaN so the
    pipeline's imputer can fill them with training medians, but only if few are missing.
    """
    required = [g.upper() for g in required_genes]
    present = [g for g in required if g in upload.index]
    missing = [g for g in required if g not in upload.index]
    frac_missing = len(missing) / max(len(required), 1)

    if len(present) == 0:
        raise UploadValidationError(
            "None of the genes required by the trained model were found in the uploaded file. "
            "Gene symbols must be HGNC-style symbols (e.g. SNCA, TH, SLC18A2)."
        )
    if frac_missing > max_missing_fraction:
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        raise UploadValidationError(
            f"{len(missing)} of {len(required)} required genes ({frac_missing:.0%}) are missing from the "
            f"upload (more than the allowed {max_missing_fraction:.0%}). Missing genes include: {preview}"
        )

    aligned = upload.reindex(required).T  # samples x genes (missing -> NaN)
    aligned.columns.name = "gene"
    notes = []
    if missing:
        notes.append(
            f"{len(missing)} required gene(s) are absent and will be imputed with training-set medians: "
            f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}. Predictions are less reliable."
        )
    if aligned.isna().any().any() and not missing:
        notes.append("Some values are missing and will be imputed with training-set medians.")
    return aligned, missing, notes


def apply_upload_transform(aligned: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Log2-transform uploaded values only if they look like linear intensities."""
    values = aligned.to_numpy(dtype=float)
    if needs_log_transform(values):
        return np.log2(aligned.clip(lower=0) + 1.0), True
    return aligned, False
