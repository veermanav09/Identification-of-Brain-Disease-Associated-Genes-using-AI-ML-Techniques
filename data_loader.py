"""Data acquisition for NCBI GEO series (default: GSE7621).

Loading order (first that succeeds wins):
    1. processed cache            data/<GSE>_raw_cache.joblib
    2. local GEO SOFT file        data/<GSE>_family.soft.gz  (parsed with GEOparse)
    3. GEOparse download          GEOparse.get_GEO(<GSE>, destdir=data/)
    4. direct HTTPS download      requests -> data/<GSE>_family.soft.gz, then parse

The parsed result is cached to disk so the dataset is downloaded/parsed only once.
The Streamlit layer additionally wraps `load_raw_dataset` with @st.cache_data so
reruns of the script never touch the disk or the network.

Returned "raw dataset" dictionary
---------------------------------
    {
      "gse_id":       "GSE7621",
      "title":        str,
      "platform_id":  "GPL570",
      "expression":   DataFrame  probes x samples   (values as stored in GEO)
      "metadata":     DataFrame  samples x descriptive columns (title, source, characteristics...)
      "annotation":   DataFrame  probes x [gene_symbol_raw, gene_title]
    }
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import requests

from .utils import DATA_DIR, GSE_ID, ProjectError, ensure_dirs, get_logger

logger = get_logger(__name__)

RawDataset = Dict[str, Any]

_SYMBOL_COLUMN_CANDIDATES = [
    "Gene Symbol", "Gene symbol", "GENE_SYMBOL", "gene_symbol", "Symbol", "SYMBOL",
    "GeneSymbol", "ILMN_Gene", "gene_assignment",
]
_TITLE_COLUMN_CANDIDATES = ["Gene Title", "Gene title", "GENE_NAME", "Description", "DEFINITION"]


class DatasetLoadError(ProjectError):
    """Raised when the dataset cannot be obtained or parsed (message is user-safe)."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def cache_path(gse_id: str = GSE_ID, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"{gse_id}_raw_cache.joblib"


def soft_path(gse_id: str = GSE_ID, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"{gse_id}_family.soft.gz"


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _geo_ftp_prefix(gse_id: str) -> str:
    """GEO stores series in folders like GSE7nnn for GSE7621."""
    digits = gse_id.replace("GSE", "")
    stub = digits[:-3] if len(digits) > 3 else "0"
    return f"GSE{stub}nnn"


def download_family_soft(gse_id: str = GSE_ID, data_dir: Path = DATA_DIR, timeout: int = 180) -> Path:
    """Download the GEO family SOFT file directly over HTTPS (fallback to GEOparse)."""
    ensure_dirs()
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{_geo_ftp_prefix(gse_id)}/{gse_id}/soft/"
        f"{gse_id}_family.soft.gz"
    )
    target = soft_path(gse_id, data_dir)
    tmp = target.with_suffix(".part")
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    # Make sure the file is a valid gzip archive before we trust it.
    with gzip.open(tmp, "rb") as handle:
        handle.read(1024)
    tmp.replace(target)
    return target


def _get_geo_object(gse_id: str, data_dir: Path):
    """Return a GEOparse GSE object using local file, GEOparse download or direct download."""
    import GEOparse  # imported lazily so the rest of the package works without it

    ensure_dirs()
    local = soft_path(gse_id, data_dir)
    if local.exists():
        try:
            return GEOparse.get_GEO(filepath=str(local), silent=True)
        except Exception as exc:  # unreadable local file -> keep a copy aside and re-download
            logger.warning("Local SOFT file unreadable (%s); moving it aside and re-downloading.", exc)
            local.replace(local.with_suffix(".gz.bad"))

    errors: List[str] = []
    try:
        return GEOparse.get_GEO(geo=gse_id, destdir=str(data_dir), silent=True)
    except Exception as exc:
        errors.append(f"GEOparse download failed: {exc}")
        logger.warning(errors[-1])

    try:
        path = download_family_soft(gse_id, data_dir)
        return GEOparse.get_GEO(filepath=str(path), silent=True)
    except Exception as exc:
        errors.append(f"Direct download failed: {exc}")
        logger.warning(errors[-1])

    raise DatasetLoadError(
        f"Could not download {gse_id} from NCBI GEO. Check your internet connection, or download "
        f"'{gse_id}_family.soft.gz' manually from "
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{_geo_ftp_prefix(gse_id)}/{gse_id}/soft/ "
        f"and place it in the '{Path(data_dir).name}/' folder, then reload the app. "
        f"Details: {' | '.join(errors)}"
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def _join(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _build_metadata(gse) -> pd.DataFrame:
    """One row per sample with the descriptive fields GEO provides."""
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        row = {
            "sample_id": gsm_name,
            "title": _join(meta.get("title")),
            "source_name": _join(meta.get("source_name_ch1")),
            "characteristics": _join(meta.get("characteristics_ch1")),
            "description": _join(meta.get("description")),
            "platform_id": _join(meta.get("platform_id")),
        }
        # Split "key: value" characteristics into their own columns when present.
        for item in meta.get("characteristics_ch1", []) or []:
            if ":" in str(item):
                key, _, val = str(item).partition(":")
                row[f"char_{key.strip().lower().replace(' ', '_')}"] = val.strip()
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id")


def _build_annotation(gse, probe_ids: pd.Index) -> pd.DataFrame:
    """Probe -> gene annotation taken from the platform (GPL) table embedded in the SOFT file."""
    if not gse.gpls:
        raise DatasetLoadError("The GEO record does not contain a platform annotation table.")
    gpl = next(iter(gse.gpls.values()))
    table = gpl.table.copy()
    if table.empty:
        raise DatasetLoadError("The platform annotation table is empty; cannot map probes to genes.")

    id_col = _first_existing(list(table.columns), ["ID", "ID_REF"]) or table.columns[0]
    sym_col = _first_existing(list(table.columns), _SYMBOL_COLUMN_CANDIDATES)
    if sym_col is None:
        raise DatasetLoadError(
            "No gene-symbol column found in the platform annotation. "
            f"Available columns: {list(table.columns)}"
        )
    title_col = _first_existing(list(table.columns), _TITLE_COLUMN_CANDIDATES)

    annotation = pd.DataFrame({
        "gene_symbol_raw": table[sym_col].astype(str).values,
        "gene_title": table[title_col].astype(str).values if title_col else "",
    }, index=table[id_col].astype(str).values)
    annotation.index.name = "probe_id"
    annotation = annotation[~annotation.index.duplicated(keep="first")]
    return annotation.reindex(probe_ids.astype(str))


def parse_geo_object(gse) -> RawDataset:
    """Convert a GEOparse GSE object into the project's raw-dataset dictionary."""
    if not gse.gsms:
        raise DatasetLoadError("The GEO record contains no samples.")
    try:
        expression = gse.pivot_samples("VALUE")
    except Exception as exc:
        raise DatasetLoadError(f"Could not assemble the expression matrix from GEO: {exc}") from exc

    expression.index = expression.index.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression.index.name = "probe_id"

    metadata = _build_metadata(gse).reindex(expression.columns)
    annotation = _build_annotation(gse, expression.index)
    platform_id = next(iter(gse.gpls.keys()), "unknown")
    title = _join(gse.metadata.get("title"))

    return {
        "gse_id": getattr(gse, "name", GSE_ID),
        "title": title,
        "platform_id": platform_id,
        "expression": expression,
        "metadata": metadata,
        "annotation": annotation,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_raw_dataset(gse_id: str = GSE_ID, data_dir: Path = DATA_DIR, force_download: bool = False) -> RawDataset:
    """Return the raw GEO dataset, using the on-disk cache when available."""
    ensure_dirs()
    cache = cache_path(gse_id, data_dir)

    if cache.exists() and not force_download:
        try:
            raw = joblib.load(cache)
            if isinstance(raw, dict) and {"expression", "metadata", "annotation"} <= set(raw):
                logger.info("Loaded %s from cache %s", gse_id, cache.name)
                return raw
            logger.warning("Cache file has an unexpected structure; rebuilding.")
        except Exception as exc:
            logger.warning("Cache unreadable (%s); rebuilding.", exc)
        cache.unlink(missing_ok=True)

    if force_download:
        soft_path(gse_id, data_dir).unlink(missing_ok=True)

    gse = _get_geo_object(gse_id, Path(data_dir))
    raw = parse_geo_object(gse)
    try:
        joblib.dump(raw, cache, compress=3)
    except Exception as exc:  # caching problems must never break the app
        logger.warning("Could not write cache: %s", exc)
    return raw
