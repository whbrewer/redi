"""
REDI Validate Module

Validate REDI pipeline output against a known ground-truth dataset.
Computes basic statistics, similarity metrics, and optionally AIDRIN metrics.

Extracted and generalized from tests/test_nasa_power_validation.py.
"""

import contextlib
import io
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AIDRIN_FAST_METRICS = [
    "completeness",
    "duplicity",
    "outliers",
    "correlations",
    "feature_relevance",
    "class_imbalance",
]
AIDRIN_MAX_ROWS = 50000


# ---------------------------------------------------------------------------
# Utility: numpy conversion
# ---------------------------------------------------------------------------

def to_numpy(values) -> np.ndarray:
    """Convert tensor-like input to a float32 numpy array.

    Handles torch tensors (lazy import), numpy arrays, and lists.
    """
    try:
        import torch
        if torch.is_tensor(values):
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
    except ImportError:
        pass
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def compute_metrics(values) -> dict:
    """Return basic distribution metrics for a numeric sequence."""
    arr = to_numpy(values).ravel()
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def compute_kl_divergence(a_values, b_values, bins: int = 50, eps: float = 1e-8) -> float:
    """Compute KL divergence between two 1D distributions using shared bins."""
    a = np.asarray(a_values, dtype=np.float32).ravel()
    b = np.asarray(b_values, dtype=np.float32).ravel()
    min_val = float(min(a.min(), b.min()))
    max_val = float(max(a.max(), b.max()))
    if min_val == max_val:
        return 0.0
    hist_a, bin_edges = np.histogram(a, bins=bins, range=(min_val, max_val), density=True)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
    p = hist_a + eps
    q = hist_b + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def compute_similarity_metrics(a_values, b_values) -> dict:
    """Return similarity metrics between two equal-length numeric sequences.

    Returns dict with keys: mae, rmse, corr, kl.
    """
    a = to_numpy(a_values).ravel()
    b = to_numpy(b_values).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for similarity metrics: {a.shape} vs {b.shape}")
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    kl = compute_kl_divergence(a, b, bins=50)
    return {"mae": mae, "rmse": rmse, "corr": corr, "kl": kl}


# ---------------------------------------------------------------------------
# Shape adjustment
# ---------------------------------------------------------------------------

def adjust_shape(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Truncate/pad a numpy array to match *target_shape*.

    Works by flattening to 2D (rows x last-dim), aligning rows and columns,
    then reshaping to the full target shape.
    """
    if data.shape == target_shape:
        return data

    # Figure out target rows and features from the full target shape.
    target_features = target_shape[-1]
    target_rows = 1
    for dim in target_shape[:-1]:
        target_rows *= dim

    # Collapse to 2D: (total_rows, last_dim)
    flat = data.reshape(-1, data.shape[-1])

    # Row alignment
    if flat.shape[0] >= target_rows:
        flat = flat[:target_rows, :]
    else:
        pad_rows = target_rows - flat.shape[0]
        flat = np.concatenate([flat, np.zeros((pad_rows, flat.shape[1]), dtype=flat.dtype)], axis=0)

    # Column alignment
    if flat.shape[1] >= target_features:
        flat = flat[:, :target_features]
    else:
        pad_cols = target_features - flat.shape[1]
        flat = np.concatenate([flat, np.zeros((flat.shape[0], pad_cols), dtype=flat.dtype)], axis=1)

    return flat.reshape(target_shape)


# ---------------------------------------------------------------------------
# RAPS NPZ helpers
# ---------------------------------------------------------------------------

# Scalar fields that are safe to compare directly (reference-frame independent).
# Absolute timestamp fields (submit_time, start_time, end_time) are intentionally
# excluded because REDI and RAPS ground-truth files use different time epochs.
# Duration-derived fields (wait_time, run_time) are added instead.
_RAPS_SCALAR_FIELDS = [
    "nodes_required", "time_limit", "expected_run_time",
    "priority", "partition",
    "cpu_cores_required", "gpu_units_required",
    "trace_quanta",
]


def _raps_jobs_to_matrix(jobs_array: np.ndarray) -> np.ndarray:
    """Convert a RAPS jobs object array (array of dicts) into a float32 matrix.

    Jobs are sorted by id for stable row alignment between REDI output and
    ground-truth files (which may store jobs in different orders).

    Uses only reference-frame-independent fields: scalar job properties plus
    derived durations (wait_time = start - submit, run_time = end - start).
    Absolute timestamp fields are excluded because REDI and RAPS snapshots
    use different time epochs.
    """
    jobs = [j for j in jobs_array if isinstance(j, dict)]
    if not jobs:
        raise ValueError("No valid job dicts found in RAPS jobs array")

    # Sort by numeric job id for consistent row ordering across datasets
    try:
        jobs = sorted(jobs, key=lambda j: int(j.get("id", 0)))
    except (TypeError, ValueError):
        pass  # leave unsorted if ids aren't numeric

    rows = []
    for job in jobs:
        row = []
        for field in _RAPS_SCALAR_FIELDS:
            val = job.get(field, 0)
            try:
                row.append(float(val) if val is not None else 0.0)
            except (TypeError, ValueError):
                row.append(0.0)
        # Derived duration fields (reference-frame independent)
        try:
            wait = float(job.get("start_time", 0)) - float(job.get("submit_time", 0))
        except (TypeError, ValueError):
            wait = 0.0
        try:
            run = float(job.get("end_time", 0)) - float(job.get("start_time", 0))
        except (TypeError, ValueError):
            run = 0.0
        row.extend([wait, run])
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def _install_raps_stub() -> None:
    """Install a meta path finder so any 'raps.*' import resolves to a stub.

    This lets numpy unpickle RAPS snapshot files without the raps package
    installed. The stub turns every raps class into a generic container.
    Safe to call multiple times.
    """
    import importlib.util
    import sys
    import types

    if any(getattr(f, "_is_raps_stub", False) for f in sys.meta_path):
        return

    class _GenericRapsObj:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

    class _RapsAttrModule(types.ModuleType):
        def __getattr__(self, name):
            return _GenericRapsObj

    class _RapsModLoader:
        def create_module(self, spec):
            mod = _RapsAttrModule(spec.name)
            mod.__path__ = []
            mod.__file__ = None
            mod.__spec__ = spec
            mod.__loader__ = self
            mod.__package__ = spec.name
            return mod
        def exec_module(self, module):
            pass

    class _RapsMetaFinder:
        _is_raps_stub = True
        def find_spec(self, fullname, path, target=None):
            if fullname == "raps" or fullname.startswith("raps."):
                return importlib.util.spec_from_loader(fullname, _RapsModLoader())
            return None

    sys.meta_path.insert(0, _RapsMetaFinder())


def _load_npz_with_raps_fallback(path: Path) -> np.ndarray:
    """Load an NPZ file, handling RAPS-format object arrays.

    Tries without pickle first (fast path for numeric arrays). If the file
    contains RAPS job dicts, loads with pickle and converts to a float matrix.
    Installs a RAPS module stub so ground-truth files saved by RAPS can be
    unpickled without the raps package being installed.
    """
    # Try strict (no pickle) first — covers plain numeric NPZs
    try:
        data = np.load(path)
        if "X" in data:
            return data["X"].astype(np.float32, copy=False)
        keys = list(data.keys())
        if keys:
            arr = data[keys[0]]
            if np.issubdtype(arr.dtype, np.number):
                return arr.astype(np.float32, copy=False)
    except Exception:
        pass

    # Install RAPS stub so pickle can resolve raps.* classes
    _install_raps_stub()

    # Fall back to pickle — needed for RAPS format
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # RAPS snapshot: look for 'jobs' array of dicts
    if "jobs" in keys:
        try:
            jobs = data["jobs"]
            if jobs.ndim == 1 and len(jobs) > 0 and isinstance(jobs[0], dict):
                return _raps_jobs_to_matrix(jobs)
        except Exception:
            pass

    # Generic fallback: first loadable numeric array (skip scalars)
    for key in keys:
        try:
            arr = data[key]
            if arr.shape == ():
                continue
            if np.issubdtype(arr.dtype, np.number):
                return arr.astype(np.float32, copy=False)
            # Object array of dicts
            if arr.dtype == object and arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], dict):
                return _raps_jobs_to_matrix(arr)
        except Exception:
            continue

    raise ValueError(f"No loadable numeric arrays found in {path}")


# ---------------------------------------------------------------------------
# RAPS sample comparison
# ---------------------------------------------------------------------------

_RAPS_SAMPLE_FIELDS = _RAPS_SCALAR_FIELDS + ["submit_time", "start_time", "end_time"]


def _load_raps_jobs_dict(path: Path) -> Optional[Dict[str, dict]]:
    """Load a RAPS NPZ file and return {job_id: job_dict}, or None if not RAPS format."""
    if path.suffix.lower() != ".npz":
        return None
    _install_raps_stub()
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "jobs" not in data.keys():
        return None
    try:
        jobs = data["jobs"]
        if jobs.ndim != 1 or len(jobs) == 0 or not isinstance(jobs[0], dict):
            return None
        return {str(j.get("id", i)): j for i, j in enumerate(jobs)}
    except Exception:
        return None


def raps_sample_comparison(
    redi_path: str,
    gt_path: str,
    n: int = 10,
    seed: int = 42,
) -> Optional[List[dict]]:
    """Compare n randomly sampled jobs between a REDI RAPS output and a ground-truth NPZ.

    Returns a list of per-job dicts with keys: id, fields (list of field comparison dicts).
    Returns None if either file is not in RAPS NPZ format.
    """
    import random

    redi_jobs = _load_raps_jobs_dict(Path(redi_path))
    gt_jobs   = _load_raps_jobs_dict(Path(gt_path))
    if redi_jobs is None or gt_jobs is None:
        return None

    shared = sorted(set(redi_jobs) & set(gt_jobs), key=lambda x: int(x) if x.isdigit() else x)
    if not shared:
        return None

    rng = random.Random(seed)
    sample_ids = rng.sample(shared, min(n, len(shared)))
    sample_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    results = []
    for jid in sample_ids:
        r, g = redi_jobs[jid], gt_jobs[jid]
        fields = []
        for field in _RAPS_SAMPLE_FIELDS:
            rv = r.get(field, None)
            gv = g.get(field, None)
            fields.append({"field": field, "redi": rv, "gt": gv, "match": rv == gv})
        # Derived durations
        for name, key_a, key_b in [("wait_time", "start_time", "submit_time"),
                                    ("run_time",  "end_time",   "start_time")]:
            try:
                rv = float(r.get(key_a, 0)) - float(r.get(key_b, 0))
                gv = float(g.get(key_a, 0)) - float(g.get(key_b, 0))
                fields.append({"field": name, "redi": rv, "gt": gv, "match": rv == gv})
            except (TypeError, ValueError):
                pass
        results.append({"id": jid, "fields": fields})
    return results


_RAPS_STAT_FIELDS = [
    "nodes_required", "time_limit", "expected_run_time",
    "priority", "wait_time", "run_time", "trace_quanta",
]


def raps_field_stats(
    redi_path: str,
    gt_path: str,
) -> Optional[List[dict]]:
    """Compute per-field stats over all aligned jobs between REDI and ground-truth RAPS NPZ.

    Returns a list of dicts (one per field) with keys:
      field, redi_mean, redi_std, redi_min, redi_max,
              gt_mean,   gt_std,   gt_min,   gt_max, mae, match
    Returns None if either file is not RAPS NPZ format.
    """
    redi_jobs = _load_raps_jobs_dict(Path(redi_path))
    gt_jobs   = _load_raps_jobs_dict(Path(gt_path))
    if redi_jobs is None or gt_jobs is None:
        return None

    shared = sorted(set(redi_jobs) & set(gt_jobs), key=lambda x: int(x) if x.isdigit() else x)
    if not shared:
        return None

    def _get(job, field):
        if field == "wait_time":
            return float(job.get("start_time", 0)) - float(job.get("submit_time", 0))
        if field == "run_time":
            return float(job.get("end_time", 0)) - float(job.get("start_time", 0))
        v = job.get(field, None)
        return float(v) if v is not None else 0.0

    results = []
    for field in _RAPS_STAT_FIELDS:
        redi_vals = np.array([_get(redi_jobs[jid], field) for jid in shared], dtype=np.float32)
        gt_vals   = np.array([_get(gt_jobs[jid],   field) for jid in shared], dtype=np.float32)
        mae = float(np.mean(np.abs(redi_vals - gt_vals)))
        results.append({
            "field":    field,
            "redi_mean": float(redi_vals.mean()),
            "redi_std":  float(redi_vals.std()),
            "redi_min":  float(redi_vals.min()),
            "redi_max":  float(redi_vals.max()),
            "gt_mean":   float(gt_vals.mean()),
            "gt_std":    float(gt_vals.std()),
            "gt_min":    float(gt_vals.min()),
            "gt_max":    float(gt_vals.max()),
            "mae":       mae,
            "match":     mae == 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: str, fmt: Optional[str] = None) -> np.ndarray:
    """Load ground truth data from a file.

    Supported formats: .pt, .npz, .npy, .csv, .parquet.
    Auto-detects format from extension if *fmt* is None.
    Returns a numpy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    if fmt is None:
        fmt = path.suffix.lower().lstrip(".")

    if fmt == "pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load .pt files") from exc
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, dict) and "X" in payload:
            tensor = payload["X"]
        elif isinstance(payload, torch.Tensor):
            tensor = payload
        elif hasattr(payload, "tensors") and payload.tensors:
            tensor = payload.tensors[0]
        elif isinstance(payload, (list, tuple)) and payload:
            first = payload[0]
            if isinstance(first, torch.Tensor):
                tensor = first
            else:
                raise TypeError(f"Cannot extract tensor from .pt payload: {type(payload)}")
        else:
            raise TypeError(f"Cannot extract tensor from .pt payload: {type(payload)}")
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    elif fmt == "npz":
        return _load_npz_with_raps_fallback(path)

    elif fmt == "npy":
        return np.load(path).astype(np.float32, copy=False)

    elif fmt == "csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.select_dtypes(include=[np.number]).values.astype(np.float32)

    elif fmt == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.select_dtypes(include=[np.number]).values.astype(np.float32)

    else:
        raise ValueError(f"Unsupported ground truth format: {fmt}")


# ---------------------------------------------------------------------------
# Load REDI output
# ---------------------------------------------------------------------------

def load_redi_output(output_dir: str) -> np.ndarray:
    """Load REDI pipeline output from an output directory or a direct file path.

    When given a file, loads it directly (NPZ, NPY, Parquet, CSV supported).
    When given a directory, reads metadata.json to determine format and
    concatenates the appropriate split files. Returns a numpy array.
    """
    path = Path(output_dir)

    # Direct file path — load it as ground-truth-style input
    if path.is_file():
        fmt = path.suffix.lower().lstrip(".")
        if fmt == "npz":
            return _load_npz_with_raps_fallback(path)
        elif fmt == "npy":
            return np.load(path).astype(np.float32, copy=False)
        elif fmt in ("parquet", "pq"):
            import pandas as pd
            df = pd.read_parquet(path)
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)
        elif fmt == "csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format for direct load: {fmt}")

    output_dir = path
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {output_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    output_format = metadata.get("output_format", "npz")

    if output_format == "npz":
        arrays = []
        for split_name in ("train", "val", "test"):
            split_path = output_dir / f"{split_name}.npz"
            if split_path.exists():
                arrays.append(_load_npz_with_raps_fallback(split_path))
        if not arrays:
            # Fall back: check saved_files in metadata for any .npz
            for saved in metadata.get("saved_files", []):
                saved_path = Path(saved)
                if not saved_path.is_absolute():
                    saved_path = output_dir / saved_path.name
                if saved_path.suffix == ".npz" and saved_path.exists():
                    arrays.append(_load_npz_with_raps_fallback(saved_path))
                    break
        if not arrays:
            # Last resort: glob for any .npz in the directory
            npz_files = sorted(output_dir.glob("*.npz"))
            for npz_path in npz_files:
                arrays.append(_load_npz_with_raps_fallback(npz_path))
                break
        if not arrays:
            raise FileNotFoundError(f"No split .npz files found in {output_dir}")
        return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)

    elif output_format == "pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load .pt output") from exc
        data_path = output_dir / "data.pt"
        if not data_path.exists():
            raise FileNotFoundError(f"data.pt not found in {output_dir}")
        payload = torch.load(data_path, weights_only=False)
        if isinstance(payload, dict) and "X" in payload:
            tensor = payload["X"]
        elif isinstance(payload, torch.Tensor):
            tensor = payload
        else:
            raise TypeError(f"Cannot extract tensor from data.pt: {type(payload)}")
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    elif output_format in ("parquet", "csv"):
        import pandas as pd
        frames = []
        for split_name in ("train", "val", "test"):
            ext = "parquet" if output_format == "parquet" else "csv"
            split_path = output_dir / f"{split_name}.{ext}"
            if split_path.exists():
                if output_format == "parquet":
                    frames.append(pd.read_parquet(split_path))
                else:
                    frames.append(pd.read_csv(split_path))
        if not frames:
            raise FileNotFoundError(f"No split {output_format} files found in {output_dir}")
        df = pd.concat(frames, ignore_index=True)
        return df.select_dtypes(include=[np.number]).values.astype(np.float32)

    else:
        raise ValueError(f"Unsupported REDI output format: {output_format}")


# ---------------------------------------------------------------------------
# AIDRIN functions (lazy imports)
# ---------------------------------------------------------------------------

def _normalize_payload_for_pt(payload) -> dict:
    """Normalize multiple payload forms into a torch.save()-able .pt payload."""
    import torch
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, torch.Tensor):
        return {"X": payload}
    if hasattr(payload, "tensors") and payload.tensors:
        norm = {"X": payload.tensors[0]}
        if len(payload.tensors) > 1:
            norm["y"] = payload.tensors[1]
        return norm
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if isinstance(first, torch.Tensor):
            norm = {"X": first}
            if len(payload) > 1 and isinstance(payload[1], torch.Tensor):
                norm["y"] = payload[1]
            return norm
    raise TypeError(f"Unsupported payload type for .pt normalization: {type(payload).__name__}")


def _trim_payload_rows(payload, max_rows: int = AIDRIN_MAX_ROWS):
    """Trim payload row count for expensive AIDRIN metrics."""
    import torch
    if isinstance(payload, dict):
        x = payload.get("X")
        if not isinstance(x, torch.Tensor) or x.shape[0] <= max_rows:
            return payload
        trimmed = dict(payload)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == x.shape[0]:
                trimmed[key] = value[:max_rows]
        return trimmed
    if isinstance(payload, torch.Tensor) and payload.ndim >= 1 and payload.shape[0] > max_rows:
        return payload[:max_rows]
    return payload


def compute_aidrin_metrics(payload) -> dict:
    """Run AIDRIN metrics on a payload by materializing a temporary .pt file.

    Requires torch and aidrin.headless. Returns AIDRIN results dict.
    """
    import torch
    try:
        from aidrin.headless import run_max_metrics
    except ImportError as exc:
        raise ImportError("aidrin is required for AIDRIN metrics: pip install aidrin") from exc

    normalized = _trim_payload_rows(_normalize_payload_for_pt(payload))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(normalized, tmp_path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_max_metrics(
                str(tmp_path),
                file_type=".pt",
                metrics=AIDRIN_FAST_METRICS,
                verbose=False,
                strip_visualizations=True,
                save_images=False,
            )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _collect_numeric_signature(obj, prefix: str = "", out: Optional[dict] = None) -> dict:
    """Collect a stable numeric signature from nested metric output objects."""
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_numeric_signature(obj[key], next_prefix, out)
        return out

    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj)
        next_prefix = prefix or "sequence"
        if arr.size == 0:
            return out
        if np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(np.float32, copy=False).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return out
            out[f"{next_prefix}.count"] = float(arr.size)
            out[f"{next_prefix}.mean"] = float(arr.mean())
            out[f"{next_prefix}.std"] = float(arr.std())
            out[f"{next_prefix}.min"] = float(arr.min())
            out[f"{next_prefix}.max"] = float(arr.max())
        return out

    if isinstance(obj, (bool, np.bool_)):
        return out
    if isinstance(obj, (int, float, np.integer, np.floating)):
        if np.isfinite(float(obj)):
            out[prefix or "value"] = float(obj)
    return out


def compute_aidrin_similarity(aidrin_a: dict, aidrin_b: dict) -> dict:
    """Compute similarity over shared numeric AIDRIN metric signatures."""
    a_sig = _collect_numeric_signature(aidrin_a.get("results", {}))
    b_sig = _collect_numeric_signature(aidrin_b.get("results", {}))
    shared_keys = sorted(set(a_sig.keys()).intersection(b_sig.keys()))
    if not shared_keys:
        return {
            "overlap_count": 0,
            "a_only_count": len(a_sig),
            "b_only_count": len(b_sig),
            "similarity": None,
        }
    a_vec = np.asarray([a_sig[key] for key in shared_keys], dtype=np.float32)
    b_vec = np.asarray([b_sig[key] for key in shared_keys], dtype=np.float32)
    similarity = compute_similarity_metrics(a_vec, b_vec)
    similarity["overlap_count"] = len(shared_keys)
    similarity["a_only_count"] = len(set(a_sig.keys()) - set(b_sig.keys()))
    similarity["b_only_count"] = len(set(b_sig.keys()) - set(a_sig.keys()))
    return similarity


def summarize_aidrin_results(aidrin_metrics: dict) -> dict:
    """Create a compact, printable dict of key AIDRIN metric outputs."""
    results = aidrin_metrics.get("results", {})
    summary = {}

    completeness = results.get("completeness", {})
    if isinstance(completeness, dict):
        comp_scores = completeness.get("Completeness scores", {})
        summary["completeness"] = {
            "overall": completeness.get("Overall Completeness"),
            "features": len(comp_scores) if isinstance(comp_scores, dict) else None,
        }

    duplicity = results.get("duplicity", {})
    if isinstance(duplicity, dict):
        dup_scores = duplicity.get("Duplicity scores", {})
        overall_dup = None
        if isinstance(dup_scores, dict):
            overall_dup = dup_scores.get("Overall duplicity of the dataset")
        summary["duplicity"] = {"overall": overall_dup}

    outliers = results.get("outliers", {})
    if isinstance(outliers, dict):
        out_scores = outliers.get("Outlier scores", {})
        overall_out = None
        if isinstance(out_scores, dict):
            overall_out = out_scores.get("Overall outlier score")
        summary["outliers"] = {"overall": overall_out}

    correlations = results.get("correlations", {})
    if isinstance(correlations, dict):
        num = correlations.get("Correlations Analysis Numerical", {})
        num_scores = num.get("Correlation Scores", {}) if isinstance(num, dict) else {}
        cat = correlations.get("Correlations Analysis Categorical", {})
        cat_scores = cat.get("Theil's U Scores", {}) if isinstance(cat, dict) else {}
        summary["correlations"] = {
            "numerical_pairs": len(num_scores) if isinstance(num_scores, dict) else 0,
            "categorical_pairs": len(cat_scores) if isinstance(cat_scores, dict) else 0,
        }

    feature_rel = results.get("feature_relevance", {})
    if isinstance(feature_rel, dict):
        pearson = feature_rel.get("Pearson Correlation to Target", {})
        summary["feature_relevance"] = {
            "features_scored": len(pearson) if isinstance(pearson, dict) else 0,
        }

    class_imb = results.get("class_imbalance", {})
    if isinstance(class_imb, dict):
        imbalance = class_imb.get("Imbalance degree", {})
        summary["class_imbalance"] = {
            "score": imbalance.get("Imbalance Degree Score")
            if isinstance(imbalance, dict)
            else None,
        }

    if aidrin_metrics.get("skipped"):
        summary["skipped"] = aidrin_metrics["skipped"]

    return summary


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Container for validation comparison results."""

    redi_dir: str
    ground_truth_path: str
    redi_shape: tuple
    ground_truth_shape: tuple
    aligned: bool
    redi_stats: dict
    ground_truth_stats: dict
    similarity: dict
    per_feature: Optional[list] = None
    aidrin: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "redi_dir": self.redi_dir,
            "ground_truth_path": self.ground_truth_path,
            "redi_shape": list(self.redi_shape),
            "ground_truth_shape": list(self.ground_truth_shape),
            "aligned": self.aligned,
            "redi_stats": self.redi_stats,
            "ground_truth_stats": self.ground_truth_stats,
            "similarity": self.similarity,
            "per_feature": self.per_feature,
            "aidrin": self.aidrin,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_validation(
    redi_dir: str,
    config: dict,
    aidrin: bool = False,
    per_feature: bool = False,
) -> ValidationResult:
    """Run validation comparing REDI output to ground truth.

    Args:
        redi_dir: Path to REDI output directory (must contain metadata.json).
        config: Validation config dict with at least 'ground_truth' key.
            Optional keys: 'format', 'alignment', 'metrics'.
        aidrin: If True, run AIDRIN metric comparison.
        per_feature: If True, compute per-feature similarity breakdown.

    Returns:
        ValidationResult with all comparison data.
    """
    ground_truth_path = config.get("ground_truth")
    if not ground_truth_path:
        raise ValueError("config must contain 'ground_truth' key with path to ground truth file")

    gt_format = config.get("format")
    alignment_cfg = config.get("alignment", {})
    alignment_method = alignment_cfg.get("method", "truncate_pad")

    # Load data
    logger.info("Loading REDI output from %s", redi_dir)
    redi_data = load_redi_output(redi_dir)
    redi_shape = redi_data.shape

    logger.info("Loading ground truth from %s", ground_truth_path)
    gt_data = load_ground_truth(ground_truth_path, fmt=gt_format)
    gt_shape = gt_data.shape

    logger.info("REDI shape: %s, Ground truth shape: %s", redi_shape, gt_shape)

    # Align shapes
    aligned = False
    target_shape = alignment_cfg.get("target_shape")
    if target_shape is not None:
        target_shape = tuple(target_shape)
    else:
        target_shape = gt_shape

    if redi_data.shape != target_shape:
        if alignment_method == "error":
            raise ValueError(
                f"Shape mismatch: REDI {redi_data.shape} vs target {target_shape}. "
                f"Set alignment.method to 'truncate_pad' to auto-align."
            )
        redi_data = adjust_shape(redi_data, target_shape)
        aligned = True
        logger.info("Aligned REDI data to shape: %s", redi_data.shape)
    else:
        aligned = redi_data.shape == target_shape

    # Also align ground truth if needed
    if gt_data.shape != target_shape:
        gt_data = adjust_shape(gt_data, target_shape)

    # Compute basic stats
    metrics_cfg = config.get("metrics", {})
    redi_stats = compute_metrics(redi_data) if metrics_cfg.get("basic_stats", True) else {}
    gt_stats = compute_metrics(gt_data) if metrics_cfg.get("basic_stats", True) else {}

    # Compute similarity
    similarity = {}
    if metrics_cfg.get("similarity", True):
        similarity = compute_similarity_metrics(redi_data.ravel(), gt_data.ravel())

    # Per-feature similarity
    per_feature_results = None
    if per_feature or metrics_cfg.get("per_feature", False):
        n_features = target_shape[-1]
        redi_2d = redi_data.reshape(-1, n_features)
        gt_2d = gt_data.reshape(-1, n_features)
        per_feature_results = []
        for i in range(n_features):
            feat_sim = compute_similarity_metrics(redi_2d[:, i], gt_2d[:, i])
            feat_sim["feature_index"] = i
            per_feature_results.append(feat_sim)

    # AIDRIN comparison
    aidrin_results = None
    if aidrin:
        try:
            import torch
            logger.info("Running AIDRIN metrics on REDI output...")
            redi_tensor = torch.tensor(redi_data)
            aidrin_redi = compute_aidrin_metrics({"X": redi_tensor})

            logger.info("Running AIDRIN metrics on ground truth...")
            gt_tensor = torch.tensor(gt_data)
            aidrin_gt = compute_aidrin_metrics({"X": gt_tensor})

            logger.info("Computing AIDRIN similarity...")
            aidrin_sim = compute_aidrin_similarity(aidrin_redi, aidrin_gt)

            aidrin_results = {
                "redi_summary": summarize_aidrin_results(aidrin_redi),
                "ground_truth_summary": summarize_aidrin_results(aidrin_gt),
                "similarity": aidrin_sim,
            }
        except ImportError as exc:
            logger.warning("AIDRIN not available: %s", exc)
            aidrin_results = {"error": str(exc)}

    return ValidationResult(
        redi_dir=str(redi_dir),
        ground_truth_path=str(ground_truth_path),
        redi_shape=redi_shape,
        ground_truth_shape=gt_shape,
        aligned=aligned,
        redi_stats=redi_stats,
        ground_truth_stats=gt_stats,
        similarity=similarity,
        per_feature=per_feature_results,
        aidrin=aidrin_results,
    )
