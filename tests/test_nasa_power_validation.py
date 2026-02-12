"""Validation tests for NASA POWER outputs."""
import contextlib
import io
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import requests

AIDRIN_FAST_METRICS = [
    "completeness",
    "duplicity",
    "outliers",
    "correlations",
    "feature_relevance",
    "class_imbalance",
]
AIDRIN_MAX_ROWS = 50000


def load_nasa_power_config():
    """Load NASA POWER config and resolve repo-relative paths."""
    from redi.config import load_yaml_config

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "examples" / "nasa_power_config.yaml"
    config = load_yaml_config(str(config_path))
    input_path = Path(config["input_path"])
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return config, input_path, output_dir


def download_raw_json():
    base_url = (
        "https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/resolve/main"
    )
    raw_path = "raw/usa/usa_0_data.json"
    raw_url = f"{base_url}/{raw_path}"

    _, input_path, _ = load_nasa_power_config()
    input_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.info("Downloading RAW JSON from %s", raw_url)
        response = requests.get(raw_url, timeout=60)
        response.raise_for_status()
        input_path.write_bytes(response.content)
        logging.info("Saved RAW JSON to %s", input_path)
    else:
        logging.info("Using cached RAW JSON at %s", input_path)

    return json.loads(input_path.read_text())


def download_ready_pt():
    base_url = (
        "https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/resolve/main"
    )
    ready_path = "pytorch/weather_dataset_daily_0.pt"
    ready_url = f"{base_url}/{ready_path}"

    _, input_path, _ = load_nasa_power_config()
    output_path = input_path.parent / "weather_dataset_daily_0.pt"

    if not output_path.exists():
        logging.info("Downloading READY PT from %s", ready_url)
        response = requests.get(ready_url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logging.info("Saved READY PT to %s", output_path)
    else:
        logging.info("Using cached READY PT at %s", output_path)

    torch = pytest.importorskip("torch")
    return torch.load(output_path, weights_only=False)


def load_redi_output():
    """Load the newest NASA POWER Redi output and print the tensor shape."""
    _, _, output_dir = load_nasa_power_config()

    if not output_dir.exists():
        pytest.skip(f"Output directory not found: {output_dir}")

    output_dirs = [path for path in output_dir.glob("usa_0_data_*") if path.is_dir()]
    if not output_dirs:
        pytest.skip(f"No output directories matching usa_0_data_* in {output_dir}")

    latest_dir = max(output_dirs, key=lambda path: path.stat().st_mtime)
    data_path = latest_dir / "data.pt"
    if not data_path.exists():
        pytest.skip(f"data.pt not found in {latest_dir}")

    torch = pytest.importorskip("torch")
    payload = torch.load(data_path)
    tensor = payload["X"] if isinstance(payload, dict) and "X" in payload else payload

    return tensor


def _extract_shape(payload):
    if hasattr(payload, "shape"):
        return payload.shape
    if isinstance(payload, dict) and "X" in payload and hasattr(payload["X"], "shape"):
        return payload["X"].shape
    if hasattr(payload, "tensors") and payload.tensors:
        return payload.tensors[0].shape
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if hasattr(first, "shape"):
            return first.shape
    return None


def _to_tensor(payload):
    torch = pytest.importorskip("torch")
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and "X" in payload and isinstance(payload["X"], torch.Tensor):
        return payload["X"]
    if hasattr(payload, "tensors") and payload.tensors:
        return payload.tensors[0]
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError("Unsupported payload for tensor conversion")


def _adjust_redi_to_ready_shape(redi_payload, ready_shape):
    torch = pytest.importorskip("torch")
    redi_tensor = _to_tensor(redi_payload)
    # Target layout matches the ready dataset: (locations, timesteps, features).
    target_samples, target_steps, target_features = ready_shape
    # Flatten target into a single row count for easier slicing/padding.
    target_rows = target_samples * target_steps

    # Collapse any existing structure to 2D so we can re-shape deterministically.
    flat = redi_tensor.reshape(-1, redi_tensor.shape[-1])
    # If we have more rows than needed, truncate to the exact target size.
    if flat.shape[0] >= target_rows:
        flat = flat[:target_rows, :]
    else:
        # If we have fewer rows, pad with zeros to reach the target count.
        pad_rows = target_rows - flat.shape[0]
        flat = torch.cat([flat, torch.zeros(pad_rows, flat.shape[1])], dim=0)

    # Align feature dimension to the ready dataset's feature count.
    if flat.shape[1] >= target_features:
        # Drop extra features on the right to fit the target.
        flat = flat[:, :target_features]
    else:
        # Append zero-filled feature columns if we have too few.
        pad_cols = target_features - flat.shape[1]
        flat = torch.cat([flat, torch.zeros(flat.shape[0], pad_cols)], dim=1)

    # Final reshape into (locations, timesteps, features).
    return flat.reshape(target_samples, target_steps, target_features)




def compute_metrics(values):
    """Return basic distribution metrics for a 1D numeric sequence."""
    arr = _to_numpy(values)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_similarity_metrics(a_values, b_values):
    """Return similarity metrics between two equal-length numeric sequences."""
    a = _to_numpy(a_values)
    b = _to_numpy(b_values)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for similarity metrics: {a.shape} vs {b.shape}")
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    kl = compute_kl_divergence(a, b, bins=50)
    return {"mae": mae, "rmse": rmse, "corr": corr, "kl": kl}


def compute_kl_divergence(a_values, b_values, bins=50, eps=1e-8):
    """Compute KL divergence between two 1D distributions using shared bins."""
    a = np.asarray(a_values, dtype=np.float32)
    b = np.asarray(b_values, dtype=np.float32)
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


def _to_numpy(values):
    """Convert tensor-like input to a float32 numpy array."""
    torch = pytest.importorskip("torch")
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def _normalize_payload_for_pt(payload):
    """Normalize multiple payload forms into a torch.save()-able .pt payload."""
    torch = pytest.importorskip("torch")
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


def _trim_payload_rows(payload, max_rows=AIDRIN_MAX_ROWS):
    """Trim payload row count for expensive AIDRIN metrics."""
    torch = pytest.importorskip("torch")
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


def compute_aidrin_metrics(payload):
    """Run AIDRIN metrics on a payload by materializing a temporary .pt file."""
    torch = pytest.importorskip("torch")
    aidrin_headless = pytest.importorskip("aidrin.headless")
    run_max_metrics = getattr(aidrin_headless, "run_max_metrics", None)
    if run_max_metrics is None:
        pytest.skip("aidrin.headless.run_max_metrics is unavailable; upgrade/reinstall aidrin.")

    normalized = _trim_payload_rows(_normalize_payload_for_pt(payload))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(normalized, tmp_path)
        # AIDRIN metric internals can be very chatty (print-heavy); suppress here
        # so this test only shows its final summary lines.
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


def _collect_numeric_signature(obj, prefix="", out=None):
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


def compute_aidrin_similarity_metrics(aidrin_a, aidrin_b):
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


def _summarize_aidrin_results_for_print(aidrin_metrics):
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
            "features_scored": len(pearson) if isinstance(pearson, dict) else 0
        }

    class_imb = results.get("class_imbalance", {})
    if isinstance(class_imb, dict):
        imbalance = class_imb.get("Imbalance degree", {})
        summary["class_imbalance"] = {
            "score": imbalance.get("Imbalance Degree Score")
            if isinstance(imbalance, dict)
            else None
        }

    if aidrin_metrics.get("skipped"):
        summary["skipped"] = aidrin_metrics["skipped"]

    return summary


def test_redi_validation():
    redi_tensor = load_redi_output()
    ready_tensor = download_ready_pt()

    redi_shape = _extract_shape(redi_tensor)
    ready_shape = _extract_shape(ready_tensor)

    assert redi_shape is not None
    assert ready_shape is not None

    print()
    print("Redi shape", redi_shape)
    print("Ready shape", ready_shape)
    adjusted = _adjust_redi_to_ready_shape(redi_tensor, ready_shape)
    print("Adjusted Redi shape", adjusted.shape)

    ready_base = _to_tensor(ready_tensor)
    redi_base = _to_tensor(redi_tensor)
    redi_metrics = compute_metrics(redi_base)
    ready_metrics = compute_metrics(ready_base)

    similarity = compute_similarity_metrics(adjusted.reshape(-1), ready_base.reshape(-1))
    t0 = time.perf_counter()
    print("Running AIDRIN metrics on REDI payload...", flush=True)
    aidrin_redi_metrics = compute_aidrin_metrics(redi_tensor)
    print(f"Done AIDRIN REDI metrics in {time.perf_counter() - t0:.2f}s", flush=True)
    t1 = time.perf_counter()
    print("Running AIDRIN metrics on READY payload...", flush=True)
    aidrin_ready_metrics = compute_aidrin_metrics(ready_tensor)
    print(f"Done AIDRIN READY metrics in {time.perf_counter() - t1:.2f}s", flush=True)
    t2 = time.perf_counter()
    print("Computing AIDRIN similarity...", flush=True)
    aidrin_similarity = compute_aidrin_similarity_metrics(
        aidrin_redi_metrics,
        aidrin_ready_metrics,
    )
    print(f"Done AIDRIN similarity in {time.perf_counter() - t2:.2f}s", flush=True)

    print("REDIFIED metrics:", redi_metrics)
    print("READY metrics:", ready_metrics)
    print("Similarity metrics:", similarity)
    print("AIDRIN REDIFIED metrics:", _summarize_aidrin_results_for_print(aidrin_redi_metrics))
    print("AIDRIN READY metrics:", _summarize_aidrin_results_for_print(aidrin_ready_metrics))
    print("AIDRIN REDIFIED metrics count:", len(aidrin_redi_metrics.get("results", {})))
    print("AIDRIN READY metrics count:", len(aidrin_ready_metrics.get("results", {})))
    print("AIDRIN similarity metrics:", aidrin_similarity)
