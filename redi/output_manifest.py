"""
Output manifest for metadata.json.

``describe_outputs`` lists every file in a run's output directory with its
format, size, SHA-256 and record-level structure (array names, shapes and
dtypes; row counts and column types), read from file headers without loading
the data. Directories holding many files of one type (e.g. one ``.pt`` per
structure) collapse into a single grouped entry, and Zarr/LMDB stores are
described as one entry each.

``json_safe`` converts ``ctx.metadata`` values for JSON: small values pass
through, while bulky or non-JSON values (in-memory records, large arrays) are
replaced by short summaries, so metadata.json never serializes raw data.

``build_metadata`` assembles the metadata.json document from both.

Plugins can report exact record counts per output path (relative to the
output directory) in ``ctx.metadata["output_records"]``; otherwise the count
is inferred from the file (rows, or a leading dimension shared by all arrays).
"""

from __future__ import annotations

import csv
import enum
import hashlib
import json
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# json_safe limits
MAX_PRIMITIVE_LIST = 10_000      # longer lists of plain values are summarized
MAX_OBJECT_LIST    = 20          # longer lists of non-JSON objects are summarized
MAX_ARRAY_SIZE     = 1_000       # larger numpy arrays are summarized
MAX_STR_FALLBACK   = 500         # longer str() of unknown objects is summarized
MAX_VALUE_BYTES    = 1 << 20     # top-level metadata values above this are omitted

# describe_outputs limits
MAX_FILES_PER_GROUP = 50         # more same-suffix files in one dir -> one grouped entry

STORE_SUFFIXES = {".zarr": "zarr", ".lmdb": "lmdb"}
FORMATS = {
    ".npz": "npz", ".npy": "npy",
    ".parquet": "parquet", ".pq": "parquet",
    ".csv": "csv",
    ".h5": "hdf5", ".hdf5": "hdf5", ".nc": "netcdf",
    ".pt": "pt", ".pth": "pt",
    ".json": "json", ".jsonl": "jsonl",
    ".md": "markdown", ".sh": "shell", ".sha256": "checksums",
    ".pdf": "pdf", ".png": "png",
}

_PRIMITIVES = (str, int, float, bool, type(None))


# ---------------------------------------------------------------------------
# JSON-safe metadata values
# ---------------------------------------------------------------------------

def _summary(kind: str, **fields) -> Dict[str, Any]:
    return {"__summary__": kind, **fields}


def json_safe(value: Any, _depth: int = 0) -> Any:
    """Return a JSON-serializable version of ``value`` with bulky parts summarized."""
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if _depth > 20:
        return _summary(type(value).__name__)
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if all(isinstance(v, _PRIMITIVES) for v in items):
            if len(items) <= MAX_PRIMITIVE_LIST:
                return items
            return _summary("list", length=len(items))
        json_like = all(isinstance(v, _PRIMITIVES + (dict, list, tuple)) for v in items)
        if not json_like and len(items) > MAX_OBJECT_LIST:
            return _summary(
                "list", length=len(items),
                item_types=sorted({type(v).__name__ for v in items[:1000]}),
            )
        return [json_safe(v, _depth + 1) for v in items]
    if isinstance(value, np.ndarray):
        if value.size <= MAX_ARRAY_SIZE:
            return value.tolist()
        return _summary("ndarray", shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, enum.Enum):
        return json_safe(value.value, _depth + 1)
    text = str(value)
    if len(text) <= MAX_STR_FALLBACK:
        return text
    shape = getattr(value, "shape", None)
    return _summary(
        type(value).__name__,
        **({"shape": [int(s) for s in shape]} if isinstance(shape, tuple) else {}),
        **({"length": len(value)} if hasattr(value, "__len__") else {}),
    )


def safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-safe copy of ``ctx.metadata`` for metadata.json. Keeps the top-level
    keys REDI has always written (str/int/float/bool/list/dict values) and
    omits any value whose JSON form still exceeds MAX_VALUE_BYTES.
    """
    out = {}
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool, list, dict)):
            continue
        safe = json_safe(value)
        size = len(json.dumps(safe, default=str))
        if size > MAX_VALUE_BYTES:
            logger.warning("metadata.json: omitting %r (%d bytes as JSON)", key, size)
            safe = _summary(
                "omitted", type=type(value).__name__, json_bytes=size,
                **({"length": len(value)} if hasattr(value, "__len__") else {}),
            )
        out[key] = safe
    return out


# ---------------------------------------------------------------------------
# Per-format record readers (headers only)
# ---------------------------------------------------------------------------

def _npy_header(f) -> Dict[str, Any]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    return {"shape": [int(s) for s in shape], "dtype": str(dtype)}


def _common_leading_dim(arrays: Dict[str, Dict[str, Any]]) -> Optional[int]:
    dims = {a["shape"][0] for a in arrays.values() if a.get("shape")}
    return dims.pop() if len(dims) == 1 else None


def _records_npz(path: Path) -> Dict[str, Any]:
    arrays = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".npy"):
                with zf.open(name) as f:
                    arrays[name[:-4]] = _npy_header(f)
    return {"arrays": arrays, "n_records": _common_leading_dim(arrays)}


def _records_npy(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        header = _npy_header(f)
    return {"arrays": {path.stem: header},
            "n_records": header["shape"][0] if header["shape"] else None}


def _records_parquet(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    return {
        "n_records":    pf.metadata.num_rows,
        "n_row_groups": pf.metadata.num_row_groups,
        "columns":      {f.name: str(f.type) for f in pf.schema_arrow},
    }


def _records_csv(path: Path) -> Dict[str, Any]:
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    newlines, last = 0, b"\n"
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            newlines += block.count(b"\n")
            last = block[-1:]
    lines = newlines + (0 if last == b"\n" else 1)
    return {"n_records": max(lines - 1, 0), "columns": header}


def _records_hdf5(path: Path) -> Dict[str, Any]:
    import h5py
    arrays = {}
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                arrays[name] = {"shape": [int(s) for s in obj.shape], "dtype": str(obj.dtype)}
        h5.visititems(visit)
    return {"arrays": arrays, "n_records": _common_leading_dim(arrays)}


def _records_zarr(path: Path) -> Dict[str, Any]:
    arrays = {}
    for meta in list(path.rglob(".zarray")) + list(path.rglob("zarr.json")):
        info = json.loads(meta.read_text())
        if meta.name == "zarr.json" and info.get("node_type") != "array":
            continue
        name = meta.parent.relative_to(path).as_posix() or "."
        arrays[name] = {"shape": info.get("shape"),
                        "dtype": str(info.get("dtype", info.get("data_type")))}
    return {"arrays": arrays}


_RECORD_READERS = {
    "npz": _records_npz, "npy": _records_npy, "parquet": _records_parquet,
    "csv": _records_csv, "hdf5": _records_hdf5, "netcdf": _records_hdf5,
}


# ---------------------------------------------------------------------------
# Output directory description
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _tree_size(path: Path) -> (int, int):
    n_files = size = 0
    for p in path.rglob("*"):
        if p.is_file():
            n_files += 1
            size    += p.stat().st_size
    return n_files, size


def describe_file(path: Path, root: Path, checksum: bool = True,
                  n_records: Optional[int] = None) -> Dict[str, Any]:
    """Describe one output file: format, size, checksum and record structure."""
    fmt   = FORMATS.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "unknown")
    entry = {
        "path":       path.relative_to(root).as_posix(),
        "format":     fmt,
        "size_bytes": path.stat().st_size,
    }
    if checksum:
        entry["sha256"] = _sha256(path)
    reader = _RECORD_READERS.get(fmt)
    if reader is not None:
        try:
            entry.update({k: v for k, v in reader(path).items() if v is not None})
        except Exception as exc:   # missing optional reader dependency, corrupt file, ...
            entry["records_error"] = f"{type(exc).__name__}: {exc}"
    if n_records is not None:
        entry["n_records"] = int(n_records)
    return entry


def describe_outputs(output_dir, exclude: Iterable[str] = ("metadata.json",),
                     checksum: bool = True,
                     record_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Describe every file under ``output_dir`` (see module docstring).

    ``exclude`` names files at the top level to skip; ``record_counts`` maps a
    relative path (file, or "dir/*.ext" group) to an exact record count.
    """
    root          = Path(output_dir)
    exclude       = set(exclude)
    record_counts = {str(k).rstrip("/"): v for k, v in (record_counts or {}).items()}
    entries: List[Dict[str, Any]] = []

    if not root.is_dir():
        return {"directory": str(root), "n_files": 0, "total_size_bytes": 0, "files": []}

    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)

        # Stores (Zarr, LMDB directories) are one entry each; don't descend
        for d in sorted(dirnames):
            fmt = STORE_SUFFIXES.get(Path(d).suffix.lower())
            if fmt:
                store = here / d
                n_files, size = _tree_size(store)
                entry = {"path": store.relative_to(root).as_posix() + "/", "format": fmt,
                         "size_bytes": size, "n_files": n_files}
                if fmt == "zarr":
                    try:
                        entry.update(_records_zarr(store))
                    except Exception as exc:
                        entry["records_error"] = f"{type(exc).__name__}: {exc}"
                entries.append(entry)
        dirnames[:] = sorted(d for d in dirnames if Path(d).suffix.lower() not in STORE_SUFFIXES)

        files = sorted(f for f in filenames if not (here == root and f in exclude))
        by_suffix = defaultdict(list)
        for f in files:
            by_suffix[Path(f).suffix.lower()].append(f)

        for suffix, group in sorted(by_suffix.items()):
            if len(group) > MAX_FILES_PER_GROUP:
                rel_dir = here.relative_to(root).as_posix()
                pattern = f"{'' if rel_dir == '.' else rel_dir + '/'}*{suffix}"
                entry = {
                    "path":       pattern,
                    "format":     FORMATS.get(suffix, suffix.lstrip(".") or "unknown"),
                    "n_files":    len(group),
                    "size_bytes": sum((here / f).stat().st_size for f in group),
                    "first":      group[0],
                    "last":       group[-1],
                }
                count = record_counts.get(pattern, record_counts.get(pattern.rsplit("/", 1)[0]))
                if count is not None:
                    entry["n_records"] = int(count)
                entries.append(entry)
            else:
                for f in group:
                    path = here / f
                    rel  = path.relative_to(root).as_posix()
                    entries.append(describe_file(path, root, checksum, record_counts.get(rel)))

    entries.sort(key=lambda e: e["path"])
    return {
        "directory":        str(root),
        "n_files":          sum(e.get("n_files", 1) for e in entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "files":            entries,
    }


# ---------------------------------------------------------------------------
# metadata.json document
# ---------------------------------------------------------------------------

def build_metadata(ctx, input_file, domain, provenance: Optional[dict] = None) -> Dict[str, Any]:
    """
    Assemble metadata.json: run info, JSON-safe ctx.metadata, provenance (if
    given) and an ``outputs`` manifest of ctx.output_dir. Checksums can be
    turned off with ``manifest_checksums: false`` in the run config.
    """
    config   = ctx.config if isinstance(getattr(ctx, "config", None), dict) else {}
    metadata = {
        "input_file":       str(input_file),
        "domain":           domain,
        "completed_stages": [getattr(s, "value", s) for s in (ctx.completed_stages or [])],
        "errors":           ctx.errors,
        **safe_metadata(ctx.metadata),
    }
    if provenance is not None:
        metadata["provenance"] = provenance
    try:
        metadata["outputs"] = describe_outputs(
            ctx.output_dir,
            checksum=bool(config.get("manifest_checksums", True)),
            record_counts=ctx.metadata.get("output_records"),
        )
    except Exception as exc:
        logger.warning("metadata.json: could not describe outputs: %s", exc)
        metadata["outputs"] = {"error": f"{type(exc).__name__}: {exc}"}
    return metadata
