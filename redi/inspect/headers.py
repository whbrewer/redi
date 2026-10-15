"""Header-only metadata extraction (``redi inspect --headers-only``).

Opens files lazily and reports structure (groups, datasets, shapes, dtypes,
attributes, schemas) without loading array data, running domain detection,
or applying any transforms.
"""

import glob
import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

H5_EXTS = {".h5", ".hdf5", ".hdf", ".he5", ".nc4"}
NC_EXTS = {".nc", ".cdf"}
SUPPORTED_EXTS = H5_EXTS | NC_EXTS | {".npy", ".npz", ".parquet", ".pq", ".csv", ".tsv"}

_MAX_ATTR_CHARS = 200


def _jsonable(value: Any) -> Any:
    """Convert attribute values to short, JSON-safe representations."""
    try:
        import numpy as np
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist() if value.size <= 16 else f"<array {value.dtype} {value.shape}>"
    except ImportError:
        pass
    text = str(value)
    if len(text) > _MAX_ATTR_CHARS:
        return text[:_MAX_ATTR_CHARS] + "..."
    return value if isinstance(value, (int, float, bool, str, list)) else text


def _attrs(obj) -> Dict[str, Any]:
    out = {}
    for k in obj.attrs.keys():
        try:
            out[k] = _jsonable(obj.attrs[k])
        except Exception as e:  # unreadable attribute should not abort the scan
            out[k] = f"<unreadable: {e}>"
    return out


def _read_hdf5(path: Path) -> Dict[str, Any]:
    import h5py

    datasets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    groups: List[str] = []

    with h5py.File(path, "r") as f:
        root_attrs = _attrs(f)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "attrs": _attrs(obj),
                }
            else:
                groups.append(name)

        f.visititems(visit)

    return {"attrs": root_attrs, "groups": groups, "datasets": datasets}


def _read_netcdf(path: Path) -> Dict[str, Any]:
    # NetCDF4 files are HDF5 underneath; classic NetCDF3 needs xarray/netCDF4.
    try:
        return _read_hdf5(path)
    except Exception:
        pass
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:  # lazy: no array loads
        datasets = OrderedDict(
            (name, {
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "dims": list(var.dims),
                "attrs": {k: _jsonable(v) for k, v in var.attrs.items()},
            })
            for name, var in ds.variables.items()
        )
        return {
            "attrs": {k: _jsonable(v) for k, v in ds.attrs.items()},
            "groups": [],
            "datasets": datasets,
        }


def _read_npy_header(fp) -> Dict[str, Any]:
    from numpy.lib import format as npf

    version = npf.read_magic(fp)
    reader = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
    shape, fortran, dtype = reader(fp)
    return {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}


def _read_npy(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fp:
        return {"attrs": {}, "groups": [], "datasets": {"array": _read_npy_header(fp)}}


def _read_npz(path: Path) -> Dict[str, Any]:
    datasets = OrderedDict()
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".npy"):
                with zf.open(name) as fp:
                    datasets[name[:-4]] = _read_npy_header(fp)
    return {"attrs": {}, "groups": [], "datasets": datasets}


def _read_parquet(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    datasets = OrderedDict(
        (field.name, {"shape": [meta.num_rows], "dtype": str(field.type)})
        for field in pf.schema_arrow
    )
    kv = {}
    for k, v in (pf.schema_arrow.metadata or {}).items():
        kv[k.decode("utf-8", "replace")] = _jsonable(v)
    return {
        "attrs": kv,
        "groups": [],
        "datasets": datasets,
        "row_groups": meta.num_row_groups,
        "created_by": meta.created_by,
    }


def _read_delimited(path: Path) -> Dict[str, Any]:
    delim = "\t" if path.suffix == ".tsv" else ","
    with open(path, "r", errors="replace") as fp:
        header = fp.readline().rstrip("\r\n")
    cols = [c.strip().strip('"') for c in header.split(delim)]
    return {
        "attrs": {},
        "groups": [],
        "datasets": OrderedDict((c, {"shape": [None], "dtype": "unknown"}) for c in cols),
    }


def _read_one(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext in H5_EXTS:
        return _read_hdf5(path)
    if ext in NC_EXTS:
        return _read_netcdf(path)
    if ext == ".npy":
        return _read_npy(path)
    if ext == ".npz":
        return _read_npz(path)
    if ext in {".parquet", ".pq"}:
        return _read_parquet(path)
    if ext in {".csv", ".tsv"}:
        return _read_delimited(path)
    raise ValueError(f"unsupported format: {ext or '(none)'}")


def expand_inputs(spec: str) -> List[Path]:
    """Resolve a file, directory (recursive), or glob into supported files."""
    p = Path(spec)
    if p.is_dir():
        candidates = [q for q in p.rglob("*") if q.is_file()]
    elif p.is_file():
        candidates = [p]
    else:
        candidates = [Path(m) for m in glob.glob(spec, recursive=True) if Path(m).is_file()]
    return sorted(q for q in candidates if q.suffix.lower() in SUPPORTED_EXTS)


def _signature(info: Dict[str, Any]) -> str:
    """Schema fingerprint: names, dtypes and trailing dims (leading dim ignored)."""
    parts = [
        f"{name}:{d['dtype']}:{d['shape'][1:]}"
        for name, d in sorted(info["datasets"].items())
    ]
    return "|".join(parts)


def collect_headers(spec: str) -> Dict[str, Any]:
    """Read headers for every supported file matched by ``spec``."""
    files = expand_inputs(spec)
    records: List[Dict[str, Any]] = []
    for path in files:
        rec: Dict[str, Any] = {
            "path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
        }
        try:
            rec.update(_read_one(path))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        records.append(rec)

    # Group files that share a schema
    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for rec in records:
        if "error" in rec:
            continue
        g = groups.setdefault(_signature(rec), {"files": [], "datasets": rec["datasets"]})
        g["files"].append(rec["path"])
        rec["_sig"] = _signature(rec)

    schemas = []
    by_path = {r["path"]: r for r in records}
    for sig, g in groups.items():
        lead: Dict[str, List[int]] = {}
        for p in g["files"]:
            for name, d in by_path[p]["datasets"].items():
                if d["shape"] and d["shape"][0] is not None:
                    lead.setdefault(name, []).append(d["shape"][0])
        schemas.append({
            "n_files": len(g["files"]),
            "files": g["files"],
            "datasets": {
                name: {
                    "dtype": d["dtype"],
                    "trailing_shape": d["shape"][1:],
                    "leading_dim_min": min(lead[name]) if name in lead else None,
                    "leading_dim_max": max(lead[name]) if name in lead else None,
                    "leading_dim_total": sum(lead[name]) if name in lead else None,
                }
                for name, d in g["datasets"].items()
            },
        })
    for rec in records:
        rec.pop("_sig", None)

    return {
        "input": spec,
        "n_files": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "n_errors": sum(1 for r in records if "error" in r),
        "schemas": schemas,
        "files": records,
    }


def format_headers_text(result: Dict[str, Any], per_file: bool = False) -> str:
    """Render a human-readable summary."""
    lines = [
        f"Header scan: {result['input']}",
        f"Files: {result['n_files']}  Total size: {result['total_bytes'] / 1e6:.2f} MB  "
        f"Errors: {result['n_errors']}",
        "",
    ]
    for i, s in enumerate(result["schemas"], 1):
        first = Path(s["files"][0]).name
        last = Path(s["files"][-1]).name
        lines.append(f"Schema {i}: {s['n_files']} file(s) ({first} .. {last})")
        lines.append(f"  {'dataset':<28}{'dtype':<14}{'trailing shape':<18}{'leading dim (min/max/total)'}")
        for name, d in s["datasets"].items():
            if d["leading_dim_total"] is None:
                lead = "-"
            else:
                lead = f"{d['leading_dim_min']}/{d['leading_dim_max']}/{d['leading_dim_total']}"
            lines.append(f"  {name:<28}{d['dtype']:<14}{str(d['trailing_shape']):<18}{lead}")
        lines.append("")

    # Root attributes are usually identical across files; show from first file with any
    for rec in result["files"]:
        if rec.get("attrs"):
            lines.append(f"Attributes ({Path(rec['path']).name}):")
            for k, v in rec["attrs"].items():
                lines.append(f"  {k} = {v}")
            lines.append("")
            break

    errors = [r for r in result["files"] if "error" in r]
    if errors:
        lines.append("Errors:")
        for r in errors:
            lines.append(f"  {r['path']}: {r['error']}")
        lines.append("")

    if per_file:
        lines.append("Per-file:")
        for r in result["files"]:
            n = len(r.get("datasets", {}))
            lines.append(f"  {r['path']}  {r['size_bytes']} B  {n} dataset(s)")
    return "\n".join(lines)


def headers_to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)
