"""
Data snapshot capture for REDI observability.

Captures statistics and metadata from intermediate data files
produced by pipelines (Snakemake, Nextflow, etc.).
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class VariableStats:
    """Statistics for a single variable/array in a dataset."""
    name: str
    shape: tuple
    dtype: str
    size: int  # number of elements

    # Basic statistics (computed if numeric)
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    # Quality indicators
    null_count: int = 0
    nan_count: int = 0
    inf_count: int = 0

    # For climate/geospatial data
    units: Optional[str] = None
    long_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, filtering None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class DataSnapshot:
    """
    A snapshot of a data file at a point in the pipeline.

    Captures metadata, statistics, and provenance information
    that can be used to generate Data Cards.
    """
    # File identification
    path: str
    filename: str
    format: str  # nc, npz, npy, parquet, csv, hdf5, zarr

    # File metadata
    size_bytes: int
    checksum: str  # MD5 hash for provenance tracking
    captured_at: str  # ISO timestamp

    # Data structure
    variables: Dict[str, VariableStats] = field(default_factory=dict)
    dimensions: Dict[str, int] = field(default_factory=dict)  # dim_name -> size
    coordinates: List[str] = field(default_factory=list)

    # Global attributes (from NetCDF, HDF5, etc.)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Provenance (filled in by observability analysis)
    rule_name: Optional[str] = None
    input_files: List[str] = field(default_factory=list)

    # Climate-specific metadata
    time_range: Optional[tuple] = None  # (start, end) as ISO strings
    spatial_bounds: Optional[Dict[str, float]] = None  # lat_min, lat_max, lon_min, lon_max
    pressure_levels: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        d = {
            'path': self.path,
            'filename': self.filename,
            'format': self.format,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum,
            'captured_at': self.captured_at,
            'dimensions': self.dimensions,
            'coordinates': self.coordinates,
            'attributes': {k: _serialize_value(v) for k, v in self.attributes.items()},
            'variables': {k: v.to_dict() for k, v in self.variables.items()},
        }

        if self.rule_name:
            d['rule_name'] = self.rule_name
        if self.input_files:
            d['input_files'] = self.input_files
        if self.time_range:
            d['time_range'] = self.time_range
        if self.spatial_bounds:
            d['spatial_bounds'] = self.spatial_bounds
        if self.pressure_levels:
            d['pressure_levels'] = self.pressure_levels

        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"DataSnapshot: {self.filename}",
            f"  Format: {self.format}",
            f"  Size: {_format_bytes(self.size_bytes)}",
            f"  Variables: {len(self.variables)}",
        ]

        if self.dimensions:
            dims_str = ", ".join(f"{k}={v}" for k, v in self.dimensions.items())
            lines.append(f"  Dimensions: {dims_str}")

        if self.time_range:
            lines.append(f"  Time: {self.time_range[0]} to {self.time_range[1]}")

        if self.spatial_bounds:
            sb = self.spatial_bounds
            lines.append(f"  Spatial: lat[{sb.get('lat_min')}, {sb.get('lat_max')}], "
                        f"lon[{sb.get('lon_min')}, {sb.get('lon_max')}]")

        for name, var in list(self.variables.items())[:5]:
            stats = f"shape={var.shape}, dtype={var.dtype}"
            if var.mean is not None:
                stats += f", mean={var.mean:.4g}, std={var.std:.4g}"
            lines.append(f"  - {name}: {stats}")

        if len(self.variables) > 5:
            lines.append(f"  ... and {len(self.variables) - 5} more variables")

        return "\n".join(lines)


def capture_snapshot(
    path: Union[str, Path],
    rule_name: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    compute_stats: bool = True,
    sample_size: Optional[int] = None,
) -> DataSnapshot:
    """
    Capture a snapshot of a data file.

    Auto-detects file format and extracts relevant metadata and statistics.

    Args:
        path: Path to the data file
        rule_name: Name of the pipeline rule that created this file
        input_files: List of input files used to create this file
        compute_stats: Whether to compute statistics (min/max/mean/std)
        sample_size: If set, sample this many elements for stats (faster for large files)

    Returns:
        DataSnapshot with captured metadata and statistics
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Basic file info
    stat = path.stat()
    size_bytes = stat.st_size
    checksum = _compute_checksum(path)

    # Detect format
    suffix = path.suffix.lower()
    format_map = {
        '.nc': 'netcdf',
        '.nc4': 'netcdf',
        '.npz': 'npz',
        '.npy': 'npy',
        '.parquet': 'parquet',
        '.csv': 'csv',
        '.tsv': 'csv',
        '.h5': 'hdf5',
        '.hdf5': 'hdf5',
        '.zarr': 'zarr',
    }
    file_format = format_map.get(suffix, 'unknown')

    # Create base snapshot
    snapshot = DataSnapshot(
        path=str(path.absolute()),
        filename=path.name,
        format=file_format,
        size_bytes=size_bytes,
        checksum=checksum,
        captured_at=datetime.now().isoformat(),
        rule_name=rule_name,
        input_files=input_files or [],
    )

    # Dispatch to format-specific handler
    try:
        if file_format == 'netcdf':
            _capture_netcdf(snapshot, path, compute_stats, sample_size)
        elif file_format == 'npz':
            _capture_npz(snapshot, path, compute_stats, sample_size)
        elif file_format == 'npy':
            _capture_npy(snapshot, path, compute_stats, sample_size)
        elif file_format == 'parquet':
            _capture_parquet(snapshot, path, compute_stats, sample_size)
        elif file_format == 'csv':
            _capture_csv(snapshot, path, compute_stats, sample_size)
        elif file_format == 'hdf5':
            _capture_hdf5(snapshot, path, compute_stats, sample_size)
        elif file_format == 'zarr':
            _capture_zarr(snapshot, path, compute_stats, sample_size)
        else:
            logger.warning(f"Unknown format for {path}, capturing basic info only")
    except Exception as e:
        logger.error(f"Error capturing snapshot for {path}: {e}")
        snapshot.attributes['capture_error'] = str(e)

    return snapshot


def _compute_checksum(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of file."""
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        # For large files, just hash first and last chunks
        first_chunk = f.read(chunk_size)
        md5.update(first_chunk)

        f.seek(0, 2)  # End of file
        file_size = f.tell()

        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)
            last_chunk = f.read(chunk_size)
            md5.update(last_chunk)
            # Include file size in hash for uniqueness
            md5.update(str(file_size).encode())

    return md5.hexdigest()[:16]  # Truncate for readability


def _capture_netcdf(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture NetCDF file metadata and statistics."""
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        snapshot.attributes['error'] = 'xarray not installed'
        return

    with xr.open_dataset(path) as ds:
        # Dimensions
        snapshot.dimensions = dict(ds.sizes)

        # Coordinates
        snapshot.coordinates = list(ds.coords.keys())

        # Global attributes
        snapshot.attributes = {k: _serialize_value(v) for k, v in ds.attrs.items()}

        # Extract climate-specific metadata
        _extract_climate_metadata(snapshot, ds)

        # Variables
        for var_name in ds.data_vars:
            var = ds[var_name]

            var_stats = VariableStats(
                name=var_name,
                shape=var.shape,
                dtype=str(var.dtype),
                size=int(np.prod(var.shape)),
                units=var.attrs.get('units'),
                long_name=var.attrs.get('long_name'),
            )

            if compute_stats and np.issubdtype(var.dtype, np.number):
                try:
                    data = var.values
                    if sample_size and data.size > sample_size:
                        # Random sample for large arrays
                        flat = data.flatten()
                        indices = np.random.choice(len(flat), sample_size, replace=False)
                        data = flat[indices]

                    var_stats.nan_count = int(np.isnan(data).sum())
                    var_stats.inf_count = int(np.isinf(data).sum())

                    # Compute stats on valid values
                    valid = data[np.isfinite(data)]
                    if len(valid) > 0:
                        var_stats.min = float(np.min(valid))
                        var_stats.max = float(np.max(valid))
                        var_stats.mean = float(np.mean(valid))
                        var_stats.std = float(np.std(valid))
                except Exception as e:
                    logger.debug(f"Could not compute stats for {var_name}: {e}")

            snapshot.variables[var_name] = var_stats


def _extract_climate_metadata(snapshot: DataSnapshot, ds):
    """Extract climate-specific metadata from xarray Dataset."""
    import numpy as np

    # Time range
    if 'time' in ds.coords:
        try:
            time = ds.coords['time']
            snapshot.time_range = (
                str(time.values[0])[:19],
                str(time.values[-1])[:19]
            )
        except Exception:
            pass

    # Spatial bounds
    lat_names = ['lat', 'latitude', 'y']
    lon_names = ['lon', 'longitude', 'x']

    lat_coord = next((name for name in lat_names if name in ds.coords), None)
    lon_coord = next((name for name in lon_names if name in ds.coords), None)

    if lat_coord and lon_coord:
        try:
            lat = ds.coords[lat_coord].values
            lon = ds.coords[lon_coord].values
            snapshot.spatial_bounds = {
                'lat_min': float(np.min(lat)),
                'lat_max': float(np.max(lat)),
                'lon_min': float(np.min(lon)),
                'lon_max': float(np.max(lon)),
            }
        except Exception:
            pass

    # Pressure levels
    plev_names = ['plev', 'level', 'lev', 'pressure']
    plev_coord = next((name for name in plev_names if name in ds.coords), None)

    if plev_coord:
        try:
            plev = ds.coords[plev_coord].values
            snapshot.pressure_levels = [float(p) for p in plev]
        except Exception:
            pass


def _capture_npz(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture NPZ file metadata and statistics."""
    import numpy as np

    with np.load(path, allow_pickle=False) as npz:
        for arr_name in npz.files:
            arr = npz[arr_name]

            var_stats = VariableStats(
                name=arr_name,
                shape=arr.shape,
                dtype=str(arr.dtype),
                size=int(np.prod(arr.shape)),
            )

            if compute_stats and np.issubdtype(arr.dtype, np.number):
                try:
                    data = arr
                    if sample_size and data.size > sample_size:
                        flat = data.flatten()
                        indices = np.random.choice(len(flat), sample_size, replace=False)
                        data = flat[indices]

                    var_stats.nan_count = int(np.isnan(data).sum())
                    var_stats.inf_count = int(np.isinf(data).sum())

                    valid = data[np.isfinite(data)]
                    if len(valid) > 0:
                        var_stats.min = float(np.min(valid))
                        var_stats.max = float(np.max(valid))
                        var_stats.mean = float(np.mean(valid))
                        var_stats.std = float(np.std(valid))
                except Exception as e:
                    logger.debug(f"Could not compute stats for {arr_name}: {e}")

            snapshot.variables[arr_name] = var_stats

            # Use first array's shape for dimensions
            if not snapshot.dimensions:
                snapshot.dimensions = {f"dim_{i}": s for i, s in enumerate(arr.shape)}


def _capture_npy(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture NPY file metadata and statistics."""
    import numpy as np

    arr = np.load(path, mmap_mode='r')

    var_stats = VariableStats(
        name='data',
        shape=arr.shape,
        dtype=str(arr.dtype),
        size=int(np.prod(arr.shape)),
    )

    snapshot.dimensions = {f"dim_{i}": s for i, s in enumerate(arr.shape)}

    if compute_stats and np.issubdtype(arr.dtype, np.number):
        try:
            data = np.array(arr)  # Load into memory
            if sample_size and data.size > sample_size:
                flat = data.flatten()
                indices = np.random.choice(len(flat), sample_size, replace=False)
                data = flat[indices]

            var_stats.nan_count = int(np.isnan(data).sum())
            var_stats.inf_count = int(np.isinf(data).sum())

            valid = data[np.isfinite(data)]
            if len(valid) > 0:
                var_stats.min = float(np.min(valid))
                var_stats.max = float(np.max(valid))
                var_stats.mean = float(np.mean(valid))
                var_stats.std = float(np.std(valid))
        except Exception as e:
            logger.debug(f"Could not compute stats: {e}")

    snapshot.variables['data'] = var_stats


def _capture_parquet(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture Parquet file metadata and statistics."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        snapshot.attributes['error'] = 'pandas not installed'
        return

    df = pd.read_parquet(path)

    snapshot.dimensions = {'rows': len(df), 'columns': len(df.columns)}

    for col in df.columns:
        series = df[col]

        var_stats = VariableStats(
            name=col,
            shape=(len(series),),
            dtype=str(series.dtype),
            size=len(series),
            null_count=int(series.isna().sum()),
        )

        if compute_stats and np.issubdtype(series.dtype, np.number):
            try:
                valid = series.dropna()
                if len(valid) > 0:
                    var_stats.min = float(valid.min())
                    var_stats.max = float(valid.max())
                    var_stats.mean = float(valid.mean())
                    var_stats.std = float(valid.std())
            except Exception as e:
                logger.debug(f"Could not compute stats for {col}: {e}")

        snapshot.variables[col] = var_stats


def _capture_csv(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture CSV file metadata and statistics."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        snapshot.attributes['error'] = 'pandas not installed'
        return

    # Read with sampling for large files
    if sample_size:
        df = pd.read_csv(path, nrows=sample_size)
    else:
        df = pd.read_csv(path)

    snapshot.dimensions = {'rows': len(df), 'columns': len(df.columns)}

    for col in df.columns:
        series = df[col]

        var_stats = VariableStats(
            name=col,
            shape=(len(series),),
            dtype=str(series.dtype),
            size=len(series),
            null_count=int(series.isna().sum()),
        )

        if compute_stats and np.issubdtype(series.dtype, np.number):
            try:
                valid = series.dropna()
                if len(valid) > 0:
                    var_stats.min = float(valid.min())
                    var_stats.max = float(valid.max())
                    var_stats.mean = float(valid.mean())
                    var_stats.std = float(valid.std())
            except Exception as e:
                logger.debug(f"Could not compute stats for {col}: {e}")

        snapshot.variables[col] = var_stats


def _capture_hdf5(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture HDF5 file metadata and statistics."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        snapshot.attributes['error'] = 'h5py not installed'
        return

    def visit_datasets(group, prefix=''):
        """Recursively visit datasets in HDF5 file."""
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}" if prefix else key

            if isinstance(item, h5py.Dataset):
                var_stats = VariableStats(
                    name=full_key,
                    shape=item.shape,
                    dtype=str(item.dtype),
                    size=int(np.prod(item.shape)),
                )

                # Get attributes
                if 'units' in item.attrs:
                    var_stats.units = str(item.attrs['units'])
                if 'long_name' in item.attrs:
                    var_stats.long_name = str(item.attrs['long_name'])

                if compute_stats and np.issubdtype(item.dtype, np.number):
                    try:
                        if sample_size and item.size > sample_size:
                            # Sample from dataset
                            data = item[()].flatten()
                            indices = np.random.choice(len(data), sample_size, replace=False)
                            data = data[indices]
                        else:
                            data = item[()]

                        var_stats.nan_count = int(np.isnan(data).sum())
                        var_stats.inf_count = int(np.isinf(data).sum())

                        valid = data[np.isfinite(data)]
                        if len(valid) > 0:
                            var_stats.min = float(np.min(valid))
                            var_stats.max = float(np.max(valid))
                            var_stats.mean = float(np.mean(valid))
                            var_stats.std = float(np.std(valid))
                    except Exception as e:
                        logger.debug(f"Could not compute stats for {full_key}: {e}")

                snapshot.variables[full_key] = var_stats

            elif isinstance(item, h5py.Group):
                visit_datasets(item, full_key)

    with h5py.File(path, 'r') as f:
        # Global attributes
        snapshot.attributes = {k: _serialize_value(v) for k, v in f.attrs.items()}
        visit_datasets(f)


def _capture_zarr(snapshot: DataSnapshot, path: Path, compute_stats: bool, sample_size: Optional[int]):
    """Capture Zarr store metadata and statistics."""
    try:
        import zarr
        import numpy as np
    except ImportError:
        snapshot.attributes['error'] = 'zarr not installed'
        return

    store = zarr.open(str(path), mode='r')

    # Handle both groups and arrays
    if isinstance(store, zarr.Array):
        # Single array
        var_stats = VariableStats(
            name='data',
            shape=store.shape,
            dtype=str(store.dtype),
            size=int(np.prod(store.shape)),
        )

        if compute_stats and np.issubdtype(store.dtype, np.number):
            try:
                data = store[:]
                if sample_size and data.size > sample_size:
                    flat = data.flatten()
                    indices = np.random.choice(len(flat), sample_size, replace=False)
                    data = flat[indices]

                var_stats.nan_count = int(np.isnan(data).sum())
                var_stats.inf_count = int(np.isinf(data).sum())

                valid = data[np.isfinite(data)]
                if len(valid) > 0:
                    var_stats.min = float(np.min(valid))
                    var_stats.max = float(np.max(valid))
                    var_stats.mean = float(np.mean(valid))
                    var_stats.std = float(np.std(valid))
            except Exception as e:
                logger.debug(f"Could not compute stats: {e}")

        snapshot.variables['data'] = var_stats
        snapshot.dimensions = {f"dim_{i}": s for i, s in enumerate(store.shape)}

    else:
        # Group with multiple arrays
        snapshot.attributes = dict(store.attrs)

        for key in store.keys():
            arr = store[key]
            if isinstance(arr, zarr.Array):
                var_stats = VariableStats(
                    name=key,
                    shape=arr.shape,
                    dtype=str(arr.dtype),
                    size=int(np.prod(arr.shape)),
                )

                if compute_stats and np.issubdtype(arr.dtype, np.number):
                    try:
                        data = arr[:]
                        if sample_size and data.size > sample_size:
                            flat = data.flatten()
                            indices = np.random.choice(len(flat), sample_size, replace=False)
                            data = flat[indices]

                        var_stats.nan_count = int(np.isnan(data).sum())
                        var_stats.inf_count = int(np.isinf(data).sum())

                        valid = data[np.isfinite(data)]
                        if len(valid) > 0:
                            var_stats.min = float(np.min(valid))
                            var_stats.max = float(np.max(valid))
                            var_stats.mean = float(np.mean(valid))
                            var_stats.std = float(np.std(valid))
                    except Exception as e:
                        logger.debug(f"Could not compute stats for {key}: {e}")

                snapshot.variables[key] = var_stats


def _serialize_value(value) -> Any:
    """Convert a value to JSON-serializable format."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return value.tolist()
        return f"array(shape={value.shape}, dtype={value.dtype})"
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return f"<bytes, len={len(value)}>"
    elif isinstance(value, (list, tuple)) and len(value) > 10:
        return f"<{type(value).__name__}, len={len(value)}>"
    else:
        return value


def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
