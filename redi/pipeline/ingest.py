"""
DRAI Pipeline - Ingest Stage

Handles data acquisition, streaming, and format harmonization.
Domain-specific ingestion is delegated to domain modules.

Operations:
- Download/stream from sources
- Format detection and harmonization
- Initial validation
- Metadata extraction
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, List
from . import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error

logger = logging.getLogger(__name__)


class DetectFormat(PipelineStep):
    """Detect file format and set appropriate metadata"""

    name = "detect_format"
    stage = PipelineStage.INGEST
    description = "Detect file format and extract basic metadata"

    # Format mappings
    FORMAT_EXTENSIONS = {
        # Tabular
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".json": "json",
        ".txt": "txt",  # For manifest files (train.txt, test.txt, etc.)
        # Climate
        ".nc": "netcdf",
        ".nc4": "netcdf",
        ".grib": "grib",
        ".grib2": "grib",
        ".zarr": "zarr",
        # Materials
        ".cif": "cif",
        ".poscar": "poscar",
        ".vasp": "vasp",
        ".xyz": "xyz",
        # Life Sciences
        ".fasta": "fasta",
        ".fastq": "fastq",
        ".pdb": "pdb",
        ".mmcif": "mmcif",
        ".bam": "bam",
        ".vcf": "vcf",
        # Binary/Scientific
        ".hdf5": "hdf5",
        ".h5": "hdf5",
        ".npz": "npz",
        ".npy": "npy",
        ".tfrecord": "tfrecord",
        ".adios": "adios",
        ".bp": "adios",
    }

    def _sniff_format(self, path: Path) -> str:
        """Detect format by sniffing file content"""
        try:
            with open(path, 'rb') as f:
                header = f.read(1024)

            # Check for binary formats
            if header.startswith(b'\x89HDF'):
                return "hdf5"
            if header.startswith(b'PAR1') or b'PAR1' in header[:100]:
                return "parquet"
            if header.startswith(b'CDF'):
                return "netcdf"

            # Try to decode as text
            try:
                text = header.decode('utf-8', errors='ignore')
                first_line = text.split('\n')[0].strip()

                # Check for JSON
                if first_line.startswith('{') or first_line.startswith('['):
                    return "json"

                # Check for FASTA
                if first_line.startswith('>'):
                    return "fasta"

                # Check for CSV/TSV (has delimiters)
                if ',' in first_line or '\t' in first_line or ';' in first_line:
                    return "csv"

            except Exception:
                pass

        except Exception:
            pass

        return "unknown"

    def _detect_torque_format(self, path: Path) -> Optional[str]:
        """Detect PBS/Torque accounting log directory layout."""
        if not path.is_dir():
            return None

        # Check for torque_logs/ subdirectory (e.g., /opt/data/bluewaters)
        torque_dir = path / "torque_logs"
        if torque_dir.is_dir():
            files = [f for f in torque_dir.iterdir() if f.is_file()]
            if files and files[0].name.isdigit() and len(files[0].name) == 8:
                return "torque-pbs"

        # Check if path itself is a directory of date-named PBS log files
        date_files = [f for f in path.iterdir()
                      if f.is_file() and f.name.isdigit() and len(f.name) == 8]
        if len(date_files) >= 2:
            return "torque-pbs"

        return None

    def _detect_gcloud_v2_format(self, path: Path) -> Optional[str]:
        """Detect Google Cluster Trace v2 directory layout."""
        if not path.is_dir():
            return None

        GCV2_SUBTABLES = {"job_events", "task_events", "machine_events", "task_usage"}
        subdirs = {d.name for d in path.iterdir() if d.is_dir()}
        if not GCV2_SUBTABLES & subdirs:
            return None

        # Confirm at least one sub-table has the expected part-*.csv.gz files
        for name in GCV2_SUBTABLES & subdirs:
            parts = list((path / name).glob("part-*.csv.gz"))
            if parts:
                return "gcloud-v2"

        return None

    def _detect_sickle_format(self, path: Path) -> Optional[str]:
        """Detect SICKLE turbulence data formats (GESTS or SST-binary)."""
        if not path.is_dir():
            return None

        # Check for SST-binary format: files like u_28.440000, p_28.440000
        sst_pattern = re.compile(r'^[a-z]+_\d+\.\d+$')
        sst_files = [f for f in path.iterdir() if f.is_file() and sst_pattern.match(f.name)]
        if len(sst_files) >= 2:
            return "sst-binary"

        # Check for GESTS format: subdirectories with cube_* files
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            # Check for numbered cube directories (8192 format)
            if subdir.name.isdigit():
                cube_files = list(subdir.glob("cube_*"))
                if cube_files:
                    return "gests"
            # Check for variable name directories (e.g., velocity/, pressure/)
            elif subdir.name in ("velocity", "pressure", "enstrophy", "dissipation"):
                cube_files = list(subdir.glob("cube_*"))
                if cube_files:
                    return "gests"

        return None

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        # Handle multi-file case
        paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

        # Check all files exist
        missing = [p for p in paths if not p.exists()]
        if missing:
            ctx.add_error(f"File(s) not found: {missing[0]}")
            return ctx

        # Detect format from first file
        path = paths[0]
        ext = path.suffix.lower()
        detected_format = self.FORMAT_EXTENSIONS.get(ext, "unknown")
        logger.debug(f"     File extension: {ext}")

        # Handle directories (e.g., zarr stores, SICKLE data)
        if path.is_dir():
            logger.debug(f"     Path is directory, checking for format markers")
            if (path / ".zarray").exists() or (path / ".zgroup").exists():
                detected_format = "zarr"
                logger.debug(f"     Found zarr markers, format set to zarr")
            else:
                # Check for PBS/Torque formats
                torque_format = self._detect_torque_format(path)
                if torque_format:
                    detected_format = torque_format
                    logger.debug(f"     Detected PBS/Torque format: {torque_format}")
                else:
                    # Check for Google Cluster Trace v2
                    gcloud_format = self._detect_gcloud_v2_format(path)
                    if gcloud_format:
                        detected_format = gcloud_format
                        logger.debug(f"     Detected Google Cluster v2 format")
                    else:
                        # Check for SICKLE formats
                        sickle_format = self._detect_sickle_format(path)
                        if sickle_format:
                            detected_format = sickle_format
                            logger.debug(f"     Detected SICKLE format: {sickle_format}")

        # If format unknown, try content sniffing
        if detected_format == "unknown" and path.is_file():
            detected_format = self._sniff_format(path)
            if detected_format != "unknown":
                logger.debug(f"     Detected format from content: {detected_format}")

        # Validate all files have same format (for multi-file)
        if len(paths) > 1:
            formats = set()
            for p in paths:
                p_ext = p.suffix.lower()
                p_fmt = self.FORMAT_EXTENSIONS.get(p_ext, "unknown")
                # Sniff if unknown
                if p_fmt == "unknown" and p.is_file():
                    p_fmt = self._sniff_format(p)
                formats.add(p_fmt)

            if len(formats) > 1 and "unknown" not in formats:
                ctx.add_error(f"Mixed file formats detected: {formats}. All files must have the same format.")
                return ctx

            logger.debug(f"     Multi-file mode: {len(paths)} files")

        # Calculate total size
        total_size = sum(p.stat().st_size for p in paths if p.is_file())
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024*1024 else f"{total_size / (1024*1024):.2f} MB"

        ctx.metadata["file_format"] = detected_format
        ctx.metadata["file_extension"] = ext
        ctx.metadata["file_size_bytes"] = total_size
        ctx.metadata["file_name"] = path.name
        ctx.metadata["num_input_files"] = len(paths)
        ctx.metadata["input_files"] = [p.name for p in paths]

        logger.debug(f"     Total size: {size_str}")
        logger.debug(f"     Detected format: {detected_format}")

        if detected_format == "unknown":
            logger.warning(f"Could not detect file format for: {path.name}")
            ctx.add_error(f"Unknown file format. Supported formats: csv, json, parquet, netcdf, hdf5, fasta, cif, etc. "
                         f"Check file extension or use a supported format.")
            return ctx

        if len(paths) > 1:
            logger.info(f"Detected format: {detected_format} for {len(paths)} files")
        else:
            logger.info(f"Detected format: {detected_format} for {path.name}")
        return ctx


class LoadCSV(PipelineStep):
    """Load CSV files into pandas DataFrame"""

    name = "load_csv"
    stage = PipelineStage.INGEST
    description = "Load CSV data into memory"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "csv"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd

            paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

            if len(paths) == 1:
                # Single file
                logger.debug(f"     Reading CSV file: {paths[0]}")
                ctx.data = pd.read_csv(paths[0])
            else:
                # Multiple files - concatenate
                logger.debug(f"     Reading {len(paths)} CSV files")
                dfs = []
                for i, path in enumerate(paths):
                    logger.debug(f"     Reading file {i+1}/{len(paths)}: {path.name}")
                    df = pd.read_csv(path)
                    dfs.append(df)
                ctx.data = pd.concat(dfs, axis=0, ignore_index=True)
                logger.debug(f"     Concatenated {len(paths)} files")

            ctx.metadata["num_rows"] = len(ctx.data)
            ctx.metadata["num_columns"] = len(ctx.data.columns)
            ctx.metadata["columns"] = list(ctx.data.columns)
            ctx.metadata["dtypes"] = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}

            # Verbose logging
            logger.debug(f"     Rows: {ctx.metadata['num_rows']:,}")
            logger.debug(f"     Columns: {ctx.metadata['num_columns']}")
            logger.debug(f"     Column names: {', '.join(ctx.metadata['columns'][:10])}" +
                        (f" ... (+{len(ctx.metadata['columns'])-10} more)" if len(ctx.metadata['columns']) > 10 else ""))
            logger.debug(f"     Data types: {', '.join(f'{k}={v}' for k, v in list(ctx.metadata['dtypes'].items())[:5])}" +
                        (f" ... (+{len(ctx.metadata['dtypes'])-5} more)" if len(ctx.metadata['dtypes']) > 5 else ""))
            memory_mb = ctx.data.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.debug(f"     Memory usage: {memory_mb:.2f} MB")

            if len(paths) > 1:
                logger.info(f"Loaded {len(paths)} CSV files: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
            else:
                logger.info(f"Loaded CSV: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
        except ImportError:
            ctx.add_error(missing_dependency_error("pandas"))
        except Exception as e:
            ctx.add_error(f"Failed to load CSV: {str(e)}")
        return ctx


class LoadJSON(PipelineStep):
    """Load JSON files"""

    name = "load_json"
    stage = PipelineStage.INGEST
    description = "Load JSON data into memory"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "json"

    def _is_geojson_timeseries(self, data: list) -> bool:
        """Detect GeoJSON-style data with nested time series (e.g. NASA POWER API)"""
        if not data or not isinstance(data[0], dict):
            return False
        first = data[0]
        if "features" not in first or not isinstance(first["features"], list):
            return False
        features = first["features"]
        if not features or not isinstance(features[0], dict):
            return False
        feat = features[0]
        if feat.get("type") != "Feature" or "geometry" not in feat:
            return False
        props = feat.get("properties", {})
        param = props.get("parameter", {})
        # Check if parameter values are date-keyed dicts
        if param and isinstance(param, dict):
            first_val = next(iter(param.values()), None)
            if isinstance(first_val, dict) and len(first_val) > 0:
                first_key = next(iter(first_val))
                if isinstance(first_key, str) and first_key.isdigit() and len(first_key) == 8:
                    return True
        return False

    def _flatten_geojson_timeseries(self, data: list):
        """Flatten GeoJSON time series into a tabular DataFrame.

        Extracts lat, lon, date, and parameter values from nested GeoJSON features.
        """
        import pandas as pd

        records = []
        for item in data:
            for feature in item.get("features", []):
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates", [None, None])
                lon, lat = coords[0], coords[1]
                params = feature.get("properties", {}).get("parameter", {})
                if not params:
                    continue
                # Get all date keys from the first parameter
                first_param = next(iter(params.values()))
                for date_key in first_param:
                    row = {"latitude": lat, "longitude": lon, "date": date_key}
                    for param_name, param_values in params.items():
                        row[param_name] = param_values.get(date_key)
                    records.append(row)

        df = pd.DataFrame(records)
        # Convert date string (YYYYMMDD) to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        return df

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            logger.debug(f"     Reading JSON file: {ctx.input_path}")
            with open(ctx.input_path, 'r') as f:
                data = json.load(f)

            logger.debug(f"     JSON root type: {type(data).__name__}")

            # Try to convert to DataFrame if it's a list of records
            if isinstance(data, list):
                logger.debug(f"     JSON is array with {len(data)} elements")
                try:
                    import pandas as pd

                    # Detect GeoJSON time series (e.g. NASA POWER API)
                    if self._is_geojson_timeseries(data):
                        logger.debug(f"     Detected GeoJSON time series format, flattening...")
                        ctx.data = self._flatten_geojson_timeseries(data)
                        logger.debug(f"     Flattened to {len(ctx.data)} rows, {len(ctx.data.columns)} columns")
                    else:
                        # Check if records are flat (no nested lists/dicts)
                        sample = data[0] if data else {}
                        has_nested = any(
                            isinstance(v, (list, dict)) for v in sample.values()
                        ) if isinstance(sample, dict) else False

                        if has_nested:
                            logger.debug(f"     JSON has nested structures, using json_normalize")
                            ctx.data = pd.json_normalize(data, max_level=1)
                        else:
                            ctx.data = pd.DataFrame(data)

                    ctx.metadata["num_rows"] = len(ctx.data)
                    ctx.metadata["num_columns"] = len(ctx.data.columns)
                    ctx.metadata["columns"] = list(ctx.data.columns)
                    logger.debug(f"     Converted to DataFrame: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
                except ImportError:
                    ctx.data = data
                    ctx.metadata["num_records"] = len(data)
                    logger.debug(f"     Stored as list (pandas not available)")
            else:
                ctx.data = data
                ctx.metadata["type"] = "dict"
                ctx.metadata["keys"] = list(data.keys()) if isinstance(data, dict) else None
                if ctx.metadata["keys"]:
                    logger.debug(f"     Dict keys: {', '.join(ctx.metadata['keys'][:10])}")

            logger.info(f"Loaded JSON from {ctx.input_path.name}")
        except Exception as e:
            ctx.add_error(f"Failed to load JSON: {str(e)}")
        return ctx


class LoadParquet(PipelineStep):
    """Load Parquet files"""

    name = "load_parquet"
    stage = PipelineStage.INGEST
    description = "Load Parquet data into memory"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "parquet"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd

            paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

            if len(paths) == 1:
                # Single file
                logger.debug(f"     Reading Parquet file: {paths[0]}")
                ctx.data = pd.read_parquet(paths[0])
            else:
                # Multiple files - concatenate
                logger.debug(f"     Reading {len(paths)} Parquet files")
                dfs = []
                for i, path in enumerate(paths):
                    logger.debug(f"     Reading file {i+1}/{len(paths)}: {path.name}")
                    df = pd.read_parquet(path)
                    dfs.append(df)
                ctx.data = pd.concat(dfs, axis=0, ignore_index=True)
                logger.debug(f"     Concatenated {len(paths)} files")

            ctx.metadata["num_rows"] = len(ctx.data)
            ctx.metadata["num_columns"] = len(ctx.data.columns)
            ctx.metadata["columns"] = list(ctx.data.columns)
            ctx.metadata["dtypes"] = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}

            logger.debug(f"     Rows: {ctx.metadata['num_rows']:,}")
            logger.debug(f"     Columns: {ctx.metadata['num_columns']}")
            logger.debug(f"     Column names: {', '.join(ctx.metadata['columns'][:10])}" +
                        (f" ... (+{len(ctx.metadata['columns'])-10} more)" if len(ctx.metadata['columns']) > 10 else ""))
            memory_mb = ctx.data.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.debug(f"     Memory usage: {memory_mb:.2f} MB")

            if len(paths) > 1:
                logger.info(f"Loaded {len(paths)} Parquet files: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
            else:
                logger.info(f"Loaded Parquet: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
        except ImportError:
            ctx.add_error(missing_dependency_error("pyarrow", "parquet"))
        except Exception as e:
            ctx.add_error(f"Failed to load Parquet: {str(e)}")
        return ctx


class LoadNetCDF(PipelineStep):
    """Load NetCDF files using xarray"""

    name = "load_netcdf"
    stage = PipelineStage.INGEST
    description = "Load NetCDF data using xarray"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") in ["netcdf", "grib"]

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import xarray as xr

            paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

            if len(paths) == 1:
                # Single file
                logger.debug(f"     Reading NetCDF file: {paths[0]}")
                ctx.data = xr.open_dataset(paths[0])
            else:
                # Multiple files - load sequentially and concatenate
                logger.debug(f"     Reading {len(paths)} NetCDF files sequentially")
                file_list = [str(p) for p in paths]

                datasets = []
                failed_files = []
                for i, fpath in enumerate(file_list):
                    fname = Path(fpath).name
                    logger.debug(f"     Loading file {i+1}/{len(file_list)}: {fname}")
                    try:
                        ds = xr.open_dataset(fpath)
                        datasets.append(ds)
                    except Exception as e:
                        failed_files.append((fname, str(e)))
                        logger.warning(f"     Failed to load {fname}: {e}")

                if failed_files:
                    ctx.metadata["failed_files"] = failed_files
                    if not datasets:
                        ctx.add_error(f"All {len(file_list)} files failed to load")
                        return ctx
                    logger.warning(f"     Skipped {len(failed_files)} corrupted file(s)")

                if len(datasets) == 1:
                    ctx.data = datasets[0]
                else:
                    ctx.data = xr.concat(datasets, dim='time', data_vars='minimal')

                logger.debug(f"     Successfully combined {len(datasets)} files")

            ctx.metadata["variables"] = list(ctx.data.data_vars)
            ctx.metadata["dimensions"] = dict(ctx.data.dims)
            ctx.metadata["coordinates"] = list(ctx.data.coords)

            logger.debug(f"     Variables: {', '.join(ctx.metadata['variables'][:10])}" +
                        (f" ... (+{len(ctx.metadata['variables'])-10} more)" if len(ctx.metadata['variables']) > 10 else ""))
            logger.debug(f"     Dimensions: {ctx.metadata['dimensions']}")
            logger.debug(f"     Coordinates: {', '.join(ctx.metadata['coordinates'])}")

            if len(paths) > 1:
                logger.info(f"Loaded {len(paths)} NetCDF files: {len(ctx.metadata['variables'])} variables")
            else:
                logger.info(f"Loaded NetCDF: {len(ctx.metadata['variables'])} variables")
        except ImportError:
            ctx.add_error(missing_dependency_error("xarray/netCDF4", "climate"))
        except Exception as e:
            ctx.add_error(f"Failed to load NetCDF: {str(e)}")
        return ctx


class LoadHDF5(PipelineStep):
    """Load HDF5 files"""

    name = "load_hdf5"
    stage = PipelineStage.INGEST
    description = "Load HDF5 data into memory"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "hdf5"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import h5py
            logger.debug(f"     Reading HDF5 file: {ctx.input_path}")
            ctx.data = h5py.File(ctx.input_path, 'r')
            ctx.metadata["groups"] = list(ctx.data.keys())

            def get_structure(group, prefix=""):
                structure = {}
                for key in group.keys():
                    path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], h5py.Group):
                        structure[key] = get_structure(group[key], path)
                    else:
                        ds = group[key]
                        structure[key] = {
                            "shape": ds.shape,
                            "dtype": str(ds.dtype)
                        }
                return structure

            ctx.metadata["structure"] = get_structure(ctx.data)

            logger.debug(f"     Top-level groups: {', '.join(ctx.metadata['groups'])}")
            # Log dataset info
            dataset_count = sum(1 for k, v in ctx.metadata["structure"].items() if isinstance(v, dict) and "shape" in v)
            group_count = sum(1 for k, v in ctx.metadata["structure"].items() if isinstance(v, dict) and "shape" not in v)
            logger.debug(f"     Contains {dataset_count} datasets, {group_count} nested groups")

            logger.info(f"Loaded HDF5: {len(ctx.metadata['groups'])} top-level groups")
        except ImportError:
            ctx.add_error(missing_dependency_error("h5py", "hdf5"))
        except Exception as e:
            ctx.add_error(f"Failed to load HDF5: {str(e)}")
        return ctx


class LoadNPY(PipelineStep):
    """Load NumPy .npy files (single or multiple)"""

    name = "load_npy"
    stage = PipelineStage.INGEST
    description = "Load NumPy array files"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "npy"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

            logger.debug(f"     Loading {len(paths)} NPY file(s)")

            if len(paths) == 1:
                logger.debug(f"     Reading NPY file: {paths[0]}")
                ctx.data = np.load(paths[0])
            else:
                # Stack multiple files along axis 0
                logger.debug(f"     Reading and stacking {len(paths)} NPY files")
                arrays = []
                for i, p in enumerate(paths):
                    logger.debug(f"     Loading file {i+1}/{len(paths)}: {p.name}")
                    arr = np.load(p)
                    arrays.append(arr)
                ctx.data = np.stack(arrays, axis=0)
                logger.debug(f"     Stacked into shape: {ctx.data.shape}")

            ctx.metadata["array_shape"] = ctx.data.shape
            ctx.metadata["array_dtype"] = str(ctx.data.dtype)
            ctx.metadata["array_ndim"] = ctx.data.ndim
            ctx.metadata["num_samples"] = ctx.data.shape[0]

            # Calculate memory usage
            memory_mb = ctx.data.nbytes / (1024 * 1024)
            logger.debug(f"     Shape: {ctx.data.shape}")
            logger.debug(f"     Dtype: {ctx.data.dtype}")
            logger.debug(f"     Memory usage: {memory_mb:.2f} MB")

            if len(paths) > 1:
                logger.info(f"Loaded {len(paths)} NPY files: shape {ctx.data.shape}")
            else:
                logger.info(f"Loaded NPY: shape {ctx.data.shape}, dtype {ctx.data.dtype}")

        except Exception as e:
            ctx.add_error(f"Failed to load NPY: {str(e)}")
        return ctx


class LoadNPZ(PipelineStep):
    """Load NumPy .npz archive files"""

    name = "load_npz"
    stage = PipelineStage.INGEST
    description = "Load NumPy compressed archives"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "npz"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            logger.debug(f"     Reading NPZ file: {ctx.input_path}")
            data = np.load(ctx.input_path, allow_pickle=True)

            # Convert to regular dict for easier handling
            ctx.data = {key: data[key] for key in data.files}

            ctx.metadata["npz_keys"] = list(ctx.data.keys())
            ctx.metadata["array_shapes"] = {k: v.shape for k, v in ctx.data.items()}
            ctx.metadata["array_dtypes"] = {k: str(v.dtype) for k, v in ctx.data.items()}

            logger.debug(f"     NPZ keys: {', '.join(ctx.metadata['npz_keys'])}")
            for k, v in ctx.data.items():
                logger.debug(f"       {k}: shape {v.shape}, dtype {v.dtype}")

            # If has "X" key, extract sample count and dimensions
            if "X" in ctx.data:
                ctx.metadata["num_samples"] = ctx.data["X"].shape[0]
                ctx.metadata["array_ndim"] = ctx.data["X"].ndim
                ctx.metadata["array_shape"] = ctx.data["X"].shape
                ctx.metadata["array_dtype"] = str(ctx.data["X"].dtype)

            logger.info(f"Loaded NPZ: {len(ctx.data)} arrays - {list(ctx.data.keys())}")

        except Exception as e:
            ctx.add_error(f"Failed to load NPZ: {str(e)}")
        return ctx


class LoadManifest(PipelineStep):
    """Load files listed in a manifest (train.txt, test.txt, etc.)"""

    name = "load_manifest"
    stage = PipelineStage.INGEST
    description = "Load files from a text manifest"

    def validate(self, ctx: PipelineContext) -> bool:
        # Check if input is a .txt file containing file paths
        if ctx.metadata.get("file_format") != "txt" and ctx.input_path.suffix.lower() != ".txt":
            return False
        # Peek first line to see if it looks like a file path
        try:
            with open(ctx.input_path) as f:
                first_line = f.readline().strip()
            return first_line.endswith(('.npy', '.npz', '.h5', '.nc'))
        except Exception:
            return False

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            manifest_dir = ctx.input_path.parent

            logger.debug(f"     Reading manifest: {ctx.input_path}")

            with open(ctx.input_path) as f:
                rel_paths = [line.strip() for line in f if line.strip()]

            logger.debug(f"     Found {len(rel_paths)} file paths in manifest")

            # Resolve to absolute paths
            abs_paths = [manifest_dir / p for p in rel_paths]

            # Check for missing files
            missing = [p for p in abs_paths if not p.exists()]
            if missing:
                logger.warning(f"     Missing {len(missing)} file(s)")
                for m in missing[:3]:
                    logger.debug(f"       - {m}")
                if len(missing) > 3:
                    logger.debug(f"       ... and {len(missing) - 3} more")
                # Filter to only existing files
                abs_paths = [p for p in abs_paths if p.exists()]

            if not abs_paths:
                ctx.add_error(f"No valid files found in manifest {ctx.input_path}")
                return ctx

            # Load all files
            arrays = []
            for i, p in enumerate(abs_paths):
                logger.debug(f"     Loading file {i+1}/{len(abs_paths)}: {p.name}")
                if p.suffix == '.npy':
                    arr = np.load(p)
                    arrays.append(arr)
                elif p.suffix == '.npz':
                    data = np.load(p)
                    # Prefer 'X' key, otherwise use first array
                    if 'X' in data:
                        arrays.append(data['X'])
                    else:
                        arrays.append(data[data.files[0]])
                else:
                    logger.warning(f"     Skipping unsupported format: {p.suffix}")

            if not arrays:
                ctx.add_error("No arrays loaded from manifest")
                return ctx

            # Stack arrays along axis 0
            ctx.data = np.stack(arrays, axis=0)

            ctx.metadata["manifest_file"] = str(ctx.input_path)
            ctx.metadata["num_files"] = len(abs_paths)
            ctx.metadata["array_shape"] = ctx.data.shape
            ctx.metadata["array_dtype"] = str(ctx.data.dtype)
            ctx.metadata["array_ndim"] = ctx.data.ndim
            ctx.metadata["num_samples"] = len(arrays)

            memory_mb = ctx.data.nbytes / (1024 * 1024)
            logger.debug(f"     Stacked shape: {ctx.data.shape}")
            logger.debug(f"     Memory usage: {memory_mb:.2f} MB")

            logger.info(f"Loaded manifest: {len(arrays)} files -> shape {ctx.data.shape}")

        except Exception as e:
            ctx.add_error(f"Failed to load manifest: {str(e)}")
        return ctx


class LoadSICKLE(PipelineStep):
    """Load SICKLE turbulence data formats (GESTS and SST-binary)"""

    name = "load_sickle"
    stage = PipelineStage.INGEST
    description = "Load SICKLE turbulence simulation data (GESTS/SST-binary)"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") in ["gests", "sst-binary"]

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            from ..dataloaders import get_dataloader, LoadedData

            file_format = ctx.metadata.get("file_format")
            path = ctx.input_path

            logger.debug(f"     Loading SICKLE data ({file_format}): {path}")

            # Get the appropriate dataloader
            loader = get_dataloader(str(path), format_type=file_format, verbose=False)

            # Get metadata first (this scans the directory)
            meta = loader.metadata

            # Store metadata
            ctx.metadata["variables"] = meta.variables
            ctx.metadata["dimensions"] = meta.dimensions
            ctx.metadata["grid_shape"] = meta.grid_shape
            ctx.metadata["num_timesteps"] = meta.num_timesteps
            ctx.metadata["timesteps"] = meta.timesteps
            ctx.metadata["dtype"] = meta.dtype
            ctx.metadata["sickle_metadata"] = meta.to_dict()

            # Load the first snapshot (or all data if small enough)
            if meta.total_size_bytes < 100 * 1024 * 1024:  # < 100MB
                logger.debug(f"     Dataset small enough, loading all data")
                loaded = loader.load_all()
            else:
                logger.debug(f"     Large dataset, loading first snapshot only")
                loaded = loader.load_snapshot()

            # Store the loaded data
            ctx.data = loaded

            # Calculate some summary statistics
            num_points = 1
            for dim_size in meta.grid_shape:
                num_points *= dim_size

            logger.debug(f"     Variables: {', '.join(meta.variables)}")
            logger.debug(f"     Grid shape: {meta.grid_shape}")
            logger.debug(f"     Timesteps: {meta.num_timesteps}")
            logger.debug(f"     Total grid points: {num_points:,}")

            size_mb = meta.total_size_bytes / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
            logger.info(f"Loaded {file_format.upper()}: {len(meta.variables)} variables, "
                       f"{meta.num_timesteps} timesteps, {size_str}")

        except ImportError as e:
            ctx.add_error(f"SICKLE dataloaders not available: {e}")
        except Exception as e:
            ctx.add_error(f"Failed to load SICKLE data: {str(e)}")
        return ctx


class ValidateInput(PipelineStep):
    """Basic input validation"""

    name = "validate_input"
    stage = PipelineStage.INGEST
    description = "Validate input data is properly loaded"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.data is None:
            ctx.add_error("No data loaded - check file format and path")
            return ctx

        logger.debug(f"     Validating data of type: {type(ctx.data).__name__}")

        # Validate numpy arrays
        try:
            import numpy as np
            if isinstance(ctx.data, np.ndarray):
                logger.debug(f"     Array shape: {ctx.data.shape}")
                logger.debug(f"     Array dtype: {ctx.data.dtype}")
                if ctx.data.size == 0:
                    ctx.add_error("Array is empty")
                    return ctx
                # Check for NaN values
                nan_count = np.isnan(ctx.data).sum() if np.issubdtype(ctx.data.dtype, np.floating) else 0
                if nan_count > 0:
                    nan_pct = nan_count / ctx.data.size * 100
                    logger.debug(f"     Found {nan_count} NaN values ({nan_pct:.2f}%)")
                else:
                    logger.debug(f"     No NaN values found")
                logger.info("Input validation passed")
                return ctx
            elif isinstance(ctx.data, dict):
                # Dict with arrays (e.g., from NPZ)
                logger.debug(f"     Dict with keys: {list(ctx.data.keys())}")
                for key, value in ctx.data.items():
                    if isinstance(value, np.ndarray):
                        logger.debug(f"       {key}: shape {value.shape}, dtype {value.dtype}")
                        if value.size == 0:
                            logger.warning(f"     Array '{key}' is empty")
                logger.info("Input validation passed")
                return ctx
        except ImportError:
            pass

        # Basic validation based on data type - DataFrame
        try:
            import pandas as pd
            if isinstance(ctx.data, pd.DataFrame):
                logger.debug(f"     DataFrame shape: {ctx.data.shape}")
                if len(ctx.data) == 0:
                    ctx.add_error("DataFrame is empty")
                if len(ctx.data.columns) == 0:
                    ctx.add_error("DataFrame has no columns")
                # Check for any null values
                null_counts = ctx.data.isnull().sum()
                total_nulls = null_counts.sum()
                if total_nulls > 0:
                    cols_with_nulls = (null_counts > 0).sum()
                    logger.debug(f"     Found {total_nulls} null values across {cols_with_nulls} columns")
                else:
                    logger.debug(f"     No null values found")
        except ImportError:
            pass

        logger.info("Input validation passed")
        return ctx


# Convenience function to get standard ingest steps
def get_ingest_steps() -> List[PipelineStep]:
    """Get the standard ingest pipeline steps"""
    return [
        DetectFormat(),
        LoadCSV(),
        LoadJSON(),
        LoadParquet(),
        LoadNetCDF(),
        LoadHDF5(),
        LoadNPY(),
        LoadNPZ(),
        LoadManifest(),
        LoadSICKLE(),
        ValidateInput(),
    ]


def ingest(input_path: str, output_dir: str = "./output", domain: str = "general") -> PipelineContext:
    """Run the ingest stage standalone"""
    from . import Pipeline, create_context

    ctx = create_context(input_path, output_dir, domain)
    pipeline = Pipeline(domain)

    for step in get_ingest_steps():
        pipeline.add_step(step)

    ctx = pipeline.execute_stage(ctx, PipelineStage.INGEST)
    return ctx
