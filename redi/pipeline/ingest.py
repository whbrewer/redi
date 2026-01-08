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
from pathlib import Path
from typing import Any, Dict, Optional, List
from . import PipelineStep, PipelineStage, PipelineContext

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

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        path = ctx.input_path

        if not path.exists():
            ctx.add_error(f"File not found: {path}")
            return ctx

        # Detect format from extension
        ext = path.suffix.lower()
        detected_format = self.FORMAT_EXTENSIONS.get(ext, "unknown")
        logger.debug(f"     File extension: {ext}")

        # Handle directories (e.g., zarr stores)
        if path.is_dir():
            logger.debug(f"     Path is directory, checking for zarr markers")
            if (path / ".zarray").exists() or (path / ".zgroup").exists():
                detected_format = "zarr"
                logger.debug(f"     Found zarr markers, format set to zarr")

        file_size = path.stat().st_size if path.is_file() else 0
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"

        ctx.metadata["file_format"] = detected_format
        ctx.metadata["file_extension"] = ext
        ctx.metadata["file_size_bytes"] = file_size
        ctx.metadata["file_name"] = path.name

        logger.debug(f"     File size: {size_str}")
        logger.debug(f"     Detected format: {detected_format}")
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
            logger.debug(f"     Reading CSV file: {ctx.input_path}")
            ctx.data = pd.read_csv(ctx.input_path)
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

            logger.info(f"Loaded CSV: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
        except ImportError:
            ctx.add_error("pandas not installed - required for CSV loading")
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
            logger.debug(f"     Reading Parquet file: {ctx.input_path}")
            ctx.data = pd.read_parquet(ctx.input_path)
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

            logger.info(f"Loaded Parquet: {ctx.metadata['num_rows']} rows, {ctx.metadata['num_columns']} columns")
        except ImportError:
            ctx.add_error("pandas/pyarrow not installed - required for Parquet loading")
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
            logger.debug(f"     Reading NetCDF file: {ctx.input_path}")
            ctx.data = xr.open_dataset(ctx.input_path)
            ctx.metadata["variables"] = list(ctx.data.data_vars)
            ctx.metadata["dimensions"] = dict(ctx.data.dims)
            ctx.metadata["coordinates"] = list(ctx.data.coords)

            logger.debug(f"     Variables: {', '.join(ctx.metadata['variables'][:10])}" +
                        (f" ... (+{len(ctx.metadata['variables'])-10} more)" if len(ctx.metadata['variables']) > 10 else ""))
            logger.debug(f"     Dimensions: {ctx.metadata['dimensions']}")
            logger.debug(f"     Coordinates: {', '.join(ctx.metadata['coordinates'])}")

            logger.info(f"Loaded NetCDF: {len(ctx.metadata['variables'])} variables")
        except ImportError:
            ctx.add_error("xarray not installed - required for NetCDF loading")
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
            ctx.add_error("h5py not installed - required for HDF5 loading")
        except Exception as e:
            ctx.add_error(f"Failed to load HDF5: {str(e)}")
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

        # Basic validation based on data type
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
