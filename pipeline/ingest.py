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

        # Handle directories (e.g., zarr stores)
        if path.is_dir():
            if (path / ".zarray").exists() or (path / ".zgroup").exists():
                detected_format = "zarr"

        ctx.metadata["file_format"] = detected_format
        ctx.metadata["file_extension"] = ext
        ctx.metadata["file_size_bytes"] = path.stat().st_size if path.is_file() else 0
        ctx.metadata["file_name"] = path.name

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
            ctx.data = pd.read_csv(ctx.input_path)
            ctx.metadata["num_rows"] = len(ctx.data)
            ctx.metadata["num_columns"] = len(ctx.data.columns)
            ctx.metadata["columns"] = list(ctx.data.columns)
            ctx.metadata["dtypes"] = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}
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
            with open(ctx.input_path, 'r') as f:
                data = json.load(f)

            # Try to convert to DataFrame if it's a list of records
            if isinstance(data, list):
                try:
                    import pandas as pd
                    ctx.data = pd.DataFrame(data)
                    ctx.metadata["num_rows"] = len(ctx.data)
                    ctx.metadata["num_columns"] = len(ctx.data.columns)
                    ctx.metadata["columns"] = list(ctx.data.columns)
                except ImportError:
                    ctx.data = data
                    ctx.metadata["num_records"] = len(data)
            else:
                ctx.data = data
                ctx.metadata["type"] = "dict"
                ctx.metadata["keys"] = list(data.keys()) if isinstance(data, dict) else None

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
            ctx.data = pd.read_parquet(ctx.input_path)
            ctx.metadata["num_rows"] = len(ctx.data)
            ctx.metadata["num_columns"] = len(ctx.data.columns)
            ctx.metadata["columns"] = list(ctx.data.columns)
            ctx.metadata["dtypes"] = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}
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
            ctx.data = xr.open_dataset(ctx.input_path)
            ctx.metadata["variables"] = list(ctx.data.data_vars)
            ctx.metadata["dimensions"] = dict(ctx.data.dims)
            ctx.metadata["coordinates"] = list(ctx.data.coords)
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

        # Basic validation based on data type
        try:
            import pandas as pd
            if isinstance(ctx.data, pd.DataFrame):
                if len(ctx.data) == 0:
                    ctx.add_error("DataFrame is empty")
                if len(ctx.data.columns) == 0:
                    ctx.add_error("DataFrame has no columns")
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
