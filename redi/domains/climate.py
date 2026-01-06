"""
DRAI Climate Domain Module

Domain-specific preprocessing for climate/weather data.
Supports ERA5, CMIP6, and other gridded datasets.

Tools: xarray, xesmf, netCDF4, zarr
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages for climate processing
REQUIRED_PACKAGES = ["xarray", "netCDF4", "xesmf", "zarr"]

# Common climate variables
CLIMATE_VARIABLES = {
    "temperature": ["t2m", "t", "temp", "temperature", "tas", "ta"],
    "precipitation": ["tp", "pr", "precip", "precipitation", "rain"],
    "wind": ["u10", "v10", "u", "v", "wind", "ws", "wd"],
    "pressure": ["sp", "msl", "ps", "psl", "pressure"],
    "humidity": ["rh", "q", "humidity", "hus"],
    "geopotential": ["z", "zg", "geopotential", "phi"],
    "radiation": ["ssrd", "strd", "toa", "olr", "rsds", "rlds"],
}


def get_ingest_steps():
    """Get climate-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class LoadClimateData(PipelineStep):
        """Load climate data using xarray"""

        name = "load_climate"
        stage = PipelineStage.INGEST
        description = "Load NetCDF/Zarr climate data with xarray"

        def validate(self, ctx: PipelineContext) -> bool:
            # Auto-detect format from extension if not set
            if not ctx.metadata.get("file_format"):
                ext = ctx.input_path.suffix.lower()
                fmt = {".nc": "netcdf", ".nc4": "netcdf", ".zarr": "zarr",
                       ".grib": "grib", ".grib2": "grib"}.get(ext)
                if fmt:
                    ctx.metadata["file_format"] = fmt
            return ctx.metadata.get("file_format") in ["netcdf", "zarr", "grib"]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import xarray as xr

                path = ctx.input_path

                if ctx.metadata.get("file_format") == "zarr":
                    ctx.data = xr.open_zarr(path)
                elif ctx.metadata.get("file_format") == "grib":
                    ctx.data = xr.open_dataset(path, engine="cfgrib")
                else:
                    ctx.data = xr.open_dataset(path)

                # Extract metadata
                ctx.metadata["variables"] = list(ctx.data.data_vars)
                ctx.metadata["dimensions"] = dict(ctx.data.dims)
                ctx.metadata["coordinates"] = list(ctx.data.coords)

                # Detect time range
                if "time" in ctx.data.coords:
                    ctx.metadata["time_range"] = {
                        "start": str(ctx.data.time.values[0]),
                        "end": str(ctx.data.time.values[-1]),
                        "steps": len(ctx.data.time)
                    }

                # Detect spatial extent
                for lat_name in ["lat", "latitude"]:
                    if lat_name in ctx.data.coords:
                        ctx.metadata["lat_range"] = {
                            "min": float(ctx.data[lat_name].min()),
                            "max": float(ctx.data[lat_name].max())
                        }
                        break

                for lon_name in ["lon", "longitude"]:
                    if lon_name in ctx.data.coords:
                        ctx.metadata["lon_range"] = {
                            "min": float(ctx.data[lon_name].min()),
                            "max": float(ctx.data[lon_name].max())
                        }
                        break

                logger.info(f"Loaded climate data: {ctx.metadata['variables']}")

            except ImportError as e:
                ctx.add_error(f"Missing package: {str(e)}")
            except Exception as e:
                ctx.add_error(f"Error loading climate data: {str(e)}")

            return ctx

    return [LoadClimateData()]


def get_preprocess_steps():
    """Get climate-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class RegridClimate(PipelineStep):
        """Regrid to uniform lat/lon resolution using xESMF"""

        name = "regrid_climate"
        stage = PipelineStage.PREPROCESS
        description = "Regrid climate data to uniform resolution"

        def __init__(self, target_resolution: float = 1.0, method: str = "bilinear"):
            super().__init__()
            self.target_resolution = target_resolution
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import xarray as xr
                import xesmf as xe
                import numpy as np

                if not hasattr(ctx.data, 'dims'):
                    return ctx

                ds = ctx.data

                # Determine current coordinate names
                lat_name = next((n for n in ["lat", "latitude"] if n in ds.coords), None)
                lon_name = next((n for n in ["lon", "longitude"] if n in ds.coords), None)

                if not lat_name or not lon_name:
                    logger.warning("Cannot regrid: missing lat/lon coordinates")
                    return ctx

                # Create target grid
                lat_out = np.arange(-90, 90 + self.target_resolution, self.target_resolution)
                lon_out = np.arange(0, 360, self.target_resolution)

                ds_out = xr.Dataset({
                    'lat': (['lat'], lat_out),
                    'lon': (['lon'], lon_out),
                })

                # Rename coordinates if needed
                if lat_name != 'lat' or lon_name != 'lon':
                    ds = ds.rename({lat_name: 'lat', lon_name: 'lon'})

                # Build regridder and apply
                regridder = xe.Regridder(ds, ds_out, self.method, periodic=True)
                ctx.data = regridder(ds)

                ctx.metadata["regridded"] = {
                    "method": self.method,
                    "target_resolution": self.target_resolution,
                    "output_shape": {
                        "lat": len(lat_out),
                        "lon": len(lon_out)
                    }
                }
                logger.info(f"Regridded to {self.target_resolution}° resolution")

            except ImportError:
                ctx.add_error("xesmf not installed")
            except Exception as e:
                ctx.add_error(f"Error regridding: {str(e)}")

            return ctx

    class NormalizeClimate(PipelineStep):
        """Compute and apply per-variable normalization"""

        name = "normalize_climate"
        stage = PipelineStage.PREPROCESS
        description = "Normalize climate variables (compute mean/std)"

        def __init__(self, method: str = "zscore"):
            super().__init__()
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import xarray as xr

                if not hasattr(ctx.data, 'data_vars'):
                    return ctx

                ds = ctx.data
                stats = {}

                for var in ds.data_vars:
                    da = ds[var]
                    mean = float(da.mean())
                    std = float(da.std())

                    if self.method == "zscore" and std > 0:
                        ds[var] = (da - mean) / std
                    elif self.method == "minmax":
                        min_val = float(da.min())
                        max_val = float(da.max())
                        if max_val > min_val:
                            ds[var] = (da - min_val) / (max_val - min_val)
                        stats[var] = {"min": min_val, "max": max_val}
                        continue

                    stats[var] = {"mean": mean, "std": std}

                ctx.data = ds
                ctx.metadata["normalization_stats"] = stats
                logger.info(f"Normalized {len(stats)} climate variables")

            except Exception as e:
                ctx.add_error(f"Error normalizing: {str(e)}")

            return ctx

    return [RegridClimate(), NormalizeClimate()]


def get_transform_steps():
    """Get climate-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class SelectVariables(PipelineStep):
        """Select specific climate variables"""

        name = "select_variables"
        stage = PipelineStage.TRANSFORM
        description = "Select specific climate variables"

        def __init__(self, variables: Optional[List[str]] = None):
            super().__init__()
            self.variables = variables

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            if not hasattr(ctx.data, 'data_vars'):
                return ctx

            if self.variables:
                available = [v for v in self.variables if v in ctx.data.data_vars]
                ctx.data = ctx.data[available]
                ctx.metadata["selected_variables"] = available

            return ctx

    class ComputeAnomalies(PipelineStep):
        """Compute climate anomalies from climatology"""

        name = "compute_anomalies"
        stage = PipelineStage.TRANSFORM
        description = "Compute anomalies from climatological mean"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                if not hasattr(ctx.data, 'data_vars') or 'time' not in ctx.data.dims:
                    return ctx

                ds = ctx.data
                climatology = ds.groupby('time.month').mean('time')
                anomalies = ds.groupby('time.month') - climatology

                ctx.data = anomalies
                ctx.metadata["anomalies_computed"] = True
                logger.info("Computed climate anomalies")

            except Exception as e:
                ctx.add_error(f"Error computing anomalies: {str(e)}")

            return ctx

    return [SelectVariables(), ComputeAnomalies()]


def get_structure_steps():
    """Get climate-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class ToNumpyArrays(PipelineStep):
        """Convert xarray to numpy arrays"""

        name = "climate_to_numpy"
        stage = PipelineStage.STRUCTURE
        description = "Convert climate data to numpy arrays"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                if not hasattr(ctx.data, 'data_vars'):
                    return ctx

                ds = ctx.data

                # Group variables by shape - only stack variables with matching shapes
                shape_groups = {}
                for var in ds.data_vars:
                    shape = ds[var].shape
                    if shape not in shape_groups:
                        shape_groups[shape] = []
                    shape_groups[shape].append(var)

                if not shape_groups:
                    return ctx

                # Use the group with the most variables, or the largest arrays
                best_group = max(shape_groups.items(),
                                key=lambda x: (len(x[1]), np.prod(x[0])))
                target_shape, var_names = best_group

                arrays = [ds[var].values for var in var_names]

                if len(arrays) == 1:
                    X = arrays[0]
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                else:
                    X = np.stack(arrays, axis=-1)  # [..., n_vars]

                ctx.data = {
                    "X": X.astype(np.float32),
                    "variable_names": var_names,
                    "dimensions": list(ds.dims.keys())
                }

                # Note if some variables were excluded
                excluded = [v for v in ds.data_vars if v not in var_names]
                if excluded:
                    logger.info(f"Excluded {len(excluded)} variables with different shapes")

                ctx.metadata["array_shape"] = X.shape
                logger.info(f"Converted to numpy: shape {X.shape}, vars: {var_names}")

            except Exception as e:
                ctx.add_error(f"Error converting to numpy: {str(e)}")

            return ctx

    return [ToNumpyArrays()]


def get_output_steps():
    """Get climate-specific output steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class TrainValTestSplit(PipelineStep):
        """Split data into train/val/test sets"""

        name = "train_val_test_split"
        stage = PipelineStage.OUTPUT
        description = "Split climate data into train/validation/test sets"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np
                from sklearn.model_selection import train_test_split

                if not isinstance(ctx.data, dict) or "X" not in ctx.data:
                    return ctx

                X = ctx.data["X"]
                n_samples = X.shape[0]

                # Split 80/10/10
                indices = np.arange(n_samples)
                train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
                val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

                ctx.data["train_idx"] = train_idx
                ctx.data["val_idx"] = val_idx
                ctx.data["test_idx"] = test_idx

                ctx.metadata["split"] = {
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx)
                }

                logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

            except ImportError:
                # No sklearn, just note we didn't split
                ctx.metadata["split"] = {"note": "sklearn not available"}
            except Exception as e:
                ctx.add_error(f"Error splitting data: {str(e)}")

            return ctx

    class SaveClimateOutput(PipelineStep):
        """Save climate data as train/val/test NPZ files"""

        name = "save_climate"
        stage = PipelineStage.OUTPUT
        description = "Save climate data as train/val/test NPZ files"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                ctx.output_dir.mkdir(parents=True, exist_ok=True)
                saved_files = []

                # Handle xarray Dataset - save as Zarr
                if hasattr(ctx.data, 'to_zarr'):
                    output_path = ctx.output_dir / "data.zarr"
                    ctx.data.to_zarr(output_path, mode='w')
                    saved_files.append(str(output_path))
                    ctx.metadata["output_format"] = "zarr"
                    logger.info(f"Saved Zarr: {output_path}")

                # Handle numpy dict with splits - save train/val/test
                elif isinstance(ctx.data, dict) and "X" in ctx.data:
                    X = ctx.data["X"]
                    var_names = ctx.data.get("variable_names", [])

                    if "train_idx" in ctx.data:
                        # Save split files
                        for split_name, idx_key in [("train", "train_idx"), ("val", "val_idx"), ("test", "test_idx")]:
                            idx = ctx.data[idx_key]
                            output_path = ctx.output_dir / f"{split_name}.npz"
                            np.savez_compressed(output_path, X=X[idx], variable_names=var_names)
                            saved_files.append(str(output_path))
                        logger.info(f"Saved train/val/test NPZ files")
                    else:
                        # No split, save single file
                        output_path = ctx.output_dir / "data.npz"
                        np.savez_compressed(output_path, X=X, variable_names=var_names)
                        saved_files.append(str(output_path))
                        logger.info(f"Saved NPZ: {output_path}, shape: {X.shape}")

                    ctx.metadata["output_format"] = "npz"

                ctx.metadata["saved_files"] = saved_files

            except Exception as e:
                ctx.add_error(f"Error saving climate output: {str(e)}")

            return ctx

    return [TrainValTestSplit(), SaveClimateOutput()]


def get_full_pipeline():
    """Get the complete climate preprocessing pipeline"""
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
