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
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error
    from ..pipeline.ingest import DetectFormat

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

                # Handle multi-file case
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]
                file_format = ctx.metadata.get("file_format")

                if len(paths) == 1:
                    # Single file mode
                    path = paths[0]
                    if file_format == "zarr":
                        ctx.data = xr.open_zarr(path)
                    elif file_format == "grib":
                        ctx.data = xr.open_dataset(path, engine="cfgrib")
                    else:
                        ctx.data = xr.open_dataset(path)
                    logger.info(f"Loaded single climate file: {path.name}")
                else:
                    # Multi-file mode - load sequentially and concatenate
                    if file_format == "zarr":
                        ctx.add_error("Cannot combine multiple Zarr stores. Use a single Zarr store.")
                        return ctx

                    file_list = [str(p) for p in paths]
                    logger.info(f"Loading {len(paths)} climate files sequentially")

                    datasets = []
                    failed_files = []
                    for i, fpath in enumerate(file_list):
                        fname = Path(fpath).name
                        logger.debug(f"Loading file {i+1}/{len(file_list)}: {fname}")
                        try:
                            if file_format == "grib":
                                ds = xr.open_dataset(fpath, engine="cfgrib")
                            else:
                                ds = xr.open_dataset(fpath)
                            datasets.append(ds)
                        except Exception as e:
                            failed_files.append((fname, str(e)))
                            logger.warning(f"Failed to load {fname}: {e}")

                    if failed_files:
                        # Report failed files
                        failed_names = [f[0] for f in failed_files]
                        ctx.metadata["failed_files"] = failed_files
                        if not datasets:
                            ctx.add_error(f"All {len(file_list)} files failed to load. First error: {failed_files[0][1]}")
                            return ctx
                        logger.warning(f"Skipped {len(failed_files)} corrupted file(s): {', '.join(failed_names)}")

                    if len(datasets) == 1:
                        ctx.data = datasets[0]
                    else:
                        ctx.data = xr.concat(datasets, dim='time', data_vars='minimal')

                    logger.info(f"Successfully combined {len(datasets)} files")

                # Extract metadata
                ctx.metadata["variables"] = list(ctx.data.data_vars)
                ctx.metadata["dimensions"] = dict(ctx.data.dims)
                ctx.metadata["coordinates"] = list(ctx.data.coords)
                ctx.metadata["num_source_files"] = len(paths)

                # Detect time range
                if "time" in ctx.data.coords:
                    time_vals = ctx.data.time.values
                    ctx.metadata["time_range"] = {
                        "start": str(time_vals[0]),
                        "end": str(time_vals[-1]),
                        "steps": len(time_vals)
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

            except ImportError:
                ctx.add_error(missing_dependency_error("xarray/netCDF4", "climate"))
            except Exception as e:
                ctx.add_error(f"Error loading climate data: {str(e)}")

            return ctx

    return [DetectFormat(), LoadClimateData()]


def get_preprocess_steps():
    """Get climate-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error

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
                ctx.add_error(missing_dependency_error("xesmf", "climate"))
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

            except ImportError:
                ctx.add_error(missing_dependency_error("xarray/netCDF4", "climate"))
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
        """Save climate data in specified format"""

        name = "save_climate"
        stage = PipelineStage.OUTPUT
        description = "Save climate data as train/val/test files"

        def _save_split(self, X, var_names, split_name, output_dir, fmt):
            """Save a single split in the specified format"""
            import numpy as np

            if fmt == "npz":
                path = output_dir / f"{split_name}.npz"
                np.savez_compressed(path, X=X, variable_names=var_names)
            elif fmt == "parquet":
                import pandas as pd
                path = output_dir / f"{split_name}.parquet"
                df = pd.DataFrame(X, columns=var_names if len(var_names) == X.shape[1] else None)
                df.to_parquet(path)
            elif fmt == "csv":
                import pandas as pd
                path = output_dir / f"{split_name}.csv"
                df = pd.DataFrame(X, columns=var_names if len(var_names) == X.shape[1] else None)
                df.to_csv(path, index=False)
            elif fmt == "hdf5":
                import h5py
                path = output_dir / f"{split_name}.h5"
                with h5py.File(path, 'w') as f:
                    f.create_dataset('X', data=X)
                    f.attrs['variable_names'] = var_names
            elif fmt == "zarr":
                import zarr
                path = output_dir / f"{split_name}.zarr"
                root = zarr.open(str(path), mode='w')
                root.create_dataset('X', data=X)
                root.attrs['variable_names'] = var_names
            else:
                path = output_dir / f"{split_name}.npz"
                np.savez_compressed(path, X=X, variable_names=var_names)
            return str(path)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                ctx.output_dir.mkdir(parents=True, exist_ok=True)
                saved_files = []
                fmt = ctx.config.get("output_format", "npz")

                # Handle xarray Dataset - save as Zarr
                if hasattr(ctx.data, 'to_zarr'):
                    output_path = ctx.output_dir / "data.zarr"
                    ctx.data.to_zarr(output_path, mode='w')
                    saved_files.append(str(output_path))
                    fmt = "zarr"
                    logger.info(f"Saved Zarr: {output_path}")

                # Handle numpy dict with splits
                elif isinstance(ctx.data, dict) and "X" in ctx.data:
                    X = ctx.data["X"]
                    var_names = ctx.data.get("variable_names", [])

                    if "train_idx" in ctx.data:
                        # Save split files
                        for split_name, idx_key in [("train", "train_idx"), ("val", "val_idx"), ("test", "test_idx")]:
                            idx = ctx.data[idx_key]
                            path = self._save_split(X[idx], var_names, split_name, ctx.output_dir, fmt)
                            saved_files.append(path)
                        logger.info(f"Saved train/val/test {fmt.upper()} files")
                    else:
                        # No split, save single file
                        path = self._save_split(X, var_names, "data", ctx.output_dir, fmt)
                        saved_files.append(path)
                        logger.info(f"Saved {fmt.upper()}: {path}")

                ctx.metadata["output_format"] = fmt
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
