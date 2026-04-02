"""
redi/discover/processors/netcdf_processor.py — NetCDF file analysis.
"""

from __future__ import annotations

from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery, FileParseError, PreprocessingStep


class NetCDFProcessor(BaseProcessor):
    extension = ".nc"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            import xarray as xr
        except ImportError:
            return FileDiscovery(
                file_path=str(path),
                file_type="NetCDF",
                file_size_bytes=path.stat().st_size,
                metadata={"error": "xarray not installed"},
            )

        try:
            ds = xr.open_dataset(path)
        except Exception as exc:
            raise FileParseError(
                path=path,
                file_type="NetCDF",
                reason=str(exc),
                cause=exc,
            ) from exc

        metadata: dict = {}
        steps: list[PreprocessingStep] = []

        # Variables
        metadata["variables"] = list(ds.data_vars)
        metadata["dimensions"] = dict(ds.dims)
        metadata["coordinates"] = list(ds.coords)

        # Spatial resolution from lat/lon
        lat_name = next((c for c in ds.coords if c in ("lat", "latitude")), None)
        lon_name = next((c for c in ds.coords if c in ("lon", "longitude")), None)
        if lat_name and lon_name:
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            lat_res = round(float(abs(lat[1] - lat[0])), 4) if len(lat) > 1 else None
            lon_res = round(float(abs(lon[1] - lon[0])), 4) if len(lon) > 1 else None
            metadata["lat_range"] = [round(float(lat.min()), 2), round(float(lat.max()), 2)]
            metadata["lon_range"] = [round(float(lon.min()), 2), round(float(lon.max()), 2)]
            if lat_res is not None:
                metadata["resolution_deg"] = lat_res

        # Time range
        time_name = next((c for c in ds.coords if c in ("time",)), None)
        if time_name:
            try:
                t = ds[time_name]
                metadata["time_steps"] = int(t.sizes[time_name])
                metadata["time_start"] = str(t.values[0])[:10]
                metadata["time_end"] = str(t.values[-1])[:10]
            except Exception:
                pass

        # Per-variable units and long_name (skip coordinate bound arrays like lat_bnds)
        var_info = {}
        for var in ds.data_vars:
            if var.endswith("_bnds") or var.endswith("_bounds"):
                continue
            info = {}
            attrs = ds[var].attrs
            if "units" in attrs:
                info["units"] = attrs["units"]
            if "long_name" in attrs:
                info["long_name"] = attrs["long_name"]
            info["shape"] = list(ds[var].shape)
            var_info[var] = info
        metadata["variable_info"] = var_info

        ds.close()

        # --- Preprocessing steps ---
        res = metadata.get("resolution_deg")
        if res is not None:
            steps.append(PreprocessingStep(
                name="Verify spatial resolution",
                rationale=f"Grid is {res}° — confirm this matches your model's expected input resolution",
                suggested_method="regrid with xesmf if resolution does not match target",
            ))

        for var in var_info:
            units = metadata["variable_info"].get(var, {}).get("units", "")
            rationale = f"Raw physical values in '{units}'" if units else "Raw values detected"
            steps.append(PreprocessingStep(
                name=f"Normalize `{var}`",
                rationale=rationale + " — normalize before use in ML models",
                suggested_method="z-score or min-max normalization; compute statistics over the full dataset",
            ))

        t_steps = metadata.get("time_steps")
        if t_steps and t_steps > 1000:
            steps.append(PreprocessingStep(
                name="Chunk along time dimension",
                rationale=f"{t_steps} timesteps — large time dimension will cause slow I/O if loaded whole",
                suggested_method="chunk into fixed-size windows (e.g. zarr) suited to your model's sequence length",
            ))

        steps.append(PreprocessingStep(
            name="Create train/val/test split",
            rationale="No temporal split detected — held-out evaluation data is required for unbiased assessment",
            suggested_method="split along the time axis; avoid random shuffling to prevent data leakage",
        ))

        return FileDiscovery(
            file_path=str(path),
            file_type="NetCDF",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )
