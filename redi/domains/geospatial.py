"""
DRAI Geospatial Domain Module

Domain-specific preprocessing for geospatial data with
H3 hexagonal indexing (hexify) support.

Handles data with geographic coordinates (lat/lon), grid IDs,
or shapefiles and converts to uniform H3 hexagonal cells.

Use cases: air quality, wildfire smoke, epidemiology, urban analytics,
remote sensing, disaster response, agriculture, telecom coverage.

Tools: h3, geopandas, shapely, pandas
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = ["h3", "geopandas", "shapely"]

# Common geospatial field patterns
GEOSPATIAL_FIELDS = {
    "coordinates": ["lat", "latitude", "lon", "longitude", "lng", "x", "y"],
    "identifiers": ["grid_id", "h3_index", "hex_id", "cell_id", "fips", "geoid"],
    "values": ["pm25", "pm2.5", "smokePM", "aqi", "concentration", "exposure", "value"],
    "temporal": ["date", "time", "datetime", "timestamp", "day", "month", "year"],
    "spatial_meta": ["state", "county", "region", "zone", "grid_id_10km"],
}


def get_ingest_steps():
    """Get geospatial-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error
    from ..pipeline.ingest import DetectFormat

    class LoadGeospatialData(PipelineStep):
        """Load geospatial data from CSV, Parquet, or Shapefile"""

        name = "load_geospatial"
        stage = PipelineStage.INGEST
        description = "Load geospatial data with coordinate/grid detection"

        def validate(self, ctx: PipelineContext) -> bool:
            fmt = ctx.metadata.get("file_format")
            return fmt in ["csv", "parquet", "shp", "geojson", "gpkg"]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                file_format = ctx.metadata.get("file_format")
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

                if file_format in ("shp", "geojson", "gpkg"):
                    try:
                        import geopandas as gpd
                    except ImportError:
                        ctx.add_error(missing_dependency_error("geopandas", "geospatial"))
                        return ctx
                    df = gpd.read_file(str(paths[0]))
                    ctx.metadata["has_geometry"] = True
                elif len(paths) == 1:
                    if file_format == "parquet":
                        df = pd.read_parquet(paths[0])
                    else:
                        df = pd.read_csv(paths[0])
                else:
                    dfs = []
                    for path in paths:
                        if file_format == "parquet":
                            dfs.append(pd.read_parquet(path))
                        else:
                            dfs.append(pd.read_csv(path))
                    df = pd.concat(dfs, axis=0, ignore_index=True)
                    logger.info(f"Concatenated {len(paths)} files")

                ctx.data = df

                # Identify field types
                cols_lower = {c.lower(): c for c in df.columns}
                identified = {}
                for field_type, keywords in GEOSPATIAL_FIELDS.items():
                    for keyword in keywords:
                        for col_lower, col in cols_lower.items():
                            if keyword in col_lower:
                                identified.setdefault(field_type, []).append(col)

                ctx.metadata["identified_fields"] = identified
                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)

                # Detect if data is already hexified
                h3_cols = [c for c in df.columns if "h3" in c.lower() or "hex" in c.lower()]
                ctx.metadata["pre_hexified"] = len(h3_cols) > 0
                if h3_cols:
                    ctx.metadata["h3_columns"] = h3_cols

                logger.info(
                    f"Loaded geospatial data: {len(df)} rows, "
                    f"{len(identified)} field types identified, "
                    f"pre_hexified={ctx.metadata['pre_hexified']}"
                )

            except Exception as e:
                ctx.add_error(f"Error loading geospatial data: {str(e)}")

            return ctx

    return [DetectFormat(), LoadGeospatialData()]


def get_preprocess_steps():
    """Get geospatial-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CleanGeospatialData(PipelineStep):
        """Clean geospatial data: validate coords, parse dates, dedup"""

        name = "clean_geospatial"
        stage = PipelineStage.PREPROCESS
        description = "Validate coordinates, parse dates, remove duplicates"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                initial_count = len(df)

                # Parse date columns
                temporal_cols = ctx.metadata.get("identified_fields", {}).get("temporal", [])
                parsed_dates = []
                for col in temporal_cols:
                    if col in df.columns and df[col].dtype == "object":
                        try:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            parsed_dates.append(col)
                        except Exception:
                            pass
                ctx.metadata["parsed_dates"] = parsed_dates

                # Validate coordinate ranges if present
                coord_cols = ctx.metadata.get("identified_fields", {}).get("coordinates", [])
                lat_col = None
                lon_col = None
                for col in coord_cols:
                    col_lower = col.lower()
                    if "lat" in col_lower:
                        lat_col = col
                    elif "lon" in col_lower or "lng" in col_lower:
                        lon_col = col

                if lat_col and lat_col in df.columns:
                    df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
                if lon_col and lon_col in df.columns:
                    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]

                ctx.metadata["lat_column"] = lat_col
                ctx.metadata["lon_column"] = lon_col

                # Remove duplicates
                df = df.drop_duplicates()

                ctx.data = df
                ctx.metadata["rows_removed"] = initial_count - len(df)
                logger.info(f"Cleaned geospatial data: {initial_count} -> {len(df)} rows")

            except Exception as e:
                ctx.add_error(f"Error cleaning geospatial data: {str(e)}")

            return ctx

    return [CleanGeospatialData()]


def get_transform_steps():
    """Get geospatial-specific transform steps (hexify + aggregate + normalize)"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error

    class HexifyData(PipelineStep):
        """Convert geographic coordinates or grid cells to H3 hexagonal indices"""

        name = "hexify"
        stage = PipelineStage.TRANSFORM
        description = "Convert coordinates/grid cells to H3 hexagonal indices"

        def __init__(self, resolution: int = 8, shapefile: Optional[str] = None,
                     grid_id_field: Optional[str] = None,
                     shape_id_field: Optional[str] = None):
            super().__init__()
            self.resolution = resolution
            self.shapefile = shapefile
            self.grid_id_field = grid_id_field
            self.shape_id_field = shape_id_field

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import h3
            except ImportError:
                ctx.add_error(missing_dependency_error("h3", "geospatial"))
                return ctx

            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data

                # Check if already hexified
                if ctx.metadata.get("pre_hexified"):
                    logger.info("Data already contains H3 indices, skipping hexify")
                    return ctx

                # Override from config if available
                geo_config = ctx.config.get("geospatial", {}).get("hexify", {})
                resolution = geo_config.get("resolution", self.resolution)
                shapefile = geo_config.get("shapefile", self.shapefile)
                grid_id_field = geo_config.get("grid_id_field", self.grid_id_field)
                shape_id_field = geo_config.get("shape_id_field", self.shape_id_field)

                h3_col = f"h3_index_res{resolution}"

                if shapefile and grid_id_field:
                    # Shapefile-based hexify: map grid IDs -> centroids -> H3
                    df = self._hexify_via_shapefile(
                        df, shapefile, grid_id_field, shape_id_field or "ID",
                        resolution, h3_col
                    )
                else:
                    # Direct lat/lon hexify
                    lat_col = ctx.metadata.get("lat_column")
                    lon_col = ctx.metadata.get("lon_column")

                    if not lat_col or not lon_col:
                        # Try to find them
                        for col in df.columns:
                            cl = col.lower()
                            if "lat" in cl and not lat_col:
                                lat_col = col
                            elif ("lon" in cl or "lng" in cl) and not lon_col:
                                lon_col = col

                    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
                        df[h3_col] = df.apply(
                            lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], resolution),
                            axis=1
                        )
                    else:
                        ctx.add_error(
                            "Cannot hexify: no lat/lon columns found and no shapefile configured. "
                            "Provide a shapefile via config or ensure data has lat/lon columns."
                        )
                        return ctx

                ctx.data = df
                ctx.metadata["h3_column"] = h3_col
                ctx.metadata["h3_resolution"] = resolution
                ctx.metadata["num_unique_cells"] = df[h3_col].nunique()
                logger.info(
                    f"Hexified data at resolution {resolution}: "
                    f"{ctx.metadata['num_unique_cells']} unique H3 cells"
                )

            except Exception as e:
                ctx.add_error(f"Error in hexify: {str(e)}")

            return ctx

        def _hexify_via_shapefile(self, df, shapefile, grid_id_field,
                                  shape_id_field, resolution, h3_col):
            """Map grid IDs to H3 via shapefile centroids"""
            import h3

            try:
                import geopandas as gpd
            except ImportError:
                raise ImportError(
                    "geopandas is required for shapefile-based hexify. "
                    'Install with: pip install -e ".[geospatial]"'
                )

            gdf = gpd.read_file(shapefile)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            # Compute centroids in projected CRS for accuracy, then back to WGS84
            centroids_proj = gdf.to_crs("EPSG:3857").geometry.centroid
            centroids = gpd.GeoSeries(centroids_proj, crs="EPSG:3857").to_crs("EPSG:4326")

            # Build grid_id -> h3 mapping
            id_to_h3 = {}
            for shape_id, point in zip(gdf[shape_id_field], centroids):
                h3_idx = h3.latlng_to_cell(point.y, point.x, resolution)
                id_to_h3[str(shape_id)] = h3_idx

            # Map grid IDs to H3
            df[h3_col] = df[grid_id_field].astype(str).map(id_to_h3)

            unmapped = df[h3_col].isna().sum()
            if unmapped > 0:
                logger.warning(f"{unmapped} rows could not be mapped to H3 (missing grid IDs in shapefile)")

            return df

    class AggregateByHex(PipelineStep):
        """Aggregate values within H3 cells"""

        name = "aggregate_hex"
        stage = PipelineStage.TRANSFORM
        description = "Aggregate values within H3 hexagonal cells"

        def __init__(self, method: str = "mean", time_field: Optional[str] = None):
            super().__init__()
            self.method = method
            self.time_field = time_field

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                h3_col = ctx.metadata.get("h3_column")

                if not h3_col or h3_col not in df.columns:
                    logger.info("No H3 column found, skipping aggregation")
                    return ctx

                # Override from config
                agg_config = ctx.config.get("geospatial", {}).get("aggregate", {})
                method = agg_config.get("method", self.method)
                time_field = agg_config.get("time_field", self.time_field)

                # Find time field if not specified
                if not time_field:
                    parsed_dates = ctx.metadata.get("parsed_dates", [])
                    temporal_cols = ctx.metadata.get("identified_fields", {}).get("temporal", [])
                    candidates = parsed_dates + temporal_cols
                    for col in candidates:
                        if col in df.columns:
                            time_field = col
                            break

                # Determine groupby columns
                groupby_cols = [h3_col]
                if time_field and time_field in df.columns:
                    groupby_cols.append(time_field)

                # Get numeric columns for aggregation
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                # Exclude grid ID columns from aggregation
                exclude = ctx.metadata.get("identified_fields", {}).get("identifiers", [])
                agg_cols = [c for c in numeric_cols if c not in exclude]

                if not agg_cols:
                    logger.info("No numeric columns to aggregate")
                    return ctx

                initial_rows = len(df)

                if method == "mean":
                    df_agg = df.groupby(groupby_cols, as_index=False)[agg_cols].mean()
                elif method == "sum":
                    df_agg = df.groupby(groupby_cols, as_index=False)[agg_cols].sum()
                elif method == "count":
                    df_agg = df.groupby(groupby_cols, as_index=False)[agg_cols].count()
                else:
                    df_agg = df.groupby(groupby_cols, as_index=False)[agg_cols].mean()

                # Add count column
                counts = df.groupby(groupby_cols).size().reset_index(name="cell_count")
                df_agg = df_agg.merge(counts, on=groupby_cols, how="left")

                ctx.data = df_agg
                ctx.metadata["aggregation_method"] = method
                ctx.metadata["rows_after_aggregation"] = len(df_agg)
                logger.info(
                    f"Aggregated by H3 cell ({method}): {initial_rows} -> {len(df_agg)} rows"
                )

            except Exception as e:
                ctx.add_error(f"Error aggregating by H3 cell: {str(e)}")

            return ctx

    class NormalizeGeospatial(PipelineStep):
        """Normalize numerical features"""

        name = "normalize_geospatial"
        stage = PipelineStage.TRANSFORM
        description = "Normalize geospatial numerical features"

        def __init__(self, method: str = "zscore"):
            super().__init__()
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                # Override from config
                norm_config = ctx.config.get("normalize", {})
                if not norm_config.get("enabled", True):
                    logger.info("Normalization disabled in config")
                    return ctx
                method = norm_config.get("method", self.method)

                df = ctx.data
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

                # Exclude count/ID columns
                exclude_patterns = ["count", "cell_count", "grid_id", "h3_index"]
                feature_cols = [
                    c for c in numeric_cols
                    if not any(p in c.lower() for p in exclude_patterns)
                ]

                stats = {}
                for col in feature_cols:
                    if method == "zscore":
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[col] = (df[col] - mean) / std
                        stats[col] = {"mean": float(mean), "std": float(std)}
                    elif method == "minmax":
                        cmin = df[col].min()
                        cmax = df[col].max()
                        rng = cmax - cmin
                        if rng > 0:
                            df[col] = (df[col] - cmin) / rng
                        stats[col] = {"min": float(cmin), "max": float(cmax)}

                ctx.data = df
                ctx.metadata["normalization_stats"] = stats
                logger.info(f"Normalized {len(stats)} features ({method})")

            except Exception as e:
                ctx.add_error(f"Error normalizing: {str(e)}")

            return ctx

    return [HexifyData(), AggregateByHex(), NormalizeGeospatial()]


def get_structure_steps():
    """Get geospatial-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class PrepareGeospatialML(PipelineStep):
        """Prepare geospatial data for ML: feature arrays with optional k-ring neighbors"""

        name = "prepare_geospatial_ml"
        stage = PipelineStage.STRUCTURE
        description = "Prepare geospatial data for ML training"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                h3_col = ctx.metadata.get("h3_column")

                # Get numeric feature columns
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

                # Exclude non-feature columns
                exclude = ["cell_count"]
                feature_cols = [c for c in numeric_cols if c not in exclude]

                # Fill remaining NaN with 0
                for col in feature_cols:
                    if df[col].isna().any():
                        df[col] = df[col].fillna(0)

                X = df[feature_cols].values.astype(np.float32)
                result = {"X": X, "feature_names": feature_cols}

                # Include H3 indices as metadata
                if h3_col and h3_col in df.columns:
                    result["h3_indices"] = df[h3_col].values

                # Include temporal info if available
                temporal_cols = ctx.metadata.get("identified_fields", {}).get("temporal", [])
                for col in temporal_cols:
                    if col in df.columns:
                        result["timestamps"] = df[col].values
                        break

                ctx.data = result
                ctx.metadata["array_shape"] = X.shape
                ctx.metadata["feature_columns"] = feature_cols
                logger.info(f"Prepared ML data: X shape {X.shape}")

            except Exception as e:
                ctx.add_error(f"Error preparing for ML: {str(e)}")

            return ctx

    return [PrepareGeospatialML()]


def get_output_steps(split: bool = True):
    """Get geospatial-specific output steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.output import get_output_steps as _get_output_steps

    if split:
        return _get_output_steps(output_format="parquet", split=True)

    # No split: wrap the data dict so SaveParquet writes a single "data.parquet"
    class WrapForSingleFile(PipelineStep):
        name = "wrap_single_file"
        stage = PipelineStage.OUTPUT
        description = "Prepare single-file output (no train/val/test split)"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            if isinstance(ctx.data, dict) and "X" in ctx.data:
                ctx.data = {"data": ctx.data}
            return ctx

    return [WrapForSingleFile()] + _get_output_steps(output_format="parquet", split=False)


def get_full_pipeline(split: bool = True):
    """Get the complete geospatial preprocessing pipeline"""
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps(split=split)
    )
