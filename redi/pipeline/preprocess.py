"""
DRAI Pipeline - Preprocess Stage

Handles data cleaning, regridding, alignment, and anonymization.
Domain-specific preprocessing is delegated to domain modules.

Operations:
- Handle missing values
- Remove duplicates
- Regrid (climate)
- Align (fusion, life sciences)
- Anonymize (life sciences)
- Basic quality checks
"""

import logging
from typing import Any, Dict, List, Optional
from . import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error

logger = logging.getLogger(__name__)


class HandleMissingValues(PipelineStep):
    """Handle missing values in tabular data"""

    name = "handle_missing"
    stage = PipelineStage.PREPROCESS
    description = "Handle missing values (drop, fill, or impute)"

    def __init__(self, strategy: str = "drop", fill_value: Any = None):
        super().__init__()
        self.strategy = strategy  # "drop", "fill", "mean", "median", "mode"
        self.fill_value = fill_value

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            df = ctx.data
            missing_before = df.isna().sum().sum()
            rows_before = len(df)

            logger.debug(f"     Strategy: {self.strategy}")
            logger.debug(f"     Total missing values: {missing_before}")

            # Log per-column missing counts
            missing_per_col = df.isna().sum()
            cols_with_missing = missing_per_col[missing_per_col > 0]
            if len(cols_with_missing) > 0:
                logger.debug(f"     Columns with missing values:")
                for col, count in cols_with_missing.items():
                    pct = (count / len(df)) * 100
                    logger.debug(f"       - {col}: {count} ({pct:.1f}%)")

            if self.strategy == "drop":
                ctx.data = df.dropna()
                logger.debug(f"     Dropped {rows_before - len(ctx.data)} rows with missing values")
            elif self.strategy == "fill":
                ctx.data = df.fillna(self.fill_value)
                logger.debug(f"     Filled with value: {self.fill_value}")
            elif self.strategy == "mean":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                ctx.data = df
                logger.debug(f"     Filled {len(numeric_cols)} numeric columns with mean")
            elif self.strategy == "median":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                ctx.data = df
                logger.debug(f"     Filled {len(numeric_cols)} numeric columns with median")
            elif self.strategy == "mode":
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else None)
                ctx.data = df
                logger.debug(f"     Filled all columns with mode")

            missing_after = ctx.data.isna().sum().sum()
            ctx.metadata["missing_values_handled"] = {
                "strategy": self.strategy,
                "before": int(missing_before),
                "after": int(missing_after)
            }
            logger.debug(f"     Remaining missing values: {missing_after}")
            logger.info(f"Missing values: {missing_before} -> {missing_after} (strategy: {self.strategy})")

        except ImportError:
            ctx.add_error(missing_dependency_error("pandas"))
        except Exception as e:
            ctx.add_error(f"Error handling missing values: {str(e)}")
        return ctx


class RemoveDuplicates(PipelineStep):
    """Remove duplicate rows from tabular data"""

    name = "remove_duplicates"
    stage = PipelineStage.PREPROCESS
    description = "Remove duplicate rows"

    def __init__(self, subset: Optional[List[str]] = None, keep: str = "first"):
        super().__init__()
        self.subset = subset  # Columns to consider for duplicates
        self.keep = keep  # "first", "last", or False

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            before_count = len(ctx.data)
            logger.debug(f"     Rows before: {before_count:,}")
            logger.debug(f"     Subset columns: {self.subset if self.subset else 'all columns'}")
            logger.debug(f"     Keep strategy: {self.keep}")

            # Determine which columns to check - skip unhashable columns
            subset = self.subset
            if subset is None:
                hashable_cols = []
                for col in ctx.data.columns:
                    dtype = ctx.data[col].dtype
                    if dtype != object:
                        hashable_cols.append(col)
                    else:
                        # Sample a non-null value to check hashability
                        sample = ctx.data[col].dropna().iloc[0] if not ctx.data[col].dropna().empty else None
                        if sample is not None and isinstance(sample, (list, dict)):
                            logger.debug(f"     Skipping column '{col}' (unhashable {type(sample).__name__})")
                        else:
                            hashable_cols.append(col)
                if len(hashable_cols) < len(ctx.data.columns):
                    logger.debug(f"     Using {len(hashable_cols)}/{len(ctx.data.columns)} hashable columns for dedup")
                    subset = hashable_cols if hashable_cols else None

            if subset is not None and len(subset) == 0:
                logger.debug(f"     No hashable columns available, skipping dedup")
                ctx.metadata["duplicates_removed"] = 0
                logger.info("Skipped dedup (no hashable columns)")
                return ctx

            ctx.data = ctx.data.drop_duplicates(subset=subset, keep=self.keep)
            after_count = len(ctx.data)
            removed = before_count - after_count

            ctx.metadata["duplicates_removed"] = removed
            logger.debug(f"     Rows after: {after_count:,}")
            logger.debug(f"     Duplicates removed: {removed:,} ({(removed/before_count)*100:.1f}%)" if before_count > 0 else "     No rows to check")
            logger.info(f"Removed {removed} duplicate rows")

        except Exception as e:
            ctx.add_error(f"Error removing duplicates: {str(e)}")
        return ctx


class ValidateDataTypes(PipelineStep):
    """Validate and convert data types"""

    name = "validate_dtypes"
    stage = PipelineStage.PREPROCESS
    description = "Validate and convert data types"

    def __init__(self, type_mapping: Optional[Dict[str, str]] = None):
        super().__init__()
        self.type_mapping = type_mapping or {}

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            dtypes_before = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}
            conversions = []

            # Auto-detect datetime columns
            logger.debug(f"     Scanning {len(ctx.data.columns)} columns for datetime patterns")
            for col in ctx.data.columns:
                if ctx.data[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                            ctx.data[col] = pd.to_datetime(ctx.data[col], errors='coerce')
                            conversions.append(f"{col}: object -> datetime64")
                            logger.debug(f"     Auto-converted '{col}' to datetime")
                    except Exception:
                        pass

            # Apply explicit type mapping
            if self.type_mapping:
                logger.debug(f"     Applying {len(self.type_mapping)} explicit type mappings")
            for col, dtype in self.type_mapping.items():
                if col in ctx.data.columns:
                    try:
                        old_dtype = str(ctx.data[col].dtype)
                        ctx.data[col] = ctx.data[col].astype(dtype)
                        conversions.append(f"{col}: {old_dtype} -> {dtype}")
                        logger.debug(f"     Converted '{col}': {old_dtype} -> {dtype}")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {e}")

            ctx.metadata["dtypes_after"] = {col: str(dtype) for col, dtype in ctx.data.dtypes.items()}

            if conversions:
                logger.debug(f"     Type conversions performed: {len(conversions)}")
            else:
                logger.debug(f"     No type conversions needed")

            logger.info("Data types validated")

        except Exception as e:
            ctx.add_error(f"Error validating data types: {str(e)}")
        return ctx


class HandleOutliers(PipelineStep):
    """Detect and handle outliers using IQR or z-score"""

    name = "handle_outliers"
    stage = PipelineStage.PREPROCESS
    description = "Detect and handle statistical outliers"

    def __init__(self, method: str = "iqr", threshold: float = 1.5, action: str = "clip"):
        super().__init__()
        self.method = method  # "iqr" or "zscore"
        self.threshold = threshold  # IQR multiplier or z-score threshold
        self.action = action  # "clip", "remove", or "flag"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            numeric_cols = ctx.data.select_dtypes(include=['number']).columns
            outliers_found = {}

            logger.debug(f"     Method: {self.method}")
            logger.debug(f"     Threshold: {self.threshold}")
            logger.debug(f"     Action: {self.action}")
            logger.debug(f"     Checking {len(numeric_cols)} numeric columns")

            total_outliers = 0
            for col in numeric_cols:
                if self.method == "iqr":
                    Q1 = ctx.data[col].quantile(0.25)
                    Q3 = ctx.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - self.threshold * IQR
                    upper = Q3 + self.threshold * IQR
                elif self.method == "zscore":
                    mean = ctx.data[col].mean()
                    std = ctx.data[col].std()
                    lower = mean - self.threshold * std
                    upper = mean + self.threshold * std

                outlier_mask = (ctx.data[col] < lower) | (ctx.data[col] > upper)
                outlier_count = int(outlier_mask.sum())
                outliers_found[col] = outlier_count
                total_outliers += outlier_count

                if outlier_count > 0:
                    logger.debug(f"     {col}: {outlier_count} outliers (bounds: [{lower:.2f}, {upper:.2f}])")

                if self.action == "clip":
                    ctx.data[col] = ctx.data[col].clip(lower=lower, upper=upper)
                elif self.action == "remove":
                    ctx.data = ctx.data[~outlier_mask]
                elif self.action == "flag":
                    ctx.data[f"{col}_outlier"] = outlier_mask

            ctx.metadata["outliers"] = {
                "method": self.method,
                "threshold": self.threshold,
                "action": self.action,
                "found": outliers_found
            }
            logger.debug(f"     Total outliers found: {total_outliers}")
            logger.info(f"Outliers handled with {self.method} method")

        except Exception as e:
            ctx.add_error(f"Error handling outliers: {str(e)}")
        return ctx


class AnonymizePII(PipelineStep):
    """Anonymize personally identifiable information"""

    name = "anonymize_pii"
    stage = PipelineStage.PREPROCESS
    description = "Anonymize PII columns (hash or remove)"

    PII_KEYWORDS = [
        'name', 'email', 'phone', 'address', 'ssn', 'social_security',
        'patient', 'patient_id', 'dob', 'birth', 'ip_address',
        'credit_card', 'account_number'
    ]

    def __init__(self, columns: Optional[List[str]] = None, method: str = "hash"):
        super().__init__()
        self.columns = columns  # Explicit columns, or None for auto-detect
        self.method = method  # "hash", "remove", or "mask"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import hashlib
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Auto-detect PII columns if not specified
            pii_cols = self.columns or []
            if not pii_cols:
                logger.debug(f"     Auto-detecting PII columns using keywords: {', '.join(self.PII_KEYWORDS[:5])}...")
                for col in ctx.data.columns:
                    if any(keyword in col.lower() for keyword in self.PII_KEYWORDS):
                        pii_cols.append(col)
                        logger.debug(f"     Detected PII column: {col}")
            else:
                logger.debug(f"     Using specified PII columns: {', '.join(pii_cols)}")

            logger.debug(f"     Anonymization method: {self.method}")

            anonymized = []
            for col in pii_cols:
                if col not in ctx.data.columns:
                    logger.debug(f"     Column '{col}' not found, skipping")
                    continue

                if self.method == "hash":
                    ctx.data[col] = ctx.data[col].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else None
                    )
                    logger.debug(f"     Hashed column: {col}")
                elif self.method == "remove":
                    ctx.data = ctx.data.drop(columns=[col])
                    logger.debug(f"     Removed column: {col}")
                elif self.method == "mask":
                    ctx.data[col] = "***MASKED***"
                    logger.debug(f"     Masked column: {col}")

                anonymized.append(col)

            ctx.metadata["pii_anonymized"] = {
                "columns": anonymized,
                "method": self.method
            }
            logger.info(f"Anonymized {len(anonymized)} PII columns")

        except Exception as e:
            ctx.add_error(f"Error anonymizing PII: {str(e)}")
        return ctx


class RegridClimate(PipelineStep):
    """Regrid climate data to uniform resolution"""

    name = "regrid_climate"
    stage = PipelineStage.PREPROCESS
    description = "Regrid climate data using xESMF"

    def __init__(self, target_resolution: float = 1.0, method: str = "bilinear"):
        super().__init__()
        self.target_resolution = target_resolution  # degrees
        self.method = method  # "bilinear", "conservative", "nearest_s2d"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.domain == "climate" and ctx.metadata.get("file_format") in ["netcdf", "zarr"]

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import xarray as xr
            import xesmf as xe
            import numpy as np

            if not hasattr(ctx.data, 'dims'):  # Not an xarray object
                return ctx

            ds = ctx.data

            logger.debug(f"     Target resolution: {self.target_resolution}°")
            logger.debug(f"     Interpolation method: {self.method}")
            logger.debug(f"     Input dimensions: {dict(ds.dims)}")

            # Create target grid
            lat_out = np.arange(-90, 90 + self.target_resolution, self.target_resolution)
            lon_out = np.arange(0, 360, self.target_resolution)
            ds_out = xr.Dataset({
                'lat': (['lat'], lat_out),
                'lon': (['lon'], lon_out),
            })

            logger.debug(f"     Output grid: {len(lat_out)} lat x {len(lon_out)} lon points")

            # Build regridder
            logger.debug(f"     Building regridder...")
            regridder = xe.Regridder(ds, ds_out, self.method)
            ctx.data = regridder(ds)

            logger.debug(f"     Output dimensions: {dict(ctx.data.dims)}")

            ctx.metadata["regridded"] = {
                "target_resolution": self.target_resolution,
                "method": self.method
            }
            logger.info(f"Regridded to {self.target_resolution}° resolution")
        except ImportError:
            ctx.add_error(missing_dependency_error("xesmf", "climate"))
        except Exception as e:
            ctx.add_error(f"Error regridding: {str(e)}")
        return ctx


class NormalizeArray(PipelineStep):
    """Normalize numpy arrays (min-max or z-score)"""

    name = "normalize_array"
    stage = PipelineStage.PREPROCESS
    description = "Normalize array values"

    def __init__(self, method: str = "minmax", axis: Optional[int] = None):
        super().__init__()
        self.method = method  # "minmax" or "zscore"
        self.axis = axis  # Axis for normalization, None for global

    def validate(self, ctx: PipelineContext) -> bool:
        import numpy as np
        return isinstance(ctx.data, np.ndarray) or (
            isinstance(ctx.data, dict) and "X" in ctx.data
        )

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            # Handle dict format (NPZ with X key) or raw array
            if isinstance(ctx.data, dict):
                arr = ctx.data["X"]
                is_dict = True
            else:
                arr = ctx.data
                is_dict = False

            logger.debug(f"     Normalization method: {self.method}")
            logger.debug(f"     Input shape: {arr.shape}")
            logger.debug(f"     Input dtype: {arr.dtype}")
            logger.debug(f"     Input range: [{arr.min():.4f}, {arr.max():.4f}]")

            if self.method == "minmax":
                arr_min = arr.min(axis=self.axis, keepdims=True)
                arr_max = arr.max(axis=self.axis, keepdims=True)
                normalized = (arr - arr_min) / (arr_max - arr_min + 1e-8)
                ctx.metadata["normalization"] = {
                    "method": "minmax",
                    "min": float(arr_min.min()),
                    "max": float(arr_max.max())
                }
                logger.debug(f"     Applied min-max normalization")
            elif self.method == "zscore":
                mean = arr.mean(axis=self.axis, keepdims=True)
                std = arr.std(axis=self.axis, keepdims=True)
                normalized = (arr - mean) / (std + 1e-8)
                ctx.metadata["normalization"] = {
                    "method": "zscore",
                    "mean": float(mean.mean()),
                    "std": float(std.mean())
                }
                logger.debug(f"     Applied z-score normalization")
            else:
                ctx.add_error(f"Unknown normalization method: {self.method}")
                return ctx

            # Convert to float32 for ML compatibility
            normalized = normalized.astype(np.float32)

            logger.debug(f"     Output range: [{normalized.min():.4f}, {normalized.max():.4f}]")
            logger.debug(f"     Output dtype: {normalized.dtype}")

            # Update data
            if is_dict:
                ctx.data["X"] = normalized
            else:
                ctx.data = normalized

            logger.info(f"Normalized array using {self.method}")

        except Exception as e:
            ctx.add_error(f"Error normalizing array: {str(e)}")
        return ctx


class ComputeArrayStats(PipelineStep):
    """Compute statistics for array data"""

    name = "compute_array_stats"
    stage = PipelineStage.PREPROCESS
    description = "Compute statistics for array data"

    def validate(self, ctx: PipelineContext) -> bool:
        import numpy as np
        return isinstance(ctx.data, np.ndarray) or isinstance(ctx.data, dict)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            # Handle dict format or raw array
            if isinstance(ctx.data, dict):
                if "X" in ctx.data:
                    arr = ctx.data["X"]
                else:
                    # Use first array in dict
                    first_key = list(ctx.data.keys())[0]
                    arr = ctx.data[first_key]
            else:
                arr = ctx.data

            logger.debug(f"     Computing statistics for array shape: {arr.shape}")

            stats = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "sparsity": float((arr == 0).sum() / arr.size),
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "ndim": int(arr.ndim),
                "size": int(arr.size),
                "memory_mb": float(arr.nbytes / (1024 * 1024))
            }

            # Add percentiles
            if arr.size < 10_000_000:  # Only compute for manageable arrays
                stats["percentiles"] = {
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99))
                }

            ctx.metadata["array_stats"] = stats

            logger.debug(f"     Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            logger.debug(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            logger.debug(f"     Sparsity: {stats['sparsity']:.2%}")
            logger.debug(f"     Memory: {stats['memory_mb']:.2f} MB")

            logger.info(f"Computed array stats: shape {stats['shape']}, range [{stats['min']:.2f}, {stats['max']:.2f}]")

        except Exception as e:
            ctx.add_error(f"Error computing array stats: {str(e)}")
        return ctx


# Convenience function to get standard preprocess steps
def get_preprocess_steps(
    domain: str = "general",
    missing_strategy: str = "drop",
    handle_outliers: bool = True
) -> List[PipelineStep]:
    """Get the standard preprocess pipeline steps"""
    steps = [
        HandleMissingValues(strategy=missing_strategy),
        RemoveDuplicates(),
        ValidateDataTypes(),
    ]

    if handle_outliers:
        steps.append(HandleOutliers())

    # Domain-specific steps
    if domain == "climate":
        steps.append(RegridClimate())
    elif domain == "life_sciences":
        steps.append(AnonymizePII())

    return steps


def preprocess(ctx: PipelineContext, **kwargs) -> PipelineContext:
    """Run the preprocess stage standalone"""
    from . import Pipeline

    pipeline = Pipeline(ctx.domain)
    for step in get_preprocess_steps(ctx.domain, **kwargs):
        pipeline.add_step(step)

    return pipeline.execute_stage(ctx, PipelineStage.PREPROCESS)
