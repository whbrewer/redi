"""
DRAI HPC/Operational Domain Module

Domain-specific preprocessing for HPC job accounting,
system logs, and operational data.

Tools: pandas, scikit-learn, polars
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ["pandas", "scikit-learn", "polars"]

# Common HPC job fields
HPC_FIELDS = {
    "identifiers": ["jobid", "job_id", "job", "user", "project", "account", "allocation"],
    "resources": ["nodes", "cores", "cpus", "gpus", "memory", "mem"],
    "timing": ["walltime", "runtime", "elapsed", "wait_time", "queue_time", "submit", "start", "end"],
    "status": ["exit_code", "exitcode", "status", "state"],
    "queue": ["queue", "partition", "qos"],
    "system": ["system", "machine", "cluster", "host"],
}

# Categorical fields that need encoding
CATEGORICAL_FIELDS = ["user", "project", "queue", "partition", "status", "state", "system", "machine"]


def get_ingest_steps():
    """Get HPC-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.ingest import DetectFormat

    class LoadHPCData(PipelineStep):
        """Load HPC job accounting data"""

        name = "load_hpc"
        stage = PipelineStage.INGEST
        description = "Load HPC job accounting CSV"

        def validate(self, ctx: PipelineContext) -> bool:
            # Domain check is optional since it may not be set during ingest
            fmt_ok = ctx.metadata.get("file_format") in ["csv", "parquet"]
            domain_ok = ctx.domain in ["hpc", "general"] or not ctx.domain
            return fmt_ok and domain_ok

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]
                file_format = ctx.metadata.get("file_format")

                if len(paths) == 1:
                    # Single file
                    if file_format == "parquet":
                        df = pd.read_parquet(paths[0])
                    else:
                        df = pd.read_csv(paths[0])
                else:
                    # Multiple files - concatenate
                    dfs = []
                    for i, path in enumerate(paths):
                        logger.debug(f"Loading file {i+1}/{len(paths)}: {path.name}")
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

                for field_type, keywords in HPC_FIELDS.items():
                    for keyword in keywords:
                        for col_lower, col in cols_lower.items():
                            if keyword in col_lower:
                                identified.setdefault(field_type, []).append(col)

                ctx.metadata["identified_fields"] = identified
                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)

                # Detect potential target columns (runtime prediction)
                runtime_cols = identified.get("timing", [])
                if runtime_cols:
                    ctx.metadata["target_candidates"] = runtime_cols

                logger.info(f"Loaded HPC data: {len(df)} jobs, {len(identified)} field types identified")

            except Exception as e:
                ctx.add_error(f"Error loading HPC data: {str(e)}")

            return ctx

    return [DetectFormat(), LoadHPCData()]


def get_preprocess_steps():
    """Get HPC-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CleanHPCData(PipelineStep):
        """Clean HPC job data"""

        name = "clean_hpc"
        stage = PipelineStage.PREPROCESS
        description = "Clean HPC job records"

        def __init__(self, remove_failed: bool = False):
            super().__init__()
            self.remove_failed = remove_failed

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                initial_count = len(df)

                # Remove jobs with zero/negative runtime
                timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                for col in timing_cols:
                    if col in df.columns:
                        df = df[df[col] > 0]

                # Optionally remove failed jobs
                if self.remove_failed:
                    status_cols = ctx.metadata.get("identified_fields", {}).get("status", [])
                    for col in status_cols:
                        if col in df.columns:
                            # Keep exit_code == 0
                            if "exit" in col.lower():
                                df = df[df[col] == 0]

                # Remove duplicates
                df = df.drop_duplicates()

                ctx.data = df
                ctx.metadata["rows_removed"] = initial_count - len(df)
                logger.info(f"Cleaned HPC data: {initial_count} -> {len(df)} jobs")

            except Exception as e:
                ctx.add_error(f"Error cleaning HPC data: {str(e)}")

            return ctx

    class ParseTimestamps(PipelineStep):
        """Parse timestamp columns"""

        name = "parse_timestamps"
        stage = PipelineStage.PREPROCESS
        description = "Parse and standardize timestamps"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                timestamp_keywords = ["submit", "start", "end", "time", "date"]

                parsed = []
                for col in df.columns:
                    if any(kw in col.lower() for kw in timestamp_keywords):
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                parsed.append(col)
                            except Exception:
                                pass

                ctx.data = df
                ctx.metadata["parsed_timestamps"] = parsed
                logger.info(f"Parsed {len(parsed)} timestamp columns")

            except Exception as e:
                ctx.add_error(f"Error parsing timestamps: {str(e)}")

            return ctx

    return [CleanHPCData(), ParseTimestamps()]


def get_transform_steps():
    """Get HPC-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class EngineerHPCFeatures(PipelineStep):
        """Engineer HPC-specific features"""

        name = "engineer_hpc_features"
        stage = PipelineStage.TRANSFORM
        description = "Create derived HPC features"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                created = []
                cols_lower = {c.lower(): c for c in df.columns}

                # Compute wait time if submit and start exist
                if "submit" in cols_lower and "start" in cols_lower:
                    submit_col = cols_lower["submit"]
                    start_col = cols_lower["start"]
                    if pd.api.types.is_datetime64_any_dtype(df[submit_col]):
                        df["wait_time_seconds"] = (df[start_col] - df[submit_col]).dt.total_seconds()
                        created.append("wait_time_seconds")

                # Compute core-hours
                timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                resource_cols = ctx.metadata.get("identified_fields", {}).get("resources", [])

                for time_col in timing_cols:
                    if "runtime" in time_col.lower() or "elapsed" in time_col.lower():
                        for res_col in resource_cols:
                            if "core" in res_col.lower() or "cpu" in res_col.lower():
                                # Runtime in seconds, cores -> core-hours
                                df["core_hours"] = df[time_col] * df[res_col] / 3600
                                created.append("core_hours")
                                break
                        break

                # Node utilization (if nodes requested vs allocated)
                if "nodes" in cols_lower:
                    nodes_col = cols_lower["nodes"]
                    if "cores" in cols_lower:
                        cores_col = cols_lower["cores"]
                        # Approximate cores per node
                        df["cores_per_node"] = df[cores_col] / df[nodes_col].replace(0, 1)
                        created.append("cores_per_node")

                # Time features from submit time
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[f"{col}_hour"] = df[col].dt.hour
                        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                        df[f"{col}_month"] = df[col].dt.month
                        created.extend([f"{col}_hour", f"{col}_dayofweek", f"{col}_month"])

                ctx.data = df
                ctx.metadata["engineered_features"] = created
                logger.info(f"Engineered {len(created)} HPC features")

            except Exception as e:
                ctx.add_error(f"Error engineering features: {str(e)}")

            return ctx

    class EncodeHPCCategoricals(PipelineStep):
        """Encode categorical HPC fields"""

        name = "encode_hpc_categoricals"
        stage = PipelineStage.TRANSFORM
        description = "Encode categorical fields (user, queue, etc.)"

        def __init__(self, method: str = "label", max_categories: int = 100):
            super().__init__()
            self.method = method
            self.max_categories = max_categories

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                encoded_cols = []

                for col in df.columns:
                    # Check if it's a categorical column
                    if df[col].dtype == 'object':
                        n_unique = df[col].nunique()

                        if n_unique > self.max_categories:
                            # High cardinality - use frequency encoding
                            freq = df[col].value_counts(normalize=True)
                            df[f"{col}_freq"] = df[col].map(freq)
                            df = df.drop(columns=[col])
                            encoded_cols.append(f"{col}_freq (frequency)")
                        elif self.method == "label":
                            df[col], _ = pd.factorize(df[col])
                            encoded_cols.append(f"{col} (label)")
                        elif self.method == "onehot" and n_unique <= 20:
                            dummies = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                            encoded_cols.append(f"{col} (onehot)")

                ctx.data = df
                ctx.metadata["encoded_categoricals"] = encoded_cols
                logger.info(f"Encoded {len(encoded_cols)} categorical columns")

            except Exception as e:
                ctx.add_error(f"Error encoding categoricals: {str(e)}")

            return ctx

    class NormalizeHPCFeatures(PipelineStep):
        """Normalize numerical features"""

        name = "normalize_hpc"
        stage = PipelineStage.TRANSFORM
        description = "Normalize HPC numerical features"

        def __init__(self, method: str = "zscore"):
            super().__init__()
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Exclude datetime-derived features from normalization
                exclude_patterns = ["_hour", "_dayofweek", "_month", "exit"]
                feature_cols = [c for c in numeric_cols
                               if not any(p in c.lower() for p in exclude_patterns)]

                stats = {}
                for col in feature_cols:
                    if self.method == "zscore":
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[col] = (df[col] - mean) / std
                        stats[col] = {"mean": float(mean), "std": float(std)}
                    elif self.method == "log":
                        # Log transform for skewed distributions
                        if df[col].min() >= 0:
                            df[col] = np.log1p(df[col])
                            stats[col] = {"transform": "log1p"}

                ctx.data = df
                ctx.metadata["normalization_stats"] = stats
                logger.info(f"Normalized {len(stats)} features")

            except Exception as e:
                ctx.add_error(f"Error normalizing: {str(e)}")

            return ctx

    return [EngineerHPCFeatures(), EncodeHPCCategoricals(), NormalizeHPCFeatures()]


def get_structure_steps():
    """Get HPC-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class PrepareForML(PipelineStep):
        """Prepare data for ML model training"""

        name = "prepare_hpc_ml"
        stage = PipelineStage.STRUCTURE
        description = "Prepare HPC data for ML training"

        def __init__(self, target_column: Optional[str] = None):
            super().__init__()
            self.target_column = target_column

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data

                # Determine target column
                target = self.target_column
                if not target:
                    # Default to first timing column (runtime prediction)
                    timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                    for col in timing_cols:
                        if "runtime" in col.lower() or "elapsed" in col.lower():
                            target = col
                            break

                # Get numeric feature columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Exclude non-feature columns
                exclude = ["jobid", "job_id", target] if target else ["jobid", "job_id"]
                feature_cols = [c for c in numeric_cols if c.lower() not in [e.lower() for e in exclude]]

                # Remove any columns with all NaN
                feature_cols = [c for c in feature_cols if df[c].notna().any()]

                # Fill remaining NaN with median
                for col in feature_cols:
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].median())

                X = df[feature_cols].values.astype(np.float32)

                result = {"X": X, "feature_names": feature_cols}

                if target and target in df.columns:
                    y = df[target].values.astype(np.float32)
                    result["y"] = y
                    ctx.metadata["target_column"] = target

                ctx.data = result
                ctx.metadata["array_shape"] = X.shape
                ctx.metadata["feature_columns"] = feature_cols
                logger.info(f"Prepared ML data: X shape {X.shape}, target={target}")

            except Exception as e:
                ctx.add_error(f"Error preparing for ML: {str(e)}")

            return ctx

    return [PrepareForML()]


def get_output_steps():
    """Get HPC-specific output steps"""
    from ..pipeline.output import get_output_steps
    return get_output_steps(output_format="npz")


def get_raps_pipeline(column_overrides=None, power_config=None, output_format="npz",
                      system_name="unknown"):
    """Get the RAPS Job extraction pipeline.

    Args:
        column_overrides: Dict mapping original_column_name → canonical_field_name
        power_config: Dict like {"cpu": {"idle": 100, "max": 500}} for power→utilization
        output_format: Output format — npz (default), json, parquet, csv
        system_name: RAPS system name stored in the NPZ snapshot (e.g. "adastraMI250")
    """
    from .hpc_job_mapper import AutoMapColumns, DeriveJobFields, BuildJobDicts, SaveJobDicts
    from ..pipeline.ingest import DetectFormat

    # Reuse existing ingest/preprocess, then switch to RAPS-specific steps
    ingest = get_ingest_steps()                          # DetectFormat + LoadHPCData
    preprocess = get_preprocess_steps()                  # CleanHPCData + ParseTimestamps
    raps_steps = [
        AutoMapColumns(overrides=column_overrides),      # map columns → canonical fields
        DeriveJobFields(),                               # derive missing fields
        BuildJobDicts(power_config=power_config),        # build job dicts per row
        SaveJobDicts(output_format=output_format, system_name=system_name),
    ]
    return ingest + preprocess + raps_steps


def get_full_pipeline(raps_jobs=False, **kwargs):
    """Get the complete HPC preprocessing pipeline.

    Args:
        raps_jobs: If True, return the RAPS Job extraction pipeline instead of
                   the default ML-prep pipeline.
        **kwargs: Passed to get_raps_pipeline() when raps_jobs=True
                  (column_overrides, power_config, output_format).
    """
    if raps_jobs:
        return get_raps_pipeline(**kwargs)
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
