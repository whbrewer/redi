"""
DRAI Pipeline - Transform Stage

Handles feature engineering, normalization, and encoding.
Domain-specific transformations are delegated to domain modules.

Operations:
- Feature engineering
- Normalization/standardization
- Categorical encoding
- Domain-specific feature extraction
"""

import logging
from typing import Any, Dict, List, Optional, Union
from . import PipelineStep, PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


class NormalizeNumeric(PipelineStep):
    """Normalize numerical columns"""

    name = "normalize_numeric"
    stage = PipelineStage.TRANSFORM
    description = "Normalize numerical columns (z-score or min-max)"

    def __init__(
        self,
        method: str = "zscore",
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__()
        self.method = method  # "zscore", "minmax", "robust"
        self.columns = columns  # None = all numeric columns
        self.exclude = list(exclude or [])  # never normalized (e.g. the target)

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Select columns
            if self.columns:
                cols = [c for c in self.columns if c in ctx.data.columns]
                logger.debug(f"     Using specified columns: {len(cols)} columns")
            else:
                cols = ctx.data.select_dtypes(include=['number']).columns.tolist()
                logger.debug(f"     Auto-selected {len(cols)} numeric columns")
            cols = [c for c in cols if c not in self.exclude]

            logger.debug(f"     Normalization method: {self.method}")

            stats = {}
            for col in cols:
                if self.method == "zscore":
                    mean = ctx.data[col].mean()
                    std = ctx.data[col].std()
                    if std > 0:
                        ctx.data[col] = (ctx.data[col] - mean) / std
                    stats[col] = {"mean": float(mean), "std": float(std)}
                    logger.debug(f"     {col}: mean={mean:.4f}, std={std:.4f}")

                elif self.method == "minmax":
                    min_val = ctx.data[col].min()
                    max_val = ctx.data[col].max()
                    if max_val > min_val:
                        ctx.data[col] = (ctx.data[col] - min_val) / (max_val - min_val)
                    stats[col] = {"min": float(min_val), "max": float(max_val)}
                    logger.debug(f"     {col}: min={min_val:.4f}, max={max_val:.4f}")

                elif self.method == "robust":
                    median = ctx.data[col].median()
                    q1 = ctx.data[col].quantile(0.25)
                    q3 = ctx.data[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        ctx.data[col] = (ctx.data[col] - median) / iqr
                    stats[col] = {"median": float(median), "iqr": float(iqr)}
                    logger.debug(f"     {col}: median={median:.4f}, iqr={iqr:.4f}")

            ctx.metadata["normalization"] = {
                "method": self.method,
                "columns": cols,
                "stats": stats
            }
            logger.info(f"Normalized {len(cols)} columns using {self.method}")

        except Exception as e:
            ctx.add_error(f"Error normalizing: {str(e)}")
        return ctx


class EncodeCategorical(PipelineStep):
    """Encode categorical columns"""

    name = "encode_categorical"
    stage = PipelineStage.TRANSFORM
    description = "Encode categorical columns (one-hot, label, or target)"

    def __init__(
        self,
        method: str = "onehot",
        columns: Optional[List[str]] = None,
        max_categories: int = 50,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__()
        self.method = method  # "onehot", "label", "frequency"
        self.columns = columns
        self.max_categories = max_categories
        self.exclude = list(exclude or [])  # never encoded (e.g. the target)

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Select categorical columns
            if self.columns:
                cols = [c for c in self.columns if c in ctx.data.columns]
                logger.debug(f"     Using specified columns: {len(cols)} columns")
            else:
                cols = ctx.data.select_dtypes(include=['object', 'category']).columns.tolist()
                logger.debug(f"     Auto-selected {len(cols)} categorical columns")
            cols = [c for c in cols if c not in self.exclude]

            logger.debug(f"     Encoding method: {self.method}")
            logger.debug(f"     Max categories for one-hot: {self.max_categories}")

            encoded_info = {}
            for col in cols:
                n_unique = ctx.data[col].nunique()
                logger.debug(f"     {col}: {n_unique} unique values")

                # Skip high-cardinality columns for one-hot
                if self.method == "onehot" and n_unique > self.max_categories:
                    logger.warning(f"Skipping {col}: {n_unique} categories > {self.max_categories}")
                    continue

                if self.method == "onehot":
                    dummies = pd.get_dummies(ctx.data[col], prefix=col, dummy_na=True)
                    ctx.data = pd.concat([ctx.data.drop(columns=[col]), dummies], axis=1)
                    encoded_info[col] = {"method": "onehot", "categories": n_unique}
                    logger.debug(f"     {col}: one-hot encoded -> {len(dummies.columns)} columns")

                elif self.method == "label":
                    ctx.data[col], uniques = pd.factorize(ctx.data[col])
                    encoded_info[col] = {"method": "label", "mapping": dict(enumerate(uniques))}
                    logger.debug(f"     {col}: label encoded (0 to {len(uniques)-1})")

                elif self.method == "frequency":
                    freq_map = ctx.data[col].value_counts(normalize=True).to_dict()
                    ctx.data[col] = ctx.data[col].map(freq_map)
                    encoded_info[col] = {"method": "frequency"}
                    logger.debug(f"     {col}: frequency encoded")

            ctx.metadata["categorical_encoding"] = encoded_info
            logger.info(f"Encoded {len(encoded_info)} categorical columns")

        except Exception as e:
            ctx.add_error(f"Error encoding categoricals: {str(e)}")
        return ctx


class EngineerFeatures(PipelineStep):
    """Generic feature engineering"""

    name = "engineer_features"
    stage = PipelineStage.TRANSFORM
    description = "Create derived features from existing columns"

    def __init__(self, features: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        # Features format: [{"name": "col1_col2_ratio", "expr": "col1 / col2"}, ...]
        self.features = features or []

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            logger.debug(f"     Feature definitions: {len(self.features)}")

            created = []
            for feat in self.features:
                name = feat.get("name")
                expr = feat.get("expr")
                if name and expr:
                    try:
                        ctx.data[name] = ctx.data.eval(expr)
                        created.append(name)
                        logger.debug(f"     Created '{name}' = {expr}")
                    except Exception as e:
                        logger.warning(f"Failed to create feature {name}: {e}")
                        logger.debug(f"     Failed '{name}' = {expr}: {e}")

            ctx.metadata["engineered_features"] = created
            logger.info(f"Created {len(created)} engineered features")

        except Exception as e:
            ctx.add_error(f"Error engineering features: {str(e)}")
        return ctx


class AutoFeatureEngineering(PipelineStep):
    """Automatic feature engineering based on column types"""

    name = "auto_feature_engineering"
    stage = PipelineStage.TRANSFORM
    description = "Automatically create derived features"

    def __init__(self, create_interactions: bool = False, create_datetime_features: bool = True):
        super().__init__()
        self.create_interactions = create_interactions
        self.create_datetime_features = create_datetime_features

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            created = []
            logger.debug(f"     Create datetime features: {self.create_datetime_features}")
            logger.debug(f"     Create interactions: {self.create_interactions}")

            # Datetime features
            if self.create_datetime_features:
                datetime_cols = ctx.data.select_dtypes(include=['datetime64']).columns
                logger.debug(f"     Found {len(datetime_cols)} datetime columns")
                for col in datetime_cols:
                    ctx.data[f"{col}_year"] = ctx.data[col].dt.year
                    ctx.data[f"{col}_month"] = ctx.data[col].dt.month
                    ctx.data[f"{col}_day"] = ctx.data[col].dt.day
                    ctx.data[f"{col}_dayofweek"] = ctx.data[col].dt.dayofweek
                    ctx.data[f"{col}_hour"] = ctx.data[col].dt.hour
                    new_features = [
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_hour"
                    ]
                    created.extend(new_features)
                    logger.debug(f"     {col}: created {len(new_features)} datetime features")

            # Log transforms for skewed numeric columns
            numeric_cols = ctx.data.select_dtypes(include=['number']).columns
            skewed_cols = []
            for col in numeric_cols:
                if col.endswith('_log'):
                    continue
                # Check skewness
                skewness = ctx.data[col].skew()
                if skewness > 1.0:
                    min_val = ctx.data[col].min()
                    if min_val >= 0:
                        ctx.data[f"{col}_log"] = np.log1p(ctx.data[col])
                        created.append(f"{col}_log")
                        skewed_cols.append((col, skewness))

            if skewed_cols:
                logger.debug(f"     Applied log transform to {len(skewed_cols)} skewed columns:")
                for col, skew in skewed_cols[:5]:
                    logger.debug(f"       - {col} (skewness: {skew:.2f})")
                if len(skewed_cols) > 5:
                    logger.debug(f"       ... and {len(skewed_cols) - 5} more")

            ctx.metadata["auto_engineered_features"] = created
            logger.info(f"Auto-created {len(created)} features")

        except Exception as e:
            ctx.add_error(f"Error in auto feature engineering: {str(e)}")
        return ctx


class SelectFeatures(PipelineStep):
    """Feature selection based on variance or correlation"""

    name = "select_features"
    stage = PipelineStage.TRANSFORM
    description = "Select features based on variance or correlation"

    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.01,
        target_column: Optional[str] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__()
        self.method = method  # "variance", "correlation", "mutual_info"
        self.threshold = threshold
        self.target_column = target_column
        self.exclude = list(exclude or [])  # never dropped (e.g. the target)

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            numeric_cols = [c for c in ctx.data.select_dtypes(include=['number']).columns.tolist()
                            if c not in self.exclude]
            removed = []

            logger.debug(f"     Selection method: {self.method}")
            logger.debug(f"     Threshold: {self.threshold}")
            logger.debug(f"     Numeric columns to evaluate: {len(numeric_cols)}")

            if self.method == "variance":
                # Remove low-variance features
                logger.debug(f"     Checking variance of each column...")
                for col in numeric_cols:
                    variance = ctx.data[col].var()
                    if variance < self.threshold:
                        ctx.data = ctx.data.drop(columns=[col])
                        removed.append(col)
                        logger.debug(f"     Removed '{col}' (variance: {variance:.6f})")

            elif self.method == "correlation" and self.target_column:
                # Select features correlated with target
                logger.debug(f"     Target column: {self.target_column}")
                if self.target_column in ctx.data.columns:
                    corr = ctx.data[numeric_cols].corrwith(ctx.data[self.target_column]).abs()
                    low_corr = corr[corr < self.threshold].index.tolist()
                    removed = [c for c in low_corr if c != self.target_column]
                    ctx.data = ctx.data.drop(columns=removed)
                    for col in removed[:5]:
                        logger.debug(f"     Removed '{col}' (correlation: {corr[col]:.4f})")
                    if len(removed) > 5:
                        logger.debug(f"     ... and {len(removed) - 5} more columns")

            ctx.metadata["feature_selection"] = {
                "method": self.method,
                "threshold": self.threshold,
                "removed": removed,
                "remaining": len(ctx.data.columns)
            }
            logger.debug(f"     Columns remaining: {len(ctx.data.columns)}")
            logger.info(f"Removed {len(removed)} features, {len(ctx.data.columns)} remaining")

        except Exception as e:
            ctx.add_error(f"Error in feature selection: {str(e)}")
        return ctx


class EncodeTarget(PipelineStep):
    """Label-encode a non-numeric target column so it can be saved as a numeric array.

    A numeric target is left untouched. Codes are assigned in sorted label order, so the
    mapping is deterministic, and it is recorded in ``metadata["target_encoding"]``.
    """

    name = "encode_target"
    stage = PipelineStage.TRANSFORM
    description = "Label-encode a non-numeric target column"

    def __init__(self, target_column: str):
        super().__init__()
        self.target_column = target_column

    def validate(self, ctx: PipelineContext) -> bool:
        import pandas as pd
        return isinstance(ctx.data, pd.DataFrame)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            col = self.target_column
            if col not in ctx.data.columns:
                ctx.add_error(
                    f"Target column '{col}' not found in the data "
                    f"(columns: {', '.join(map(str, list(ctx.data.columns)[:10]))}...)"
                )
                return ctx

            series = ctx.data[col]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                ctx.metadata["target_encoding"] = {"column": col, "method": "none"}
                return ctx

            labels = sorted(series.dropna().unique().tolist(), key=str)
            codes = {label: i for i, label in enumerate(labels)}
            ctx.data[col] = series.map(codes)
            if ctx.data[col].isna().any():
                ctx.add_error(f"Target column '{col}' has missing values after preprocessing")
            else:
                ctx.data[col] = ctx.data[col].astype("int64")
            ctx.metadata["target_encoding"] = {
                "column": col,
                "method": "label",
                "mapping": {str(i): str(label) for label, i in codes.items()},
            }
            logger.info(f"Label-encoded target '{col}' ({len(labels)} classes)")
        except Exception as e:
            ctx.add_error(f"Error encoding target: {str(e)}")
        return ctx


# Variance below this means the column holds a single value (float noise aside).
CONSTANT_VARIANCE = 1e-12


# Convenience function to get standard transform steps
def get_transform_steps(
    domain: str = "general",
    normalize: bool = True,
    encode_categoricals: bool = True,
    auto_features: bool = False,
    target_column: Optional[str] = None,
    drop_constant: bool = False,
) -> List[PipelineStep]:
    """Get the standard transform pipeline steps.

    ``target_column`` keeps that column out of normalization, encoding and feature
    selection, and label-encodes it if it is not numeric. ``drop_constant`` removes
    numeric columns with a single value. Both default off, which leaves the steps
    exactly as they were.
    """
    steps = []
    protected = [target_column] if target_column else []

    if target_column:
        steps.append(EncodeTarget(target_column))

    if drop_constant:
        steps.append(SelectFeatures(method="variance", threshold=CONSTANT_VARIANCE, exclude=protected))

    if auto_features:
        steps.append(AutoFeatureEngineering())

    if normalize:
        steps.append(NormalizeNumeric(exclude=protected))

    if encode_categoricals:
        steps.append(EncodeCategorical(exclude=protected))

    return steps


def transform(ctx: PipelineContext, **kwargs) -> PipelineContext:
    """Run the transform stage standalone"""
    from . import Pipeline

    pipeline = Pipeline(ctx.domain)
    for step in get_transform_steps(ctx.domain, **kwargs):
        pipeline.add_step(step)

    return pipeline.execute_stage(ctx, PipelineStage.TRANSFORM)
