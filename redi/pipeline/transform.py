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

    def __init__(self, method: str = "zscore", columns: Optional[List[str]] = None):
        super().__init__()
        self.method = method  # "zscore", "minmax", "robust"
        self.columns = columns  # None = all numeric columns

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Select columns
            if self.columns:
                cols = [c for c in self.columns if c in ctx.data.columns]
            else:
                cols = ctx.data.select_dtypes(include=['number']).columns.tolist()

            stats = {}
            for col in cols:
                if self.method == "zscore":
                    mean = ctx.data[col].mean()
                    std = ctx.data[col].std()
                    if std > 0:
                        ctx.data[col] = (ctx.data[col] - mean) / std
                    stats[col] = {"mean": float(mean), "std": float(std)}

                elif self.method == "minmax":
                    min_val = ctx.data[col].min()
                    max_val = ctx.data[col].max()
                    if max_val > min_val:
                        ctx.data[col] = (ctx.data[col] - min_val) / (max_val - min_val)
                    stats[col] = {"min": float(min_val), "max": float(max_val)}

                elif self.method == "robust":
                    median = ctx.data[col].median()
                    q1 = ctx.data[col].quantile(0.25)
                    q3 = ctx.data[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        ctx.data[col] = (ctx.data[col] - median) / iqr
                    stats[col] = {"median": float(median), "iqr": float(iqr)}

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
        max_categories: int = 50
    ):
        super().__init__()
        self.method = method  # "onehot", "label", "frequency"
        self.columns = columns
        self.max_categories = max_categories

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Select categorical columns
            if self.columns:
                cols = [c for c in self.columns if c in ctx.data.columns]
            else:
                cols = ctx.data.select_dtypes(include=['object', 'category']).columns.tolist()

            encoded_info = {}
            for col in cols:
                n_unique = ctx.data[col].nunique()

                # Skip high-cardinality columns for one-hot
                if self.method == "onehot" and n_unique > self.max_categories:
                    logger.warning(f"Skipping {col}: {n_unique} categories > {self.max_categories}")
                    continue

                if self.method == "onehot":
                    dummies = pd.get_dummies(ctx.data[col], prefix=col, dummy_na=True)
                    ctx.data = pd.concat([ctx.data.drop(columns=[col]), dummies], axis=1)
                    encoded_info[col] = {"method": "onehot", "categories": n_unique}

                elif self.method == "label":
                    ctx.data[col], uniques = pd.factorize(ctx.data[col])
                    encoded_info[col] = {"method": "label", "mapping": dict(enumerate(uniques))}

                elif self.method == "frequency":
                    freq_map = ctx.data[col].value_counts(normalize=True).to_dict()
                    ctx.data[col] = ctx.data[col].map(freq_map)
                    encoded_info[col] = {"method": "frequency"}

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

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            created = []
            for feat in self.features:
                name = feat.get("name")
                expr = feat.get("expr")
                if name and expr:
                    try:
                        ctx.data[name] = ctx.data.eval(expr)
                        created.append(name)
                    except Exception as e:
                        logger.warning(f"Failed to create feature {name}: {e}")

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

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            created = []

            # Datetime features
            if self.create_datetime_features:
                datetime_cols = ctx.data.select_dtypes(include=['datetime64']).columns
                for col in datetime_cols:
                    ctx.data[f"{col}_year"] = ctx.data[col].dt.year
                    ctx.data[f"{col}_month"] = ctx.data[col].dt.month
                    ctx.data[f"{col}_day"] = ctx.data[col].dt.day
                    ctx.data[f"{col}_dayofweek"] = ctx.data[col].dt.dayofweek
                    ctx.data[f"{col}_hour"] = ctx.data[col].dt.hour
                    created.extend([
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_hour"
                    ])

            # Log transforms for skewed numeric columns
            numeric_cols = ctx.data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col.endswith('_log'):
                    continue
                # Check skewness
                if ctx.data[col].skew() > 1.0:
                    min_val = ctx.data[col].min()
                    if min_val >= 0:
                        ctx.data[f"{col}_log"] = np.log1p(ctx.data[col])
                        created.append(f"{col}_log")

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
        target_column: Optional[str] = None
    ):
        super().__init__()
        self.method = method  # "variance", "correlation", "mutual_info"
        self.threshold = threshold
        self.target_column = target_column

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            numeric_cols = ctx.data.select_dtypes(include=['number']).columns.tolist()
            removed = []

            if self.method == "variance":
                # Remove low-variance features
                for col in numeric_cols:
                    variance = ctx.data[col].var()
                    if variance < self.threshold:
                        ctx.data = ctx.data.drop(columns=[col])
                        removed.append(col)

            elif self.method == "correlation" and self.target_column:
                # Select features correlated with target
                if self.target_column in ctx.data.columns:
                    corr = ctx.data[numeric_cols].corrwith(ctx.data[self.target_column]).abs()
                    low_corr = corr[corr < self.threshold].index.tolist()
                    removed = [c for c in low_corr if c != self.target_column]
                    ctx.data = ctx.data.drop(columns=removed)

            ctx.metadata["feature_selection"] = {
                "method": self.method,
                "threshold": self.threshold,
                "removed": removed,
                "remaining": len(ctx.data.columns)
            }
            logger.info(f"Removed {len(removed)} features, {len(ctx.data.columns)} remaining")

        except Exception as e:
            ctx.add_error(f"Error in feature selection: {str(e)}")
        return ctx


# Convenience function to get standard transform steps
def get_transform_steps(
    domain: str = "general",
    normalize: bool = True,
    encode_categoricals: bool = True,
    auto_features: bool = False
) -> List[PipelineStep]:
    """Get the standard transform pipeline steps"""
    steps = []

    if auto_features:
        steps.append(AutoFeatureEngineering())

    if normalize:
        steps.append(NormalizeNumeric())

    if encode_categoricals:
        steps.append(EncodeCategorical())

    return steps


def transform(ctx: PipelineContext, **kwargs) -> PipelineContext:
    """Run the transform stage standalone"""
    from . import Pipeline

    pipeline = Pipeline(ctx.domain)
    for step in get_transform_steps(ctx.domain, **kwargs):
        pipeline.add_step(step)

    return pipeline.execute_stage(ctx, PipelineStage.TRANSFORM)
