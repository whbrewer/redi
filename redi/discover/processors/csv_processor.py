"""
redi/discover/processors/csv_processor.py — CSV/TSV file analysis.
"""

from __future__ import annotations

from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery, FileParseError, PreprocessingStep

# Ratio of unique values to total rows below which a column is considered
# categorical rather than free-form text.
_CATEGORICAL_RATIO = 0.05
_CATEGORICAL_MAX_UNIQUE = 20
# Maximum rows loaded for analysis (avoids memory issues on huge files).
_MAX_ROWS = 100_000


class CsvProcessor(BaseProcessor):
    extension = ".csv"

    def analyze(self, path: Path) -> FileDiscovery:
        import pandas as pd

        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep, nrows=_MAX_ROWS, low_memory=False)
        except Exception as exc:
            raise FileParseError(
                path=path,
                file_type="CSV",
                reason=str(exc),
                cause=exc,
            ) from exc

        n_rows, n_cols = df.shape
        column_types: dict[str, str] = {}
        column_stats: dict[str, dict] = {}
        steps: list[PreprocessingStep] = []

        for col in df.columns:
            col_type = self._classify_column(df[col], n_rows)
            column_types[col] = col_type
            column_stats[col] = self._compute_stats(df[col], col_type, n_rows)
            steps.extend(self._suggest_steps(col, col_type, column_stats[col]))

        metadata = {
            "shape": [n_rows, n_cols],
            "column_titles": list(df.columns),
            "column_types": column_types,
            "column_stats": column_stats,
        }

        return FileDiscovery(
            file_path=str(path),
            file_type="CSV",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _classify_column(self, series, n_rows: int) -> str:
        import pandas as pd

        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        # Try datetime parse on a sample
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(50)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample)
                    return "datetime"
                except (ValueError, TypeError):
                    pass
            n_unique = series.nunique(dropna=True)
            if n_unique <= _CATEGORICAL_MAX_UNIQUE or (n_rows > 0 and n_unique / n_rows < _CATEGORICAL_RATIO):
                return "categorical"
            return "text"
        return "unknown"

    def _compute_stats(self, series, col_type: str, n_rows: int) -> dict:
        pct_missing = round(series.isna().sum() / n_rows * 100, 2) if n_rows > 0 else 0.0

        if col_type == "numeric":
            desc = series.describe()
            return {
                "mean": self._safe_float(desc.get("mean")),
                "std": self._safe_float(desc.get("std")),
                "min": self._safe_float(desc.get("min")),
                "max": self._safe_float(desc.get("max")),
                "pct_missing": pct_missing,
            }

        if col_type == "categorical":
            top = series.value_counts(dropna=True).head(10).to_dict()
            return {
                "n_unique": int(series.nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in top.items()},
                "pct_missing": pct_missing,
            }

        if col_type == "boolean":
            vc = series.value_counts(dropna=True).to_dict()
            return {"value_counts": {str(k): int(v) for k, v in vc.items()}, "pct_missing": pct_missing}

        if col_type == "datetime":
            return {"pct_missing": pct_missing}

        # text / unknown
        return {"pct_missing": pct_missing}

    def _safe_float(self, val) -> float | None:
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return None

    def _suggest_steps(self, col: str, col_type: str, stats: dict) -> list[PreprocessingStep]:
        steps: list[PreprocessingStep] = []
        pct_missing = stats.get("pct_missing", 0.0)

        if col_type == "numeric":
            mn = stats.get("min")
            mx = stats.get("max")
            std = stats.get("std")
            mean = stats.get("mean")

            # Choose normalization method
            if mn is not None and mx is not None and mx > mn:
                value_range = mx - mn
                if std is not None and mean is not None and value_range > 0:
                    cv = std / abs(mean) if mean != 0 else float("inf")
                    method = "z-score standardization" if cv > 1.0 else "min-max scaling to [0, 1]"
                else:
                    method = "min-max scaling to [0, 1]"
            else:
                method = "z-score standardization"

            steps.append(PreprocessingStep(
                name=f"Normalize `{col}`",
                rationale=f"Numeric column; range [{mn}, {mx}]",
                suggested_method=method,
            ))

            if pct_missing > 0:
                impute_method = "median imputation" if (std or 0) > 0 else "mean imputation"
                steps.append(PreprocessingStep(
                    name=f"Impute missing values in `{col}`",
                    rationale=f"{pct_missing}% of values are missing",
                    suggested_method=impute_method,
                ))

        elif col_type == "categorical":
            n_unique = stats.get("n_unique", 0)
            if n_unique <= 2:
                method = "binary (0/1) encoding"
            elif n_unique <= 10:
                method = "one-hot encoding"
            else:
                method = "label encoding or target encoding"

            steps.append(PreprocessingStep(
                name=f"Encode `{col}`",
                rationale=f"Categorical column with {n_unique} unique values",
                suggested_method=method,
            ))

            if n_unique > 5:
                steps.append(PreprocessingStep(
                    name=f"Consider `{col}` as shard key",
                    rationale=f"{n_unique} categories — sharding by this column preserves class distribution",
                    suggested_method="stratified sharding",
                ))

            if pct_missing > 0:
                steps.append(PreprocessingStep(
                    name=f"Impute missing values in `{col}`",
                    rationale=f"{pct_missing}% of values are missing",
                    suggested_method="fill with most-frequent value or dedicated 'unknown' category",
                ))

        elif col_type == "boolean":
            steps.append(PreprocessingStep(
                name=f"Encode `{col}`",
                rationale="Boolean column",
                suggested_method="binary (0/1) encoding",
            ))

        elif col_type == "datetime":
            steps.append(PreprocessingStep(
                name=f"Extract features from `{col}`",
                rationale="Datetime column — raw timestamps are rarely useful directly",
                suggested_method="extract year, month, day-of-week, hour; optionally sort by this column",
            ))

        return steps
