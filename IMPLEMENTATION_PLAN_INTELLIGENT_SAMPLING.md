# Implementation Plan: Intelligent Sampling for REDI Pipeline

## Overview

Add intelligent sampling as a pipeline step to reduce large datasets while preserving important data characteristics for ML training. This is especially critical for HPC/turbulence data where 3D grids can be hundreds of GB.

## Pipeline Placement

```
INGEST → PREPROCESS → [SAMPLING] → TRANSFORM → STRUCTURE → OUTPUT
```

Sampling occurs AFTER preprocessing (data is clean) and BEFORE transform (avoid wasting compute on data we'll discard).

---

## Files to Create

### 1. `redi/pipeline/sampling.py` (NEW FILE)

```python
"""
Intelligent sampling strategies for large-scale scientific data.
"""

from abc import abstractmethod
from typing import Optional, List, Literal
import numpy as np
import pandas as pd

from .base import PipelineStep, PipelineContext


class SamplingStep(PipelineStep):
    """Base class for sampling steps."""
    stage = "sampling"


class IntelligentSample(SamplingStep):
    """
    Reduce dataset size using various sampling strategies.

    Strategies:
    - random: Uniform random sampling
    - stratified: Maintain distribution of key variables
    - gradient: Preserve high-gradient regions (physics-aware)
    - importance: Weight by feature variance/importance
    - systematic: Every nth sample (for regular grids)
    """

    def __init__(
        self,
        method: Literal["random", "stratified", "gradient", "importance", "systematic"] = "random",
        ratio: float = 0.1,
        seed: Optional[int] = 42,
        stratify_column: Optional[str] = None,
        gradient_variables: Optional[List[str]] = None,
        importance_threshold: float = 0.5,
        min_samples: int = 1000,
    ):
        self.method = method
        self.ratio = ratio
        self.seed = seed
        self.stratify_column = stratify_column
        self.gradient_variables = gradient_variables
        self.importance_threshold = importance_threshold
        self.min_samples = min_samples

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute sampling based on configured method."""
        df = context.data

        if df is None or len(df) == 0:
            return context

        n_samples = max(self.min_samples, int(len(df) * self.ratio))
        n_samples = min(n_samples, len(df))  # Can't sample more than we have

        # Store original size for metadata
        original_size = len(df)

        if self.method == "random":
            indices = self._random_sample(df, n_samples)
        elif self.method == "stratified":
            indices = self._stratified_sample(df, n_samples)
        elif self.method == "gradient":
            indices = self._gradient_sample(df, n_samples, context)
        elif self.method == "importance":
            indices = self._importance_sample(df, n_samples)
        elif self.method == "systematic":
            indices = self._systematic_sample(df, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

        context.data = df.iloc[indices].reset_index(drop=True)

        # Track in metadata
        context.metadata["sampling"] = {
            "method": self.method,
            "original_size": original_size,
            "sampled_size": len(context.data),
            "ratio_achieved": len(context.data) / original_size,
            "seed": self.seed,
        }

        return context

    def _random_sample(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """Uniform random sampling."""
        rng = np.random.default_rng(self.seed)
        return rng.choice(len(df), size=n, replace=False)

    def _stratified_sample(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """Stratified sampling maintaining class distribution."""
        if self.stratify_column is None or self.stratify_column not in df.columns:
            # Fall back to random if no valid stratify column
            return self._random_sample(df, n)

        from sklearn.model_selection import train_test_split

        # Use train_test_split with stratify to get proportional samples
        _, sample_idx = train_test_split(
            np.arange(len(df)),
            test_size=n / len(df),
            stratify=df[self.stratify_column],
            random_state=self.seed,
        )
        return sample_idx

    def _gradient_sample(
        self, df: pd.DataFrame, n: int, context: PipelineContext
    ) -> np.ndarray:
        """
        Gradient-based importance sampling for physics data.

        Preserves regions with high spatial gradients where interesting
        physics (shocks, turbulence, boundaries) occurs.
        """
        rng = np.random.default_rng(self.seed)

        # Determine which columns to compute gradients on
        if self.gradient_variables:
            grad_cols = [c for c in self.gradient_variables if c in df.columns]
        else:
            # Use all numeric columns
            grad_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not grad_cols:
            return self._random_sample(df, n)

        # Compute importance scores based on local variance/gradient magnitude
        importance = np.zeros(len(df))

        for col in grad_cols:
            values = df[col].values
            # Use rolling difference as gradient proxy for 1D
            grad = np.abs(np.diff(values, prepend=values[0]))
            # Normalize to [0, 1]
            if grad.max() > grad.min():
                grad = (grad - grad.min()) / (grad.max() - grad.min())
            importance += grad

        importance /= len(grad_cols)  # Average across variables

        # Add small uniform component to ensure coverage
        importance = 0.8 * importance + 0.2

        # Normalize to probability distribution
        probs = importance / importance.sum()

        # Sample according to importance
        return rng.choice(len(df), size=n, replace=False, p=probs)

    def _importance_sample(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """Sample based on feature variance (high variance = more important)."""
        rng = np.random.default_rng(self.seed)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return self._random_sample(df, n)

        # Compute per-row importance as sum of squared deviations from mean
        importance = np.zeros(len(df))
        for col in numeric_cols:
            values = df[col].values
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                importance += ((values - mean) / std) ** 2

        # Add uniform component
        importance = 0.7 * (importance / importance.max()) + 0.3
        probs = importance / importance.sum()

        return rng.choice(len(df), size=n, replace=False, p=probs)

    def _systematic_sample(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """Systematic sampling: every kth element."""
        step = max(1, len(df) // n)
        indices = np.arange(0, len(df), step)[:n]
        return indices


class GridSample3D(SamplingStep):
    """
    Specialized sampling for 3D grid data (HPC/turbulence).

    Works on flattened 3D arrays and preserves spatial structure.
    """

    def __init__(
        self,
        method: Literal["uniform", "gradient", "adaptive"] = "gradient",
        ratio: float = 0.1,
        grid_shape: Optional[tuple] = None,
        seed: int = 42,
        preserve_boundaries: bool = True,
        boundary_ratio: float = 0.2,
    ):
        self.method = method
        self.ratio = ratio
        self.grid_shape = grid_shape
        self.seed = seed
        self.preserve_boundaries = preserve_boundaries
        self.boundary_ratio = boundary_ratio

    def run(self, context: PipelineContext) -> PipelineContext:
        """Sample 3D grid data."""
        # Implementation for 3D numpy arrays
        # Check if we have array data in context
        if hasattr(context, 'arrays') and context.arrays:
            context.arrays = self._sample_arrays(context.arrays)
        elif context.data is not None:
            # Fall back to DataFrame sampling
            sampler = IntelligentSample(
                method="gradient" if self.method == "gradient" else "random",
                ratio=self.ratio,
                seed=self.seed,
            )
            return sampler.run(context)

        return context

    def _sample_arrays(self, arrays: dict) -> dict:
        """Sample from dictionary of numpy arrays."""
        rng = np.random.default_rng(self.seed)
        sampled = {}

        for name, arr in arrays.items():
            if arr.ndim < 3:
                sampled[name] = arr
                continue

            shape = arr.shape[-3:]  # Last 3 dims are spatial
            n_points = np.prod(shape)
            n_samples = int(n_points * self.ratio)

            if self.method == "uniform":
                # Random voxel selection
                flat_indices = rng.choice(n_points, size=n_samples, replace=False)
            elif self.method == "gradient":
                flat_indices = self._gradient_sample_3d(arr, n_samples, rng)
            else:
                flat_indices = rng.choice(n_points, size=n_samples, replace=False)

            # Store indices for reconstruction
            sampled[name] = arr.flatten()[flat_indices] if arr.ndim == 3 else arr
            sampled[f"{name}_indices"] = flat_indices
            sampled[f"{name}_original_shape"] = shape

        return sampled

    def _gradient_sample_3d(
        self, arr: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Gradient-based sampling for 3D arrays."""
        # Compute gradient magnitude
        if arr.ndim == 3:
            gx = np.gradient(arr, axis=0)
            gy = np.gradient(arr, axis=1)
            gz = np.gradient(arr, axis=2)
            grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        else:
            # For 4D+ arrays, use first 3D slice
            grad_mag = np.abs(np.gradient(arr.flatten()[:np.prod(arr.shape[-3:])]))

        grad_flat = grad_mag.flatten()

        # Normalize to probabilities
        grad_flat = grad_flat - grad_flat.min() + 1e-8
        probs = grad_flat / grad_flat.sum()

        # Mix with uniform for coverage
        uniform = np.ones_like(probs) / len(probs)
        probs = 0.7 * probs + 0.3 * uniform
        probs = probs / probs.sum()

        return rng.choice(len(probs), size=n_samples, replace=False, p=probs)
```

---

## Files to Modify

### 2. `redi/pipeline/__init__.py`

Add sampling to stage definitions and exports:

```python
# Add to STAGES dict (after PREPROCESS, before TRANSFORM):
STAGES = {
    "ingest": 1,
    "preprocess": 2,
    "sampling": 3,    # NEW
    "transform": 4,   # was 3
    "structure": 5,   # was 4
    "output": 6,      # was 5
}

# Add to imports:
from .sampling import IntelligentSample, GridSample3D, SamplingStep

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "SamplingStep",
    "IntelligentSample",
    "GridSample3D",
]
```

### 3. `redi/config.py`

Add sampling configuration schema (likely already partially there from YAML):

```python
# Add to SubsampleConfig or create SamplingConfig:
@dataclass
class SamplingConfig:
    enabled: bool = False
    method: str = "random"  # random, stratified, gradient, importance, systematic
    ratio: float = 0.1
    seed: int = 42
    stratify_column: Optional[str] = None
    gradient_variables: Optional[List[str]] = None
    min_samples: int = 1000
    # 3D grid specific
    grid_shape: Optional[Tuple[int, int, int]] = None
    preserve_boundaries: bool = True
```

### 4. `redi/domains/hpc.py`

Wire sampling into HPC pipeline (most relevant for turbulence):

```python
def get_full_pipeline(config: PipelineConfig) -> Pipeline:
    steps = []

    # Ingest steps...
    steps.extend(get_ingest_steps(config))

    # Preprocess steps...
    steps.extend(get_preprocess_steps(config))

    # NEW: Sampling steps
    if config.subsample and config.subsample.enabled:
        if config.sickle and config.sickle.grid_shape:
            # Use 3D-aware sampling for grid data
            steps.append(GridSample3D(
                method=config.subsample.method,
                ratio=config.subsample.ratio,
                grid_shape=tuple(config.sickle.grid_shape),
                seed=config.subsample.seed,
            ))
        else:
            # Use general intelligent sampling
            steps.append(IntelligentSample(
                method=config.subsample.method,
                ratio=config.subsample.ratio,
                seed=config.subsample.seed,
            ))

    # Transform steps...
    steps.extend(get_transform_steps(config))

    # ... rest of pipeline
```

### 5. Other domain files (optional)

Similar updates to:
- `redi/domains/climate.py` - temporal-aware sampling
- `redi/domains/materials.py` - structure-preserving sampling
- `redi/domains/life_sciences.py` - stratified by class
- `redi/domains/fusion.py` - similar to HPC

---

## Implementation Order

1. **Create `redi/pipeline/sampling.py`** with base classes and methods
2. **Update `redi/pipeline/__init__.py`** to export new classes
3. **Update `redi/config.py`** if SamplingConfig needs enhancement
4. **Update `redi/domains/hpc.py`** to wire in sampling
5. **Test with P1F4R32 config** we created earlier
6. **Add unit tests** in `tests/test_sampling.py`

---

## Test Cases

### `tests/test_sampling.py`

```python
import pytest
import numpy as np
import pandas as pd
from redi.pipeline.sampling import IntelligentSample, GridSample3D
from redi.pipeline.base import PipelineContext


class TestIntelligentSample:

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
        })

    @pytest.fixture
    def context(self, sample_df):
        ctx = PipelineContext()
        ctx.data = sample_df
        return ctx

    def test_random_sample(self, context):
        sampler = IntelligentSample(method="random", ratio=0.1, seed=42)
        result = sampler.run(context)
        assert len(result.data) == 100
        assert "sampling" in result.metadata

    def test_stratified_sample(self, context):
        sampler = IntelligentSample(
            method="stratified",
            ratio=0.1,
            stratify_column="category",
            seed=42
        )
        result = sampler.run(context)
        # Check distribution is roughly preserved
        orig_dist = context.data['category'].value_counts(normalize=True)
        new_dist = result.data['category'].value_counts(normalize=True)
        for cat in ['A', 'B', 'C']:
            assert abs(orig_dist[cat] - new_dist[cat]) < 0.1

    def test_gradient_sample(self, context):
        # Add gradient structure
        context.data['x'] = np.concatenate([
            np.zeros(500),
            np.linspace(0, 10, 500)  # High gradient region
        ])
        sampler = IntelligentSample(method="gradient", ratio=0.1, seed=42)
        result = sampler.run(context)
        # Should oversample from high-gradient region
        high_grad_count = (result.data.index >= 500).sum()
        assert high_grad_count > 50  # More than uniform would give

    def test_min_samples(self, context):
        sampler = IntelligentSample(method="random", ratio=0.01, min_samples=500)
        result = sampler.run(context)
        assert len(result.data) == 500  # Respects min_samples


class TestGridSample3D:

    def test_gradient_sampling_3d(self):
        # Create 3D array with localized feature
        arr = np.zeros((64, 64, 64))
        arr[30:34, 30:34, 30:34] = 10.0  # Sharp feature

        sampler = GridSample3D(method="gradient", ratio=0.1, seed=42)
        # Test gradient computation
        indices = sampler._gradient_sample_3d(arr, 1000, np.random.default_rng(42))

        # Convert flat indices to 3D
        coords = np.unravel_index(indices, arr.shape)

        # Should oversample near the feature
        near_feature = sum(
            1 for x, y, z in zip(*coords)
            if 25 <= x <= 38 and 25 <= y <= 38 and 25 <= z <= 38
        )
        assert near_feature > 100  # More than uniform
```

---

## Config YAML Example (already created)

The `examples/p1f4r32_turbulence_config.yaml` we created has the subsample section:

```yaml
subsample:
  enabled: true
  method: gradient
  ratio: 0.25
  seed: 42
```

This will work once the implementation is complete.

---

## Verification Steps

After implementation:

```bash
# Run tests
pytest tests/test_sampling.py -v

# Run pipeline on test data
python -m redi.pipeline --config examples/p1f4r32_turbulence_config.yaml

# Check output metadata for sampling stats
cat output/p1f4r32_ready/metadata.json | jq '.sampling'
```

---

## Notes

- The gradient sampling is key for turbulence data - preserves shocks, vortices, boundary layers
- For very large 3D data, may need chunked/streaming implementation
- Consider adding visualization of sampling distribution for validation
- The 3D sampler stores indices so data can be reconstructed to original grid positions
