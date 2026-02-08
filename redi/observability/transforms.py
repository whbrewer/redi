"""
Transformation parameter detection for REDI observability.

Automatically detects and extracts parameters from common ML preprocessing steps:
- Normalization (mean, std)
- Train/test splits
- Regridding parameters
- Unit conversions
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .snapshot import DataSnapshot, capture_snapshot

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """Detected normalization parameters."""
    method: str  # 'zscore', 'minmax', 'custom'
    mean_file: Optional[str] = None
    std_file: Optional[str] = None
    mean_values: Dict[str, float] = field(default_factory=dict)  # var_name -> mean
    std_values: Dict[str, float] = field(default_factory=dict)   # var_name -> std
    min_values: Dict[str, float] = field(default_factory=dict)   # for minmax
    max_values: Dict[str, float] = field(default_factory=dict)   # for minmax
    computed_on: Optional[str] = None  # 'full', 'train', 'unknown'

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'mean_file': self.mean_file,
            'std_file': self.std_file,
            'mean_values': self.mean_values,
            'std_values': self.std_values,
            'min_values': self.min_values,
            'max_values': self.max_values,
            'computed_on': self.computed_on,
        }

    def summary(self) -> str:
        lines = [f"Normalization: {self.method}"]
        if self.mean_file:
            lines.append(f"  Mean file: {self.mean_file}")
        if self.std_file:
            lines.append(f"  Std file: {self.std_file}")
        if self.mean_values:
            lines.append(f"  Variables: {len(self.mean_values)}")
            for var, mean in list(self.mean_values.items())[:3]:
                std = self.std_values.get(var, 'N/A')
                lines.append(f"    {var}: mean={mean:.4g}, std={std:.4g}" if isinstance(std, float) else f"    {var}: mean={mean:.4g}")
            if len(self.mean_values) > 3:
                lines.append(f"    ... and {len(self.mean_values) - 3} more")
        if self.computed_on:
            lines.append(f"  Computed on: {self.computed_on}")
        return "\n".join(lines)


@dataclass
class SplitParams:
    """Detected train/val/test split parameters."""
    method: str  # 'random', 'temporal', 'stratified', 'unknown'
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    split_files: Dict[str, str] = field(default_factory=dict)  # 'train' -> path

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'split_files': self.split_files,
        }


@dataclass
class RegridParams:
    """Detected regridding parameters."""
    source_resolution: Optional[Tuple[float, float]] = None  # (lat_res, lon_res)
    target_resolution: Optional[Tuple[float, float]] = None
    method: str = 'unknown'  # 'bilinear', 'conservative', 'nearest'
    source_grid: Optional[str] = None
    target_grid: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'source_resolution': self.source_resolution,
            'target_resolution': self.target_resolution,
            'method': self.method,
            'source_grid': self.source_grid,
            'target_grid': self.target_grid,
        }


@dataclass
class TransformationReport:
    """Summary of all detected transformations."""
    normalization: Optional[NormalizationParams] = None
    splits: Optional[SplitParams] = None
    regrid: Optional[RegridParams] = None
    other: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {}
        if self.normalization:
            result['normalization'] = self.normalization.to_dict()
        if self.splits:
            result['splits'] = self.splits.to_dict()
        if self.regrid:
            result['regrid'] = self.regrid.to_dict()
        if self.other:
            result['other'] = self.other
        return result

    def to_markdown(self) -> str:
        """Generate markdown section for Data Card."""
        lines = ["## Transformation Parameters", ""]

        if self.normalization:
            lines.extend([
                "### Normalization",
                f"- **Method:** {self.normalization.method}",
            ])
            if self.normalization.mean_file:
                lines.append(f"- **Mean file:** `{Path(self.normalization.mean_file).name}`")
            if self.normalization.std_file:
                lines.append(f"- **Std file:** `{Path(self.normalization.std_file).name}`")
            if self.normalization.computed_on:
                lines.append(f"- **Computed on:** {self.normalization.computed_on}")

            if self.normalization.mean_values:
                lines.extend(["", "| Variable | Mean | Std |", "|----------|------|-----|"])
                for var in list(self.normalization.mean_values.keys())[:10]:
                    mean = self.normalization.mean_values[var]
                    std = self.normalization.std_values.get(var, '-')
                    std_str = f"{std:.4g}" if isinstance(std, float) else str(std)
                    lines.append(f"| {var} | {mean:.4g} | {std_str} |")
                if len(self.normalization.mean_values) > 10:
                    lines.append(f"| *... {len(self.normalization.mean_values) - 10} more* | | |")
            lines.append("")

        if self.splits:
            lines.extend([
                "### Train/Val/Test Split",
                f"- **Method:** {self.splits.method}",
            ])
            if self.splits.train_size:
                total = (self.splits.train_size or 0) + (self.splits.val_size or 0) + (self.splits.test_size or 0)
                lines.append(f"- **Train:** {self.splits.train_size:,} samples ({self.splits.train_size/total*100:.1f}%)" if total else f"- **Train:** {self.splits.train_size:,} samples")
            if self.splits.val_size:
                lines.append(f"- **Val:** {self.splits.val_size:,} samples")
            if self.splits.test_size:
                lines.append(f"- **Test:** {self.splits.test_size:,} samples")
            lines.append("")

        if self.regrid:
            lines.extend([
                "### Regridding",
                f"- **Method:** {self.regrid.method}",
            ])
            if self.regrid.source_resolution:
                lines.append(f"- **Source resolution:** {self.regrid.source_resolution[0]}° x {self.regrid.source_resolution[1]}°")
            if self.regrid.target_resolution:
                lines.append(f"- **Target resolution:** {self.regrid.target_resolution[0]}° x {self.regrid.target_resolution[1]}°")
            lines.append("")

        return "\n".join(lines) if len(lines) > 2 else ""


def detect_transformations(data_dir: Path, snapshots: Optional[List[DataSnapshot]] = None) -> TransformationReport:
    """
    Scan a data directory and detect transformation parameters.

    Args:
        data_dir: Directory containing pipeline outputs
        snapshots: Optional pre-captured snapshots to analyze

    Returns:
        TransformationReport with detected parameters
    """
    report = TransformationReport()

    # Detect normalization parameters
    report.normalization = _detect_normalization(data_dir, snapshots)

    # Detect train/val/test splits
    report.splits = _detect_splits(data_dir, snapshots)

    # Detect regridding
    report.regrid = _detect_regrid(data_dir, snapshots)

    return report


def _detect_normalization(data_dir: Path, snapshots: Optional[List[DataSnapshot]] = None) -> Optional[NormalizationParams]:
    """Detect normalization parameters from files."""
    import numpy as np

    params = NormalizationParams(method='unknown')

    # Pattern 1: Look for normalize_mean.npz / normalize_std.npz (ClimaX style)
    mean_patterns = ['normalize_mean.npz', 'normalization_mean.npz', 'mean.npz', 'stats_mean.npz']
    std_patterns = ['normalize_std.npz', 'normalization_std.npz', 'std.npz', 'stats_std.npz']

    mean_file = None
    std_file = None

    for pattern in mean_patterns:
        candidates = list(data_dir.rglob(pattern))
        if candidates:
            mean_file = candidates[0]
            break

    for pattern in std_patterns:
        candidates = list(data_dir.rglob(pattern))
        if candidates:
            std_file = candidates[0]
            break

    if mean_file and mean_file.exists():
        params.mean_file = str(mean_file)
        params.method = 'zscore'
        try:
            with np.load(mean_file) as npz:
                for key in npz.files:
                    arr = npz[key]
                    if arr.ndim == 0:
                        params.mean_values[key] = float(arr)
                    elif arr.size == 1:
                        params.mean_values[key] = float(arr.flat[0])
                    else:
                        # For multi-dimensional arrays, store the global mean
                        params.mean_values[key] = float(np.mean(arr))
        except Exception as e:
            logger.warning(f"Could not read mean file {mean_file}: {e}")

    if std_file and std_file.exists():
        params.std_file = str(std_file)
        try:
            with np.load(std_file) as npz:
                for key in npz.files:
                    arr = npz[key]
                    if arr.ndim == 0:
                        params.std_values[key] = float(arr)
                    elif arr.size == 1:
                        params.std_values[key] = float(arr.flat[0])
                    else:
                        params.std_values[key] = float(np.mean(arr))
        except Exception as e:
            logger.warning(f"Could not read std file {std_file}: {e}")

    # Pattern 2: Look for combined stats file
    stats_patterns = ['stats.npz', 'normalization.npz', 'normalize.npz', 'scaler.npz']
    for pattern in stats_patterns:
        candidates = list(data_dir.rglob(pattern))
        if candidates:
            stats_file = candidates[0]
            try:
                with np.load(stats_file) as npz:
                    keys = list(npz.files)
                    if 'mean' in keys and 'std' in keys:
                        params.method = 'zscore'
                        params.mean_file = str(stats_file)
                        params.std_file = str(stats_file)
                        mean_arr = npz['mean']
                        std_arr = npz['std']
                        params.mean_values['global'] = float(np.mean(mean_arr))
                        params.std_values['global'] = float(np.mean(std_arr))
                    elif 'min' in keys and 'max' in keys:
                        params.method = 'minmax'
                        params.min_values['global'] = float(np.mean(npz['min']))
                        params.max_values['global'] = float(np.mean(npz['max']))
            except Exception as e:
                logger.debug(f"Could not read stats file {stats_file}: {e}")

    # Pattern 3: Look for sklearn-style pickle/joblib files
    scaler_patterns = ['scaler.pkl', 'scaler.joblib', 'normalizer.pkl', 'normalizer.joblib']
    for pattern in scaler_patterns:
        candidates = list(data_dir.rglob(pattern))
        if candidates:
            params.method = 'sklearn_scaler'
            params.other = {'scaler_file': str(candidates[0])}
            break

    # Pattern 4: Look for JSON config files that might contain normalization info
    config_patterns = ['config.json', 'params.json', 'preprocessing.json', 'transform_config.json']
    for pattern in config_patterns:
        candidates = list(data_dir.rglob(pattern))
        for config_file in candidates:
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    if 'normalization' in config or 'normalize' in config:
                        norm_config = config.get('normalization') or config.get('normalize')
                        if isinstance(norm_config, dict):
                            if 'mean' in norm_config:
                                params.mean_values = norm_config['mean'] if isinstance(norm_config['mean'], dict) else {'global': norm_config['mean']}
                            if 'std' in norm_config:
                                params.std_values = norm_config['std'] if isinstance(norm_config['std'], dict) else {'global': norm_config['std']}
                            if 'method' in norm_config:
                                params.method = norm_config['method']
            except Exception:
                pass

    # Return None if nothing found
    if not params.mean_file and not params.std_file and not params.mean_values:
        return None

    # Try to detect what data the normalization was computed on
    params.computed_on = _detect_computed_on(data_dir, params)

    return params


def _detect_computed_on(data_dir: Path, params: NormalizationParams) -> Optional[str]:
    """
    Detect whether normalization stats were computed on training data or full dataset.

    Returns:
        'train' - Stats likely computed on training set only (safe)
        'full' - Stats likely computed on full dataset (potential leakage)
        'unknown' - Cannot determine
    """
    import os

    # Pattern 1: Check if normalize files are inside a "train" directory
    if params.mean_file:
        mean_path = Path(params.mean_file)
        path_parts = [p.lower() for p in mean_path.parts]
        if 'train' in path_parts or 'training' in path_parts:
            return 'train'

    # Pattern 2: Check for naming patterns suggesting train-only stats
    train_stat_patterns = ['train_mean', 'train_stats', 'training_mean', 'train_normalize']
    if params.mean_file:
        filename = Path(params.mean_file).stem.lower()
        if any(p in filename for p in train_stat_patterns):
            return 'train'

    # Pattern 3: Check file timestamps relative to split directories
    # If normalize files are newer than train dir creation, stats were computed after split
    try:
        train_dirs = list(data_dir.rglob('train/')) + list(data_dir.rglob('training/'))
        if train_dirs and params.mean_file:
            mean_mtime = Path(params.mean_file).stat().st_mtime
            train_dir = train_dirs[0]
            train_mtime = train_dir.stat().st_mtime

            # If mean file was modified after train dir was created, likely computed on train
            # (This is a heuristic and not always accurate)
            if mean_mtime > train_mtime:
                return 'train'
    except (OSError, FileNotFoundError):
        pass

    # Pattern 4: Look for config files that might indicate the computation scope
    config_patterns = ['config.json', 'config.yaml', 'config.yml', 'params.json']
    for pattern in config_patterns:
        for config_file in data_dir.rglob(pattern):
            try:
                if config_file.suffix == '.json':
                    import json
                    with open(config_file) as f:
                        config = json.load(f)
                else:
                    import yaml
                    with open(config_file) as f:
                        config = yaml.safe_load(f)

                # Look for normalization settings
                norm_config = config.get('normalization') or config.get('normalize') or {}
                if isinstance(norm_config, dict):
                    computed_on = norm_config.get('computed_on') or norm_config.get('fit_on')
                    if computed_on:
                        return computed_on.lower()
            except Exception:
                pass

    # Pattern 5: If we found splits but stats are at root level, likely full dataset
    has_splits = bool(list(data_dir.rglob('train/')) or list(data_dir.rglob('val/')) or list(data_dir.rglob('test/')))
    if has_splits and params.mean_file:
        mean_path = Path(params.mean_file)
        # If mean file is at or near root (not inside train), likely computed on full
        rel_path = mean_path.relative_to(data_dir) if mean_path.is_relative_to(data_dir) else mean_path
        if len(rel_path.parts) <= 2:  # At root or one level deep
            return 'full'

    return 'unknown'


def _detect_splits(data_dir: Path, snapshots: Optional[List[DataSnapshot]] = None) -> Optional[SplitParams]:
    """Detect train/val/test split information."""
    params = SplitParams(method='unknown')
    found_splits = False

    # Pattern 1: Look for train/val/test directories or files
    split_names = {
        'train': ['train', 'training'],
        'val': ['val', 'valid', 'validation', 'dev'],
        'test': ['test', 'testing', 'eval'],
    }

    for split_type, patterns in split_names.items():
        for pattern in patterns:
            # Check for directories
            for d in data_dir.rglob(f'{pattern}/'):
                if d.is_dir():
                    params.split_files[split_type] = str(d)
                    found_splits = True
                    # Count files in directory
                    files = list(d.glob('*'))
                    if split_type == 'train':
                        params.train_size = len(files)
                    elif split_type == 'val':
                        params.val_size = len(files)
                    elif split_type == 'test':
                        params.test_size = len(files)
                    break

            # Check for files with split in name
            for f in data_dir.rglob(f'*{pattern}*.npz'):
                params.split_files[split_type] = str(f)
                found_splits = True
                break
            for f in data_dir.rglob(f'*{pattern}*.npy'):
                params.split_files[split_type] = str(f)
                found_splits = True
                break

    # Pattern 2: Look for split info in snapshots
    if snapshots:
        for snapshot in snapshots:
            filename = snapshot.filename.lower()
            if 'train' in filename:
                params.split_files['train'] = snapshot.path
                # Try to get size from first dimension
                if snapshot.variables:
                    first_var = list(snapshot.variables.values())[0]
                    if first_var.shape:
                        params.train_size = first_var.shape[0]
                found_splits = True
            elif 'val' in filename or 'valid' in filename:
                params.split_files['val'] = snapshot.path
                if snapshot.variables:
                    first_var = list(snapshot.variables.values())[0]
                    if first_var.shape:
                        params.val_size = first_var.shape[0]
                found_splits = True
            elif 'test' in filename:
                params.split_files['test'] = snapshot.path
                if snapshot.variables:
                    first_var = list(snapshot.variables.values())[0]
                    if first_var.shape:
                        params.test_size = first_var.shape[0]
                found_splits = True

    # Try to infer split method from file naming
    if found_splits:
        # Check for temporal patterns in filenames (years, dates)
        for path in params.split_files.values():
            if re.search(r'(19|20)\d{2}', path):
                params.method = 'temporal'
                break
        else:
            params.method = 'unknown'

    # Calculate ratios if we have sizes
    total = (params.train_size or 0) + (params.val_size or 0) + (params.test_size or 0)
    if total > 0:
        if params.train_size:
            params.train_ratio = params.train_size / total
        if params.val_size:
            params.val_ratio = params.val_size / total
        if params.test_size:
            params.test_ratio = params.test_size / total

    return params if found_splits else None


def _detect_regrid(data_dir: Path, snapshots: Optional[List[DataSnapshot]] = None) -> Optional[RegridParams]:
    """Detect regridding parameters from file metadata."""
    params = RegridParams()
    found_regrid = False

    # Pattern 1: Look for resolution in filenames (e.g., 1.40625deg, 5.625deg)
    res_pattern = re.compile(r'(\d+\.?\d*)deg')

    for f in data_dir.rglob('*.nc'):
        match = res_pattern.search(f.name)
        if match:
            res = float(match.group(1))
            params.target_resolution = (res, res)
            found_regrid = True
            break

    # Pattern 2: Infer from snapshots with lat/lon coordinates
    if snapshots:
        for snapshot in snapshots:
            if snapshot.spatial_bounds:
                # Try to infer resolution from grid size
                lat_range = snapshot.spatial_bounds.get('lat_max', 90) - snapshot.spatial_bounds.get('lat_min', -90)
                lon_range = snapshot.spatial_bounds.get('lon_max', 360) - snapshot.spatial_bounds.get('lon_min', 0)

                # Look for lat/lon dimension sizes
                lat_size = snapshot.dimensions.get('lat') or snapshot.dimensions.get('latitude') or snapshot.dimensions.get('y')
                lon_size = snapshot.dimensions.get('lon') or snapshot.dimensions.get('longitude') or snapshot.dimensions.get('x')

                if lat_size and lon_size:
                    lat_res = lat_range / lat_size
                    lon_res = lon_range / lon_size
                    params.target_resolution = (round(lat_res, 4), round(lon_res, 4))
                    found_regrid = True
                    break

    # Pattern 3: Look for regrid config files
    config_patterns = ['regrid_config.json', 'regrid.json', 'grid_config.json']
    for pattern in config_patterns:
        candidates = list(data_dir.rglob(pattern))
        if candidates:
            try:
                with open(candidates[0]) as f:
                    config = json.load(f)
                    if 'method' in config:
                        params.method = config['method']
                    if 'target_resolution' in config:
                        params.target_resolution = tuple(config['target_resolution'])
                    found_regrid = True
            except Exception:
                pass

    # Pattern 4: Look for xesmf weight files
    weight_files = list(data_dir.rglob('*weights*.nc')) + list(data_dir.rglob('*regrid*.nc'))
    if weight_files:
        params.method = 'xesmf'
        found_regrid = True

    return params if found_regrid else None
