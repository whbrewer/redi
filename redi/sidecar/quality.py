"""
Data Quality Checks for REDI Sidecar.

Detects potential issues in data transformations:
- Distribution shift
- Data leakage
- Missing values introduced
- Outliers
- NaN/Inf values
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

from .snapshot import DataSnapshot, VariableStats

logger = logging.getLogger(__name__)


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityIssue:
    """A detected quality issue."""
    check_name: str
    severity: Severity
    message: str
    variable: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'check': self.check_name,
            'severity': self.severity.value,
            'message': self.message,
            'variable': self.variable,
            'details': self.details,
        }

    def __str__(self) -> str:
        icon = {'info': 'ℹ', 'warning': '⚠', 'error': '✗'}[self.severity.value]
        var_str = f" [{self.variable}]" if self.variable else ""
        return f"{icon} {self.check_name}{var_str}: {self.message}"


@dataclass
class QualityReport:
    """Summary of quality checks for a snapshot or pipeline."""
    snapshot_path: str
    issues: List[QualityIssue] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.severity == Severity.ERROR)
        warnings = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        infos = sum(1 for i in self.issues if i.severity == Severity.INFO)

        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"{status} - {errors} errors, {warnings} warnings, {infos} info"

    def to_dict(self) -> dict:
        return {
            'snapshot_path': self.snapshot_path,
            'passed': self.passed,
            'checks_run': self.checks_run,
            'issues': [i.to_dict() for i in self.issues],
            'summary': {
                'errors': sum(1 for i in self.issues if i.severity == Severity.ERROR),
                'warnings': sum(1 for i in self.issues if i.severity == Severity.WARNING),
                'info': sum(1 for i in self.issues if i.severity == Severity.INFO),
            }
        }


def check_snapshot(snapshot: DataSnapshot) -> QualityReport:
    """
    Run all quality checks on a single snapshot.

    Args:
        snapshot: The data snapshot to check

    Returns:
        QualityReport with any detected issues
    """
    report = QualityReport(snapshot_path=snapshot.path)

    # Run all checks
    checks = [
        _check_nan_values,
        _check_inf_values,
        _check_missing_values,
        _check_constant_variables,
        _check_outliers,
        _check_shape_consistency,
    ]

    for check in checks:
        check_name = check.__name__.replace('_check_', '')
        report.checks_run.append(check_name)
        try:
            issues = check(snapshot)
            report.issues.extend(issues)
        except Exception as e:
            logger.warning(f"Check {check_name} failed: {e}")

    return report


def check_transformation(
    before: DataSnapshot,
    after: DataSnapshot,
    expected_transform: Optional[str] = None
) -> QualityReport:
    """
    Check quality of a transformation between two snapshots.

    Args:
        before: Snapshot before transformation
        after: Snapshot after transformation
        expected_transform: Name of expected transformation (e.g., 'normalize', 'regrid')

    Returns:
        QualityReport with any detected issues
    """
    report = QualityReport(snapshot_path=after.path)

    # Run transformation-specific checks
    checks = [
        _check_distribution_shift,
        _check_shape_change,
        _check_nan_introduced,
        _check_data_loss,
    ]

    for check in checks:
        check_name = check.__name__.replace('_check_', '')
        report.checks_run.append(check_name)
        try:
            issues = check(before, after)
            report.issues.extend(issues)
        except Exception as e:
            logger.warning(f"Check {check_name} failed: {e}")

    return report


# =============================================================================
# Single Snapshot Checks
# =============================================================================

def _check_nan_values(snapshot: DataSnapshot) -> List[QualityIssue]:
    """Check for NaN values in variables."""
    issues = []
    for name, var in snapshot.variables.items():
        if var.nan_count > 0:
            pct = (var.nan_count / var.size) * 100
            severity = Severity.ERROR if pct > 50 else Severity.WARNING if pct > 5 else Severity.INFO
            issues.append(QualityIssue(
                check_name="nan_values",
                severity=severity,
                message=f"{var.nan_count:,} NaN values ({pct:.1f}%)",
                variable=name,
                details={'nan_count': var.nan_count, 'percentage': pct}
            ))
    return issues


def _check_inf_values(snapshot: DataSnapshot) -> List[QualityIssue]:
    """Check for infinite values in variables."""
    issues = []
    for name, var in snapshot.variables.items():
        if var.inf_count > 0:
            pct = (var.inf_count / var.size) * 100
            issues.append(QualityIssue(
                check_name="inf_values",
                severity=Severity.ERROR,
                message=f"{var.inf_count:,} infinite values ({pct:.1f}%)",
                variable=name,
                details={'inf_count': var.inf_count, 'percentage': pct}
            ))
    return issues


def _check_missing_values(snapshot: DataSnapshot) -> List[QualityIssue]:
    """Check for null/missing values."""
    issues = []
    for name, var in snapshot.variables.items():
        if var.null_count > 0:
            pct = (var.null_count / var.size) * 100
            severity = Severity.WARNING if pct > 10 else Severity.INFO
            issues.append(QualityIssue(
                check_name="missing_values",
                severity=severity,
                message=f"{var.null_count:,} missing values ({pct:.1f}%)",
                variable=name,
                details={'null_count': var.null_count, 'percentage': pct}
            ))
    return issues


def _check_constant_variables(snapshot: DataSnapshot) -> List[QualityIssue]:
    """Check for variables with zero variance (constant)."""
    issues = []
    for name, var in snapshot.variables.items():
        if var.std is not None and var.std == 0:
            issues.append(QualityIssue(
                check_name="constant_variable",
                severity=Severity.WARNING,
                message=f"Constant value (std=0), value={var.mean}",
                variable=name,
                details={'value': var.mean}
            ))
    return issues


def _check_outliers(snapshot: DataSnapshot, sigma_threshold: float = 6.0) -> List[QualityIssue]:
    """Check for extreme outliers (values far from mean)."""
    issues = []
    for name, var in snapshot.variables.items():
        if var.mean is None or var.std is None or var.std == 0:
            continue

        # Check if min/max are extreme outliers
        if var.min is not None:
            z_min = abs(var.min - var.mean) / var.std
            if z_min > sigma_threshold:
                issues.append(QualityIssue(
                    check_name="outlier",
                    severity=Severity.WARNING,
                    message=f"Extreme minimum: {var.min:.4g} ({z_min:.1f}σ from mean)",
                    variable=name,
                    details={'value': var.min, 'z_score': z_min, 'type': 'min'}
                ))

        if var.max is not None:
            z_max = abs(var.max - var.mean) / var.std
            if z_max > sigma_threshold:
                issues.append(QualityIssue(
                    check_name="outlier",
                    severity=Severity.WARNING,
                    message=f"Extreme maximum: {var.max:.4g} ({z_max:.1f}σ from mean)",
                    variable=name,
                    details={'value': var.max, 'z_score': z_max, 'type': 'max'}
                ))

    return issues


def _check_shape_consistency(snapshot: DataSnapshot) -> List[QualityIssue]:
    """Check that all variables have consistent shapes where expected."""
    issues = []

    # Group variables by number of dimensions
    by_ndim = {}
    for name, var in snapshot.variables.items():
        ndim = len(var.shape)
        if ndim not in by_ndim:
            by_ndim[ndim] = []
        by_ndim[ndim].append((name, var.shape))

    # For climate data, check that all variables with same ndim have same spatial dims
    for ndim, vars_list in by_ndim.items():
        if len(vars_list) > 1 and ndim >= 2:
            # Check if last 2 dims (usually lat/lon) are consistent
            spatial_shapes = set(v[1][-2:] for v in vars_list if len(v[1]) >= 2)
            if len(spatial_shapes) > 1:
                issues.append(QualityIssue(
                    check_name="shape_inconsistency",
                    severity=Severity.INFO,
                    message=f"Variables with {ndim}D have different spatial shapes: {spatial_shapes}",
                    details={'shapes': list(spatial_shapes)}
                ))

    return issues


# =============================================================================
# Transformation Checks (before/after)
# =============================================================================

def _check_distribution_shift(before: DataSnapshot, after: DataSnapshot) -> List[QualityIssue]:
    """Check for significant distribution shifts between snapshots."""
    issues = []

    # Find common variables
    common_vars = set(before.variables.keys()) & set(after.variables.keys())

    for name in common_vars:
        var_before = before.variables[name]
        var_after = after.variables[name]

        if var_before.mean is None or var_after.mean is None:
            continue
        if var_before.std is None or var_after.std is None or var_before.std == 0:
            continue

        # Check mean shift (in units of original std)
        mean_shift = abs(var_after.mean - var_before.mean) / var_before.std

        if mean_shift > 3.0:
            issues.append(QualityIssue(
                check_name="distribution_shift",
                severity=Severity.WARNING,
                message=f"Mean shifted by {mean_shift:.1f}σ ({var_before.mean:.4g} → {var_after.mean:.4g})",
                variable=name,
                details={
                    'mean_before': var_before.mean,
                    'mean_after': var_after.mean,
                    'shift_sigma': mean_shift
                }
            ))

        # Check std change
        if var_after.std is not None and var_before.std > 0:
            std_ratio = var_after.std / var_before.std
            if std_ratio < 0.1 or std_ratio > 10:
                issues.append(QualityIssue(
                    check_name="variance_change",
                    severity=Severity.WARNING,
                    message=f"Std changed by {std_ratio:.2f}x ({var_before.std:.4g} → {var_after.std:.4g})",
                    variable=name,
                    details={
                        'std_before': var_before.std,
                        'std_after': var_after.std,
                        'ratio': std_ratio
                    }
                ))

    return issues


def _check_shape_change(before: DataSnapshot, after: DataSnapshot) -> List[QualityIssue]:
    """Check for unexpected shape changes."""
    issues = []

    # Compare dimensions
    dims_before = set(before.dimensions.keys())
    dims_after = set(after.dimensions.keys())

    added_dims = dims_after - dims_before
    removed_dims = dims_before - dims_after

    if added_dims:
        issues.append(QualityIssue(
            check_name="dimensions_added",
            severity=Severity.INFO,
            message=f"New dimensions: {added_dims}",
            details={'added': list(added_dims)}
        ))

    if removed_dims:
        issues.append(QualityIssue(
            check_name="dimensions_removed",
            severity=Severity.INFO,
            message=f"Removed dimensions: {removed_dims}",
            details={'removed': list(removed_dims)}
        ))

    # Check common dimensions for size changes
    common_dims = dims_before & dims_after
    for dim in common_dims:
        size_before = before.dimensions[dim]
        size_after = after.dimensions[dim]
        if size_before != size_after:
            issues.append(QualityIssue(
                check_name="dimension_size_change",
                severity=Severity.INFO,
                message=f"Dimension '{dim}' changed: {size_before} → {size_after}",
                variable=dim,
                details={'before': size_before, 'after': size_after}
            ))

    return issues


def _check_nan_introduced(before: DataSnapshot, after: DataSnapshot) -> List[QualityIssue]:
    """Check if NaN values were introduced by transformation."""
    issues = []

    common_vars = set(before.variables.keys()) & set(after.variables.keys())

    for name in common_vars:
        var_before = before.variables[name]
        var_after = after.variables[name]

        if var_after.nan_count > var_before.nan_count:
            new_nans = var_after.nan_count - var_before.nan_count
            issues.append(QualityIssue(
                check_name="nan_introduced",
                severity=Severity.WARNING,
                message=f"{new_nans:,} new NaN values introduced",
                variable=name,
                details={
                    'nan_before': var_before.nan_count,
                    'nan_after': var_after.nan_count,
                    'introduced': new_nans
                }
            ))

    return issues


def _check_data_loss(before: DataSnapshot, after: DataSnapshot) -> List[QualityIssue]:
    """Check for significant data loss (large reduction in size)."""
    issues = []

    # Check overall size
    size_ratio = after.size_bytes / before.size_bytes if before.size_bytes > 0 else 1

    if size_ratio < 0.1:
        issues.append(QualityIssue(
            check_name="data_loss",
            severity=Severity.WARNING,
            message=f"File size reduced to {size_ratio:.1%} of original",
            details={
                'size_before': before.size_bytes,
                'size_after': after.size_bytes,
                'ratio': size_ratio
            }
        ))

    # Check for lost variables
    vars_before = set(before.variables.keys())
    vars_after = set(after.variables.keys())
    lost_vars = vars_before - vars_after

    if lost_vars:
        issues.append(QualityIssue(
            check_name="variables_lost",
            severity=Severity.INFO,
            message=f"Variables removed: {lost_vars}",
            details={'lost': list(lost_vars)}
        ))

    return issues


def detect_normalization_params(snapshot: DataSnapshot) -> Optional[Dict[str, Any]]:
    """
    Try to detect if this snapshot contains normalization parameters.

    Looks for files named like 'normalize_mean.npz', 'normalize_std.npz',
    or variables named 'mean', 'std', etc.

    Returns:
        Dict with detected normalization info, or None if not found
    """
    filename = snapshot.filename.lower()

    # Check filename patterns
    if 'normalize' in filename or 'norm' in filename:
        if 'mean' in filename:
            return {
                'type': 'normalization_stats',
                'stat': 'mean',
                'variables': {k: v.to_dict() for k, v in snapshot.variables.items()}
            }
        elif 'std' in filename:
            return {
                'type': 'normalization_stats',
                'stat': 'std',
                'variables': {k: v.to_dict() for k, v in snapshot.variables.items()}
            }

    # Check for mean/std variables in the snapshot
    var_names = set(v.lower() for v in snapshot.variables.keys())
    if 'mean' in var_names and 'std' in var_names:
        return {
            'type': 'normalization_stats',
            'stat': 'combined',
            'mean_var': next(v for v in snapshot.variables.keys() if v.lower() == 'mean'),
            'std_var': next(v for v in snapshot.variables.keys() if v.lower() == 'std'),
        }

    return None


def check_transformation_report(transform_report) -> List[QualityIssue]:
    """
    Check a TransformationReport for potential data quality issues.

    Args:
        transform_report: TransformationReport from transforms.py

    Returns:
        List of QualityIssue objects
    """
    issues = []

    if transform_report is None:
        return issues

    # Check for data leakage in normalization
    if transform_report.normalization:
        norm = transform_report.normalization
        computed_on = norm.computed_on

        if computed_on == 'full':
            issues.append(QualityIssue(
                check_name="potential_data_leakage",
                severity=Severity.WARNING,
                message="Normalization appears to be computed on full dataset, not just training set. "
                        "This may cause data leakage if train/val/test splits exist.",
                details={
                    'normalization_method': norm.method,
                    'mean_file': norm.mean_file,
                    'computed_on': computed_on,
                }
            ))
        elif computed_on == 'train':
            issues.append(QualityIssue(
                check_name="normalization_scope",
                severity=Severity.INFO,
                message="Normalization computed on training set only (no data leakage detected).",
                details={
                    'normalization_method': norm.method,
                    'computed_on': computed_on,
                }
            ))
        elif computed_on == 'unknown':
            issues.append(QualityIssue(
                check_name="normalization_scope_unknown",
                severity=Severity.INFO,
                message="Could not determine if normalization was computed on training set or full dataset.",
                details={
                    'normalization_method': norm.method,
                    'computed_on': computed_on,
                }
            ))

    # Check for missing test split when train/val exist
    if transform_report.splits:
        splits = transform_report.splits
        has_train = splits.train_size is not None or 'train' in splits.split_files
        has_val = splits.val_size is not None or 'val' in splits.split_files
        has_test = splits.test_size is not None or 'test' in splits.split_files

        if has_train and has_val and not has_test:
            issues.append(QualityIssue(
                check_name="missing_test_split",
                severity=Severity.WARNING,
                message="Train and validation splits detected, but no test split found.",
                details={
                    'train_size': splits.train_size,
                    'val_size': splits.val_size,
                    'test_size': splits.test_size,
                }
            ))

        # Check for very small validation set
        if has_train and has_val and splits.val_ratio and splits.val_ratio < 0.05:
            issues.append(QualityIssue(
                check_name="small_validation_set",
                severity=Severity.WARNING,
                message=f"Validation set is only {splits.val_ratio:.1%} of total data. "
                        "Consider using a larger validation set for reliable metrics.",
                details={
                    'val_ratio': splits.val_ratio,
                    'val_size': splits.val_size,
                }
            ))

    return issues
