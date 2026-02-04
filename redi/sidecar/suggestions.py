"""
Workflow Improvement Suggestions for REDI Sidecar.

Analyzes detected issues and workflow structure to provide
actionable recommendations for improving data pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

from .quality import QualityIssue, QualityReport, Severity

logger = logging.getLogger(__name__)


class SuggestionCategory(Enum):
    """Categories of suggestions."""
    DATA_QUALITY = "data_quality"
    DATA_LEAKAGE = "data_leakage"
    SPLITS = "splits"
    NORMALIZATION = "normalization"
    REPRODUCIBILITY = "reproducibility"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    BEST_PRACTICES = "best_practices"


@dataclass
class Suggestion:
    """An actionable suggestion for workflow improvement."""
    category: SuggestionCategory
    title: str
    description: str
    priority: int  # 1 = high, 2 = medium, 3 = low
    related_issue: Optional[str] = None
    code_example: Optional[str] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'category': self.category.value,
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'related_issue': self.related_issue,
            'code_example': self.code_example,
            'references': self.references,
        }

    def __str__(self) -> str:
        priority_icon = {1: '🔴', 2: '🟡', 3: '🟢'}[self.priority]
        return f"{priority_icon} [{self.category.value}] {self.title}"


@dataclass
class SuggestionsReport:
    """Collection of suggestions for a workflow."""
    workflow_file: str
    suggestions: List[Suggestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'workflow_file': self.workflow_file,
            'suggestions': [s.to_dict() for s in self.suggestions],
            'summary': {
                'total': len(self.suggestions),
                'high_priority': sum(1 for s in self.suggestions if s.priority == 1),
                'medium_priority': sum(1 for s in self.suggestions if s.priority == 2),
                'low_priority': sum(1 for s in self.suggestions if s.priority == 3),
            }
        }

    def by_priority(self) -> List[Suggestion]:
        """Return suggestions sorted by priority (high first)."""
        return sorted(self.suggestions, key=lambda s: s.priority)

    def by_category(self) -> Dict[str, List[Suggestion]]:
        """Return suggestions grouped by category."""
        result = {}
        for s in self.suggestions:
            cat = s.category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(s)
        return result


def generate_suggestions(
    quality_issues: List[QualityIssue],
    transformation_report: Optional[Any] = None,
    workflow_info: Optional[Dict[str, Any]] = None,
    data_snapshots: Optional[List[Any]] = None,
) -> List[Suggestion]:
    """
    Generate actionable suggestions based on detected issues and workflow analysis.

    Args:
        quality_issues: List of QualityIssue from quality checks
        transformation_report: TransformationReport from transforms.py
        workflow_info: Dict with workflow metadata (rules, stages, etc.)
        data_snapshots: List of DataSnapshot objects

    Returns:
        List of Suggestion objects
    """
    suggestions = []

    # Generate suggestions from quality issues
    suggestions.extend(_suggestions_from_issues(quality_issues))

    # Generate suggestions from transformation report
    if transformation_report:
        suggestions.extend(_suggestions_from_transforms(transformation_report))

    # Generate suggestions from workflow structure
    if workflow_info:
        suggestions.extend(_suggestions_from_workflow(workflow_info))

    # Generate suggestions from data snapshots
    if data_snapshots:
        suggestions.extend(_suggestions_from_snapshots(data_snapshots))

    # Add general best practice suggestions if not already covered
    suggestions.extend(_general_suggestions(suggestions, workflow_info))

    # Deduplicate by title
    seen_titles = set()
    unique_suggestions = []
    for s in suggestions:
        if s.title not in seen_titles:
            seen_titles.add(s.title)
            unique_suggestions.append(s)

    return unique_suggestions


def _suggestions_from_issues(issues: List[QualityIssue]) -> List[Suggestion]:
    """Generate suggestions based on detected quality issues."""
    suggestions = []

    for issue in issues:
        check = issue.check_name

        # Data leakage
        if check == "potential_data_leakage":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_LEAKAGE,
                title="Compute normalization on training data only",
                description=(
                    "Normalization statistics (mean, std) should be computed only on the "
                    "training set to prevent data leakage. Apply these statistics to "
                    "validation and test sets without recomputing."
                ),
                priority=1,
                related_issue=issue.message,
                code_example="""# Correct approach:
train_mean = train_data.mean()
train_std = train_data.std()

# Apply to all splits using TRAINING statistics
train_normalized = (train_data - train_mean) / train_std
val_normalized = (val_data - train_mean) / train_std
test_normalized = (test_data - train_mean) / train_std""",
                references=[
                    "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"
                ]
            ))

        # Missing test split
        elif check == "missing_test_split":
            suggestions.append(Suggestion(
                category=SuggestionCategory.SPLITS,
                title="Add a held-out test split",
                description=(
                    "A test split is essential for unbiased final model evaluation. "
                    "The test set should only be used once, after all hyperparameter "
                    "tuning is complete. Recommended split: 70% train, 15% validation, 15% test."
                ),
                priority=1,
                related_issue=issue.message,
                code_example="""# Example split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# For time series, use temporal splits (not random)
# train: earliest data, val: middle, test: most recent""",
            ))

        # Small validation set
        elif check == "small_validation_set":
            suggestions.append(Suggestion(
                category=SuggestionCategory.SPLITS,
                title="Increase validation set size",
                description=(
                    "A validation set smaller than 10% may give unreliable performance "
                    "estimates during hyperparameter tuning. Consider increasing to 15-20% "
                    "or using k-fold cross-validation for small datasets."
                ),
                priority=2,
                related_issue=issue.message,
            ))

        # NaN values
        elif check == "nan_values" and issue.severity in (Severity.WARNING, Severity.ERROR):
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Handle NaN values before training",
                description=(
                    f"Variable '{issue.variable}' contains NaN values. Consider: "
                    "(1) imputation with mean/median/mode, (2) forward/backward fill for time series, "
                    "(3) dropping rows with NaN, or (4) using models that handle NaN natively."
                ),
                priority=1 if issue.severity == Severity.ERROR else 2,
                related_issue=issue.message,
                code_example="""# Option 1: Drop NaN
df = df.dropna()

# Option 2: Fill with mean (compute on train only!)
train_mean = train_df['column'].mean()
df['column'] = df['column'].fillna(train_mean)

# Option 3: Forward fill (time series)
df = df.ffill()""",
            ))

        # Inf values
        elif check == "inf_values":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Remove or cap infinite values",
                description=(
                    f"Variable '{issue.variable}' contains infinite values. These will cause "
                    "training to fail. Cap to a maximum value or investigate the source."
                ),
                priority=1,
                related_issue=issue.message,
                code_example="""import numpy as np
# Replace inf with NaN, then handle NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Or cap to a maximum value
max_val = df[df != np.inf].max()
df = df.clip(upper=max_val)""",
            ))

        # Outliers
        elif check == "outlier":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Investigate and handle outliers",
                description=(
                    f"Variable '{issue.variable}' has extreme outliers. Investigate if these "
                    "are data errors or valid extreme values. Consider clipping, log transform, "
                    "or robust scaling."
                ),
                priority=2,
                related_issue=issue.message,
                code_example="""# Option 1: Clip to percentiles
lower, upper = df['col'].quantile([0.01, 0.99])
df['col'] = df['col'].clip(lower, upper)

# Option 2: Robust scaling (uses median/IQR)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['col'] = scaler.fit_transform(df[['col']])""",
            ))

        # NaN introduced by transformation
        elif check == "nan_introduced":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Fix transformation that introduces NaN",
                description=(
                    "A transformation step is introducing NaN values. This often happens with "
                    "division by zero, log of negative numbers, or regridding to finer resolution. "
                    "Add validation after each transformation step."
                ),
                priority=1,
                related_issue=issue.message,
            ))

        # Distribution shift
        elif check == "distribution_shift":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Verify large distribution shift is intentional",
                description=(
                    f"Variable '{issue.variable}' has a large distribution shift after "
                    "transformation. If this is unexpected, check for bugs in the transformation. "
                    "If intentional (e.g., normalization), document the expected behavior."
                ),
                priority=2,
                related_issue=issue.message,
            ))

        # Constant variables
        elif check == "constant_variable":
            suggestions.append(Suggestion(
                category=SuggestionCategory.DATA_QUALITY,
                title="Remove or investigate constant variables",
                description=(
                    f"Variable '{issue.variable}' has zero variance (constant value). "
                    "Constant features provide no information for learning. Consider removing "
                    "or investigating if this is a data processing error."
                ),
                priority=2,
                related_issue=issue.message,
            ))

    return suggestions


def _suggestions_from_transforms(transform_report) -> List[Suggestion]:
    """Generate suggestions based on transformation analysis."""
    suggestions = []

    if transform_report is None:
        return suggestions

    # Check normalization
    if transform_report.normalization:
        norm = transform_report.normalization

        if norm.computed_on == 'unknown':
            suggestions.append(Suggestion(
                category=SuggestionCategory.REPRODUCIBILITY,
                title="Document normalization scope",
                description=(
                    "Could not determine if normalization was computed on training data only. "
                    "Add a config file or naming convention (e.g., 'train_mean.npz') to make "
                    "this explicit for reproducibility."
                ),
                priority=2,
            ))

        if not norm.mean_file or not norm.std_file:
            suggestions.append(Suggestion(
                category=SuggestionCategory.REPRODUCIBILITY,
                title="Save normalization statistics to files",
                description=(
                    "Save normalization mean and std to separate files (e.g., normalize_mean.npz, "
                    "normalize_std.npz) for reproducibility. This allows applying the same "
                    "normalization to new data."
                ),
                priority=2,
                code_example="""import numpy as np
# Save statistics
np.savez('normalize_mean.npz', **{var: mean for var, mean in means.items()})
np.savez('normalize_std.npz', **{var: std for var, std in stds.items()})""",
            ))

    # Check splits
    if transform_report.splits:
        splits = transform_report.splits

        if splits.method == 'unknown':
            suggestions.append(Suggestion(
                category=SuggestionCategory.REPRODUCIBILITY,
                title="Document split method",
                description=(
                    "The train/val/test split method is not documented. For time series data, "
                    "use temporal splits. For i.i.d. data, random splits with a fixed seed. "
                    "Add this to your config or Data Card."
                ),
                priority=2,
            ))

    # Check regrid
    if transform_report.regrid:
        regrid = transform_report.regrid

        if regrid.method == 'unknown':
            suggestions.append(Suggestion(
                category=SuggestionCategory.DOCUMENTATION,
                title="Document regridding method",
                description=(
                    "The regridding method (bilinear, conservative, nearest) is not documented. "
                    "Different methods have different properties for preserving quantities. "
                    "Document the choice in your Data Card."
                ),
                priority=3,
            ))

    return suggestions


def _suggestions_from_workflow(workflow_info: Dict[str, Any]) -> List[Suggestion]:
    """Generate suggestions based on workflow structure."""
    suggestions = []

    rules = workflow_info.get('rules', [])
    stage_mapping = workflow_info.get('stage_mapping', {})

    # Check for missing stages
    expected_stages = {'ingest', 'preprocess', 'transform', 'output'}
    present_stages = set(k for k, v in stage_mapping.items() if v)

    missing_stages = expected_stages - present_stages

    if 'transform' in missing_stages:
        suggestions.append(Suggestion(
            category=SuggestionCategory.BEST_PRACTICES,
            title="Add explicit transformation stage",
            description=(
                "No explicit transformation stage detected. Consider adding steps for: "
                "normalization, feature engineering, encoding categoricals, or handling "
                "missing values."
            ),
            priority=3,
        ))

    # Check for validation steps
    rule_names = [r.get('name', '').lower() for r in rules]
    has_validation = any('valid' in name or 'check' in name or 'qc' in name
                         for name in rule_names)

    if not has_validation:
        suggestions.append(Suggestion(
            category=SuggestionCategory.BEST_PRACTICES,
            title="Add data validation steps",
            description=(
                "No validation or QC steps detected in the workflow. Consider adding "
                "validation after each major transformation to catch issues early."
            ),
            priority=2,
            code_example="""# Snakemake example
rule validate_data:
    input: "{dataset}/processed/data.nc"
    output: "{dataset}/processed/validation_report.json"
    shell:
        \"\"\"
        python scripts/validate.py {input} --output {output}
        \"\"\"
""",
        ))

    return suggestions


def _suggestions_from_snapshots(snapshots: List[Any]) -> List[Suggestion]:
    """Generate suggestions based on data snapshots."""
    suggestions = []

    if not snapshots:
        return suggestions

    # Check for very large files that might benefit from chunking
    large_files = [s for s in snapshots if s.size_bytes > 1e9]  # > 1GB
    if large_files:
        suggestions.append(Suggestion(
            category=SuggestionCategory.PERFORMANCE,
            title="Consider chunked processing for large files",
            description=(
                f"Found {len(large_files)} files larger than 1GB. Consider using "
                "chunked/lazy loading (Dask, Zarr) to reduce memory usage and enable "
                "parallel processing."
            ),
            priority=3,
            code_example="""# Using xarray with Dask
import xarray as xr
ds = xr.open_dataset('large_file.nc', chunks={'time': 100})

# Using Zarr for efficient storage
ds.to_zarr('output.zarr', mode='w')""",
        ))

    return suggestions


def _general_suggestions(
    existing: List[Suggestion],
    workflow_info: Optional[Dict[str, Any]]
) -> List[Suggestion]:
    """Add general best practice suggestions if not already covered."""
    suggestions = []

    existing_titles = {s.title for s in existing}

    # Reproducibility suggestions
    if "Document normalization scope" not in existing_titles:
        suggestions.append(Suggestion(
            category=SuggestionCategory.REPRODUCIBILITY,
            title="Record random seeds for reproducibility",
            description=(
                "Set and record random seeds for any stochastic operations "
                "(train/test splits, data augmentation, shuffling). This ensures "
                "experiments can be exactly reproduced."
            ),
            priority=3,
            code_example="""import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Document in config or Data Card
# random_seed: 42""",
        ))

    # Documentation suggestions
    suggestions.append(Suggestion(
        category=SuggestionCategory.DOCUMENTATION,
        title="Generate a Data Card for the output dataset",
        description=(
            "Create a Data Card documenting: data source, preprocessing steps, "
            "normalization parameters, split ratios, and known limitations. "
            "REDI can generate this automatically with --data-dir."
        ),
        priority=3,
        references=[
            "https://arxiv.org/abs/2204.01075"  # Data Cards paper
        ],
    ))

    return suggestions


def format_suggestions_markdown(suggestions: List[Suggestion]) -> str:
    """Format suggestions as markdown for Data Card."""
    if not suggestions:
        return "No suggestions - workflow follows best practices.\n"

    lines = ["## Suggestions for Improvement\n"]

    # Group by priority
    high = [s for s in suggestions if s.priority == 1]
    medium = [s for s in suggestions if s.priority == 2]
    low = [s for s in suggestions if s.priority == 3]

    if high:
        lines.append("### High Priority\n")
        for s in high:
            lines.append(f"#### 🔴 {s.title}\n")
            lines.append(f"{s.description}\n")
            if s.code_example:
                lines.append(f"\n```python\n{s.code_example}\n```\n")
            lines.append("")

    if medium:
        lines.append("### Medium Priority\n")
        for s in medium:
            lines.append(f"#### 🟡 {s.title}\n")
            lines.append(f"{s.description}\n")
            if s.code_example:
                lines.append(f"\n```python\n{s.code_example}\n```\n")
            lines.append("")

    if low:
        lines.append("### Low Priority\n")
        for s in low:
            lines.append(f"#### 🟢 {s.title}\n")
            lines.append(f"{s.description}\n")
            lines.append("")

    return "\n".join(lines)


def format_suggestions_cli(suggestions: List[Suggestion], use_color: bool = True) -> str:
    """Format suggestions for CLI output."""
    if not suggestions:
        return "  No suggestions - workflow follows best practices."

    lines = []

    # Sort by priority
    for s in sorted(suggestions, key=lambda x: x.priority):
        if use_color:
            priority_icon = {1: '\033[91m●\033[0m', 2: '\033[93m●\033[0m', 3: '\033[92m●\033[0m'}[s.priority]
        else:
            priority_icon = {1: '[HIGH]', 2: '[MED]', 3: '[LOW]'}[s.priority]

        lines.append(f"  {priority_icon} {s.title}")
        # Wrap description to 70 chars
        desc_lines = _wrap_text(s.description, 70)
        for dl in desc_lines:
            lines.append(f"      {dl}")

    return "\n".join(lines)


def _wrap_text(text: str, width: int) -> List[str]:
    """Simple text wrapper."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= width:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_len = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines
