"""
REDI Sidecar - Observability layer for data pipelines.

Captures data snapshots, tracks transformations, and generates Data Cards
for existing workflows without modifying them.
"""

from .snapshot import DataSnapshot, VariableStats, capture_snapshot
from .quality import (
    QualityIssue,
    QualityReport,
    Severity,
    check_snapshot,
    check_transformation,
    check_transformation_report,
    detect_normalization_params,
)
from .transforms import (
    NormalizationParams,
    SplitParams,
    RegridParams,
    TransformationReport,
    detect_transformations,
)
from .snakemake_log import (
    RuleExecution,
    SnakemakeLogReport,
    parse_snakemake_log,
    find_latest_log,
    find_all_logs,
)
from .nextflow_parser import (
    NextflowProcess,
    NextflowWorkflow,
    NextflowInclude,
    NextflowParseResult,
    parse_nextflow_file,
    map_processes_to_stages,
    find_nextflow_logs,
)
from .suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionsReport,
    generate_suggestions,
    format_suggestions_markdown,
    format_suggestions_cli,
)

__all__ = [
    "DataSnapshot",
    "VariableStats",
    "capture_snapshot",
    "QualityIssue",
    "QualityReport",
    "Severity",
    "check_snapshot",
    "check_transformation",
    "check_transformation_report",
    "detect_normalization_params",
    "NormalizationParams",
    "SplitParams",
    "RegridParams",
    "TransformationReport",
    "detect_transformations",
    "RuleExecution",
    "SnakemakeLogReport",
    "parse_snakemake_log",
    "find_latest_log",
    "find_all_logs",
    "NextflowProcess",
    "NextflowWorkflow",
    "NextflowInclude",
    "NextflowParseResult",
    "parse_nextflow_file",
    "map_processes_to_stages",
    "find_nextflow_logs",
    "Suggestion",
    "SuggestionCategory",
    "SuggestionsReport",
    "generate_suggestions",
    "format_suggestions_markdown",
    "format_suggestions_cli",
]
