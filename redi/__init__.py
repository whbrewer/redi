"""
REDI - Readiness Engine for Data Integration

A modular pipeline for transforming raw scientific data into AI-ready datasets.
"""

__version__ = "0.1.0"
__author__ = "Wes Brewer"

from .analyze import analyze_dataset, get_recommendations, generate_report, DataCard
from .pipeline import Pipeline, PipelineContext, PipelineStage, PipelineStep, create_context

__all__ = [
    "analyze_dataset",
    "get_recommendations",
    "generate_report",
    "DataCard",
    "Pipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineStep",
    "create_context",
    "__version__",
]
