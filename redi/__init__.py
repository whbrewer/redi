"""
REDI - Readiness Engine for Data Integration

A modular pipeline for transforming raw scientific data into AI-ready datasets.
"""

__version__ = "0.1.0"
__author__ = "Wes Brewer"

from .analyze import analyze_dataset, get_recommendations, generate_report, DataCard
from .pipeline import Pipeline, PipelineContext, PipelineStage, PipelineStep, create_context

# Data loaders (optional - requires numpy)
try:
    from .dataloaders import (
        GESTSDataLoader,
        SSTBinaryDataLoader,
        ZarrDataLoader,
        get_dataloader,
        load_sickle_data,
        LoadedData,
        DatasetMetadata,
    )
    _HAS_DATALOADERS = True
except ImportError:
    _HAS_DATALOADERS = False

__all__ = [
    # Analysis
    "analyze_dataset",
    "get_recommendations",
    "generate_report",
    "DataCard",
    # Pipeline
    "Pipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineStep",
    "create_context",
    # Version
    "__version__",
]

# Add dataloader exports if available
if _HAS_DATALOADERS:
    __all__.extend([
        "GESTSDataLoader",
        "SSTBinaryDataLoader",
        "ZarrDataLoader",
        "get_dataloader",
        "load_sickle_data",
        "LoadedData",
        "DatasetMetadata",
    ])
