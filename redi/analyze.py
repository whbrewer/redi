#!/usr/bin/env python3
"""
REDI Analysis Module — backwards-compatibility shim.

The implementation has moved to:
  - redi.datacard  (data model: enums, dataclasses, DataCard)
  - redi.assess    (analysis logic: DataAnalyzer, PipelineRecommender, convenience functions)

This module re-exports everything so existing ``from redi.analyze import X``
statements continue to work.
"""

# Data model
from .datacard import (  # noqa: F401
    MaturityLevel,
    ScientificDomain,
    DatasetOverview,
    Provenance,
    DatasetCharacteristics,
    SensitiveAttributes,
    Transformations,
    Annotations,
    Limitations,
    DataCard,
)

# Assessment logic
from .assess import (  # noqa: F401
    DOMAIN_FILE_PATTERNS,
    DOMAIN_PIPELINES,
    DataAnalyzer,
    PipelineRecommender,
    analyze_dataset,
    get_recommendations,
    generate_report,
)
