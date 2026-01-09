"""FAIR Assessment Module for REDI.

This module provides FAIR (Findable, Accessible, Interoperable, Reusable)
principles assessment for scientific datasets.

Example usage:

    from redi.fair import assess_fair, FAIRAssessor, FAIRMetadata

    # Quick assessment
    score = assess_fair("data.nc")
    print(score.to_markdown())

    # With external metadata
    metadata = FAIRMetadata(
        persistent_identifier="10.5281/zenodo.123456",
        license="CC-BY-4.0"
    )
    score = assess_fair("data.nc", metadata={"persistent_identifier": "10.5281/zenodo.123456"})

    # Full control
    assessor = FAIRAssessor(domain="climate")
    score = assessor.assess("data.nc", metadata=metadata)
"""

from typing import Any, Dict, Optional

from .principles import (
    ComplianceLevel,
    DomainStandardResult,
    FAIRCategory,
    FAIRScore,
    FAIR_PRINCIPLES,
    PrincipleScore,
    create_all_principle_scores,
    create_principle_score,
)
from .assessment import FAIRAssessor, FAIRMetadata, load_fair_metadata
from .recommendations import FAIRRecommender, FAIR_RECOMMENDATIONS
from .standards import (
    StandardDefinition,
    get_domain_standards,
    get_standard,
    get_standards_for_extension,
    list_all_standards,
)


__all__ = [
    # Core classes
    "FAIRAssessor",
    "FAIRMetadata",
    "FAIRScore",
    "FAIRRecommender",
    # Enums and types
    "ComplianceLevel",
    "FAIRCategory",
    "PrincipleScore",
    "DomainStandardResult",
    "StandardDefinition",
    # Constants
    "FAIR_PRINCIPLES",
    "FAIR_RECOMMENDATIONS",
    # Functions
    "assess_fair",
    "generate_fair_report",
    "load_fair_metadata",
    "create_principle_score",
    "create_all_principle_scores",
    "get_domain_standards",
    "get_standard",
    "get_standards_for_extension",
    "list_all_standards",
]


def assess_fair(
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
) -> FAIRScore:
    """Assess FAIR compliance of a dataset.

    This is a convenience function that combines DataCard generation
    and FAIR assessment in a single call.

    Args:
        file_path: Path to the dataset file
        metadata: Optional dictionary with FAIR metadata (DOI, license, etc.)
        domain: Optional domain override (auto-detected if not provided)

    Returns:
        FAIRScore with assessment results

    Example:
        >>> score = assess_fair("data.nc")
        >>> print(f"Overall: {score.overall_score:.0%}")

        >>> score = assess_fair(
        ...     "data.nc",
        ...     metadata={"persistent_identifier": "10.5281/zenodo.123456"}
        ... )
    """
    # Try to get DataCard from REDI analysis
    data_card = None
    detected_domain = domain

    try:
        from ..analyze import DataAnalyzer
        analyzer = DataAnalyzer(file_path)
        data_card = analyzer.analyze()
        if not detected_domain:
            detected_domain = data_card.detected_domain
    except Exception:
        pass  # DataCard is optional

    # Default domain
    if not detected_domain:
        detected_domain = "general"

    # Create metadata object
    fair_metadata = None
    if metadata:
        fair_metadata = FAIRMetadata.from_dict(metadata)

    # Run assessment
    assessor = FAIRAssessor(domain=detected_domain)
    return assessor.assess(
        file_path=file_path,
        data_card=data_card,
        metadata=fair_metadata,
    )


def generate_fair_report(
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    include_recommendations: bool = True,
) -> str:
    """Generate a complete FAIR assessment report in markdown format.

    Args:
        file_path: Path to the dataset file
        metadata: Optional dictionary with FAIR metadata
        domain: Optional domain override
        include_recommendations: Include detailed recommendations

    Returns:
        Markdown-formatted FAIR report

    Example:
        >>> report = generate_fair_report("data.nc")
        >>> print(report)
    """
    score = assess_fair(file_path, metadata=metadata, domain=domain)
    report = score.to_markdown()

    if include_recommendations:
        recommender = FAIRRecommender(score, domain=score.domain)
        report += "\n\n" + recommender.to_markdown()

    return report
