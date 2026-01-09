"""FAIR Principles definitions and data structures.

This module defines the 15 FAIR principles (F1-F4, A1-A2, I1-I3, R1-R1.3)
and the data structures used to represent assessment scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FAIRCategory(Enum):
    """The four FAIR categories."""
    FINDABLE = "findable"
    ACCESSIBLE = "accessible"
    INTEROPERABLE = "interoperable"
    REUSABLE = "reusable"


class ComplianceLevel(Enum):
    """Compliance levels for FAIR principle scoring.

    Each principle is scored on a 0-3 scale:
    - 0: Not assessed (insufficient information)
    - 1: Not compliant
    - 2: Partially compliant
    - 3: Fully compliant
    """
    NOT_ASSESSED = 0
    NOT_COMPLIANT = 1
    PARTIALLY_COMPLIANT = 2
    FULLY_COMPLIANT = 3


# FAIR Principles definitions
FAIR_PRINCIPLES = {
    # Findable
    "F1": {
        "id": "F1",
        "category": FAIRCategory.FINDABLE,
        "name": "Globally Unique Identifier",
        "description": "(Meta)data are assigned a globally unique and persistent identifier",
        "guidance": "Use DOI, ARK, Handle, or other persistent identifier systems",
    },
    "F2": {
        "id": "F2",
        "category": FAIRCategory.FINDABLE,
        "name": "Rich Metadata",
        "description": "Data are described with rich metadata",
        "guidance": "Provide comprehensive metadata including title, description, keywords, creators, dates",
    },
    "F3": {
        "id": "F3",
        "category": FAIRCategory.FINDABLE,
        "name": "Identifier in Metadata",
        "description": "Metadata clearly and explicitly include the identifier of the data they describe",
        "guidance": "Embed the data identifier within the metadata record itself",
    },
    "F4": {
        "id": "F4",
        "category": FAIRCategory.FINDABLE,
        "name": "Searchable Registration",
        "description": "(Meta)data are registered or indexed in a searchable resource",
        "guidance": "Register in domain repositories, data catalogs, or general registries like DataCite",
    },

    # Accessible
    "A1": {
        "id": "A1",
        "category": FAIRCategory.ACCESSIBLE,
        "name": "Retrievable by Identifier",
        "description": "(Meta)data are retrievable by their identifier using a standardized communications protocol",
        "guidance": "Provide access via standard protocols (HTTPS, FTP, OPeNDAP)",
    },
    "A1.1": {
        "id": "A1.1",
        "category": FAIRCategory.ACCESSIBLE,
        "name": "Open Protocol",
        "description": "The protocol is open, free, and universally implementable",
        "guidance": "Use open protocols like HTTPS rather than proprietary ones",
    },
    "A1.2": {
        "id": "A1.2",
        "category": FAIRCategory.ACCESSIBLE,
        "name": "Authentication Support",
        "description": "The protocol allows for an authentication and authorization procedure, where necessary",
        "guidance": "Document authentication requirements clearly if data access is restricted",
    },
    "A2": {
        "id": "A2",
        "category": FAIRCategory.ACCESSIBLE,
        "name": "Metadata Persistence",
        "description": "Metadata are accessible, even when the data are no longer available",
        "guidance": "Maintain metadata records (landing pages) even if data is removed",
    },

    # Interoperable
    "I1": {
        "id": "I1",
        "category": FAIRCategory.INTEROPERABLE,
        "name": "Formal Language",
        "description": "(Meta)data use a formal, accessible, shared, and broadly applicable language for knowledge representation",
        "guidance": "Use standard file formats (NetCDF, HDF5, JSON-LD) with well-defined schemas",
    },
    "I2": {
        "id": "I2",
        "category": FAIRCategory.INTEROPERABLE,
        "name": "FAIR Vocabularies",
        "description": "(Meta)data use vocabularies that follow FAIR principles",
        "guidance": "Use domain-standard vocabularies (CF Conventions, Dublin Core, Schema.org)",
    },
    "I3": {
        "id": "I3",
        "category": FAIRCategory.INTEROPERABLE,
        "name": "Qualified References",
        "description": "(Meta)data include qualified references to other (meta)data",
        "guidance": "Link to related datasets, publications, or resources using identifiers",
    },

    # Reusable
    "R1": {
        "id": "R1",
        "category": FAIRCategory.REUSABLE,
        "name": "Rich Attributes",
        "description": "(Meta)data are richly described with a plurality of accurate and relevant attributes",
        "guidance": "Include detailed descriptions of variables, units, methods, and provenance",
    },
    "R1.1": {
        "id": "R1.1",
        "category": FAIRCategory.REUSABLE,
        "name": "Clear License",
        "description": "(Meta)data are released with a clear and accessible data usage license",
        "guidance": "Use standard licenses (CC-BY, CC0, MIT) and include license in metadata",
    },
    "R1.2": {
        "id": "R1.2",
        "category": FAIRCategory.REUSABLE,
        "name": "Detailed Provenance",
        "description": "(Meta)data are associated with detailed provenance",
        "guidance": "Document data sources, processing steps, authorship, and creation dates",
    },
    "R1.3": {
        "id": "R1.3",
        "category": FAIRCategory.REUSABLE,
        "name": "Community Standards",
        "description": "(Meta)data meet domain-relevant community standards",
        "guidance": "Follow domain conventions (CF for climate, CIF for crystallography, etc.)",
    },
}


@dataclass
class PrincipleScore:
    """Score for a single FAIR principle."""

    principle_id: str
    category: FAIRCategory
    name: str
    description: str
    score: ComplianceLevel = ComplianceLevel.NOT_ASSESSED
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def score_value(self) -> int:
        """Return numeric score value (0-3)."""
        return self.score.value

    @property
    def score_percentage(self) -> float:
        """Return score as percentage (0-100)."""
        if self.score == ComplianceLevel.NOT_ASSESSED:
            return 0.0
        return (self.score.value / 3.0) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "principle_id": self.principle_id,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "score": self.score.name,
            "score_value": self.score_value,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
        }


@dataclass
class DomainStandardResult:
    """Result of domain-specific standard validation."""

    standard_name: str
    version: Optional[str] = None
    compliance_percentage: float = 0.0
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "standard_name": self.standard_name,
            "version": self.version,
            "compliance_percentage": self.compliance_percentage,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }


@dataclass
class FAIRScore:
    """Complete FAIR assessment scores."""

    dataset_name: str = ""
    dataset_path: str = ""
    domain: str = "general"
    assessment_date: str = field(default_factory=lambda: datetime.now().isoformat())
    assessor_version: str = "1.0.0"

    # Principle scores
    principles: Dict[str, PrincipleScore] = field(default_factory=dict)

    # Aggregated scores
    overall_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Domain-specific standards
    domain_standards: Dict[str, DomainStandardResult] = field(default_factory=dict)

    def calculate_scores(self) -> None:
        """Calculate overall and category scores from principle scores."""
        if not self.principles:
            return

        # Calculate category scores
        category_totals: Dict[str, List[float]] = {
            cat.value: [] for cat in FAIRCategory
        }

        for principle in self.principles.values():
            if principle.score != ComplianceLevel.NOT_ASSESSED:
                category_totals[principle.category.value].append(
                    principle.score.value / 3.0
                )

        # Calculate averages per category
        for category, scores in category_totals.items():
            if scores:
                self.category_scores[category] = sum(scores) / len(scores)
            else:
                self.category_scores[category] = 0.0

        # Calculate overall score (average of category scores)
        assessed_categories = [s for s in self.category_scores.values() if s > 0]
        if assessed_categories:
            self.overall_score = sum(assessed_categories) / len(assessed_categories)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "domain": self.domain,
            "assessment_date": self.assessment_date,
            "assessor_version": self.assessor_version,
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "principles": {k: v.to_dict() for k, v in self.principles.items()},
            "domain_standards": {k: v.to_dict() for k, v in self.domain_standards.items()},
        }

    def to_markdown(self) -> str:
        """Generate markdown report of FAIR assessment."""
        lines = [
            "# FAIR Assessment Report",
            "",
            f"**Dataset:** {self.dataset_name}",
            f"**Path:** {self.dataset_path}",
            f"**Domain:** {self.domain}",
            f"**Assessment Date:** {self.assessment_date[:10]}",
            "",
            f"## Overall Score: {self.overall_score:.0%}",
            "",
            "| Category | Score | Status |",
            "|----------|-------|--------|",
        ]

        # Category scores table
        for category in FAIRCategory:
            score = self.category_scores.get(category.value, 0.0)
            status = self._score_to_status(score)
            lines.append(f"| {category.value.capitalize()} | {score:.0%} | {status} |")

        lines.append("")

        # Detailed principle scores by category
        lines.append("## Detailed Assessment")
        lines.append("")

        for category in FAIRCategory:
            lines.append(f"### {category.value.capitalize()}")
            lines.append("")
            lines.append("| Principle | Score | Evidence |")
            lines.append("|-----------|-------|----------|")

            for pid, principle in self.principles.items():
                if principle.category == category:
                    score_name = principle.score.name.replace("_", " ").title()
                    evidence = "; ".join(principle.evidence[:2]) if principle.evidence else "-"
                    if len(evidence) > 50:
                        evidence = evidence[:47] + "..."
                    lines.append(f"| {pid}: {principle.name} | {score_name} | {evidence} |")

            lines.append("")

        # Domain standards if present
        if self.domain_standards:
            lines.append("## Domain Standards Compliance")
            lines.append("")
            lines.append("| Standard | Compliance |")
            lines.append("|----------|------------|")

            for name, result in self.domain_standards.items():
                version = f" {result.version}" if result.version else ""
                lines.append(f"| {result.standard_name}{version} | {result.compliance_percentage:.0%} |")

            lines.append("")

        # Recommendations
        high_priority = []
        medium_priority = []

        for principle in self.principles.values():
            if principle.score == ComplianceLevel.NOT_COMPLIANT:
                high_priority.extend(principle.recommendations)
            elif principle.score == ComplianceLevel.PARTIALLY_COMPLIANT:
                medium_priority.extend(principle.recommendations)

        if high_priority or medium_priority:
            lines.append("## Recommendations")
            lines.append("")

            if high_priority:
                lines.append("### High Priority")
                lines.append("")
                for i, rec in enumerate(high_priority[:5], 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")

            if medium_priority:
                lines.append("### Medium Priority")
                lines.append("")
                for i, rec in enumerate(medium_priority[:5], 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _score_to_status(score: float) -> str:
        """Convert score to status string."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Needs Improvement"
        elif score > 0:
            return "Poor"
        else:
            return "Not Assessed"


def create_principle_score(principle_id: str) -> PrincipleScore:
    """Create a PrincipleScore from a principle ID."""
    if principle_id not in FAIR_PRINCIPLES:
        raise ValueError(f"Unknown principle ID: {principle_id}")

    p = FAIR_PRINCIPLES[principle_id]
    return PrincipleScore(
        principle_id=p["id"],
        category=p["category"],
        name=p["name"],
        description=p["description"],
    )


def create_all_principle_scores() -> Dict[str, PrincipleScore]:
    """Create PrincipleScore objects for all FAIR principles."""
    return {pid: create_principle_score(pid) for pid in FAIR_PRINCIPLES}
