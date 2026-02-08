#!/usr/bin/env python3
"""
REDI Data Card Module

Data model for Data Cards: enums, dataclasses, and serialization.
Extracted from analyze.py for separation of concerns.
"""

import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MaturityLevel(Enum):
    """5-level data readiness taxonomy from DRAI paper"""
    RAW = 1
    CLEANED = 2
    LABELED = 3
    FEATURE_ENGINEERED = 4
    FULLY_AI_READY = 5

    def description(self) -> str:
        descriptions = {
            1: "Raw data as originally collected, no processing applied",
            2: "Missing values handled, outliers addressed, basic quality checks passed",
            3: "Target variables defined, annotations added where needed",
            4: "Domain-specific features extracted, transformations applied",
            5: "Sharded, normalized, in AI-optimized format (HDF5/TFRecord/NPZ/Zarr)"
        }
        return descriptions[self.value]


class ScientificDomain(Enum):
    """Scientific domains with domain-specific preprocessing patterns"""
    CLIMATE = "climate"
    MATERIALS = "materials"
    LIFE_SCIENCES = "life_sciences"
    FUSION = "fusion"
    HPC = "hpc"
    GENERAL = "general"


# =============================================================================
# DATA CARD STRUCTURE
# =============================================================================

@dataclass
class DatasetOverview:
    """Section 1: Dataset Overview"""
    name: str = ""
    version: str = ""
    description: str = ""
    purpose: str = ""
    size_bytes: int = 0
    num_records: int = 0
    num_features: int = 0
    file_format: str = ""
    date_created: str = ""
    date_modified: str = ""


@dataclass
class Provenance:
    """Section 2: Provenance"""
    authors: List[str] = field(default_factory=list)
    organization: str = ""
    funding_sources: List[str] = field(default_factory=list)
    collection_method: str = ""
    data_sources: List[str] = field(default_factory=list)
    preprocessing_applied: List[str] = field(default_factory=list)
    license: str = ""
    citation: str = ""


@dataclass
class DatasetCharacteristics:
    """Section 3: Dataset Characteristics"""
    domain: str = ""
    data_types: List[str] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)
    temporal_coverage: str = ""
    spatial_coverage: str = ""
    missing_data_strategy: str = ""
    class_balance: str = ""


@dataclass
class SensitiveAttributes:
    """Section 4: Sensitive Human Attributes"""
    contains_pii: bool = False
    pii_types: List[str] = field(default_factory=list)
    anonymization_applied: bool = False
    anonymization_method: str = ""
    consent_obtained: bool = False
    irb_approved: bool = False


@dataclass
class Transformations:
    """Section 5: Transformations Applied"""
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_engineering: List[str] = field(default_factory=list)
    normalization_method: str = ""
    encoding_methods: List[str] = field(default_factory=list)
    train_val_test_split: str = ""


@dataclass
class Annotations:
    """Section 6: Annotations/Labels"""
    has_labels: bool = False
    label_types: List[str] = field(default_factory=list)
    annotation_method: str = ""
    annotator_info: str = ""
    label_quality_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Limitations:
    """Section 7: Limitations and Caveats"""
    known_biases: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)
    not_suitable_for: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)


@dataclass
class DataCard:
    """Complete Data Card structure"""
    overview: DatasetOverview = field(default_factory=DatasetOverview)
    provenance: Provenance = field(default_factory=Provenance)
    characteristics: DatasetCharacteristics = field(default_factory=DatasetCharacteristics)
    sensitive_attributes: SensitiveAttributes = field(default_factory=SensitiveAttributes)
    transformations: Transformations = field(default_factory=Transformations)
    annotations: Annotations = field(default_factory=Annotations)
    limitations: Limitations = field(default_factory=Limitations)
    readiness_level: int = 1
    detected_domain: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate a markdown representation of the Data Card"""
        md = []
        md.append(f"# Data Card: {self.overview.name or 'Unnamed Dataset'}\n")

        md.append("## 1. Dataset Overview\n")
        md.append(f"- **Name:** {self.overview.name}")
        md.append(f"- **Version:** {self.overview.version}")
        md.append(f"- **Description:** {self.overview.description}")
        md.append(f"- **Purpose:** {self.overview.purpose}")
        md.append(f"- **Size:** {self._format_size(self.overview.size_bytes)}")
        md.append(f"- **Records:** {self.overview.num_records:,}")
        md.append(f"- **Features:** {self.overview.num_features}")
        md.append(f"- **Format:** {self.overview.file_format}\n")

        md.append("## 2. Provenance\n")
        md.append(f"- **Authors:** {', '.join(self.provenance.authors) or 'Not specified'}")
        md.append(f"- **Organization:** {self.provenance.organization}")
        md.append(f"- **Collection Method:** {self.provenance.collection_method}")
        md.append(f"- **Data Sources:** {', '.join(self.provenance.data_sources) or 'Not specified'}")
        md.append(f"- **License:** {self.provenance.license}\n")

        md.append("## 3. Dataset Characteristics\n")
        md.append(f"- **Domain:** {self.characteristics.domain}")
        md.append(f"- **Data Types:** {', '.join(self.characteristics.data_types)}")
        md.append(f"- **Temporal Coverage:** {self.characteristics.temporal_coverage}")
        md.append(f"- **Spatial Coverage:** {self.characteristics.spatial_coverage}")
        md.append(f"- **Missing Data Strategy:** {self.characteristics.missing_data_strategy}\n")

        if self.characteristics.features:
            md.append("### Features\n")
            md.append("| Name | Type | Description |")
            md.append("|------|------|-------------|")
            for feat in self.characteristics.features[:20]:
                md.append(f"| {feat.get('name', '')} | {feat.get('type', '')} | {feat.get('description', '')} |")
            if len(self.characteristics.features) > 20:
                md.append(f"| ... | ... | ({len(self.characteristics.features) - 20} more features) |")
            md.append("")

        md.append("## 4. Sensitive Attributes\n")
        md.append(f"- **Contains PII:** {'Yes' if self.sensitive_attributes.contains_pii else 'No'}")
        if self.sensitive_attributes.contains_pii:
            md.append(f"- **PII Types:** {', '.join(self.sensitive_attributes.pii_types)}")
            md.append(f"- **Anonymization Applied:** {'Yes' if self.sensitive_attributes.anonymization_applied else 'No'}")
            md.append(f"- **Anonymization Method:** {self.sensitive_attributes.anonymization_method}\n")
        else:
            md.append("")

        md.append("## 5. Transformations\n")
        if self.transformations.preprocessing_steps:
            md.append("### Preprocessing Steps")
            for step in self.transformations.preprocessing_steps:
                md.append(f"- {step}")
            md.append("")
        md.append(f"- **Normalization:** {self.transformations.normalization_method}")
        md.append(f"- **Train/Val/Test Split:** {self.transformations.train_val_test_split}\n")

        md.append("## 6. Annotations\n")
        md.append(f"- **Has Labels:** {'Yes' if self.annotations.has_labels else 'No'}")
        if self.annotations.has_labels:
            md.append(f"- **Label Types:** {', '.join(self.annotations.label_types)}")
            md.append(f"- **Annotation Method:** {self.annotations.annotation_method}\n")
        else:
            md.append("")

        md.append("## 7. Limitations\n")
        if self.limitations.known_biases:
            md.append("### Known Biases")
            for bias in self.limitations.known_biases:
                md.append(f"- {bias}")
            md.append("")
        if self.limitations.data_quality_issues:
            md.append("### Data Quality Issues")
            for issue in self.limitations.data_quality_issues:
                md.append(f"- {issue}")
            md.append("")
        if self.limitations.not_suitable_for:
            md.append("### Not Suitable For")
            for item in self.limitations.not_suitable_for:
                md.append(f"- {item}")
            md.append("")

        md.append("## 8. AI Readiness Assessment\n")
        level = MaturityLevel(self.readiness_level)
        md.append(f"- **Current Level:** {level.value} - {level.name}")
        md.append(f"- **Description:** {level.description()}")
        md.append(f"- **Detected Domain:** {self.detected_domain}\n")

        return "\n".join(md)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
