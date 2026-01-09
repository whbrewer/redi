"""FAIR Assessment module.

This module provides the FAIRAssessor class for evaluating datasets
against FAIR principles.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .principles import (
    ComplianceLevel,
    FAIRCategory,
    FAIRScore,
    PrincipleScore,
    create_all_principle_scores,
)


# Known persistent identifier patterns
DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s]+")
ARK_PATTERN = re.compile(r"ark:/\d{5}/\S+")
HANDLE_PATTERN = re.compile(r"hdl:\d+/\S+")
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# SPDX license identifiers (common ones)
SPDX_LICENSES = {
    "cc0-1.0", "cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0",
    "mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", "bsd-2-clause",
    "lgpl-3.0", "mpl-2.0", "unlicense", "public domain",
}

# File formats ranked by interoperability
FORMAT_INTEROPERABILITY = {
    # High interoperability (formal, well-defined schemas)
    ".nc": 3, ".nc4": 3, ".hdf5": 3, ".h5": 3, ".zarr": 3,
    ".json": 2, ".jsonld": 3, ".xml": 2, ".rdf": 3,
    ".csv": 2, ".tsv": 2, ".parquet": 3,
    ".cif": 3, ".pdb": 3, ".fasta": 2, ".fastq": 2,
    # Lower interoperability
    ".xlsx": 1, ".xls": 1, ".mat": 1, ".pkl": 1, ".pickle": 1,
    ".npy": 2, ".npz": 2,
    # Binary/proprietary
    ".bin": 1, ".dat": 1,
}


@dataclass
class FAIRMetadata:
    """External metadata for complete FAIR assessment.

    This class allows users to provide additional metadata that cannot
    be automatically extracted from the data file itself.
    """

    # Findable
    persistent_identifier: Optional[str] = None
    identifier_type: Optional[str] = None  # "doi", "ark", "handle", "uuid"
    is_registered: bool = False
    registry_url: Optional[str] = None
    registry_name: Optional[str] = None

    # Accessible
    access_url: Optional[str] = None
    access_protocol: Optional[str] = None  # "https", "ftp", "s3", "opendap"
    requires_authentication: bool = False
    authentication_type: Optional[str] = None
    authentication_documented: bool = False
    metadata_landing_page: Optional[str] = None

    # Interoperable
    vocabulary_standards: List[str] = field(default_factory=list)
    references_other_data: List[str] = field(default_factory=list)
    schema_url: Optional[str] = None

    # Reusable
    license: Optional[str] = None
    license_url: Optional[str] = None
    provenance_documented: bool = False
    community_standards: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FAIRMetadata":
        """Create FAIRMetadata from dictionary."""
        return cls(
            persistent_identifier=data.get("persistent_identifier"),
            identifier_type=data.get("identifier_type"),
            is_registered=data.get("is_registered", False),
            registry_url=data.get("registry_url"),
            registry_name=data.get("registry_name"),
            access_url=data.get("access_url"),
            access_protocol=data.get("access_protocol"),
            requires_authentication=data.get("requires_authentication", False),
            authentication_type=data.get("authentication_type"),
            authentication_documented=data.get("authentication_documented", False),
            metadata_landing_page=data.get("metadata_landing_page"),
            vocabulary_standards=data.get("vocabulary_standards", []),
            references_other_data=data.get("references_other_data", []),
            schema_url=data.get("schema_url"),
            license=data.get("license"),
            license_url=data.get("license_url"),
            provenance_documented=data.get("provenance_documented", False),
            community_standards=data.get("community_standards", []),
        )


class FAIRAssessor:
    """Assesses FAIR compliance of datasets."""

    def __init__(self, domain: str = "general"):
        """Initialize FAIR assessor.

        Args:
            domain: Scientific domain for domain-specific standards
        """
        self.domain = domain
        self._validators = {}
        self._load_domain_validators()

    def _load_domain_validators(self) -> None:
        """Load domain-specific validators."""
        try:
            from .validators import get_validator
            self._validators = get_validator(self.domain)
        except ImportError:
            self._validators = {}

    def assess(
        self,
        file_path: str,
        data_card: Optional[Any] = None,
        metadata: Optional[FAIRMetadata] = None,
    ) -> FAIRScore:
        """Run complete FAIR assessment.

        Args:
            file_path: Path to the dataset file
            data_card: Optional DataCard from REDI analysis
            metadata: Optional external metadata for complete assessment

        Returns:
            FAIRScore with assessment results
        """
        path = Path(file_path)
        metadata = metadata or FAIRMetadata()

        # Initialize score
        score = FAIRScore(
            dataset_name=path.name,
            dataset_path=str(path),
            domain=self.domain,
            principles=create_all_principle_scores(),
        )

        # Run assessments for each category
        self._assess_findable(score, path, data_card, metadata)
        self._assess_accessible(score, path, data_card, metadata)
        self._assess_interoperable(score, path, data_card, metadata)
        self._assess_reusable(score, path, data_card, metadata)

        # Run domain-specific standards
        self._assess_domain_standards(score, path, data_card)

        # Calculate aggregated scores
        score.calculate_scores()

        return score

    def _assess_findable(
        self,
        score: FAIRScore,
        path: Path,
        data_card: Optional[Any],
        metadata: FAIRMetadata,
    ) -> None:
        """Assess Findable principles (F1-F4)."""

        # F1: Globally unique persistent identifier
        f1 = score.principles["F1"]
        if metadata.persistent_identifier:
            pid = metadata.persistent_identifier
            if DOI_PATTERN.search(pid):
                f1.score = ComplianceLevel.FULLY_COMPLIANT
                f1.evidence.append(f"DOI found: {pid}")
            elif ARK_PATTERN.search(pid) or HANDLE_PATTERN.search(pid):
                f1.score = ComplianceLevel.FULLY_COMPLIANT
                f1.evidence.append(f"Persistent identifier: {pid}")
            elif UUID_PATTERN.search(pid):
                f1.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f1.evidence.append(f"UUID found: {pid}")
                f1.recommendations.append(
                    "Consider registering for a DOI to improve findability"
                )
            else:
                f1.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f1.evidence.append(f"Identifier provided: {pid}")
        else:
            # Check if file has embedded identifier
            embedded_id = self._check_embedded_identifier(path, data_card)
            if embedded_id:
                f1.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f1.evidence.append(f"Embedded identifier: {embedded_id}")
                f1.recommendations.append(
                    "Register dataset with a DOI provider (Zenodo, DataCite)"
                )
            else:
                f1.score = ComplianceLevel.NOT_COMPLIANT
                f1.evidence.append("No persistent identifier found")
                f1.recommendations.append(
                    "Assign a globally unique persistent identifier (DOI recommended)"
                )

        # F2: Rich metadata
        f2 = score.principles["F2"]
        if data_card:
            completeness = self._calculate_datacard_completeness(data_card)
            if completeness >= 0.8:
                f2.score = ComplianceLevel.FULLY_COMPLIANT
                f2.evidence.append(f"DataCard {completeness:.0%} complete")
            elif completeness >= 0.5:
                f2.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f2.evidence.append(f"DataCard {completeness:.0%} complete")
                f2.recommendations.append(
                    "Complete remaining DataCard sections for richer metadata"
                )
            else:
                f2.score = ComplianceLevel.NOT_COMPLIANT
                f2.evidence.append(f"DataCard only {completeness:.0%} complete")
                f2.recommendations.append(
                    "Add comprehensive metadata: title, description, keywords, authors"
                )
        else:
            # Check basic file metadata
            has_metadata = self._check_file_metadata(path)
            if has_metadata:
                f2.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f2.evidence.append("File contains embedded metadata")
                f2.recommendations.append(
                    "Generate a DataCard for comprehensive metadata documentation"
                )
            else:
                f2.score = ComplianceLevel.NOT_COMPLIANT
                f2.evidence.append("Limited metadata available")
                f2.recommendations.append(
                    "Create rich metadata describing the dataset"
                )

        # F3: Metadata includes identifier
        f3 = score.principles["F3"]
        if metadata.persistent_identifier:
            # Check if identifier is in metadata/DataCard
            if data_card and hasattr(data_card, "overview"):
                f3.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f3.evidence.append("Identifier provided separately from metadata")
                f3.recommendations.append(
                    "Embed identifier directly in dataset metadata"
                )
            else:
                f3.score = ComplianceLevel.PARTIALLY_COMPLIANT
                f3.evidence.append("Identifier provided externally")
        else:
            f3.score = ComplianceLevel.NOT_COMPLIANT
            f3.evidence.append("No identifier to include in metadata")
            f3.recommendations.append(
                "Include dataset identifier in metadata record"
            )

        # F4: Registered in searchable resource
        f4 = score.principles["F4"]
        if metadata.is_registered and metadata.registry_url:
            f4.score = ComplianceLevel.FULLY_COMPLIANT
            registry = metadata.registry_name or "registry"
            f4.evidence.append(f"Registered in {registry}")
        elif metadata.is_registered:
            f4.score = ComplianceLevel.PARTIALLY_COMPLIANT
            f4.evidence.append("Reported as registered")
            f4.recommendations.append(
                "Provide registry URL for verification"
            )
        else:
            f4.score = ComplianceLevel.NOT_COMPLIANT
            f4.evidence.append("Not registered in searchable resource")
            f4.recommendations.append(
                "Register in domain repository (Zenodo, Dryad, domain-specific)"
            )

    def _assess_accessible(
        self,
        score: FAIRScore,
        path: Path,
        data_card: Optional[Any],
        metadata: FAIRMetadata,
    ) -> None:
        """Assess Accessible principles (A1-A2)."""

        # A1: Retrievable by identifier using standardized protocol
        a1 = score.principles["A1"]
        if metadata.access_url:
            protocol = metadata.access_protocol or self._detect_protocol(
                metadata.access_url
            )
            if protocol in ("https", "http", "ftp", "s3", "opendap"):
                a1.score = ComplianceLevel.FULLY_COMPLIANT
                a1.evidence.append(f"Accessible via {protocol.upper()}")
            else:
                a1.score = ComplianceLevel.PARTIALLY_COMPLIANT
                a1.evidence.append(f"Access URL provided: {metadata.access_url}")
        elif path.exists():
            a1.score = ComplianceLevel.PARTIALLY_COMPLIANT
            a1.evidence.append("Data accessible via local file path")
            a1.recommendations.append(
                "Provide standardized access URL (HTTPS preferred)"
            )
        else:
            a1.score = ComplianceLevel.NOT_COMPLIANT
            a1.evidence.append("No access method documented")
            a1.recommendations.append(
                "Document how to retrieve data using standard protocol"
            )

        # A1.1: Protocol is open, free, universally implementable
        a11 = score.principles["A1.1"]
        protocol = metadata.access_protocol or (
            self._detect_protocol(metadata.access_url)
            if metadata.access_url
            else None
        )
        if protocol in ("https", "http"):
            a11.score = ComplianceLevel.FULLY_COMPLIANT
            a11.evidence.append("Uses open HTTPS protocol")
        elif protocol in ("ftp", "s3", "opendap"):
            a11.score = ComplianceLevel.PARTIALLY_COMPLIANT
            a11.evidence.append(f"Uses {protocol.upper()} protocol")
            if protocol == "s3":
                a11.recommendations.append(
                    "Consider providing HTTPS access alongside S3"
                )
        elif path.exists():
            a11.score = ComplianceLevel.PARTIALLY_COMPLIANT
            a11.evidence.append("Local file system access")
        else:
            a11.score = ComplianceLevel.NOT_ASSESSED
            a11.evidence.append("Protocol not specified")

        # A1.2: Protocol allows authentication where needed
        a12 = score.principles["A1.2"]
        if not metadata.requires_authentication:
            a12.score = ComplianceLevel.FULLY_COMPLIANT
            a12.evidence.append("No authentication required (open access)")
        elif metadata.authentication_documented:
            a12.score = ComplianceLevel.FULLY_COMPLIANT
            auth_type = metadata.authentication_type or "documented"
            a12.evidence.append(f"Authentication method: {auth_type}")
        elif metadata.requires_authentication:
            a12.score = ComplianceLevel.PARTIALLY_COMPLIANT
            a12.evidence.append("Authentication required but not documented")
            a12.recommendations.append(
                "Document authentication requirements and procedure"
            )
        else:
            a12.score = ComplianceLevel.NOT_ASSESSED
            a12.evidence.append("Authentication requirements unknown")

        # A2: Metadata accessible even when data unavailable
        a2 = score.principles["A2"]
        if metadata.metadata_landing_page:
            a2.score = ComplianceLevel.FULLY_COMPLIANT
            a2.evidence.append("Metadata landing page exists")
        elif metadata.is_registered:
            a2.score = ComplianceLevel.PARTIALLY_COMPLIANT
            a2.evidence.append("Registered (likely has landing page)")
            a2.recommendations.append(
                "Verify metadata landing page persists"
            )
        else:
            a2.score = ComplianceLevel.NOT_ASSESSED
            a2.evidence.append("Metadata persistence unknown")
            a2.recommendations.append(
                "Ensure metadata remains accessible via landing page"
            )

    def _assess_interoperable(
        self,
        score: FAIRScore,
        path: Path,
        data_card: Optional[Any],
        metadata: FAIRMetadata,
    ) -> None:
        """Assess Interoperable principles (I1-I3)."""

        # I1: Formal, accessible, shared language
        i1 = score.principles["I1"]
        ext = path.suffix.lower()
        interop_score = FORMAT_INTEROPERABILITY.get(ext, 1)

        if interop_score >= 3:
            i1.score = ComplianceLevel.FULLY_COMPLIANT
            i1.evidence.append(f"Uses well-defined format: {ext}")
        elif interop_score >= 2:
            i1.score = ComplianceLevel.PARTIALLY_COMPLIANT
            i1.evidence.append(f"Uses standard format: {ext}")
            i1.recommendations.append(
                "Consider self-describing format (NetCDF, HDF5) for better interoperability"
            )
        else:
            i1.score = ComplianceLevel.NOT_COMPLIANT
            i1.evidence.append(f"Format has limited interoperability: {ext}")
            i1.recommendations.append(
                "Convert to standard format with well-defined schema"
            )

        # I2: FAIR vocabularies
        i2 = score.principles["I2"]
        if metadata.vocabulary_standards:
            i2.score = ComplianceLevel.FULLY_COMPLIANT
            standards = ", ".join(metadata.vocabulary_standards[:3])
            i2.evidence.append(f"Uses vocabularies: {standards}")
        else:
            # Check domain-specific defaults
            domain_vocabs = self._get_domain_vocabularies()
            if domain_vocabs:
                i2.score = ComplianceLevel.PARTIALLY_COMPLIANT
                i2.evidence.append("Domain may have standard vocabularies")
                i2.recommendations.append(
                    f"Verify compliance with: {', '.join(domain_vocabs)}"
                )
            else:
                i2.score = ComplianceLevel.NOT_ASSESSED
                i2.evidence.append("Vocabulary standards not specified")
                i2.recommendations.append(
                    "Use domain-standard vocabularies for metadata"
                )

        # I3: Qualified references to other data
        i3 = score.principles["I3"]
        if metadata.references_other_data:
            i3.score = ComplianceLevel.FULLY_COMPLIANT
            i3.evidence.append(
                f"References {len(metadata.references_other_data)} other datasets"
            )
        elif data_card and hasattr(data_card, "provenance"):
            sources = getattr(data_card.provenance, "data_sources", [])
            if sources:
                i3.score = ComplianceLevel.PARTIALLY_COMPLIANT
                i3.evidence.append(f"Data sources listed: {len(sources)}")
                i3.recommendations.append(
                    "Use persistent identifiers for data references"
                )
            else:
                i3.score = ComplianceLevel.NOT_COMPLIANT
                i3.evidence.append("No references to related data")
                i3.recommendations.append(
                    "Link to related datasets using identifiers"
                )
        else:
            i3.score = ComplianceLevel.NOT_ASSESSED
            i3.evidence.append("References not documented")

    def _assess_reusable(
        self,
        score: FAIRScore,
        path: Path,
        data_card: Optional[Any],
        metadata: FAIRMetadata,
    ) -> None:
        """Assess Reusable principles (R1-R1.3)."""

        # R1: Rich attributes
        r1 = score.principles["R1"]
        if data_card:
            # Check characteristics completeness
            has_features = bool(
                getattr(data_card.characteristics, "features", None)
            )
            has_temporal = bool(
                getattr(data_card.characteristics, "temporal_coverage", "")
            )
            has_spatial = bool(
                getattr(data_card.characteristics, "spatial_coverage", "")
            )
            attr_count = sum([has_features, has_temporal, has_spatial])

            if attr_count >= 2:
                r1.score = ComplianceLevel.FULLY_COMPLIANT
                r1.evidence.append("Rich dataset attributes documented")
            elif attr_count >= 1:
                r1.score = ComplianceLevel.PARTIALLY_COMPLIANT
                r1.evidence.append("Some attributes documented")
                r1.recommendations.append(
                    "Add more descriptive attributes (temporal, spatial coverage)"
                )
            else:
                r1.score = ComplianceLevel.NOT_COMPLIANT
                r1.evidence.append("Limited attributes")
                r1.recommendations.append(
                    "Document dataset attributes comprehensively"
                )
        else:
            r1.score = ComplianceLevel.NOT_ASSESSED
            r1.evidence.append("Attributes not documented")

        # R1.1: Clear license
        r11 = score.principles["R1.1"]
        license_str = metadata.license or (
            getattr(data_card.provenance, "license", "") if data_card else ""
        )
        if license_str:
            license_lower = license_str.lower().replace(" ", "-")
            if any(spdx in license_lower for spdx in SPDX_LICENSES):
                r11.score = ComplianceLevel.FULLY_COMPLIANT
                r11.evidence.append(f"SPDX license: {license_str}")
            else:
                r11.score = ComplianceLevel.PARTIALLY_COMPLIANT
                r11.evidence.append(f"License stated: {license_str}")
                r11.recommendations.append(
                    "Use standard SPDX license identifier"
                )
        else:
            r11.score = ComplianceLevel.NOT_COMPLIANT
            r11.evidence.append("No license specified")
            r11.recommendations.append(
                "Add clear data usage license (CC-BY-4.0, CC0, etc.)"
            )

        # R1.2: Detailed provenance
        r12 = score.principles["R1.2"]
        if data_card and hasattr(data_card, "provenance"):
            prov = data_card.provenance
            has_authors = bool(getattr(prov, "authors", []))
            has_sources = bool(getattr(prov, "data_sources", []))
            has_method = bool(getattr(prov, "collection_method", ""))
            prov_count = sum([has_authors, has_sources, has_method])

            if prov_count >= 2:
                r12.score = ComplianceLevel.FULLY_COMPLIANT
                r12.evidence.append("Detailed provenance documented")
            elif prov_count >= 1:
                r12.score = ComplianceLevel.PARTIALLY_COMPLIANT
                r12.evidence.append("Partial provenance documented")
                r12.recommendations.append(
                    "Add authors, sources, and collection method"
                )
            else:
                r12.score = ComplianceLevel.NOT_COMPLIANT
                r12.evidence.append("Provenance not documented")
                r12.recommendations.append(
                    "Document data provenance (authors, sources, methods)"
                )
        elif metadata.provenance_documented:
            r12.score = ComplianceLevel.PARTIALLY_COMPLIANT
            r12.evidence.append("Provenance reported as documented")
        else:
            r12.score = ComplianceLevel.NOT_COMPLIANT
            r12.evidence.append("No provenance information")
            r12.recommendations.append(
                "Document data provenance comprehensively"
            )

        # R1.3: Community standards
        r13 = score.principles["R1.3"]
        if metadata.community_standards:
            r13.score = ComplianceLevel.FULLY_COMPLIANT
            standards = ", ".join(metadata.community_standards[:2])
            r13.evidence.append(f"Follows: {standards}")
        else:
            # Domain-specific check
            domain_standards = self._get_domain_standards()
            if domain_standards:
                r13.score = ComplianceLevel.PARTIALLY_COMPLIANT
                r13.evidence.append("Domain standards available")
                r13.recommendations.append(
                    f"Verify compliance with: {', '.join(domain_standards)}"
                )
            else:
                r13.score = ComplianceLevel.NOT_ASSESSED
                r13.evidence.append("Community standards not specified")

    def _assess_domain_standards(
        self,
        score: FAIRScore,
        path: Path,
        data_card: Optional[Any],
    ) -> None:
        """Run domain-specific standard validations."""
        if not self._validators:
            return

        for validator in self._validators.values():
            try:
                result = validator.validate(path, data_card)
                if result:
                    score.domain_standards[result.standard_name] = result
            except Exception:
                pass  # Skip failed validations

    def _check_embedded_identifier(
        self, path: Path, data_card: Optional[Any]
    ) -> Optional[str]:
        """Check for identifier embedded in file or DataCard."""
        # Check DataCard first
        if data_card and hasattr(data_card, "overview"):
            name = getattr(data_card.overview, "name", "")
            if DOI_PATTERN.search(name) or UUID_PATTERN.search(name):
                return name

        # Try to extract from common metadata locations
        try:
            ext = path.suffix.lower()
            if ext in (".nc", ".nc4"):
                import xarray as xr
                ds = xr.open_dataset(path)
                for attr in ("id", "uuid", "doi", "identifier"):
                    if attr in ds.attrs:
                        return str(ds.attrs[attr])
                ds.close()
            elif ext in (".h5", ".hdf5"):
                import h5py
                with h5py.File(path, "r") as f:
                    for attr in ("id", "uuid", "doi", "identifier"):
                        if attr in f.attrs:
                            return str(f.attrs[attr])
        except Exception:
            pass

        return None

    def _calculate_datacard_completeness(self, data_card: Any) -> float:
        """Calculate DataCard completeness as a percentage."""
        total_fields = 0
        filled_fields = 0

        sections = [
            "overview",
            "provenance",
            "characteristics",
            "sensitive_attributes",
            "transformations",
            "annotations",
            "limitations",
        ]

        for section_name in sections:
            section = getattr(data_card, section_name, None)
            if section:
                for field_name, value in vars(section).items():
                    if not field_name.startswith("_"):
                        total_fields += 1
                        if value and value not in ([], {}, "", 0, False):
                            filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    def _check_file_metadata(self, path: Path) -> bool:
        """Check if file has embedded metadata."""
        try:
            ext = path.suffix.lower()
            if ext in (".nc", ".nc4"):
                import xarray as xr
                ds = xr.open_dataset(path)
                has_attrs = len(ds.attrs) > 0
                ds.close()
                return has_attrs
            elif ext in (".h5", ".hdf5"):
                import h5py
                with h5py.File(path, "r") as f:
                    return len(f.attrs) > 0
            elif ext in (".json", ".geojson"):
                import json
                with open(path) as f:
                    data = json.load(f)
                return "metadata" in data or "properties" in data
        except Exception:
            pass
        return False

    def _detect_protocol(self, url: str) -> Optional[str]:
        """Detect protocol from URL."""
        if not url:
            return None
        url_lower = url.lower()
        if url_lower.startswith("https://"):
            return "https"
        elif url_lower.startswith("http://"):
            return "http"
        elif url_lower.startswith("ftp://"):
            return "ftp"
        elif url_lower.startswith("s3://"):
            return "s3"
        elif "opendap" in url_lower or "/thredds/" in url_lower:
            return "opendap"
        return None

    def _get_domain_vocabularies(self) -> List[str]:
        """Get recommended vocabularies for domain."""
        domain_vocabs = {
            "climate": ["CF Conventions", "ACDD", "CMIP6 CV"],
            "materials": ["OPTIMADE", "CIF dictionary", "AFLOW schema"],
            "life_sciences": ["UniProt", "Gene Ontology", "FHIR"],
            "fusion": ["IMAS", "MDSplus schema"],
            "hpc": ["XSEDE schema", "PBS schema"],
        }
        return domain_vocabs.get(self.domain, [])

    def _get_domain_standards(self) -> List[str]:
        """Get community standards for domain."""
        domain_standards = {
            "climate": ["CF Conventions", "CMIP6 DRS"],
            "materials": ["CIF standard", "OPTIMADE API"],
            "life_sciences": ["PDB format", "FASTA standard"],
            "fusion": ["IMAS"],
            "hpc": ["PBS/Slurm schemas"],
        }
        return domain_standards.get(self.domain, [])


def load_fair_metadata(config_path: str) -> FAIRMetadata:
    """Load FAIR metadata from YAML configuration file.

    Args:
        config_path: Path to YAML file with FAIR metadata

    Returns:
        FAIRMetadata instance
    """
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"FAIR config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Handle nested 'fair' key
    if "fair" in data:
        data = data["fair"]

    return FAIRMetadata.from_dict(data)
