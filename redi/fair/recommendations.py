"""FAIR Improvement Recommendations.

This module provides detailed recommendations for improving FAIR compliance
based on assessment scores.
"""

from typing import Dict, List, Any

from .principles import ComplianceLevel, FAIRScore, FAIRCategory


# Detailed recommendations for each principle and score level
FAIR_RECOMMENDATIONS: Dict[str, Dict[int, List[str]]] = {
    # Findable
    "F1": {
        1: [
            "Register your dataset with a DOI provider (Zenodo, DataCite, Figshare)",
            "If using institutional repository, ensure it provides persistent identifiers",
            "Add the identifier to your dataset's metadata file",
        ],
        2: [
            "Upgrade from local UUID to a globally resolvable DOI",
            "Ensure identifier follows a recognized standard (DOI, ARK, Handle)",
            "Verify identifier resolves to your dataset landing page",
        ],
    },
    "F2": {
        1: [
            "Create a comprehensive Data Card documenting your dataset",
            "Include: title, description, keywords, creators, dates, format",
            "Add domain-specific metadata (temporal coverage, spatial extent)",
            "Use REDI's --analyze flag to auto-generate initial metadata",
        ],
        2: [
            "Complete remaining sections of your Data Card",
            "Add detailed feature descriptions and units",
            "Include provenance and collection methodology",
        ],
    },
    "F3": {
        1: [
            "Embed the dataset identifier directly in your data file metadata",
            "For NetCDF: add 'id' or 'doi' global attribute",
            "For HDF5: add identifier to root-level attributes",
            "Include identifier in accompanying README or Data Card",
        ],
        2: [
            "Ensure identifier appears in both data file and external metadata",
            "Verify consistency between embedded and registered identifiers",
        ],
    },
    "F4": {
        1: [
            "Register your dataset in a searchable repository:",
            "  - General: Zenodo, Dryad, Figshare",
            "  - Climate: ESGF, Copernicus CDS, NCAR RDA",
            "  - Materials: Materials Project, AFLOW, NOMAD",
            "  - Life Sciences: NCBI, UniProt, PDB",
            "Ensure the repository is indexed by search engines",
        ],
        2: [
            "Verify your dataset appears in repository search results",
            "Add comprehensive keywords to improve discoverability",
            "Consider cross-listing in multiple relevant repositories",
        ],
    },

    # Accessible
    "A1": {
        1: [
            "Provide a standardized access URL for your data",
            "Use HTTPS for web access (preferred)",
            "Consider cloud storage with public URLs (S3, GCS, Azure)",
            "For large datasets, consider OPeNDAP or THREDDS servers",
        ],
        2: [
            "Document the access URL in your metadata",
            "Ensure URL is stable and won't change",
            "Test that the URL works from external networks",
        ],
    },
    "A1.1": {
        1: [
            "Migrate from proprietary protocols to open standards",
            "Use HTTPS for primary access",
            "If using S3/cloud storage, provide presigned URLs or public access",
        ],
        2: [
            "Provide alternative access methods (HTTPS alongside S3)",
            "Document any special client requirements",
        ],
    },
    "A1.2": {
        1: [
            "If authentication is required, document the process clearly",
            "Specify authentication type (API key, OAuth, institutional)",
            "Provide registration/account creation instructions",
        ],
        2: [
            "Add authentication details to your metadata/Data Card",
            "Create a quick-start guide for data access",
        ],
    },
    "A2": {
        1: [
            "Ensure your repository maintains metadata landing pages",
            "Use repositories with data persistence policies",
            "Create a separate metadata record that persists independently",
        ],
        2: [
            "Verify landing page URL resolves correctly",
            "Check repository's data retention policy",
        ],
    },

    # Interoperable
    "I1": {
        1: [
            "Convert data to a standard, self-describing format:",
            "  - Climate: NetCDF-4, Zarr",
            "  - Materials: CIF, HDF5 with schema",
            "  - Life Sciences: PDB, FASTA, validated JSON",
            "  - General: Parquet, HDF5, validated JSON",
            "Avoid proprietary binary formats without documentation",
        ],
        2: [
            "Add schema documentation for your data format",
            "Consider upgrading to more self-describing format",
            "Include format specification reference in metadata",
        ],
    },
    "I2": {
        1: [
            "Use domain-standard vocabularies and ontologies:",
            "  - Climate: CF Conventions standard names",
            "  - Materials: OPTIMADE property definitions",
            "  - Life Sciences: Gene Ontology, UniProt terms",
            "Reference vocabulary URIs in your metadata",
        ],
        2: [
            "Ensure all terms map to vocabulary definitions",
            "Document any custom terms or extensions",
            "Validate against vocabulary schemas where possible",
        ],
    },
    "I3": {
        1: [
            "Add references to related datasets using persistent identifiers",
            "Link to source data, derived products, or related studies",
            "Include citation information for referenced data",
        ],
        2: [
            "Use DOIs or other PIDs for all data references",
            "Specify relationship type (derived from, related to, etc.)",
        ],
    },

    # Reusable
    "R1": {
        1: [
            "Document all dataset attributes comprehensively:",
            "  - Variable names, units, and descriptions",
            "  - Temporal and spatial coverage/resolution",
            "  - Data collection methods and instruments",
            "  - Quality flags and uncertainty estimates",
        ],
        2: [
            "Add missing attribute documentation",
            "Include units for all numerical variables",
            "Document any data transformations applied",
        ],
    },
    "R1.1": {
        1: [
            "Add a clear data usage license to your dataset",
            "Recommended licenses for open data:",
            "  - CC0 1.0: Public domain dedication",
            "  - CC-BY 4.0: Attribution required",
            "  - CC-BY-SA 4.0: Share-alike",
            "Include license in metadata AND data file",
        ],
        2: [
            "Use standard SPDX license identifier",
            "Provide license URL in metadata",
            "Clarify any usage restrictions or conditions",
        ],
    },
    "R1.2": {
        1: [
            "Document dataset provenance comprehensively:",
            "  - Authors/creators with ORCIDs",
            "  - Data sources and original collection",
            "  - Processing pipeline and transformations",
            "  - Software versions used",
            "  - Date of creation/modification",
        ],
        2: [
            "Add ORCIDs for all authors",
            "Include version history if applicable",
            "Document any data cleaning or QC steps",
        ],
    },
    "R1.3": {
        1: [
            "Ensure data meets domain community standards:",
            "  - Climate: CF Conventions, CMIP6 DRS",
            "  - Materials: CIF validation, OPTIMADE compliance",
            "  - Life Sciences: PDB validation, FASTA format",
            "Run validation tools and fix reported issues",
        ],
        2: [
            "Address any validation warnings",
            "Document deviations from standards",
            "Consider community review of your data",
        ],
    },
}

# Domain-specific recommendations
DOMAIN_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "climate": {
        "vocabularies": [
            "Use CF Conventions standard_name attribute for all variables",
            "Follow ACDD (Attribute Convention for Data Discovery) 1.3",
            "Use CMIP6 controlled vocabularies where applicable",
        ],
        "formats": [
            "Use NetCDF-4 with CF Conventions",
            "Consider Zarr for cloud-optimized storage",
            "Include time bounds for temporal aggregations",
        ],
        "repositories": [
            "Submit to ESGF for climate model output",
            "Use Copernicus Climate Data Store for observations",
            "Consider NCAR Research Data Archive",
        ],
    },
    "materials": {
        "vocabularies": [
            "Use OPTIMADE property definitions",
            "Follow Materials Project naming conventions",
            "Include AFLOW UIDs for structure references",
        ],
        "formats": [
            "Use validated CIF for crystal structures",
            "Consider OPTIMADE JSON for database interchange",
            "Include space group and lattice parameters",
        ],
        "repositories": [
            "Submit to Materials Project or AFLOW",
            "Use NOMAD for computational materials data",
            "Consider Zenodo for general materials datasets",
        ],
    },
    "life_sciences": {
        "vocabularies": [
            "Use Gene Ontology terms for biological processes",
            "Follow UniProt naming conventions for proteins",
            "Include taxonomy IDs (NCBI Taxonomy)",
        ],
        "formats": [
            "Use validated PDB format for structures",
            "Follow FASTA standards for sequences",
            "Consider mmCIF for macromolecular data",
        ],
        "repositories": [
            "Submit structures to Protein Data Bank (PDB)",
            "Use NCBI databases for sequences",
            "Consider BioStudies for experimental data",
        ],
    },
    "fusion": {
        "vocabularies": [
            "Follow IMAS (Integrated Modelling & Analysis Suite) schema",
            "Use MDSplus signal naming conventions",
            "Include machine/shot identifiers",
        ],
        "formats": [
            "Use HDF5 with IMAS-compliant schema",
            "Include signal metadata (units, error bars)",
            "Document temporal alignment methods",
        ],
        "repositories": [
            "Submit to ITER IMAS database",
            "Use fusion facility data archives",
            "Consider Zenodo for processed datasets",
        ],
    },
    "hpc": {
        "vocabularies": [
            "Follow scheduler-specific schemas (Slurm, PBS)",
            "Use XSEDE/ACCESS terminology for allocations",
            "Include standard resource metrics",
        ],
        "formats": [
            "Use Parquet for large log datasets",
            "Include timestamps in ISO 8601 format",
            "Document any anonymization applied",
        ],
        "repositories": [
            "Check with your HPC facility for data policies",
            "Use Zenodo for research datasets",
            "Consider Open Science Framework",
        ],
    },
}


class FAIRRecommender:
    """Generate improvement recommendations for FAIR compliance."""

    def __init__(self, fair_score: FAIRScore, domain: str = "general"):
        """Initialize recommender.

        Args:
            fair_score: FAIR assessment results
            domain: Scientific domain for domain-specific recommendations
        """
        self.score = fair_score
        self.domain = domain

    def get_recommendations(self) -> Dict[str, List[str]]:
        """Get all recommendations organized by priority.

        Returns:
            Dictionary with high_priority, medium_priority, and domain_specific lists
        """
        high_priority: List[str] = []
        medium_priority: List[str] = []

        for principle_id, principle in self.score.principles.items():
            if principle.score == ComplianceLevel.NOT_COMPLIANT:
                # Get detailed recommendations for this principle
                recs = FAIR_RECOMMENDATIONS.get(principle_id, {}).get(1, [])
                high_priority.extend(recs)
                # Also include any from the assessment
                high_priority.extend(principle.recommendations)
            elif principle.score == ComplianceLevel.PARTIALLY_COMPLIANT:
                recs = FAIR_RECOMMENDATIONS.get(principle_id, {}).get(2, [])
                medium_priority.extend(recs)
                medium_priority.extend(principle.recommendations)

        # Deduplicate while preserving order
        high_priority = list(dict.fromkeys(high_priority))
        medium_priority = list(dict.fromkeys(medium_priority))

        return {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "domain_specific": self._get_domain_recommendations(),
        }

    def _get_domain_recommendations(self) -> List[str]:
        """Get domain-specific recommendations."""
        if self.domain not in DOMAIN_RECOMMENDATIONS:
            return []

        domain_recs = DOMAIN_RECOMMENDATIONS[self.domain]
        recommendations = []

        # Check which categories need improvement
        for category in FAIRCategory:
            cat_score = self.score.category_scores.get(category.value, 0)
            if cat_score < 0.8:
                # Add relevant domain recommendations
                if category == FAIRCategory.INTEROPERABLE:
                    recommendations.extend(domain_recs.get("vocabularies", []))
                    recommendations.extend(domain_recs.get("formats", []))
                elif category == FAIRCategory.FINDABLE:
                    recommendations.extend(domain_recs.get("repositories", []))
                elif category == FAIRCategory.REUSABLE:
                    recommendations.extend(domain_recs.get("vocabularies", []))

        return list(dict.fromkeys(recommendations))[:10]

    def get_quick_wins(self) -> List[str]:
        """Get easy-to-implement improvements.

        Returns:
            List of quick win recommendations
        """
        quick_wins = []

        # License is usually easy to add
        r11 = self.score.principles.get("R1.1")
        if r11 and r11.score != ComplianceLevel.FULLY_COMPLIANT:
            quick_wins.append(
                "Add a CC-BY-4.0 or CC0 license to your metadata"
            )

        # DOI registration is straightforward
        f1 = self.score.principles.get("F1")
        if f1 and f1.score != ComplianceLevel.FULLY_COMPLIANT:
            quick_wins.append(
                "Register for a DOI via Zenodo (free, takes ~5 minutes)"
            )

        # Metadata completeness
        f2 = self.score.principles.get("F2")
        if f2 and f2.score != ComplianceLevel.FULLY_COMPLIANT:
            quick_wins.append(
                "Run 'redi --analyze' to generate a Data Card template"
            )

        return quick_wins

    def to_markdown(self) -> str:
        """Generate markdown report of recommendations.

        Returns:
            Markdown-formatted recommendations
        """
        recs = self.get_recommendations()
        lines = ["# FAIR Improvement Recommendations", ""]

        # Quick wins
        quick_wins = self.get_quick_wins()
        if quick_wins:
            lines.append("## Quick Wins")
            lines.append("")
            for win in quick_wins:
                lines.append(f"- {win}")
            lines.append("")

        # High priority
        if recs["high_priority"]:
            lines.append("## High Priority")
            lines.append("")
            lines.append("These issues significantly impact FAIR compliance:")
            lines.append("")
            for i, rec in enumerate(recs["high_priority"][:10], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Medium priority
        if recs["medium_priority"]:
            lines.append("## Medium Priority")
            lines.append("")
            lines.append("These improvements would enhance compliance:")
            lines.append("")
            for i, rec in enumerate(recs["medium_priority"][:10], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Domain specific
        if recs["domain_specific"]:
            lines.append(f"## {self.domain.replace('_', ' ').title()} Domain Recommendations")
            lines.append("")
            for rec in recs["domain_specific"]:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)
