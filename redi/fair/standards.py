"""Domain-specific FAIR standards registry.

This module defines domain-specific standards and provides lookup functions.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StandardDefinition:
    """Definition of a domain-specific standard."""

    name: str
    version: Optional[str] = None
    description: str = ""
    url: Optional[str] = None
    required_checks: List[str] = field(default_factory=list)
    optional_checks: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=list)


# Climate domain standards
CLIMATE_STANDARDS: Dict[str, StandardDefinition] = {
    "cf_conventions": StandardDefinition(
        name="CF Conventions",
        version="1.10",
        description="Climate and Forecast Conventions for NetCDF data",
        url="https://cfconventions.org/",
        required_checks=[
            "conventions_attribute",
            "standard_name_table",
            "units_compliant",
            "coordinate_variables",
            "axis_attribute",
        ],
        optional_checks=[
            "bounds_variables",
            "cell_methods",
            "ancillary_variables",
            "flag_values",
        ],
        file_extensions=[".nc", ".nc4"],
    ),
    "acdd": StandardDefinition(
        name="Attribute Convention for Data Discovery",
        version="1.3",
        description="Metadata attributes for dataset discovery",
        url="https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3",
        required_checks=[
            "title",
            "summary",
            "keywords",
            "conventions",
            "creator_name",
            "institution",
            "date_created",
        ],
        optional_checks=[
            "creator_email",
            "creator_url",
            "publisher_name",
            "geospatial_bounds",
            "time_coverage_start",
            "time_coverage_end",
            "license",
            "doi",
        ],
        file_extensions=[".nc", ".nc4"],
    ),
    "cmip6_drs": StandardDefinition(
        name="CMIP6 Data Reference Syntax",
        description="Naming and directory conventions for CMIP6 data",
        url="https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6aT7nD9O4R9mDk",
        required_checks=[
            "filename_convention",
            "global_attributes",
            "variable_attributes",
        ],
        file_extensions=[".nc"],
    ),
}

# Materials domain standards
MATERIALS_STANDARDS: Dict[str, StandardDefinition] = {
    "cif": StandardDefinition(
        name="Crystallographic Information File",
        version="2.0",
        description="Standard format for crystallographic data",
        url="https://www.iucr.org/resources/cif",
        required_checks=[
            "valid_syntax",
            "data_block",
            "cell_parameters",
            "space_group",
            "atom_sites",
        ],
        optional_checks=[
            "symmetry_operations",
            "aniso_displacement",
            "publication_info",
        ],
        file_extensions=[".cif"],
    ),
    "optimade": StandardDefinition(
        name="OPTIMADE API",
        version="1.1.0",
        description="Open Databases Integration for Materials Design",
        url="https://www.optimade.org/",
        required_checks=[
            "structure_id",
            "structure_type",
            "chemical_formula",
            "lattice_vectors",
            "cartesian_site_positions",
            "species",
        ],
        optional_checks=[
            "nelements",
            "elements",
            "dimension_types",
            "nperiodic_dimensions",
        ],
    ),
    "aflow": StandardDefinition(
        name="AFLOW Schema",
        description="Automatic FLOW for Materials Discovery database schema",
        url="http://aflowlib.org/",
        required_checks=[
            "auid",
            "compound",
            "spacegroup",
        ],
        optional_checks=[
            "enthalpy_formation",
            "energy_cell",
            "volume_cell",
            "pressure",
        ],
    ),
}

# Life sciences domain standards
LIFE_SCIENCES_STANDARDS: Dict[str, StandardDefinition] = {
    "pdb": StandardDefinition(
        name="Protein Data Bank Format",
        version="3.3",
        description="Standard format for macromolecular structures",
        url="https://www.wwpdb.org/documentation/file-format",
        required_checks=[
            "header_record",
            "title_record",
            "atom_records",
            "end_record",
        ],
        optional_checks=[
            "compnd_record",
            "source_record",
            "author_record",
            "hetatm_records",
            "conect_records",
        ],
        file_extensions=[".pdb", ".ent"],
    ),
    "fasta": StandardDefinition(
        name="FASTA Format",
        description="Standard format for biological sequences",
        url="https://www.ncbi.nlm.nih.gov/BLAST/fasta.shtml",
        required_checks=[
            "header_line",
            "valid_sequence",
            "valid_alphabet",
        ],
        optional_checks=[
            "sequence_id",
            "description",
        ],
        file_extensions=[".fasta", ".fa", ".fna", ".faa"],
    ),
    "mmcif": StandardDefinition(
        name="Macromolecular CIF",
        description="Extended CIF for macromolecular structures",
        url="https://mmcif.wwpdb.org/",
        required_checks=[
            "entry_id",
            "cell",
            "atom_site",
        ],
        optional_checks=[
            "entity",
            "struct",
            "citation",
        ],
        file_extensions=[".cif", ".mmcif"],
    ),
}

# Fusion domain standards
FUSION_STANDARDS: Dict[str, StandardDefinition] = {
    "imas": StandardDefinition(
        name="ITER Integrated Modelling & Analysis Suite",
        description="Standard data model for fusion data",
        url="https://imas.iter.org/",
        required_checks=[
            "ids_structure",
            "time_array",
            "shot_number",
        ],
        optional_checks=[
            "provenance",
            "code_parameters",
            "equilibrium_data",
        ],
        file_extensions=[".h5", ".hdf5"],
    ),
    "mdsplus": StandardDefinition(
        name="MDSplus",
        description="Data system for fusion experiments",
        url="https://www.mdsplus.org/",
        required_checks=[
            "tree_structure",
            "signal_data",
        ],
        optional_checks=[
            "units",
            "dimensions",
            "expressions",
        ],
    ),
}

# HPC domain standards
HPC_STANDARDS: Dict[str, StandardDefinition] = {
    "slurm_sacct": StandardDefinition(
        name="Slurm Accounting",
        description="Slurm job accounting data format",
        url="https://slurm.schedmd.com/sacct.html",
        required_checks=[
            "job_id",
            "user",
            "state",
            "submit_time",
            "start_time",
            "end_time",
        ],
        optional_checks=[
            "partition",
            "nodes",
            "cpus",
            "memory",
            "timelimit",
        ],
    ),
    "pbs": StandardDefinition(
        name="PBS Professional Accounting",
        description="PBS job accounting data format",
        url="https://www.altair.com/pbs-professional/",
        required_checks=[
            "job_id",
            "user",
            "queue",
            "ctime",
            "qtime",
        ],
        optional_checks=[
            "resources_used",
            "exit_status",
        ],
    ),
}

# Registry of all domain standards
DOMAIN_STANDARDS_REGISTRY: Dict[str, Dict[str, StandardDefinition]] = {
    "climate": CLIMATE_STANDARDS,
    "materials": MATERIALS_STANDARDS,
    "life_sciences": LIFE_SCIENCES_STANDARDS,
    "fusion": FUSION_STANDARDS,
    "hpc": HPC_STANDARDS,
    "general": {},
}


def get_domain_standards(domain: str) -> Dict[str, StandardDefinition]:
    """Get standards for a specific domain.

    Args:
        domain: Scientific domain name

    Returns:
        Dictionary of standard definitions
    """
    return DOMAIN_STANDARDS_REGISTRY.get(domain.lower(), {})


def get_standard(domain: str, standard_name: str) -> Optional[StandardDefinition]:
    """Get a specific standard definition.

    Args:
        domain: Scientific domain name
        standard_name: Name of the standard

    Returns:
        StandardDefinition or None if not found
    """
    domain_standards = get_domain_standards(domain)
    return domain_standards.get(standard_name.lower().replace(" ", "_"))


def get_standards_for_extension(extension: str) -> List[StandardDefinition]:
    """Get all standards applicable to a file extension.

    Args:
        extension: File extension (e.g., ".nc")

    Returns:
        List of applicable standards
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"

    applicable = []
    for domain_standards in DOMAIN_STANDARDS_REGISTRY.values():
        for standard in domain_standards.values():
            if ext in standard.file_extensions:
                applicable.append(standard)

    return applicable


def list_all_standards() -> List[Dict[str, Any]]:
    """List all registered standards.

    Returns:
        List of standard information dictionaries
    """
    standards = []
    for domain, domain_standards in DOMAIN_STANDARDS_REGISTRY.items():
        for name, standard in domain_standards.items():
            standards.append({
                "domain": domain,
                "key": name,
                "name": standard.name,
                "version": standard.version,
                "description": standard.description,
                "url": standard.url,
            })
    return standards
