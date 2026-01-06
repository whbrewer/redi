"""
DRAI Domains Module

Domain-specific preprocessing handlers for:
- Climate (xarray, xesmf, NetCDF, Zarr)
- Materials Science (pymatgen, matminer, torch_geometric)
- Life Sciences (biopython, presidio)
- Fusion Energy (MDSplus, ADIOS, h5py)
- HPC/Operational (pandas, sklearn)
"""

from typing import Dict, List, Type, Optional
from enum import Enum


class Domain(Enum):
    """Supported scientific domains"""
    CLIMATE = "climate"
    MATERIALS = "materials"
    LIFE_SCIENCES = "life_sciences"
    FUSION = "fusion"
    HPC = "hpc"
    GENERAL = "general"


# Domain detection keywords
DOMAIN_KEYWORDS: Dict[Domain, List[str]] = {
    Domain.CLIMATE: [
        "temperature", "precipitation", "wind", "pressure", "humidity",
        "geopotential", "lat", "lon", "latitude", "longitude",
        "era5", "cmip", "sst", "toa", "olr", "u10", "v10", "t2m"
    ],
    Domain.MATERIALS: [
        "compound", "formula", "lattice", "crystal", "band_gap", "enthalpy",
        "formation_energy", "density", "auid", "aflow", "element", "magpie",
        "mendeleev", "atomic", "electronegativity", "valence", "cif",
        "structure", "spacegroup", "composition", "alloy", "material"
    ],
    Domain.LIFE_SCIENCES: [
        "sequence", "protein", "gene", "amino", "nucleotide", "residue",
        "chain", "patient", "sample", "alignment", "fasta", "pdb",
        "dna", "rna", "genome", "transcript", "variant"
    ],
    Domain.FUSION: [
        "plasma", "disruption", "tokamak", "magnetic", "confinement",
        "shot", "pulse", "diagnostic", "equilibrium", "density",
        "ip", "bt", "ne", "te", "ti", "q95", "betap"
    ],
    Domain.HPC: [
        "job", "node", "core", "walltime", "runtime", "queue", "allocation",
        "user", "project", "exit_code", "gpu", "cobalt", "slurm", "pbs",
        "cpu", "memory", "submit", "start", "end", "wait"
    ]
}

# Domain file extensions
DOMAIN_EXTENSIONS: Dict[Domain, List[str]] = {
    Domain.CLIMATE: [".nc", ".nc4", ".grib", ".grib2", ".zarr"],
    Domain.MATERIALS: [".cif", ".poscar", ".vasp", ".xyz", ".pdb"],
    Domain.LIFE_SCIENCES: [".fasta", ".fastq", ".pdb", ".mmcif", ".bam", ".vcf"],
    Domain.FUSION: [".hdf5", ".h5", ".adios", ".bp"],
    Domain.HPC: [],  # Generic formats, detected by content
    Domain.GENERAL: []
}


def detect_domain(
    columns: Optional[List[str]] = None,
    file_extension: Optional[str] = None
) -> Domain:
    """Detect domain from column names or file extension"""

    # Check extension first
    if file_extension:
        ext = file_extension.lower()
        for domain, extensions in DOMAIN_EXTENSIONS.items():
            if ext in extensions:
                return domain

    # Score by column name keywords
    if columns:
        scores = {domain: 0 for domain in Domain}
        columns_lower = [c.lower() for c in columns]

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for col in columns_lower:
                    if keyword_lower in col:
                        scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain

    return Domain.GENERAL


def get_domain_pipeline(domain: Domain):
    """Get the domain-specific pipeline configuration"""
    from . import climate, materials, life_sciences, fusion, hpc

    domain_modules = {
        Domain.CLIMATE: climate,
        Domain.MATERIALS: materials,
        Domain.LIFE_SCIENCES: life_sciences,
        Domain.FUSION: fusion,
        Domain.HPC: hpc,
    }

    return domain_modules.get(domain)
