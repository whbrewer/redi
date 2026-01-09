"""Domain-specific FAIR validators.

This module provides validators for domain-specific standards.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from ..principles import DomainStandardResult


class Validator(Protocol):
    """Protocol for domain validators."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate file against domain standard.

        Args:
            file_path: Path to data file
            data_card: Optional DataCard with metadata

        Returns:
            DomainStandardResult or None if not applicable
        """
        ...


def get_validator(domain: str) -> Dict[str, Any]:
    """Get validators for a domain.

    Args:
        domain: Scientific domain name

    Returns:
        Dictionary of validator instances
    """
    validators: Dict[str, Any] = {}

    if domain == "climate":
        try:
            from .climate import CFConventionsValidator, ACDDValidator
            validators["cf_conventions"] = CFConventionsValidator()
            validators["acdd"] = ACDDValidator()
        except ImportError:
            pass

    elif domain == "materials":
        try:
            from .materials import CIFValidator, OPTIMADEValidator
            validators["cif"] = CIFValidator()
            validators["optimade"] = OPTIMADEValidator()
        except ImportError:
            pass

    elif domain == "life_sciences":
        try:
            from .life_sciences import PDBValidator, FASTAValidator
            validators["pdb"] = PDBValidator()
            validators["fasta"] = FASTAValidator()
        except ImportError:
            pass

    elif domain == "fusion":
        try:
            from .fusion import IMASValidator
            validators["imas"] = IMASValidator()
        except ImportError:
            pass

    elif domain == "hpc":
        try:
            from .hpc import SlurmValidator, PBSValidator
            validators["slurm"] = SlurmValidator()
            validators["pbs"] = PBSValidator()
        except ImportError:
            pass

    return validators
