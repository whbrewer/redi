"""Materials science domain FAIR validators.

Validates materials data against CIF and OPTIMADE standards.
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from ..principles import DomainStandardResult


# CIF required data items for crystal structures
CIF_REQUIRED_ITEMS = {
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
}

CIF_RECOMMENDED_ITEMS = {
    "_symmetry_space_group_name_H-M", "_symmetry_Int_Tables_number",
    "_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
    "_chemical_formula_sum", "_chemical_formula_structural",
}

# OPTIMADE required properties for structures
OPTIMADE_REQUIRED = {
    "id", "type", "attributes",
}

OPTIMADE_STRUCTURE_REQUIRED = {
    "lattice_vectors", "cartesian_site_positions", "species_at_sites", "species",
}


class CIFValidator:
    """Validates CIF (Crystallographic Information File) format."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate CIF file.

        Args:
            file_path: Path to CIF file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext != ".cif":
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            content = file_path.read_text(errors="ignore")

            # Check for data block
            if re.search(r"^data_\w+", content, re.MULTILINE):
                passed.append("Contains valid data block")
            else:
                failed.append("Missing data_ block header")

            # Check required cell parameters
            cell_params_found = 0
            for item in CIF_REQUIRED_ITEMS:
                if item in content:
                    cell_params_found += 1

            if cell_params_found == len(CIF_REQUIRED_ITEMS):
                passed.append("All cell parameters present")
            elif cell_params_found > 0:
                passed.append(f"{cell_params_found}/6 cell parameters present")
                warnings.append("Some cell parameters missing")
            else:
                failed.append("Missing cell parameters")

            # Check for space group
            if "_symmetry_space_group_name" in content or "_space_group" in content:
                passed.append("Space group information present")
            else:
                warnings.append("Space group information not found")

            # Check for atom sites
            if "_atom_site" in content:
                passed.append("Atom site data present")
                # Count atom sites
                atom_count = len(re.findall(r"_atom_site_label\s+(\S+)", content))
                if atom_count > 0:
                    passed.append(f"{atom_count} atom sites defined")
            else:
                failed.append("Missing atom site data")

            # Check for chemical formula
            if "_chemical_formula" in content:
                passed.append("Chemical formula present")
            else:
                warnings.append("Chemical formula not specified")

            # Check CIF version/audit
            if "_audit" in content:
                passed.append("Audit information present")

            # Calculate compliance
            total = len(passed) + len(failed)
            compliance = len(passed) / total if total > 0 else 0.0

            return DomainStandardResult(
                standard_name="CIF",
                version="2.0",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except Exception as e:
            return DomainStandardResult(
                standard_name="CIF",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )


class OPTIMADEValidator:
    """Validates OPTIMADE API compliance for materials data."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate OPTIMADE compliance.

        This validator checks JSON files for OPTIMADE structure format
        or CSV/Parquet files for OPTIMADE-like properties.

        Args:
            file_path: Path to data file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            if ext == ".json":
                return self._validate_json(file_path)
            elif ext in (".csv", ".parquet"):
                return self._validate_tabular(file_path, data_card)
            else:
                return None

        except Exception as e:
            return DomainStandardResult(
                standard_name="OPTIMADE",
                compliance_percentage=0.0,
                failed_checks=[f"Error: {str(e)}"],
            )

    def _validate_json(self, file_path: Path) -> Optional[DomainStandardResult]:
        """Validate JSON file for OPTIMADE structure format."""
        import json

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        with open(file_path) as f:
            data = json.load(f)

        # Check if it's OPTIMADE response format
        if isinstance(data, dict):
            if "data" in data:
                passed.append("Has OPTIMADE response structure")
                entries = data["data"]
                if isinstance(entries, list) and len(entries) > 0:
                    # Check first entry
                    entry = entries[0]
                    for field in OPTIMADE_REQUIRED:
                        if field in entry:
                            passed.append(f"Has required field: {field}")
                        else:
                            failed.append(f"Missing required field: {field}")

                    # Check structure attributes
                    if "attributes" in entry:
                        attrs = entry["attributes"]
                        for field in OPTIMADE_STRUCTURE_REQUIRED:
                            if field in attrs:
                                passed.append(f"Has structure field: {field}")
                            else:
                                warnings.append(f"Missing structure field: {field}")
            else:
                warnings.append("Not in OPTIMADE response format")

        total = len(passed) + len(failed)
        compliance = len(passed) / total if total > 0 else 0.0

        return DomainStandardResult(
            standard_name="OPTIMADE",
            version="1.1.0",
            compliance_percentage=compliance * 100,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
        )

    def _validate_tabular(
        self, file_path: Path, data_card: Optional[Any]
    ) -> Optional[DomainStandardResult]:
        """Check tabular data for OPTIMADE-like properties."""
        passed: List[str] = []
        warnings: List[str] = []

        # Get columns from data_card if available
        columns = set()
        if data_card and hasattr(data_card, "characteristics"):
            features = getattr(data_card.characteristics, "features", [])
            columns = {f.get("name", "").lower() for f in features}

        # Check for OPTIMADE-like properties
        optimade_props = {
            "nelements", "elements", "chemical_formula_reduced",
            "chemical_formula_anonymous", "dimension_types",
            "lattice_vectors", "cartesian_site_positions",
        }

        found = columns & {p.lower() for p in optimade_props}
        if found:
            passed.append(f"Has {len(found)} OPTIMADE-like properties")
        else:
            warnings.append("No OPTIMADE property names found")

        if not passed and not warnings:
            return None

        return DomainStandardResult(
            standard_name="OPTIMADE",
            version="1.1.0",
            compliance_percentage=50.0 if passed else 0.0,
            passed_checks=passed,
            warnings=warnings,
        )
