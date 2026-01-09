"""Fusion domain FAIR validators.

Validates fusion data against IMAS and related standards.
"""

from pathlib import Path
from typing import Any, List, Optional

from ..principles import DomainStandardResult


# Common IMAS IDS (Interface Data Structures) names
IMAS_IDS_NAMES = {
    "equilibrium", "core_profiles", "core_sources", "core_transport",
    "magnetics", "pf_active", "pf_passive", "wall",
    "interferometer", "thomson_scattering", "charge_exchange",
    "bolometer", "langmuir_probes", "reflectometer",
    "mhd", "mhd_linear", "disruption",
    "summary", "pulse_schedule", "machine_description",
}

# Expected attributes in IMAS-compliant HDF5
IMAS_REQUIRED_ATTRS = {"shot", "run", "ids_properties"}
IMAS_RECOMMENDED_ATTRS = {"time", "homogeneous_time", "code"}


class IMASValidator:
    """Validates fusion data against IMAS schema."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate file against IMAS conventions.

        Args:
            file_path: Path to HDF5 file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".h5", ".hdf5"):
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            import h5py

            with h5py.File(file_path, "r") as f:
                # Check for IMAS-like structure
                root_groups = list(f.keys())
                root_attrs = dict(f.attrs)

                # Check for shot/run information
                if "shot" in root_attrs or "pulse" in root_attrs:
                    passed.append("Has shot/pulse identifier")
                elif any("shot" in str(k).lower() for k in root_groups):
                    passed.append("Has shot information in groups")
                else:
                    warnings.append("No shot identifier found")

                # Check for IMAS IDS groups
                ids_found = []
                for group in root_groups:
                    if group.lower() in IMAS_IDS_NAMES:
                        ids_found.append(group)

                if ids_found:
                    passed.append(f"Contains IDS: {', '.join(ids_found[:3])}")
                    if len(ids_found) > 3:
                        passed.append(f"({len(ids_found)} IDS total)")

                # Check for time arrays
                has_time = self._check_for_time_data(f)
                if has_time:
                    passed.append("Contains time array data")
                else:
                    warnings.append("No time array found")

                # Check for provenance/code information
                if "code" in root_groups or "provenance" in root_attrs:
                    passed.append("Has provenance/code information")

                # Check for units metadata
                units_found = self._count_units_attrs(f)
                if units_found > 0:
                    passed.append(f"Has units metadata ({units_found} datasets)")
                else:
                    warnings.append("No units metadata found")

                # Check HDF5 structure depth (IMAS is hierarchical)
                max_depth = self._check_hierarchy_depth(f)
                if max_depth >= 3:
                    passed.append("Has hierarchical structure (IMAS-like)")

            # Calculate compliance
            total = len(passed) + len(failed)
            if total == 0:
                # No IMAS-specific features found
                return None

            compliance = len(passed) / total if total > 0 else 0.0

            return DomainStandardResult(
                standard_name="IMAS",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except ImportError:
            return None
        except Exception as e:
            return DomainStandardResult(
                standard_name="IMAS",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )

    def _check_for_time_data(self, f: Any) -> bool:
        """Check if file contains time array data."""
        def search_time(group, depth=0):
            if depth > 5:
                return False
            for key in group.keys():
                if "time" in key.lower():
                    return True
                item = group[key]
                if hasattr(item, "keys"):
                    if search_time(item, depth + 1):
                        return True
            return False

        return search_time(f)

    def _count_units_attrs(self, f: Any) -> int:
        """Count datasets with units attributes."""
        count = 0

        def search_units(group, depth=0):
            nonlocal count
            if depth > 5:
                return
            for key in group.keys():
                item = group[key]
                if hasattr(item, "attrs") and "units" in item.attrs:
                    count += 1
                if hasattr(item, "keys"):
                    search_units(item, depth + 1)

        search_units(f)
        return count

    def _check_hierarchy_depth(self, f: Any) -> int:
        """Check maximum depth of HDF5 hierarchy."""
        def get_depth(group, current=0):
            max_d = current
            for key in list(group.keys())[:10]:  # Limit for performance
                item = group[key]
                if hasattr(item, "keys"):
                    max_d = max(max_d, get_depth(item, current + 1))
            return max_d

        return get_depth(f)
