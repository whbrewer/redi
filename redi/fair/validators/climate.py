"""Climate domain FAIR validators.

Validates climate data against CF Conventions and ACDD standards.
"""

from pathlib import Path
from typing import Any, List, Optional

from ..principles import DomainStandardResult


# CF Conventions standard names (subset of common ones)
CF_STANDARD_NAMES = {
    "air_temperature", "surface_air_temperature", "sea_surface_temperature",
    "air_pressure", "surface_air_pressure", "sea_level_pressure",
    "precipitation_flux", "precipitation_amount", "rainfall_rate",
    "eastward_wind", "northward_wind", "wind_speed", "wind_from_direction",
    "specific_humidity", "relative_humidity",
    "geopotential", "geopotential_height",
    "sea_surface_height", "sea_water_temperature", "sea_water_salinity",
    "latitude", "longitude", "time", "height", "altitude", "depth",
    "air_potential_temperature", "equivalent_potential_temperature",
    "cloud_area_fraction", "cloud_base_altitude", "cloud_top_altitude",
}

# ACDD 1.3 required and recommended attributes
ACDD_REQUIRED = {
    "title", "summary", "keywords", "Conventions",
}

ACDD_HIGHLY_RECOMMENDED = {
    "id", "naming_authority", "source", "processing_level",
    "creator_name", "creator_email", "creator_url",
    "institution", "project",
    "date_created", "date_modified",
    "geospatial_lat_min", "geospatial_lat_max",
    "geospatial_lon_min", "geospatial_lon_max",
    "time_coverage_start", "time_coverage_end",
}

ACDD_RECOMMENDED = {
    "publisher_name", "publisher_email", "publisher_url",
    "license", "standard_name_vocabulary",
    "history", "comment", "references",
}


class CFConventionsValidator:
    """Validates NetCDF files against CF Conventions."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate file against CF Conventions.

        Args:
            file_path: Path to NetCDF file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".nc", ".nc4"):
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            import xarray as xr
            ds = xr.open_dataset(file_path)

            # Check Conventions attribute
            conventions = ds.attrs.get("Conventions", ds.attrs.get("conventions", ""))
            if "CF" in str(conventions).upper():
                passed.append("Conventions attribute contains CF reference")
            else:
                failed.append("Missing or invalid Conventions attribute")

            # Check for standard_name attributes
            vars_with_standard_name = 0
            vars_without_standard_name = []
            for var_name, var in ds.data_vars.items():
                if "standard_name" in var.attrs:
                    sn = var.attrs["standard_name"]
                    if sn in CF_STANDARD_NAMES:
                        vars_with_standard_name += 1
                    else:
                        warnings.append(
                            f"Variable '{var_name}' has non-standard standard_name: {sn}"
                        )
                        vars_with_standard_name += 1  # Still has standard_name
                else:
                    vars_without_standard_name.append(var_name)

            if vars_with_standard_name > 0:
                passed.append(f"{vars_with_standard_name} variables have standard_name")
            if vars_without_standard_name:
                if len(vars_without_standard_name) <= 3:
                    warnings.append(
                        f"Variables without standard_name: {', '.join(vars_without_standard_name)}"
                    )
                else:
                    warnings.append(
                        f"{len(vars_without_standard_name)} variables without standard_name"
                    )

            # Check units
            vars_with_units = 0
            for var_name, var in ds.data_vars.items():
                if "units" in var.attrs:
                    vars_with_units += 1

            if vars_with_units > 0:
                passed.append(f"{vars_with_units} variables have units")
            if vars_with_units < len(ds.data_vars):
                missing = len(ds.data_vars) - vars_with_units
                warnings.append(f"{missing} variables missing units attribute")

            # Check coordinate variables
            has_lat = any(c in ds.coords for c in ("lat", "latitude"))
            has_lon = any(c in ds.coords for c in ("lon", "longitude"))
            has_time = "time" in ds.coords

            if has_lat and has_lon:
                passed.append("Has latitude and longitude coordinates")
            else:
                warnings.append("Missing latitude or longitude coordinates")

            if has_time:
                passed.append("Has time coordinate")

            # Check axis attributes
            coords_with_axis = sum(
                1 for c in ds.coords.values()
                if "axis" in c.attrs
            )
            if coords_with_axis > 0:
                passed.append(f"{coords_with_axis} coordinates have axis attribute")

            ds.close()

            # Calculate compliance
            total_checks = len(passed) + len(failed)
            compliance = len(passed) / total_checks if total_checks > 0 else 0.0

            return DomainStandardResult(
                standard_name="CF Conventions",
                version="1.10",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except ImportError:
            return None
        except Exception as e:
            return DomainStandardResult(
                standard_name="CF Conventions",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )


class ACDDValidator:
    """Validates NetCDF files against ACDD conventions."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate file against ACDD.

        Args:
            file_path: Path to NetCDF file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".nc", ".nc4"):
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            import xarray as xr
            ds = xr.open_dataset(file_path)
            attrs = {k.lower(): v for k, v in ds.attrs.items()}

            # Check required attributes
            for attr in ACDD_REQUIRED:
                if attr.lower() in attrs:
                    passed.append(f"Has required attribute: {attr}")
                else:
                    failed.append(f"Missing required attribute: {attr}")

            # Check highly recommended
            hr_present = 0
            for attr in ACDD_HIGHLY_RECOMMENDED:
                if attr.lower() in attrs:
                    hr_present += 1

            if hr_present > 0:
                passed.append(
                    f"{hr_present}/{len(ACDD_HIGHLY_RECOMMENDED)} highly recommended attributes"
                )
            if hr_present < len(ACDD_HIGHLY_RECOMMENDED) // 2:
                missing = [
                    a for a in ACDD_HIGHLY_RECOMMENDED
                    if a.lower() not in attrs
                ][:5]
                warnings.append(f"Consider adding: {', '.join(missing)}")

            # Check recommended
            rec_present = sum(
                1 for attr in ACDD_RECOMMENDED
                if attr.lower() in attrs
            )
            if rec_present > 0:
                passed.append(f"{rec_present} recommended attributes present")

            ds.close()

            # Calculate compliance (weighted: required=3, highly_rec=2, rec=1)
            req_score = sum(1 for a in ACDD_REQUIRED if a.lower() in attrs) / len(ACDD_REQUIRED)
            hr_score = hr_present / len(ACDD_HIGHLY_RECOMMENDED)
            rec_score = rec_present / len(ACDD_RECOMMENDED) if ACDD_RECOMMENDED else 1

            compliance = (req_score * 3 + hr_score * 2 + rec_score) / 6

            return DomainStandardResult(
                standard_name="ACDD",
                version="1.3",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except ImportError:
            return None
        except Exception as e:
            return DomainStandardResult(
                standard_name="ACDD",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )
