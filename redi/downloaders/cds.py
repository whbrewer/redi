"""
CDS Downloader for ERA5 Data

Downloads reanalysis data from the Copernicus Climate Data Store (CDS).
Supports ERA5, ERA5-Land, and other reanalysis products.

Authentication:
- CDS requires an API key
- Register at: https://cds.climate.copernicus.eu/
- Set CDS_API_KEY environment variable or create ~/.cdsapirc

Usage:
    redi run --download --source cds \\
        --variable 2m_temperature --year 2020 \\
        -o ./output
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseDownloader, DownloadResult, register_downloader

logger = logging.getLogger(__name__)

# ERA5 variable mappings (CDS names -> standard names)
ERA5_VARIABLES = {
    # Surface variables
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "2m_dewpoint_temperature": "d2m",
    # Radiation
    "surface_solar_radiation_downwards": "ssrd",
    "surface_thermal_radiation_downwards": "strd",
    "top_net_solar_radiation": "tsr",
    "toa_incident_solar_radiation": "tisr",
    # Pressure levels
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "specific_humidity": "q",
    "relative_humidity": "r",
}

# Standard pressure levels (hPa)
ERA5_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000
]


@register_downloader("cds")
class CDSDownloader(BaseDownloader):
    """Download ERA5 data from Copernicus Climate Data Store"""

    name = "cds"
    description = "Copernicus Climate Data Store (ERA5, ERA5-Land)"

    def __init__(
        self,
        output_dir: str = "./downloads",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(output_dir, **kwargs)
        self.api_key = api_key or os.environ.get("CDS_API_KEY")
        self.api_url = api_url or os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")

    def _get_client(self):
        """Get CDS API client"""
        try:
            import cdsapi
            return cdsapi.Client()
        except ImportError:
            raise ImportError(
                "cdsapi not installed. Install with: pip install cdsapi\n"
                "Then configure: https://cds.climate.copernicus.eu/api-how-to"
            )

    def search(
        self,
        variable: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[List[int]] = None,
        day: Optional[List[int]] = None,
        time: Optional[List[str]] = None,
        pressure_level: Optional[List[int]] = None,
        product_type: str = "reanalysis",
        dataset: str = "reanalysis-era5-single-levels",
        area: Optional[List[float]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Build CDS request parameters.

        Note: CDS doesn't have a search API like ESGF. This method
        builds the request parameters for download.

        Args:
            variable: Variable name (e.g., '2m_temperature')
            year: Year (e.g., 2020) or list of years
            month: Month(s) 1-12 (default: all)
            day: Day(s) 1-31 (default: all)
            time: Hour(s) in 'HH:00' format (default: all)
            pressure_level: Pressure levels in hPa (for pressure-level data)
            product_type: 'reanalysis' or 'ensemble_members'
            dataset: CDS dataset name
            area: Bounding box [North, West, South, East]

        Returns:
            List with single request dict (for compatibility with base class)
        """
        if not variable:
            raise ValueError("variable is required for CDS downloads")

        # Build request
        request = {
            "product_type": product_type,
            "variable": variable,
            "format": "netcdf",
        }

        # Handle year
        if year:
            request["year"] = str(year) if isinstance(year, int) else [str(y) for y in year]
        else:
            request["year"] = "2020"  # Default

        # Handle month
        if month:
            request["month"] = [f"{m:02d}" for m in month]
        else:
            request["month"] = [f"{m:02d}" for m in range(1, 13)]

        # Handle day
        if day:
            request["day"] = [f"{d:02d}" for d in day]
        else:
            request["day"] = [f"{d:02d}" for d in range(1, 32)]

        # Handle time
        if time:
            request["time"] = time
        else:
            request["time"] = [f"{h:02d}:00" for h in range(0, 24, 6)]  # 6-hourly default

        # Handle pressure levels
        if pressure_level:
            request["pressure_level"] = [str(p) for p in pressure_level]
            if "single-levels" in dataset:
                dataset = "reanalysis-era5-pressure-levels"

        # Handle area
        if area:
            request["area"] = area

        # Determine output filename
        year_str = request["year"] if isinstance(request["year"], str) else request["year"][0]
        filename = f"era5_{variable}_{year_str}.nc"

        return [{
            "id": f"cds:{dataset}:{variable}:{year_str}",
            "dataset": dataset,
            "request": request,
            "filename": filename,
            "variable": variable,
            "year": year_str,
        }]

    def download(
        self,
        datasets: List[Dict[str, Any]],
        **kwargs
    ) -> DownloadResult:
        """Download data from CDS.

        Args:
            datasets: List of request dicts from search()

        Returns:
            DownloadResult with downloaded file paths
        """
        result = DownloadResult(success=True)

        if not datasets:
            result.success = False
            result.message = "No datasets to download"
            return result

        try:
            client = self._get_client()
        except ImportError as e:
            result.success = False
            result.message = str(e)
            return result

        for ds in datasets:
            dataset_name = ds["dataset"]
            request = ds["request"]
            filename = ds["filename"]
            filepath = self.output_dir / filename

            logger.info(f"Requesting from CDS: {filename}")
            logger.debug(f"Dataset: {dataset_name}")
            logger.debug(f"Request: {request}")

            try:
                # This is a blocking call that waits for the data
                client.retrieve(dataset_name, request, str(filepath))

                if filepath.exists():
                    size = filepath.stat().st_size
                    result.files.append(filepath)
                    result.total_size_bytes += size
                    logger.info(f"Downloaded: {filename} ({size / 1024 / 1024:.1f} MB)")
                else:
                    result.errors.append(f"File not created: {filename}")

            except Exception as e:
                result.errors.append(f"Failed to download {filename}: {e}")
                logger.error(f"CDS download failed: {e}")

        if result.errors:
            result.success = len(result.files) > 0
            result.message = f"Downloaded {len(result.files)} files with {len(result.errors)} errors"
        else:
            result.message = f"Successfully downloaded {len(result.files)} files"

        return result


def list_era5_variables() -> Dict[str, str]:
    """List common ERA5 variables"""
    return ERA5_VARIABLES


def list_era5_pressure_levels() -> List[int]:
    """List available ERA5 pressure levels"""
    return ERA5_PRESSURE_LEVELS
