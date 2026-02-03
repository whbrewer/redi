"""
ESGF Downloader for CMIP6 Data

Downloads climate model data from the Earth System Grid Federation (ESGF).
Supports CMIP6, CMIP5, and other MIP data.

Authentication:
- ESGF requires an OpenID account for some data
- Set ESGF_OPENID environment variable or use --esgf-openid flag
- Register at: https://esgf-node.llnl.gov/user/add/

Usage:
    redi run --download --source esgf \\
        --variable tas --model TaiESM1 --experiment historical \\
        -o ./output
"""

import hashlib
import logging
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import BaseDownloader, DownloadResult, register_downloader

logger = logging.getLogger(__name__)

# ESGF search API endpoints (use LLNL as primary)
ESGF_SEARCH_URLS = [
    "https://esgf-node.llnl.gov/esg-search/search",
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-index1.ceda.ac.uk/esg-search/search",
]

# Standard CMIP6 variables
CMIP6_VARIABLES = {
    # Surface variables
    "tas": "Near-Surface Air Temperature",
    "pr": "Precipitation",
    "psl": "Sea Level Pressure",
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind",
    "huss": "Near-Surface Specific Humidity",
    "ts": "Surface Temperature",
    "tos": "Sea Surface Temperature",
    # Radiation
    "rsds": "Surface Downwelling Shortwave Radiation",
    "rlds": "Surface Downwelling Longwave Radiation",
    "rsus": "Surface Upwelling Shortwave Radiation",
    "rlus": "Surface Upwelling Longwave Radiation",
    # Pressure levels
    "ta": "Air Temperature",
    "ua": "Eastward Wind",
    "va": "Northward Wind",
    "zg": "Geopotential Height",
    "hus": "Specific Humidity",
}

# Common CMIP6 models
CMIP6_MODELS = [
    "TaiESM1", "MPI-ESM1-2-HR", "AWI-CM-1-1-MR", "HAMMOZ-Consortium",
    "CMCC-CM2-SR5", "EC-Earth3", "GFDL-ESM4", "MIROC6", "NorESM2-LM",
    "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CESM2", "CNRM-ESM2-1",
]


@register_downloader("esgf")
class ESGFDownloader(BaseDownloader):
    """Download CMIP6 data from ESGF"""

    name = "esgf"
    description = "Earth System Grid Federation (CMIP6, CMIP5)"

    def __init__(
        self,
        output_dir: str = "./downloads",
        openid: Optional[str] = None,
        node: Optional[str] = None,
        **kwargs
    ):
        super().__init__(output_dir, **kwargs)
        self.openid = openid or os.environ.get("ESGF_OPENID")
        self.search_url = node or ESGF_SEARCH_URLS[0]

    def search(
        self,
        variable: Optional[str] = None,
        model: Optional[str] = None,
        experiment: str = "historical",
        frequency: str = "6hr",
        project: str = "CMIP6",
        activity_id: str = "CMIP",
        variant_label: str = "r1i1p1f1",
        grid_label: Optional[str] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search ESGF for datasets matching criteria.

        Args:
            variable: Variable name (e.g., 'tas', 'pr')
            model: Model/source_id (e.g., 'TaiESM1')
            experiment: Experiment ID (e.g., 'historical', 'ssp585')
            frequency: Time frequency (e.g., '6hr', 'day', 'mon')
            project: MIP project (default: 'CMIP6')
            activity_id: Activity (default: 'CMIP')
            variant_label: Variant (default: 'r1i1p1f1')
            grid_label: Grid label (e.g., 'gn', 'gr')
            limit: Maximum results to return

        Returns:
            List of dataset metadata dicts
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed, using urllib fallback")
            return self._search_urllib(
                variable=variable, model=model, experiment=experiment,
                frequency=frequency, project=project, activity_id=activity_id,
                variant_label=variant_label, grid_label=grid_label, limit=limit
            )

        # Build query parameters
        params = {
            "type": "File",
            "project": project,
            "experiment_id": experiment,
            "frequency": frequency,
            "variant_label": variant_label,
            "format": "application/solr+json",
            "limit": limit,
            "distrib": "true",
        }

        if variable:
            params["variable_id"] = variable
        if model:
            params["source_id"] = model
        if activity_id:
            params["activity_id"] = activity_id
        if grid_label:
            params["grid_label"] = grid_label

        # Add any extra kwargs
        params.update(kwargs)

        logger.info(f"Searching ESGF: variable={variable}, model={model}, experiment={experiment}")
        logger.debug(f"Query params: {params}")

        try:
            response = requests.get(self.search_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"ESGF search failed: {e}")
            return []

        # Parse results
        docs = data.get("response", {}).get("docs", [])
        logger.info(f"Found {len(docs)} files")

        results = []
        for doc in docs:
            # Extract download URL
            url_list = doc.get("url", [])
            http_url = None
            for url_entry in url_list:
                if "HTTPServer" in url_entry or "http://" in url_entry or "https://" in url_entry:
                    # Format: "url|mime|service"
                    http_url = url_entry.split("|")[0]
                    break

            if not http_url:
                continue

            results.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "url": http_url,
                "size": doc.get("size", 0),
                "checksum": doc.get("checksum", [""])[0] if doc.get("checksum") else "",
                "checksum_type": doc.get("checksum_type", [""])[0] if doc.get("checksum_type") else "",
                "variable": doc.get("variable_id", [""])[0] if isinstance(doc.get("variable_id"), list) else doc.get("variable_id", ""),
                "model": doc.get("source_id", [""])[0] if isinstance(doc.get("source_id"), list) else doc.get("source_id", ""),
                "experiment": doc.get("experiment_id", [""])[0] if isinstance(doc.get("experiment_id"), list) else doc.get("experiment_id", ""),
                "frequency": doc.get("frequency", [""])[0] if isinstance(doc.get("frequency"), list) else doc.get("frequency", ""),
                "time_range": doc.get("datetime_start", "") + " to " + doc.get("datetime_stop", ""),
            })

        return results

    def _search_urllib(self, **kwargs) -> List[Dict[str, Any]]:
        """Fallback search using urllib (no requests dependency)"""
        import json
        from urllib.parse import urlencode

        params = {
            "type": "File",
            "project": kwargs.get("project", "CMIP6"),
            "experiment_id": kwargs.get("experiment", "historical"),
            "frequency": kwargs.get("frequency", "6hr"),
            "variant_label": kwargs.get("variant_label", "r1i1p1f1"),
            "format": "application/solr+json",
            "limit": kwargs.get("limit", 100),
            "distrib": "true",
        }

        if kwargs.get("variable"):
            params["variable_id"] = kwargs["variable"]
        if kwargs.get("model"):
            params["source_id"] = kwargs["model"]

        url = f"{self.search_url}?{urlencode(params)}"
        logger.debug(f"Search URL: {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            logger.error(f"ESGF search failed: {e}")
            return []

        docs = data.get("response", {}).get("docs", [])
        results = []
        for doc in docs:
            url_list = doc.get("url", [])
            http_url = None
            for url_entry in url_list:
                if "HTTPServer" in url_entry:
                    http_url = url_entry.split("|")[0]
                    break

            if http_url:
                results.append({
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "url": http_url,
                    "size": doc.get("size", 0),
                    "variable": doc.get("variable_id", [""])[0] if isinstance(doc.get("variable_id"), list) else doc.get("variable_id", ""),
                    "model": doc.get("source_id", [""])[0] if isinstance(doc.get("source_id"), list) else doc.get("source_id", ""),
                })

        return results

    def download(
        self,
        datasets: List[Dict[str, Any]],
        parallel: int = 4,
        verify_checksum: bool = True,
        skip_existing: bool = True,
        **kwargs
    ) -> DownloadResult:
        """Download datasets from ESGF.

        Args:
            datasets: List of dataset metadata from search()
            parallel: Number of parallel downloads
            verify_checksum: Verify file checksums after download
            skip_existing: Skip files that already exist

        Returns:
            DownloadResult with downloaded file paths
        """
        result = DownloadResult(success=True)

        if not datasets:
            result.success = False
            result.message = "No datasets to download"
            return result

        logger.info(f"Downloading {len(datasets)} files to {self.output_dir}")

        # Filter out existing files if skip_existing
        to_download = []
        for ds in datasets:
            filename = Path(ds["url"]).name
            filepath = self.output_dir / filename

            if skip_existing and filepath.exists():
                expected_size = ds.get("size", 0)
                actual_size = filepath.stat().st_size
                if expected_size == 0 or actual_size == expected_size:
                    logger.debug(f"Skipping existing: {filename}")
                    result.files.append(filepath)
                    result.total_size_bytes += actual_size
                    continue

            to_download.append(ds)

        if not to_download:
            result.message = f"All {len(datasets)} files already exist"
            logger.info(result.message)
            return result

        logger.info(f"Downloading {len(to_download)} new files ({len(datasets) - len(to_download)} skipped)")

        # Download files
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(self._download_file, ds, verify_checksum): ds
                    for ds in to_download
                }
                for future in as_completed(futures):
                    ds = futures[future]
                    try:
                        filepath, size = future.result()
                        if filepath:
                            result.files.append(filepath)
                            result.total_size_bytes += size
                    except Exception as e:
                        result.errors.append(f"Failed to download {ds.get('title', ds['url'])}: {e}")
        else:
            for ds in to_download:
                try:
                    filepath, size = self._download_file(ds, verify_checksum)
                    if filepath:
                        result.files.append(filepath)
                        result.total_size_bytes += size
                except Exception as e:
                    result.errors.append(f"Failed to download {ds.get('title', ds['url'])}: {e}")

        if result.errors:
            result.success = len(result.files) > 0
            result.message = f"Downloaded {len(result.files)} files with {len(result.errors)} errors"
        else:
            result.message = f"Successfully downloaded {len(result.files)} files"

        logger.info(result.message)
        return result

    def _download_file(self, ds: Dict[str, Any], verify_checksum: bool = True) -> tuple:
        """Download a single file"""
        url = ds["url"]
        filename = Path(url).name
        filepath = self.output_dir / filename

        logger.info(f"Downloading: {filename}")

        # Try wget first (handles redirects and auth better)
        if self._download_wget(url, filepath):
            size = filepath.stat().st_size
            if verify_checksum and ds.get("checksum"):
                if not self._verify_checksum(filepath, ds["checksum"], ds.get("checksum_type", "sha256")):
                    logger.error(f"Checksum mismatch for {filename}")
                    filepath.unlink()
                    return None, 0
            return filepath, size

        # Fallback to urllib
        try:
            urllib.request.urlretrieve(url, filepath)
            size = filepath.stat().st_size
            return filepath, size
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            raise

    def _download_wget(self, url: str, filepath: Path) -> bool:
        """Download using wget (better for large files)"""
        try:
            cmd = [
                "wget", "-q", "--show-progress",
                "-O", str(filepath),
                "--tries=3",
                "--timeout=60",
                url
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _verify_checksum(self, filepath: Path, expected: str, checksum_type: str = "sha256") -> bool:
        """Verify file checksum"""
        if checksum_type.lower() in ("sha256", "sha-256"):
            hasher = hashlib.sha256()
        elif checksum_type.lower() in ("md5",):
            hasher = hashlib.md5()
        else:
            logger.warning(f"Unknown checksum type: {checksum_type}")
            return True

        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        actual = hasher.hexdigest()
        return actual.lower() == expected.lower()


def list_cmip6_variables() -> Dict[str, str]:
    """List common CMIP6 variables"""
    return CMIP6_VARIABLES


def list_cmip6_models() -> List[str]:
    """List common CMIP6 models"""
    return CMIP6_MODELS
