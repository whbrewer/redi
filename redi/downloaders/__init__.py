"""
REDI Downloaders Module

Provides data acquisition from various scientific data sources.
Each domain has specific data sources:
- Climate: ESGF (CMIP6), CDS (ERA5)
- Materials: Materials Project, AFLOW
- Life Sciences: UniProt, PDB, NCBI
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation"""
    success: bool
    files: List[Path] = field(default_factory=list)
    total_size_bytes: int = 0
    message: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDownloader(ABC):
    """Base class for data downloaders"""

    name: str = "base"
    description: str = "Base downloader"

    def __init__(self, output_dir: str = "./downloads", **kwargs):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = kwargs

    @abstractmethod
    def search(self, **query) -> List[Dict[str, Any]]:
        """Search for available datasets matching query criteria.

        Returns list of dataset metadata dicts with at least:
        - id: unique identifier
        - size: size in bytes (if known)
        - url: download URL
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, datasets: List[Dict[str, Any]], **kwargs) -> DownloadResult:
        """Download the specified datasets.

        Args:
            datasets: List of dataset metadata from search()
            **kwargs: Additional download options

        Returns:
            DownloadResult with downloaded file paths
        """
        raise NotImplementedError

    def search_and_download(self, **query) -> DownloadResult:
        """Convenience method to search and download in one call"""
        datasets = self.search(**query)
        if not datasets:
            return DownloadResult(
                success=False,
                message=f"No datasets found matching query: {query}"
            )
        return self.download(datasets)


# Registry of available downloaders
_DOWNLOADERS: Dict[str, type] = {}


def register_downloader(name: str):
    """Decorator to register a downloader class"""
    def decorator(cls):
        _DOWNLOADERS[name] = cls
        return cls
    return decorator


def get_downloader(name: str, **kwargs) -> BaseDownloader:
    """Get a downloader instance by name"""
    if name not in _DOWNLOADERS:
        available = ", ".join(_DOWNLOADERS.keys())
        raise ValueError(f"Unknown downloader: {name}. Available: {available}")
    return _DOWNLOADERS[name](**kwargs)


def list_downloaders() -> Dict[str, str]:
    """List available downloaders with descriptions"""
    return {name: cls.description for name, cls in _DOWNLOADERS.items()}


# Import downloaders to register them
from .esgf import ESGFDownloader
from .cds import CDSDownloader
