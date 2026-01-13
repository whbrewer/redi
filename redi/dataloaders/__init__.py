"""
REDI Dataloaders Module

Provides data loaders for scientific simulation formats used in the SICKLE
project for turbulence research:

- GESTS: GPU-based Extreme-Scale Turbulence Simulations (forced isotropic turbulence)
- SST-Binary: Stably Stratified Turbulence (buoyancy-influenced turbulence)

Usage:
    >>> from redi.dataloaders import get_dataloader
    >>> loader = get_dataloader("/path/to/data", format_type="auto")
    >>> print(loader.metadata)
    >>> data = loader.load_snapshot()

Or directly:
    >>> from redi.dataloaders import GESTSDataLoader, SSTBinaryDataLoader
    >>> loader = GESTSDataLoader("/path/to/gests/data")
    >>> velocity = loader.load_variable("velocity")

References:
    - SICKLE: Sparse Intelligent Curation frameworK for Learning Efficiently
      https://github.com/at-aaims/sickle
    - Brewer et al. (2025) "Intelligent Sampling of Extreme-Scale Turbulence
      Datasets for Accurate and Efficient Spatiotemporal Model Training"
"""

from .base import (
    BaseDataLoader,
    DatasetMetadata,
    LoadedData,
    get_dataloader,
)
from .gests import GESTSDataLoader
from .sst_binary import SSTBinaryDataLoader


__all__ = [
    # Base classes
    "BaseDataLoader",
    "DatasetMetadata",
    "LoadedData",
    # Factory function
    "get_dataloader",
    # Concrete loaders
    "GESTSDataLoader",
    "SSTBinaryDataLoader",
]


# Version info
__version__ = "0.1.0"


def load_sickle_data(path: str, format_type: str = "auto", **kwargs):
    """
    Convenience function to load SICKLE-format data.

    This is a high-level function that automatically detects the data format
    and returns a LoadedData object with the first snapshot.

    Args:
        path: Path to the data directory
        format_type: "auto", "gests", or "sst-binary"
        **kwargs: Additional arguments passed to the dataloader

    Returns:
        LoadedData object containing data, coordinates, and metadata

    Example:
        >>> from redi.dataloaders import load_sickle_data
        >>> data = load_sickle_data("/path/to/simulation/data")
        >>> print(data.metadata)
        >>> velocity = data.get_variable("velocity")
    """
    loader = get_dataloader(path, format_type=format_type, **kwargs)
    return loader.load_snapshot()
