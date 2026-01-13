"""
Base DataLoader class for REDI.

Provides a common interface for loading scientific data formats
used in turbulence simulations (SICKLE project).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""

    name: str = ""
    path: str = ""
    file_format: str = ""
    num_variables: int = 0
    variables: List[str] = field(default_factory=list)
    dimensions: Dict[str, int] = field(default_factory=dict)
    dtype: str = "float32"
    total_size_bytes: int = 0
    num_timesteps: int = 0
    timesteps: List[float] = field(default_factory=list)
    grid_shape: Tuple[int, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "file_format": self.file_format,
            "num_variables": self.num_variables,
            "variables": self.variables,
            "dimensions": self.dimensions,
            "dtype": self.dtype,
            "total_size_bytes": self.total_size_bytes,
            "num_timesteps": self.num_timesteps,
            "timesteps": self.timesteps,
            "grid_shape": self.grid_shape,
            "extra": self.extra,
        }


@dataclass
class LoadedData:
    """Container for loaded data and associated metadata."""

    data: Dict[str, np.ndarray] = field(default_factory=dict)
    coordinates: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)

    def get_variable(self, name: str) -> Optional[np.ndarray]:
        """Get a specific variable by name."""
        return self.data.get(name)

    def list_variables(self) -> List[str]:
        """List all loaded variable names."""
        return list(self.data.keys())

    def get_shape(self, name: str) -> Optional[Tuple[int, ...]]:
        """Get shape of a specific variable."""
        arr = self.data.get(name)
        return arr.shape if arr is not None else None


class BaseDataLoader(ABC):
    """
    Abstract base class for SICKLE-format data loaders.

    Subclasses must implement:
        - _scan_directory(): Discover available files and build metadata
        - load_variable(): Load a specific variable
        - load_snapshot(): Load all variables for a given timestep
    """

    def __init__(self, path: str, verbose: bool = False):
        """
        Initialize the data loader.

        Args:
            path: Path to the data directory or file
            verbose: Enable verbose output
        """
        self.path = Path(path)
        self.verbose = verbose
        self._metadata: Optional[DatasetMetadata] = None

    @property
    def metadata(self) -> DatasetMetadata:
        """Get dataset metadata, scanning if not already done."""
        if self._metadata is None:
            self._metadata = self._scan_directory()
        return self._metadata

    @abstractmethod
    def _scan_directory(self) -> DatasetMetadata:
        """
        Scan the data directory and build metadata.

        Returns:
            DatasetMetadata with information about the dataset
        """
        pass

    @abstractmethod
    def load_variable(
        self,
        variable: str,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load a specific variable.

        Args:
            variable: Name of the variable to load
            timestep: Optional timestep index or value
            subcube: Optional tuple of slices for extracting a subcube

        Returns:
            Numpy array containing the variable data
        """
        pass

    @abstractmethod
    def load_snapshot(
        self,
        timestep: Optional[Union[int, float]] = None,
        variables: Optional[List[str]] = None
    ) -> LoadedData:
        """
        Load all (or specified) variables for a given timestep.

        Args:
            timestep: Timestep to load (uses first available if None)
            variables: Optional list of specific variables to load

        Returns:
            LoadedData container with data and metadata
        """
        pass

    def load_all(self) -> LoadedData:
        """
        Load all available data (all timesteps and variables).

        Note: This can be memory-intensive for large datasets.

        Returns:
            LoadedData container with all data
        """
        meta = self.metadata
        all_data = {}

        for var in meta.variables:
            var_data = []
            for ts in meta.timesteps:
                var_data.append(self.load_variable(var, timestep=ts))
            all_data[var] = np.stack(var_data, axis=0) if len(var_data) > 1 else var_data[0]

        return LoadedData(
            data=all_data,
            metadata=meta
        )

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")


def get_dataloader(path: str, format_type: str = "auto", **kwargs) -> BaseDataLoader:
    """
    Factory function to get the appropriate dataloader.

    Args:
        path: Path to the data
        format_type: Either "auto", "gests", "sst-binary", or "zarr"
        **kwargs: Additional arguments passed to the dataloader

    Returns:
        Appropriate BaseDataLoader subclass instance
    """
    from .gests import GESTSDataLoader
    from .sst_binary import SSTBinaryDataLoader
    from .zarr_loader import ZarrDataLoader

    if format_type == "auto":
        # Try to auto-detect format
        path_obj = Path(path)

        # Check for Zarr format (.zarr directory with .zarray or .zgroup)
        if path_obj.is_dir():
            if (path_obj / ".zarray").exists() or (path_obj / ".zgroup").exists():
                return ZarrDataLoader(path, **kwargs)
            # Also check for zarr v3 format (zarr.json)
            if (path_obj / "zarr.json").exists():
                return ZarrDataLoader(path, **kwargs)
            # Check by extension
            if path_obj.suffix == ".zarr":
                return ZarrDataLoader(path, **kwargs)

        # Check for SST-binary format (files like u_28.440000)
        if path_obj.is_dir():
            files = list(path_obj.glob("*_*.*"))
            if files:
                # Check if files match SST pattern (var_timestep)
                import re
                sst_pattern = re.compile(r'^[a-z]+_\d+\.\d+$')
                for f in files[:5]:
                    if sst_pattern.match(f.name):
                        return SSTBinaryDataLoader(path, **kwargs)

            # Check for GESTS format (subdirectories with cube_* files)
            subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
            for subdir in subdirs:
                cube_files = list(subdir.glob("cube_*"))
                if cube_files:
                    return GESTSDataLoader(path, **kwargs)

        raise ValueError(
            f"Could not auto-detect format for path: {path}. "
            "Please specify format_type='gests', 'sst-binary', or 'zarr'"
        )

    elif format_type in ("gests", "gests+"):
        return GESTSDataLoader(path, **kwargs)

    elif format_type in ("sst-binary", "sst"):
        return SSTBinaryDataLoader(path, **kwargs)

    elif format_type == "zarr":
        return ZarrDataLoader(path, **kwargs)

    else:
        raise ValueError(f"Unknown format type: {format_type}")
