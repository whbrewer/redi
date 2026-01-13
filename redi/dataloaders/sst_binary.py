"""
SST-Binary Data Loader for REDI.

Loads Stably Stratified Turbulence (SST) binary data format from the SICKLE
project. This format is used for DNS simulations of turbulence influenced
by buoyancy (Taylor-Green vortex initialization).

File format:
    - Binary files stored directly in data directory
    - Filenames: {variable}_{timestep} (e.g., u_28.440000, p_28.440000)
    - Variables: u, v, w (velocity), r/rho (density), p (pressure), pv (potential vorticity)
    - Data stored in binary format with configurable grid dimensions
    - Grid dimensions encoded in directory name or config

Reference:
    De Bruyn Kops and collaborators - DNS of stably stratified turbulence
    Riley & de Bruyn Kops (2003), Portwood et al. (2016)
"""

import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataLoader, DatasetMetadata, LoadedData


class SSTBinaryDataLoader(BaseDataLoader):
    """
    Data loader for SST (Stably Stratified Turbulence) binary format.

    This loader supports:
        - Variable grid sizes (512³, 1024³, 4096×1024×4096, etc.)
        - Time-dependent snapshots
        - Variables: u, v, w (velocity), r/rho (density), p (pressure), pv (potential vorticity)
        - Memory-mapped reading for large files

    Example usage:
        >>> loader = SSTBinaryDataLoader("/path/to/sst/data", grid_shape=(512, 512, 256))
        >>> print(loader.metadata)
        >>> velocity_u = loader.load_variable("u", timestep=28.44)
        >>> snapshot = loader.load_snapshot(timestep=28.44)
    """

    # Standard variable names
    VARIABLES = {"u", "v", "w", "r", "rho", "p", "pv", "ee"}

    # Variable descriptions
    VARIABLE_DESCRIPTIONS = {
        "u": "velocity component (x-direction)",
        "v": "velocity component (y-direction)",
        "w": "velocity component (z-direction)",
        "r": "density perturbation",
        "rho": "density",
        "p": "pressure",
        "pv": "potential vorticity",
        "ee": "dissipation rate (epsilon)",
    }

    def __init__(
        self,
        path: str,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the SST binary data loader.

        Args:
            path: Path to the SST data directory
            grid_shape: Grid dimensions (nx, ny, nz). If None, attempts to infer from path.
            verbose: Enable verbose output
        """
        super().__init__(path, verbose)
        self._grid_shape = grid_shape

    def _parse_grid_from_path(self) -> Optional[Tuple[int, int, int]]:
        """Try to parse grid dimensions from the directory name."""
        # Common patterns in SST directory names
        # E.g., P1F4R32_nx512ny512nz256_6vars
        path_str = str(self.path)

        # Pattern: nx{N}ny{N}nz{N}
        pattern1 = re.search(r'nx(\d+)ny(\d+)nz(\d+)', path_str)
        if pattern1:
            return (int(pattern1.group(1)), int(pattern1.group(2)), int(pattern1.group(3)))

        # Pattern: {N}x{N}x{N}
        pattern2 = re.search(r'(\d+)x(\d+)x(\d+)', path_str)
        if pattern2:
            return (int(pattern2.group(1)), int(pattern2.group(2)), int(pattern2.group(3)))

        return None

    def _scan_directory(self) -> DatasetMetadata:
        """Scan the SST data directory and extract metadata."""
        self._log(f"Scanning SST-binary directory: {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        variables = []
        timesteps = set()
        total_size = 0

        # Pattern for SST files: {variable}_{timestep}
        # Timestep format: decimal number (e.g., 28.440000)
        file_pattern = re.compile(r'^([a-z]+)_(\d+\.\d+)$')

        for f in self.path.iterdir():
            if f.is_file():
                match = file_pattern.match(f.name)
                if match:
                    var_name = match.group(1)
                    ts = float(match.group(2))

                    if var_name not in variables:
                        variables.append(var_name)
                    timesteps.add(ts)
                    total_size += f.stat().st_size

        # Try to infer grid shape
        grid_shape = self._grid_shape
        if grid_shape is None:
            # First try to parse from path
            grid_shape = self._parse_grid_from_path()

        if grid_shape is None and total_size > 0 and variables:
            # Try to infer from file size
            sample_file = self.path / f"{variables[0]}_{min(timesteps):.6f}"
            if sample_file.exists():
                file_size = sample_file.stat().st_size
                # Assume float32 or float64
                for dtype_size in [4, 8]:
                    total_points = file_size // dtype_size
                    # Check if it could be cubic
                    dim = int(round(total_points ** (1/3)))
                    if dim ** 3 == total_points:
                        grid_shape = (dim, dim, dim)
                        self._log(f"Inferred cubic grid shape: {grid_shape}")
                        break
                    # Check for common anisotropic grids (including 514 variants for SST)
                    common_grids = [
                        (514, 512, 256),   # P1F4R32 SST dataset
                        (512, 512, 256),
                        (1026, 1024, 512),  # Larger SST with padding
                        (1024, 1024, 512),
                        (4098, 1024, 4096),  # P1F100 with padding
                        (4096, 1024, 4096),
                    ]
                    for nx, ny, nz in common_grids:
                        if nx * ny * nz == total_points:
                            grid_shape = (nx, ny, nz)
                            self._log(f"Inferred grid shape: {grid_shape}")
                            break
                    if grid_shape:
                        break

        # Default grid if couldn't infer
        if grid_shape is None:
            grid_shape = (512, 512, 256)
            self._log(f"Using default grid shape: {grid_shape}")

        self._grid_shape = grid_shape

        sorted_timesteps = sorted(timesteps)

        return DatasetMetadata(
            name=self.path.name,
            path=str(self.path),
            file_format="sst-binary",
            num_variables=len(variables),
            variables=sorted(variables),
            dimensions={
                "x": grid_shape[0],
                "y": grid_shape[1],
                "z": grid_shape[2],
            },
            dtype="float32",
            total_size_bytes=total_size,
            num_timesteps=len(sorted_timesteps),
            timesteps=sorted_timesteps,
            grid_shape=grid_shape,
            extra={
                "format_version": "sst-binary",
                "variable_descriptions": {
                    v: self.VARIABLE_DESCRIPTIONS.get(v, "Unknown")
                    for v in variables
                },
            }
        )

    def _get_file_for_variable(self, variable: str, timestep: float) -> Path:
        """Get the file path for a variable at a specific timestep."""
        # Format timestep to 6 decimal places
        filename = f"{variable}_{timestep:.6f}"
        return self.path / filename

    def load_variable(
        self,
        variable: str,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load a specific variable from the SST dataset.

        Args:
            variable: Variable name (u, v, w, r, p, pv, etc.)
            timestep: Timestep value (uses first available if None)
            subcube: Optional tuple of slices for extracting a subcube region

        Returns:
            Numpy array with shape (nx, ny, nz)
        """
        meta = self.metadata

        # Determine timestep
        if timestep is None:
            timestep = meta.timesteps[0] if meta.timesteps else 0.0
        else:
            timestep = float(timestep)

        file_path = self._get_file_for_variable(variable, timestep)
        if not file_path.exists():
            # Try to find closest timestep
            closest_ts = min(meta.timesteps, key=lambda t: abs(t - timestep), default=None)
            if closest_ts is not None and abs(closest_ts - timestep) < 0.001:
                file_path = self._get_file_for_variable(variable, closest_ts)
            else:
                raise FileNotFoundError(
                    f"No file found for variable '{variable}' at timestep {timestep:.6f}"
                )

        self._log(f"Loading {variable} from {file_path}")

        grid_shape = self._grid_shape
        dtype = np.float32

        # Use memory mapping for large files
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=grid_shape, order='F')

        # Convert to regular array if subcube is requested
        if subcube is not None:
            data = np.array(data[subcube[0], subcube[1], subcube[2]])
        else:
            data = np.array(data)

        # Transpose from Fortran to C order
        data = np.transpose(data, (2, 1, 0))  # (x, y, z) -> (z, y, x)

        return data

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
        meta = self.metadata

        if timestep is None:
            timestep = meta.timesteps[0] if meta.timesteps else 0.0

        if variables is None:
            variables = meta.variables

        data = {}
        for var in variables:
            if var in meta.variables:
                try:
                    data[var] = self.load_variable(var, timestep=timestep)
                except Exception as e:
                    self._log(f"Warning: Could not load {var}: {e}")

        # Generate coordinate arrays
        coords = {
            "x": np.linspace(0, 1, meta.grid_shape[0]),
            "y": np.linspace(0, 1, meta.grid_shape[1]),
            "z": np.linspace(0, 1, meta.grid_shape[2]),
        }

        return LoadedData(
            data=data,
            coordinates=coords,
            metadata=meta
        )

    def load_velocity_field(
        self,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load the full velocity field (u, v, w) as a single array.

        Args:
            timestep: Timestep to load
            subcube: Optional subcube region

        Returns:
            Numpy array with shape (3, nz, ny, nx) containing [u, v, w]
        """
        u = self.load_variable("u", timestep=timestep, subcube=subcube)
        v = self.load_variable("v", timestep=timestep, subcube=subcube)
        w = self.load_variable("w", timestep=timestep, subcube=subcube)

        return np.stack([u, v, w], axis=0)

    def get_statistics(self, variable: str, timestep: Optional[float] = None) -> Dict[str, float]:
        """
        Compute basic statistics for a variable.

        Args:
            variable: Variable name
            timestep: Optional timestep

        Returns:
            Dictionary with min, max, mean, std statistics
        """
        data = self.load_variable(variable, timestep=timestep)

        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }

    def list_timesteps(self) -> List[float]:
        """Get list of available timesteps."""
        return self.metadata.timesteps


# For backward compatibility
DataLoader = SSTBinaryDataLoader
