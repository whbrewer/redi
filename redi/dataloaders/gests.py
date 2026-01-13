"""
GESTS Data Loader for REDI.

Loads GPU-based Extreme-Scale Turbulence Simulations (GESTS) data format
from the SICKLE project. This format is used for forced isotropic turbulence
simulations at extreme scales (2048³ to 8192³ grids).

File format:
    - Binary files stored in subdirectories by variable name
    - Filenames: cube_{var_prefix}.{id}
    - Data stored in Fortran order as float32
    - Velocity is stored as 3-component vector (u, v, w)
    - Other variables (pressure, enstrophy, dissipation) are scalars

Reference:
    Yeung et al. (2025) "GPU-enabled extreme-scale turbulence simulations"
    Computer Physics Communications 306, 109364.
"""

import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataLoader, DatasetMetadata, LoadedData


class GESTSDataLoader(BaseDataLoader):
    """
    Data loader for GESTS (GPU-based Extreme-Scale Turbulence Simulations) format.

    This loader supports:
        - 2048³ and 8192³ grid resolutions
        - Variables: velocity (vector), pressure, enstrophy, dissipation (scalars)
        - Memory-mapped reading for large files
        - Subcube extraction for partial data loading

    Example usage:
        >>> loader = GESTSDataLoader("/path/to/gests/data")
        >>> print(loader.metadata)
        >>> velocity = loader.load_variable("velocity")
        >>> snapshot = loader.load_snapshot(timestep=0)
    """

    # Map user-friendly names to file prefixes
    VARIABLE_MAP = {
        "enstrophy": "enst",
        "velocity": "uvw",
        "pressure": "p",
        "dissipation": "diss",
    }

    # Reverse map for file prefix to variable name
    PREFIX_TO_VAR = {v: k for k, v in VARIABLE_MAP.items()}

    # Variables that are vectors (3 components)
    VECTOR_VARIABLES = {"velocity"}

    def __init__(
        self,
        path: str,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the GESTS data loader.

        Args:
            path: Path to the GESTS data directory
            grid_shape: Optional grid dimensions (nx, ny, nz). If None, attempts to infer.
            verbose: Enable verbose output
        """
        super().__init__(path, verbose)
        self._grid_shape = grid_shape
        self._num_cubes = 8  # For 8192 dataset, data is split into 8 cubes

    def _scan_directory(self) -> DatasetMetadata:
        """Scan the GESTS data directory and extract metadata."""
        self._log(f"Scanning GESTS directory: {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        variables = []
        timesteps = set()
        total_size = 0
        grid_shape = self._grid_shape

        # Look for variable subdirectories or cube files
        # Format 1: variable subdirectories (e.g., velocity/cube_uvw.0)
        # Format 2: numbered subdirectories for 8192 (e.g., 0/cube_enst.0)

        subdirs = [d for d in self.path.iterdir() if d.is_dir()]

        for subdir in subdirs:
            subdir_name = subdir.name

            # Check if it's a numbered cube directory (8192 format)
            if subdir_name.isdigit():
                cube_files = list(subdir.glob("cube_*"))
                for cf in cube_files:
                    # Extract variable prefix from filename
                    match = re.match(r'cube_(\w+)\.(\d+)', cf.name)
                    if match:
                        prefix = match.group(1)
                        ts_id = int(match.group(2))
                        timesteps.add(ts_id)

                        var_name = self.PREFIX_TO_VAR.get(prefix, prefix)
                        if var_name not in variables:
                            variables.append(var_name)

                        total_size += cf.stat().st_size

            # Check if it's a variable name directory
            elif subdir_name in self.VARIABLE_MAP or subdir_name in self.PREFIX_TO_VAR:
                var_name = self.PREFIX_TO_VAR.get(subdir_name, subdir_name)
                if var_name not in variables:
                    variables.append(var_name)

                cube_files = list(subdir.glob("cube_*"))
                for cf in cube_files:
                    match = re.match(r'cube_\w+\.(\d+)', cf.name)
                    if match:
                        ts_id = int(match.group(1))
                        timesteps.add(ts_id)
                        total_size += cf.stat().st_size

        # Try to infer grid shape from file sizes if not provided
        if grid_shape is None and total_size > 0 and variables:
            # Find a scalar variable file to infer grid size
            for var in variables:
                if var not in self.VECTOR_VARIABLES:
                    files = self._get_files_for_variable(var)
                    if files:
                        file_size = files[0].stat().st_size
                        # Assume float32
                        total_points = file_size // 4
                        # Assume cubic grid
                        dim = int(round(total_points ** (1/3)))
                        if dim ** 3 == total_points:
                            grid_shape = (dim, dim, dim)
                            self._log(f"Inferred grid shape: {grid_shape}")
                        break

        # Default to 2048 if we couldn't infer
        if grid_shape is None:
            grid_shape = (2048, 2048, 2048)
            self._log(f"Using default grid shape: {grid_shape}")

        self._grid_shape = grid_shape

        sorted_timesteps = sorted(timesteps)

        return DatasetMetadata(
            name=self.path.name,
            path=str(self.path),
            file_format="gests",
            num_variables=len(variables),
            variables=variables,
            dimensions={
                "x": grid_shape[0],
                "y": grid_shape[1],
                "z": grid_shape[2],
            },
            dtype="float32",
            total_size_bytes=total_size,
            num_timesteps=len(sorted_timesteps),
            timesteps=[float(t) for t in sorted_timesteps],
            grid_shape=grid_shape,
            extra={
                "format_version": "gests",
                "vector_variables": list(self.VECTOR_VARIABLES),
            }
        )

    def _get_files_for_variable(self, variable: str, timestep: Optional[int] = None) -> List[Path]:
        """Get file paths for a given variable and optional timestep."""
        prefix = self.VARIABLE_MAP.get(variable, variable)

        # Try standard format: variable_dir/cube_prefix.timestep
        var_dir = self.path / variable
        if var_dir.exists():
            pattern = f"cube_{prefix}.*" if timestep is None else f"cube_{prefix}.{timestep}"
            files = sorted(var_dir.glob(pattern), key=lambda x: int(x.suffix[1:]))
            if files:
                return files

        # Try 8192 format: numbered_dir/cube_prefix.id
        numbered_dirs = [d for d in self.path.iterdir() if d.is_dir() and d.name.isdigit()]
        if numbered_dirs:
            files = []
            for ndir in sorted(numbered_dirs, key=lambda x: int(x.name)):
                pattern = f"cube_{prefix}.*" if timestep is None else f"cube_{prefix}.{timestep}"
                files.extend(sorted(ndir.glob(pattern), key=lambda x: int(x.suffix[1:])))
            return files

        return []

    def load_variable(
        self,
        variable: str,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load a specific variable from the GESTS dataset.

        Args:
            variable: Variable name (velocity, pressure, enstrophy, dissipation)
            timestep: Timestep index (uses first available if None)
            subcube: Optional tuple of slices for extracting a subcube region

        Returns:
            Numpy array with shape (nx, ny, nz) for scalars or (3, nx, ny, nz) for vectors
        """
        meta = self.metadata
        is_vector = variable in self.VECTOR_VARIABLES

        # Determine timestep
        if timestep is None:
            timestep = int(meta.timesteps[0]) if meta.timesteps else 0
        else:
            timestep = int(timestep)

        files = self._get_files_for_variable(variable, timestep)
        if not files:
            raise FileNotFoundError(
                f"No files found for variable '{variable}' at timestep {timestep}"
            )

        self._log(f"Loading {variable} from {files[0]}")

        # Read binary data
        grid_shape = self._grid_shape
        dtype = np.float32

        if is_vector:
            shape = (3, *grid_shape)
        else:
            shape = grid_shape

        # Use memory mapping for large files
        data = np.memmap(files[0], dtype=dtype, mode='r', shape=shape, order='F')

        # Convert to regular array if subcube is requested
        if subcube is not None:
            if is_vector:
                data = np.array(data[:, subcube[0], subcube[1], subcube[2]])
            else:
                data = np.array(data[subcube[0], subcube[1], subcube[2]])
        else:
            data = np.array(data)

        # Transpose from Fortran to C order (x,y,z -> z,y,x for easier visualization)
        if is_vector:
            data = np.transpose(data, (0, 3, 2, 1))  # (3, x, y, z) -> (3, z, y, x)
        else:
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
            timestep = meta.timesteps[0] if meta.timesteps else 0

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

    def get_statistics(self, variable: str, timestep: Optional[int] = None) -> Dict[str, float]:
        """
        Compute basic statistics for a variable without loading entire array.

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


# For backward compatibility
DataLoader = GESTSDataLoader
