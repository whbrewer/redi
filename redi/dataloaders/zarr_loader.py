"""
Zarr Data Loader for REDI.

Loads data from Zarr stores, a format for chunked, compressed N-dimensional arrays.
Zarr is commonly used for large scientific datasets in climate science,
remote sensing, and other domains.

Features:
    - Supports both Zarr v2 and v3 formats
    - Handles hierarchical group structures
    - Lazy loading with chunked access
    - Integrates with xarray for labeled dimensions

Reference:
    https://zarr.readthedocs.io/
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataLoader, DatasetMetadata, LoadedData


class ZarrDataLoader(BaseDataLoader):
    """
    Data loader for Zarr format.

    This loader supports:
        - Zarr directory stores (.zarr directories)
        - Hierarchical group structures
        - Chunked array access for memory efficiency
        - Coordinate variables and attributes

    Example usage:
        >>> loader = ZarrDataLoader("/path/to/data.zarr")
        >>> print(loader.metadata)
        >>> temperature = loader.load_variable("temperature")
        >>> snapshot = loader.load_snapshot()
    """

    def __init__(
        self,
        path: str,
        group: Optional[str] = None,
        use_xarray: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the Zarr data loader.

        Args:
            path: Path to the Zarr store (directory)
            group: Optional group path within the store
            use_xarray: Use xarray for labeled dimension support (default True)
            verbose: Enable verbose output
        """
        super().__init__(path, verbose)
        self._group = group
        self._use_xarray = use_xarray
        self._store = None
        self._xr_dataset = None

    def _open_store(self):
        """Open the Zarr store lazily."""
        if self._store is None:
            import zarr

            store_path = str(self.path)
            if self._group:
                self._store = zarr.open_group(store_path, mode='r')[self._group]
            else:
                self._store = zarr.open(store_path, mode='r')

            self._log(f"Opened Zarr store: {store_path}")

        return self._store

    def _open_xarray(self):
        """Open with xarray for labeled dimensions."""
        if self._xr_dataset is None and self._use_xarray:
            try:
                import xarray as xr
                self._xr_dataset = xr.open_zarr(str(self.path), group=self._group)
                self._log("Opened Zarr store with xarray")
            except Exception as e:
                self._log(f"Could not open with xarray: {e}, falling back to zarr")
                self._use_xarray = False

        return self._xr_dataset

    def _scan_directory(self) -> DatasetMetadata:
        """Scan the Zarr store and extract metadata."""
        self._log(f"Scanning Zarr store: {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        # Try xarray first for richer metadata
        if self._use_xarray:
            try:
                return self._scan_with_xarray()
            except Exception as e:
                self._log(f"xarray scan failed: {e}, using zarr directly")

        return self._scan_with_zarr()

    def _scan_with_xarray(self) -> DatasetMetadata:
        """Scan using xarray for richer metadata."""
        import xarray as xr

        ds = xr.open_zarr(str(self.path), group=self._group)
        self._xr_dataset = ds

        # Extract variables (data variables, not coordinates)
        variables = list(ds.data_vars.keys())

        # Extract dimensions
        dimensions = dict(ds.sizes)

        # Calculate total size
        total_size = sum(
            ds[var].nbytes for var in variables
        )

        # Determine grid shape from first variable
        grid_shape = ()
        if variables:
            first_var = ds[variables[0]]
            grid_shape = first_var.shape

        # Look for time dimension
        timesteps = []
        time_dim = None
        for dim_name in ['time', 't', 'timestep', 'step']:
            if dim_name in ds.coords:
                time_dim = dim_name
                time_vals = ds.coords[dim_name].values
                if hasattr(time_vals, 'tolist'):
                    timesteps = [float(t) if isinstance(t, (int, float, np.number))
                                else i for i, t in enumerate(time_vals)]
                break

        # Get dtype from first variable
        dtype = "float32"
        if variables:
            dtype = str(ds[variables[0]].dtype)

        # Extract attributes
        attrs = dict(ds.attrs)

        return DatasetMetadata(
            name=self.path.name,
            path=str(self.path),
            file_format="zarr",
            num_variables=len(variables),
            variables=variables,
            dimensions=dimensions,
            dtype=dtype,
            total_size_bytes=total_size,
            num_timesteps=len(timesteps),
            timesteps=timesteps,
            grid_shape=grid_shape,
            extra={
                "format_version": "zarr",
                "coordinates": list(ds.coords.keys()),
                "attributes": attrs,
                "time_dimension": time_dim,
                "chunks": {var: ds[var].encoding.get('chunks') for var in variables},
            }
        )

    def _scan_with_zarr(self) -> DatasetMetadata:
        """Scan using zarr directly."""
        import zarr

        store = self._open_store()

        # Determine what kind of store we have
        if isinstance(store, zarr.Group):
            return self._scan_group(store)
        elif isinstance(store, zarr.Array):
            return self._scan_array(store)
        else:
            raise ValueError(f"Unexpected zarr type: {type(store)}")

    def _scan_group(self, group) -> DatasetMetadata:
        """Scan a zarr group."""
        import zarr

        variables = []
        coordinates = []
        total_size = 0
        dimensions = {}
        grid_shape = ()
        dtype = "float32"

        # Common coordinate names to exclude from variables
        coord_names = {'time', 't', 'x', 'y', 'z', 'lat', 'lon', 'latitude', 'longitude',
                      'level', 'lev', 'depth', 'height', 'grid_center_lat', 'grid_center_lon',
                      'grid_corner_lat', 'grid_corner_lon', 'grid_dims', 'grid_imask'}

        for name, item in group.items():
            if isinstance(item, zarr.Array):
                total_size += item.nbytes

                if name.lower() in coord_names or name.startswith('grid_'):
                    coordinates.append(name)
                else:
                    variables.append(name)
                    if not grid_shape:
                        grid_shape = item.shape
                        dtype = str(item.dtype)

                    # Build dimensions from array shape
                    for i, size in enumerate(item.shape):
                        dim_name = f"dim_{i}"
                        if dim_name not in dimensions:
                            dimensions[dim_name] = size

        # Look for time information
        timesteps = []
        if 'time' in group:
            time_arr = group['time'][:]
            timesteps = [float(t) if isinstance(t, (int, float, np.number))
                        else i for i, t in enumerate(time_arr)]
        elif 't' in group:
            time_arr = group['t'][:]
            timesteps = [float(t) if isinstance(t, (int, float, np.number))
                        else i for i, t in enumerate(time_arr)]

        return DatasetMetadata(
            name=self.path.name,
            path=str(self.path),
            file_format="zarr",
            num_variables=len(variables),
            variables=variables,
            dimensions=dimensions,
            dtype=dtype,
            total_size_bytes=total_size,
            num_timesteps=len(timesteps),
            timesteps=timesteps,
            grid_shape=grid_shape,
            extra={
                "format_version": "zarr",
                "coordinates": coordinates,
                "attributes": dict(group.attrs),
            }
        )

    def _scan_array(self, array) -> DatasetMetadata:
        """Scan a single zarr array."""
        return DatasetMetadata(
            name=self.path.name,
            path=str(self.path),
            file_format="zarr",
            num_variables=1,
            variables=["data"],
            dimensions={f"dim_{i}": s for i, s in enumerate(array.shape)},
            dtype=str(array.dtype),
            total_size_bytes=array.nbytes,
            num_timesteps=0,
            timesteps=[],
            grid_shape=array.shape,
            extra={
                "format_version": "zarr",
                "attributes": dict(array.attrs),
            }
        )

    def load_variable(
        self,
        variable: str,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load a specific variable from the Zarr store.

        Args:
            variable: Variable name
            timestep: Timestep index or value (uses first if None)
            subcube: Optional tuple of slices for extracting a region

        Returns:
            Numpy array with the variable data
        """
        meta = self.metadata

        # Try xarray first
        if self._use_xarray and self._xr_dataset is not None:
            return self._load_variable_xarray(variable, timestep, subcube)

        # Fall back to zarr
        store = self._open_store()

        if variable not in store:
            raise KeyError(f"Variable '{variable}' not found in Zarr store")

        arr = store[variable]

        # Handle timestep selection
        time_dim = meta.extra.get("time_dimension")
        if timestep is not None and time_dim:
            # Find timestep index
            if isinstance(timestep, int):
                ts_idx = timestep
            else:
                ts_idx = meta.timesteps.index(timestep) if timestep in meta.timesteps else 0

            # Select along time dimension (assumed to be first)
            arr = arr[ts_idx]

        # Apply subcube selection
        if subcube is not None:
            data = np.array(arr[subcube])
        else:
            data = np.array(arr)

        self._log(f"Loaded {variable} with shape {data.shape}")
        return data

    def _load_variable_xarray(
        self,
        variable: str,
        timestep: Optional[Union[int, float]] = None,
        subcube: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """Load variable using xarray."""
        ds = self._xr_dataset

        if variable not in ds:
            raise KeyError(f"Variable '{variable}' not found in dataset")

        da = ds[variable]

        # Handle timestep selection
        time_dim = self.metadata.extra.get("time_dimension")
        if timestep is not None and time_dim and time_dim in da.dims:
            if isinstance(timestep, int):
                da = da.isel({time_dim: timestep})
            else:
                # Try to select by value
                try:
                    da = da.sel({time_dim: timestep}, method='nearest')
                except:
                    da = da.isel({time_dim: 0})

        # Apply subcube selection
        if subcube is not None:
            data = da.values[subcube]
        else:
            data = da.values

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

        if timestep is None and meta.timesteps:
            timestep = 0  # Use first timestep index

        if variables is None:
            variables = meta.variables

        data = {}
        for var in variables:
            if var in meta.variables:
                try:
                    data[var] = self.load_variable(var, timestep=timestep)
                except Exception as e:
                    self._log(f"Warning: Could not load {var}: {e}")

        # Load coordinates
        coords = {}
        coord_names = meta.extra.get("coordinates", [])

        if self._use_xarray and self._xr_dataset is not None:
            for coord in coord_names:
                if coord in self._xr_dataset.coords:
                    coords[coord] = self._xr_dataset.coords[coord].values
        else:
            store = self._open_store()
            for coord in coord_names:
                if coord in store:
                    coords[coord] = np.array(store[coord])

        return LoadedData(
            data=data,
            coordinates=coords,
            metadata=meta
        )

    def load_as_xarray(self):
        """
        Load the entire store as an xarray Dataset.

        Returns:
            xarray.Dataset
        """
        if self._xr_dataset is not None:
            return self._xr_dataset

        import xarray as xr
        return xr.open_zarr(str(self.path), group=self._group)

    def get_chunk_info(self, variable: str) -> Dict[str, Any]:
        """
        Get chunking information for a variable.

        Args:
            variable: Variable name

        Returns:
            Dictionary with chunk sizes and count
        """
        store = self._open_store()

        if variable not in store:
            raise KeyError(f"Variable '{variable}' not found")

        arr = store[variable]

        return {
            "shape": arr.shape,
            "chunks": arr.chunks,
            "dtype": str(arr.dtype),
            "compressor": str(arr.compressor) if hasattr(arr, 'compressor') else None,
            "nbytes": arr.nbytes,
            "nbytes_stored": getattr(arr, 'nbytes_stored', None),
        }

    def list_groups(self) -> List[str]:
        """List all groups in the Zarr store."""
        import zarr

        store = zarr.open(str(self.path), mode='r')

        if isinstance(store, zarr.Group):
            return list(store.group_keys())
        return []

    def list_arrays(self) -> List[str]:
        """List all arrays in the Zarr store."""
        import zarr

        store = zarr.open(str(self.path), mode='r')

        if isinstance(store, zarr.Group):
            return list(store.array_keys())
        return ["data"]

    def close(self):
        """Close the Zarr store and release resources."""
        if self._xr_dataset is not None:
            self._xr_dataset.close()
            self._xr_dataset = None
        self._store = None


# For convenience
DataLoader = ZarrDataLoader
