#!/usr/bin/env python3
"""
REDI Assessment Module

Data analysis, readiness assessment, and pipeline recommendations.
Extracted from analyze.py for separation of concerns.
"""

import json
import csv
from typing import Optional, List, Dict, Any
from pathlib import Path

from .datacard import (
    MaturityLevel,
    ScientificDomain,
    DataCard,
)


# =============================================================================
# DOMAIN CONSTANTS
# =============================================================================

# Domain-specific file extensions and keywords
DOMAIN_FILE_PATTERNS = {
    ScientificDomain.CLIMATE: {
        "extensions": [".nc", ".nc4", ".grib", ".grib2", ".zarr"],
        "keywords": ["temperature", "precipitation", "wind", "pressure", "humidity",
                     "geopotential", "lat", "lon", "latitude", "longitude", "era5", "cmip"]
    },
    ScientificDomain.MATERIALS: {
        "extensions": [".cif", ".poscar", ".vasp", ".xyz"],
        "keywords": ["compound", "formula", "lattice", "crystal", "band_gap", "enthalpy",
                     "formation_energy", "density", "auid", "aflow", "element", "magpie",
                     "mendeleev", "atomicweight", "electronegativity", "valence", "cif",
                     "structure", "spacegroup", "composition", "alloy", "material"]
    },
    ScientificDomain.LIFE_SCIENCES: {
        "extensions": [".fasta", ".fastq", ".pdb", ".mmcif", ".bam", ".vcf"],
        "keywords": ["sequence", "protein", "gene", "amino", "nucleotide", "residue",
                     "chain", "patient", "sample", "alignment"]
    },
    ScientificDomain.FUSION: {
        "extensions": [".hdf5", ".h5", ".adios", ".bp"],
        "keywords": ["plasma", "disruption", "tokamak", "magnetic", "confinement",
                     "shot", "pulse", "diagnostic", "equilibrium"]
    },
    ScientificDomain.HPC: {
        "extensions": [],
        "keywords": ["job", "node", "core", "walltime", "runtime", "queue", "allocation",
                     "user", "project", "exit_code", "gpu", "cobalt", "slurm", "pbs"]
    },
    ScientificDomain.GEOSPATIAL: {
        "extensions": [".shp", ".geojson", ".gpkg", ".kml"],
        "keywords": ["h3", "hex", "hexagon", "h3_index", "grid_id", "centroid",
                     "shapefile", "polygon", "geojson", "wgs84", "epsg",
                     "smoke", "pm25", "pm2.5", "aqi", "exposure",
                     "spatial", "geospatial"]
    },
    ScientificDomain.COMBUSTION: {
        "extensions": [],
        "keywords": ["flame", "combustion", "amrex", "x_velocity", "y_velocity",
                     "rhoh", "divu", "mag_vort", "rhoRT", "fuel", "oxidizer",
                     "detonation", "deflagration"]
    }
}

# Domain-specific preprocessing pipelines
DOMAIN_PIPELINES = {
    ScientificDomain.CLIMATE: {
        "name": "Climate Data Pipeline",
        "steps": [
            ("download", "Acquire data from sources (ERA5, CMIP6, etc.)"),
            ("regrid", "Regrid to uniform spatial/temporal resolution"),
            ("normalize", "Compute and apply normalization (mean/std)"),
            ("shard", "Partition into shards (.npz, .zarr)")
        ],
        "tools": ["xarray", "xesmf", "netCDF4", "zarr"],
        "output_formats": ["NPZ", "Zarr", "TFRecord"]
    },
    ScientificDomain.MATERIALS: {
        "name": "Materials Science Pipeline",
        "steps": [
            ("download", "Fetch from databases (AFLOW, Materials Project)"),
            ("parse", "Parse structure files (CIF, POSCAR)"),
            ("featurize", "Extract composition/structure features"),
            ("encode", "Encode as graphs for GNNs if needed"),
            ("normalize", "Normalize feature values"),
            ("shard", "Partition into training shards")
        ],
        "tools": ["pymatgen", "matminer", "ase", "torch_geometric"],
        "output_formats": ["CSV", "HDF5", "PTH"]
    },
    ScientificDomain.LIFE_SCIENCES: {
        "name": "Life Sciences Pipeline",
        "steps": [
            ("encode", "Encode sequences (tokenization, embeddings)"),
            ("anonymize", "Remove/hash PII if human data present"),
            ("align", "Perform alignments (MSA for proteins, etc.)"),
            ("validate", "Validate data quality and completeness"),
            ("shard", "Partition into training shards")
        ],
        "tools": ["biopython", "mmseqs2", "openfold", "alphafold"],
        "output_formats": ["HDF5", "TFRecord", "LMDB"]
    },
    ScientificDomain.FUSION: {
        "name": "Fusion Energy Pipeline",
        "steps": [
            ("extract", "Extract signals from diagnostic systems"),
            ("align", "Temporally align multimodal signals"),
            ("engineer", "Engineer physics-based features"),
            ("label", "Add disruption/event labels"),
            ("shard", "Partition into training shards")
        ],
        "tools": ["MDSplus", "ADIOS", "h5py"],
        "output_formats": ["HDF5", "ADIOS", "NPZ"]
    },
    ScientificDomain.HPC: {
        "name": "HPC/Operational Data Pipeline",
        "steps": [
            ("ingest", "Load and validate data"),
            ("clean", "Handle missing values, remove duplicates"),
            ("normalize", "Normalize numerical features"),
            ("engineer", "Create derived features (utilization, wait times)"),
            ("encode", "Encode categorical variables"),
            ("shard", "Partition into training shards")
        ],
        "tools": ["pandas", "scikit-learn", "polars"],
        "output_formats": ["Parquet", "CSV", "HDF5"]
    },
    ScientificDomain.GEOSPATIAL: {
        "name": "Geospatial / H3 Hexify Pipeline",
        "steps": [
            ("ingest", "Load tabular or shapefile data with geographic coordinates"),
            ("hexify", "Convert coordinates/grid cells to H3 hexagonal indices"),
            ("aggregate", "Aggregate values within H3 cells (mean, sum, count)"),
            ("normalize", "Normalize numerical features"),
            ("shard", "Partition into training shards with spatial/temporal splits")
        ],
        "tools": ["h3", "geopandas", "shapely", "pandas"],
        "output_formats": ["Parquet", "NPZ", "CSV"]
    },
    ScientificDomain.COMBUSTION: {
        "name": "AMReX Combustion AMR Pipeline",
        "steps": [
            ("ingest", "Parse AMReX plotfile Header and binary FAB files per level"),
            ("preprocess", "Clip species fractions to [0,1], validate physics"),
            ("transform", "Derive vorticity/fuel proxy; normalize by reference scales"),
            ("composite", "Composite AMR levels to uniform finest-level grid"),
            ("shard", "Save snapshot as NPZ with coordinates and metadata")
        ],
        "tools": ["numpy"],
        "output_formats": ["NPZ", "HDF5"]
    },
    ScientificDomain.GENERAL: {
        "name": "General Data Pipeline",
        "steps": [
            ("ingest", "Load data from source"),
            ("clean", "Handle missing values and outliers"),
            ("transform", "Apply necessary transformations"),
            ("normalize", "Normalize/standardize features"),
            ("shard", "Partition into training shards")
        ],
        "tools": ["pandas", "numpy", "scikit-learn"],
        "output_formats": ["CSV", "Parquet", "NPZ"]
    }
}


# =============================================================================
# DATA ANALYZER
# =============================================================================

class DataAnalyzer:
    """Analyzes datasets to generate Data Cards and assess readiness"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data_card = DataCard()
        self._file_stats = {}

    def analyze(self) -> DataCard:
        """Run full analysis on the dataset"""
        self._analyze_file_metadata()
        self._detect_domain()
        self._analyze_structure()
        self._assess_readiness()
        self._detect_sensitive_data()
        return self.data_card

    def _analyze_file_metadata(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        stat = self.file_path.stat()
        self.data_card.overview.name = self.file_path.name
        self.data_card.overview.size_bytes = stat.st_size
        self.data_card.overview.file_format = self.file_path.suffix.lower()

        self._file_stats = {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": self.file_path.suffix.lower()
        }

    def _detect_domain(self) -> ScientificDomain:
        ext = self._file_stats.get("extension", "").lower()

        for domain, patterns in DOMAIN_FILE_PATTERNS.items():
            if ext in patterns["extensions"]:
                self.data_card.detected_domain = domain.value
                self.data_card.characteristics.domain = domain.value
                return domain

        if ext in [".csv", ".tsv", ".json", ".parquet"]:
            domain = self._detect_domain_from_content()
            self.data_card.detected_domain = domain.value
            self.data_card.characteristics.domain = domain.value
            return domain

        self.data_card.detected_domain = ScientificDomain.GENERAL.value
        self.data_card.characteristics.domain = ScientificDomain.GENERAL.value
        return ScientificDomain.GENERAL

    def _detect_domain_from_content(self) -> ScientificDomain:
        ext = self._file_stats.get("extension", "").lower()
        columns = []

        if ext == ".csv":
            try:
                with open(self.file_path, 'r') as f:
                    reader = csv.reader(f)
                    columns = [c.lower() for c in next(reader)]
            except Exception:
                pass
        elif ext == ".json":
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        columns = [k.lower() for k in data[0].keys()]
                    elif isinstance(data, dict):
                        columns = [k.lower() for k in data.keys()]
            except Exception:
                pass

        if not columns:
            return ScientificDomain.GENERAL

        scores = {domain: 0 for domain in ScientificDomain}
        columns_lower = [c.lower() for c in columns]
        column_text = ' '.join(columns_lower)

        for domain, patterns in DOMAIN_FILE_PATTERNS.items():
            for keyword in patterns["keywords"]:
                keyword_lower = keyword.lower()
                for col in columns_lower:
                    if keyword_lower in col:
                        scores[domain] += 1
                if keyword_lower in column_text:
                    scores[domain] += 0.5

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain
        return ScientificDomain.GENERAL

    def _analyze_structure(self):
        ext = self._file_stats.get("extension", "").lower()

        if ext == ".csv":
            self._analyze_csv()
        elif ext == ".json":
            self._analyze_json()
        elif ext in [".nc", ".nc4"]:
            self._analyze_netcdf()
        elif ext == ".npz":
            self._analyze_npz()
        elif ext == ".npy":
            self._analyze_npy()
        elif ext in [".h5", ".hdf5"]:
            self._analyze_hdf5()
        elif ext == ".parquet":
            self._analyze_parquet()
        elif ext in [".fasta", ".fastq", ".fa", ".fq"]:
            self._analyze_fasta()
        elif ext in [".pdb", ".mmcif", ".cif"]:
            self._analyze_structure_file()
        elif ext == ".zarr":
            self._analyze_zarr()
        else:
            self.data_card.overview.description = f"Binary/specialized format: {ext}"

    def _analyze_csv(self):
        try:
            with open(self.file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)

                row_count = 0
                sample_rows = []
                for row in reader:
                    row_count += 1
                    if len(sample_rows) < 100:
                        sample_rows.append(row)
                    if row_count >= 1000000:
                        break

            self.data_card.overview.num_records = row_count
            self.data_card.overview.num_features = len(headers)
            self.data_card.characteristics.data_types = ["tabular"]

            features = []
            for i, header in enumerate(headers):
                feat = {
                    "name": header,
                    "type": self._infer_column_type(sample_rows, i),
                    "description": ""
                }
                features.append(feat)

            self.data_card.characteristics.features = features

        except Exception as e:
            self.data_card.overview.description = f"Error analyzing CSV: {str(e)}"

    def _analyze_json(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                self.data_card.overview.num_records = len(data)
                if len(data) > 0 and isinstance(data[0], dict):
                    self.data_card.overview.num_features = len(data[0].keys())
                    features = [{"name": k, "type": type(v).__name__, "description": ""}
                               for k, v in data[0].items()]
                    self.data_card.characteristics.features = features
            elif isinstance(data, dict):
                self.data_card.overview.num_features = len(data.keys())
                features = [{"name": k, "type": type(v).__name__, "description": ""}
                           for k, v in data.items()]
                self.data_card.characteristics.features = features

            self.data_card.characteristics.data_types = ["json"]

        except Exception as e:
            self.data_card.overview.description = f"Error analyzing JSON: {str(e)}"

    def _analyze_netcdf(self):
        """Analyze NetCDF file using xarray"""
        try:
            import xarray as xr

            ds = xr.open_dataset(self.file_path)

            # Get dimensions
            dims = dict(ds.sizes)
            variables = list(ds.data_vars)
            coords = list(ds.coords)

            # Calculate total records (product of dimensions or largest dimension)
            if dims:
                # Use the largest dimension as record count for display
                self.data_card.overview.num_records = max(dims.values()) if dims else 0

            # Number of features = number of data variables
            self.data_card.overview.num_features = len(variables)

            # Set data types
            self.data_card.characteristics.data_types = ["gridded", "multidimensional", "netcdf"]

            # Build description
            dim_str = ", ".join(f"{k}={v}" for k, v in dims.items())
            self.data_card.overview.description = f"NetCDF dataset with {len(variables)} variables, dimensions: {dim_str}"

            # Extract features (variables)
            features = []
            for var in variables:
                da = ds[var]
                shape_str = str(da.shape)
                dtype_str = str(da.dtype)

                # Get variable attributes if available
                desc = da.attrs.get('long_name', da.attrs.get('description', ''))
                units = da.attrs.get('units', '')
                if units:
                    desc = f"{desc} [{units}]" if desc else f"[{units}]"

                features.append({
                    "name": var,
                    "type": f"{dtype_str} {shape_str}",
                    "description": desc
                })

            self.data_card.characteristics.features = features

            # Extract temporal coverage
            for time_name in ["time", "t", "TIME"]:
                if time_name in ds.coords:
                    time_coord = ds[time_name]
                    try:
                        start = str(time_coord.values[0])[:19]
                        end = str(time_coord.values[-1])[:19]
                        self.data_card.characteristics.temporal_coverage = f"{start} to {end}"
                    except (IndexError, TypeError):
                        pass
                    break

            # Extract spatial coverage
            lat_range = None
            lon_range = None

            for lat_name in ["lat", "latitude", "LAT", "LATITUDE", "grid_center_lat"]:
                if lat_name in ds.coords or lat_name in ds.data_vars:
                    lat_data = ds[lat_name]
                    lat_range = f"{float(lat_data.min()):.2f} to {float(lat_data.max()):.2f}"
                    break

            for lon_name in ["lon", "longitude", "LON", "LONGITUDE", "grid_center_lon"]:
                if lon_name in ds.coords or lon_name in ds.data_vars:
                    lon_data = ds[lon_name]
                    lon_range = f"{float(lon_data.min()):.2f} to {float(lon_data.max()):.2f}"
                    break

            if lat_range and lon_range:
                self.data_card.characteristics.spatial_coverage = f"Lat: {lat_range}, Lon: {lon_range}"

            # Store additional metadata
            self._netcdf_metadata = {
                "dimensions": dims,
                "variables": variables,
                "coordinates": coords,
                "global_attrs": dict(ds.attrs)
            }

            ds.close()

        except ImportError:
            # Fall back to stub if xarray not available
            self.data_card.characteristics.data_types = ["gridded", "multidimensional"]
            self.data_card.overview.description = "NetCDF file - install xarray for detailed analysis"
        except Exception as e:
            self.data_card.characteristics.data_types = ["gridded", "multidimensional"]
            self.data_card.overview.description = f"NetCDF file - error during analysis: {str(e)}"

    def _analyze_npz(self):
        """Analyze NumPy NPZ archive."""
        try:
            import numpy as np

            data = np.load(self.file_path, allow_pickle=False)
            try:
                arrays = dict(data)
            except ValueError:
                data.close()
                data = np.load(self.file_path, allow_pickle=True)
                arrays = dict(data)
            data.close()

            self.data_card.characteristics.data_types = ["array", "npz"]

            features = []
            total_elements = 0
            for key, arr in arrays.items():
                total_elements += arr.size
                try:
                    stat_str = (
                        f"min={arr.min():.4g}, max={arr.max():.4g}"
                        if arr.size > 0 and arr.dtype.kind in "fiu" else ""
                    )
                except Exception:
                    stat_str = ""
                features.append({
                    "name": key,
                    "type": f"{arr.dtype} {arr.shape}",
                    "description": stat_str
                })

            self.data_card.characteristics.features = features
            self.data_card.overview.num_features = len(features)
            self.data_card.overview.num_records = total_elements
            self.data_card.overview.description = (
                f"NumPy NPZ archive with {len(features)} array(s), "
                f"{total_elements:,} total elements"
            )

            # Check for split structure (train/val/test keys)
            keys_lower = {k.lower() for k in arrays.keys()}
            if keys_lower & {"x_train", "y_train", "train"}:
                self.data_card.transformations.train_val_test_split = "present"
                self.data_card.readiness_level = max(self.data_card.readiness_level, 5)

        except Exception as e:
            self.data_card.characteristics.data_types = ["array", "npz"]
            self.data_card.overview.description = f"NumPy NPZ archive - error during analysis: {str(e)}"

    def _analyze_npy(self):
        """Analyze NumPy NPY file (single array)."""
        try:
            import numpy as np

            arr = np.load(self.file_path, allow_pickle=False)
            self.data_card.characteristics.data_types = ["array", "npy"]
            self.data_card.overview.num_records = arr.shape[0] if arr.ndim > 0 else 1
            self.data_card.overview.num_features = arr.shape[1] if arr.ndim > 1 else 1

            stat_desc = ""
            if arr.size > 0 and arr.dtype.kind in "fiu":
                stat_desc = f", min={arr.min():.4g}, max={arr.max():.4g}, mean={arr.mean():.4g}"

            self.data_card.overview.description = (
                f"NumPy array: shape={arr.shape}, dtype={arr.dtype}{stat_desc}"
            )
            self.data_card.characteristics.features = [{
                "name": "array",
                "type": f"{arr.dtype} {arr.shape}",
                "description": stat_desc.lstrip(", ")
            }]

        except Exception as e:
            self.data_card.characteristics.data_types = ["array", "npy"]
            self.data_card.overview.description = f"NumPy NPY file - error during analysis: {str(e)}"

    def _analyze_hdf5(self):
        """Analyze HDF5 file."""
        try:
            import h5py

            def _collect_items(name, obj, items):
                if isinstance(obj, h5py.Dataset):
                    items.append((name, obj.shape, obj.dtype))

            items = []
            total_elements = 0
            with h5py.File(self.file_path, 'r') as f:
                f.visititems(lambda n, o: _collect_items(n, o, items))

            self.data_card.characteristics.data_types = ["array", "hdf5"]
            self.data_card.overview.num_features = len(items)

            features = []
            for name, shape, dtype in items:
                n = 1
                for s in shape:
                    n *= s
                total_elements += n
                features.append({
                    "name": name,
                    "type": f"{dtype} {shape}",
                    "description": ""
                })

            self.data_card.characteristics.features = features
            self.data_card.overview.num_records = total_elements
            self.data_card.overview.description = (
                f"HDF5 file with {len(items)} dataset(s), {total_elements:,} total elements"
            )

        except ImportError:
            self.data_card.characteristics.data_types = ["array", "hdf5"]
            self.data_card.overview.description = "HDF5 file - install h5py for detailed analysis"
        except Exception as e:
            self.data_card.characteristics.data_types = ["array", "hdf5"]
            self.data_card.overview.description = f"HDF5 file - error during analysis: {str(e)}"

    def _analyze_parquet(self):
        """Analyze Parquet file."""
        try:
            import pandas as pd

            df = pd.read_parquet(self.file_path)
            self.data_card.overview.num_records = len(df)
            self.data_card.overview.num_features = len(df.columns)
            self.data_card.characteristics.data_types = ["tabular", "parquet"]

            features = []
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                null_count = int(df[col].isna().sum())
                desc = f"null={null_count}" if null_count > 0 else ""
                features.append({
                    "name": col,
                    "type": dtype_str,
                    "description": desc
                })

            self.data_card.characteristics.features = features
            self.data_card.overview.description = (
                f"Parquet file: {len(df):,} rows × {len(df.columns)} columns"
            )

            # Re-run domain detection with actual column names
            columns = list(df.columns)
            domain = self._detect_domain_from_columns(columns)
            self.data_card.detected_domain = domain.value
            self.data_card.characteristics.domain = domain.value

        except ImportError:
            self.data_card.characteristics.data_types = ["tabular", "parquet"]
            self.data_card.overview.description = "Parquet file - install pandas/pyarrow for detailed analysis"
        except Exception as e:
            self.data_card.characteristics.data_types = ["tabular", "parquet"]
            self.data_card.overview.description = f"Parquet file - error during analysis: {str(e)}"

    def _analyze_fasta(self):
        """Analyze FASTA/FASTQ sequence file."""
        try:
            record_count = 0
            total_length = 0
            min_len = None
            max_len = None

            with open(self.file_path, 'r') as f:
                current_len = 0
                for line in f:
                    line = line.strip()
                    if line.startswith('>') or line.startswith('@'):
                        if record_count > 0:
                            total_length += current_len
                            if min_len is None or current_len < min_len:
                                min_len = current_len
                            if max_len is None or current_len > max_len:
                                max_len = current_len
                        record_count += 1
                        current_len = 0
                    elif line and not line.startswith('+') and not line.startswith('#'):
                        current_len += len(line)

                # Last record
                if record_count > 0 and current_len > 0:
                    total_length += current_len
                    if min_len is None or current_len < min_len:
                        min_len = current_len
                    if max_len is None or current_len > max_len:
                        max_len = current_len

            avg_len = total_length // record_count if record_count > 0 else 0
            ext = self._file_stats.get("extension", "").lower()
            fmt = "FASTQ" if ext in [".fastq", ".fq"] else "FASTA"

            self.data_card.characteristics.data_types = ["sequence", "biological"]
            self.data_card.characteristics.domain = ScientificDomain.LIFE_SCIENCES.value
            self.data_card.detected_domain = ScientificDomain.LIFE_SCIENCES.value
            self.data_card.overview.num_records = record_count
            self.data_card.overview.num_features = 1
            self.data_card.overview.description = (
                f"{fmt}: {record_count:,} sequences, avg length {avg_len:,} bp"
                + (f" (min={min_len}, max={max_len})" if min_len is not None else "")
            )
            self.data_card.characteristics.features = [{
                "name": "sequence",
                "type": "str",
                "description": f"avg_length={avg_len}, min={min_len}, max={max_len}"
            }]

        except Exception as e:
            self.data_card.characteristics.data_types = ["sequence", "biological"]
            self.data_card.overview.description = f"FASTA/FASTQ file - error during analysis: {str(e)}"

    def _analyze_structure_file(self):
        """Analyze PDB/mmCIF/CIF structure files."""
        ext = self._file_stats.get("extension", "").lower()

        if ext == ".cif" and "POSCAR" not in str(self.file_path):
            # Could be a crystal structure (materials) or mmCIF (protein)
            # Check content to distinguish
            try:
                with open(self.file_path, 'r') as f:
                    header = f.read(512)
                if "_atom_site" in header or "_cell" in header:
                    self._analyze_cif_crystal()
                    return
            except Exception:
                pass

        if ext in [".pdb", ".mmcif"]:
            self._analyze_pdb()
        elif ext == ".cif":
            self._analyze_cif_crystal()

    def _analyze_pdb(self):
        """Analyze PDB/mmCIF protein structure file."""
        try:
            atom_count = 0
            residue_ids = set()
            chain_ids = set()

            with open(self.file_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_count += 1
                        try:
                            chain_ids.add(line[21])
                            residue_ids.add((line[21], line[22:26].strip()))
                        except IndexError:
                            pass

            self.data_card.characteristics.data_types = ["structure", "protein"]
            self.data_card.characteristics.domain = ScientificDomain.LIFE_SCIENCES.value
            self.data_card.detected_domain = ScientificDomain.LIFE_SCIENCES.value
            self.data_card.overview.num_records = atom_count
            self.data_card.overview.description = (
                f"PDB structure: {atom_count:,} atoms, {len(residue_ids)} residues, "
                f"{len(chain_ids)} chain(s)"
            )
            self.data_card.characteristics.features = [
                {"name": "atoms", "type": "int", "description": str(atom_count)},
                {"name": "residues", "type": "int", "description": str(len(residue_ids))},
                {"name": "chains", "type": "int", "description": str(len(chain_ids))},
            ]

        except Exception as e:
            self.data_card.characteristics.data_types = ["structure", "protein"]
            self.data_card.overview.description = f"PDB file - error during analysis: {str(e)}"

    def _analyze_cif_crystal(self):
        """Analyze CIF crystal structure file."""
        try:
            formula = ""
            spacegroup = ""
            with open(self.file_path, 'r') as f:
                for line in f:
                    if "_chemical_formula" in line.lower() and not formula:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            formula = parts[1].strip("'\"")
                    if "_symmetry_space_group" in line.lower() and not spacegroup:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            spacegroup = parts[1].strip("'\"")
                    if formula and spacegroup:
                        break

            self.data_card.characteristics.data_types = ["structure", "crystal"]
            self.data_card.characteristics.domain = ScientificDomain.MATERIALS.value
            self.data_card.detected_domain = ScientificDomain.MATERIALS.value
            self.data_card.overview.num_records = 1
            self.data_card.overview.description = (
                f"CIF crystal structure"
                + (f": {formula}" if formula else "")
                + (f", space group {spacegroup}" if spacegroup else "")
            )

        except Exception as e:
            self.data_card.characteristics.data_types = ["structure", "crystal"]
            self.data_card.overview.description = f"CIF file - error during analysis: {str(e)}"

    def _analyze_zarr(self):
        """Analyze Zarr array store."""
        try:
            import zarr

            store = zarr.open(str(self.file_path), mode='r')
            self.data_card.characteristics.data_types = ["array", "zarr"]

            features = []
            total_elements = 0

            def _visit(name, obj):
                if isinstance(obj, zarr.Array):
                    total_elements_local = 1
                    for s in obj.shape:
                        total_elements_local *= s
                    features.append({
                        "name": name,
                        "type": f"{obj.dtype} {obj.shape}",
                        "description": f"chunks={obj.chunks}"
                    })
                    return total_elements_local
                return 0

            if hasattr(store, 'visititems'):
                store.visititems(lambda n, o: features.append({
                    "name": n,
                    "type": str(getattr(o, 'dtype', 'group')),
                    "description": str(getattr(o, 'shape', ''))
                }) if hasattr(o, 'shape') else None)
            else:
                for key in store.keys():
                    arr = store[key]
                    if hasattr(arr, 'shape'):
                        sz = 1
                        for s in arr.shape:
                            sz *= s
                        total_elements += sz
                        features.append({
                            "name": key,
                            "type": f"{arr.dtype} {arr.shape}",
                            "description": f"chunks={arr.chunks}" if hasattr(arr, 'chunks') else ""
                        })

            self.data_card.characteristics.features = features
            self.data_card.overview.num_features = len(features)
            self.data_card.overview.description = (
                f"Zarr store with {len(features)} array(s)"
            )

        except ImportError:
            self.data_card.characteristics.data_types = ["array", "zarr"]
            self.data_card.overview.description = "Zarr store - install zarr for detailed analysis"
        except Exception as e:
            self.data_card.characteristics.data_types = ["array", "zarr"]
            self.data_card.overview.description = f"Zarr store - error during analysis: {str(e)}"

    def _detect_domain_from_columns(self, columns: List[str]) -> 'ScientificDomain':
        """Detect domain from a list of column names (used by parquet analyzer)."""
        columns_lower = [c.lower() for c in columns]
        column_text = ' '.join(columns_lower)

        scores = {domain: 0 for domain in ScientificDomain}
        for domain, patterns in DOMAIN_FILE_PATTERNS.items():
            for keyword in patterns["keywords"]:
                kw = keyword.lower()
                for col in columns_lower:
                    if kw in col:
                        scores[domain] += 1
                if kw in column_text:
                    scores[domain] += 0.5

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return ScientificDomain.GENERAL

    def _infer_column_type(self, sample_rows: List[List[str]], col_idx: int) -> str:
        if not sample_rows:
            return "unknown"

        values = [row[col_idx] for row in sample_rows if col_idx < len(row)]

        numeric_count = 0
        int_count = 0
        for val in values[:50]:
            try:
                float(val)
                numeric_count += 1
                if '.' not in val:
                    int_count += 1
            except (ValueError, TypeError):
                pass

        if numeric_count > len(values) * 0.8:
            return "integer" if int_count > numeric_count * 0.9 else "float"

        bool_values = {'true', 'false', '0', '1', 'yes', 'no'}
        if all(str(v).lower() in bool_values for v in values[:20]):
            return "boolean"

        date_patterns = ['-', '/', ':']
        if any(any(p in str(v) for p in date_patterns) for v in values[:10]):
            return "datetime"

        return "string"

    def _detect_sensitive_data(self):
        sensitive_keywords = [
            'name', 'email', 'phone', 'address', 'ssn', 'social_security',
            'patient', 'dob', 'birth', 'gender', 'sex', 'race', 'ethnicity',
            'income', 'salary', 'password', 'credit_card', 'account'
        ]

        features = self.data_card.characteristics.features
        pii_detected = []

        for feat in features:
            feat_name = feat.get('name', '').lower()
            for keyword in sensitive_keywords:
                if keyword in feat_name:
                    pii_detected.append(feat.get('name', ''))
                    break

        if pii_detected:
            self.data_card.sensitive_attributes.contains_pii = True
            self.data_card.sensitive_attributes.pii_types = pii_detected

    def _assess_readiness(self):
        level = 1
        features = self.data_card.characteristics.features
        has_features = len(features) > 0

        if has_features:
            level = 2

        label_keywords = ['label', 'target', 'class', 'category', 'y', 'output']
        for feat in features:
            if any(kw in feat.get('name', '').lower() for kw in label_keywords):
                level = 3
                self.data_card.annotations.has_labels = True
                self.data_card.annotations.label_types.append(feat.get('name', ''))
                break

        engineered_keywords = ['_mean', '_std', '_max', '_min', '_ratio', '_normalized', '_encoded']
        for feat in features:
            if any(kw in feat.get('name', '').lower() for kw in engineered_keywords):
                level = 4
                break

        ai_formats = ['.tfrecord', '.npz', '.h5', '.hdf5', '.zarr', '.pth']
        if self._file_stats.get('extension', '').lower() in ai_formats:
            level = 5

        self.data_card.readiness_level = level


# =============================================================================
# PIPELINE RECOMMENDER
# =============================================================================

class PipelineRecommender:
    """Recommends preprocessing pipeline based on domain and readiness level"""

    def __init__(self, data_card: DataCard):
        self.data_card = data_card
        self.domain = ScientificDomain(data_card.detected_domain)
        self.current_level = MaturityLevel(data_card.readiness_level)

    def get_recommendations(self) -> Dict[str, Any]:
        pipeline = DOMAIN_PIPELINES.get(self.domain, DOMAIN_PIPELINES[ScientificDomain.GENERAL])
        needed_steps = self._get_needed_steps(pipeline['steps'])

        return {
            "domain": self.domain.value,
            "pipeline_name": pipeline['name'],
            "current_level": self.current_level.name,
            "current_level_description": self.current_level.description(),
            "target_level": MaturityLevel.FULLY_READY.name,
            "needed_steps": needed_steps,
            "recommended_tools": pipeline['tools'],
            "output_formats": pipeline['output_formats'],
            "full_pipeline": pipeline['steps']
        }

    def _get_needed_steps(self, all_steps: List[tuple]) -> List[Dict[str, str]]:
        level_to_steps = {
            MaturityLevel.RAW: 0,
            MaturityLevel.CLEANED: 1,
            MaturityLevel.LABELED: 2,
            MaturityLevel.FEATURE_ENGINEERED: 3,
            MaturityLevel.FULLY_READY: len(all_steps)
        }

        completed_steps = level_to_steps.get(self.current_level, 0)
        needed = []

        for i, (step_name, step_desc) in enumerate(all_steps):
            status = "completed" if i < completed_steps else "pending"
            needed.append({
                "step": i + 1,
                "name": step_name,
                "description": step_desc,
                "status": status
            })

        return needed

    def get_recommendations_markdown(self) -> str:
        rec = self.get_recommendations()
        md = []

        md.append(f"# Pipeline Recommendations: {rec['pipeline_name']}\n")
        md.append(f"**Detected Domain:** {rec['domain']}")
        md.append(f"**Current Readiness Level:** {rec['current_level']} - {rec['current_level_description']}")
        md.append(f"**Target Level:** {rec['target_level']}\n")

        md.append("## Pipeline Steps\n")
        md.append("| Step | Name | Description | Status |")
        md.append("|------|------|-------------|--------|")
        for step in rec['needed_steps']:
            status_icon = "[x]" if step['status'] == 'completed' else "[ ]"
            md.append(f"| {step['step']} | {step['name']} | {step['description']} | {status_icon} |")
        md.append("")

        md.append("## Recommended Tools\n")
        for tool in rec['recommended_tools']:
            md.append(f"- `{tool}`")
        md.append("")

        md.append("## Output Formats\n")
        for fmt in rec['output_formats']:
            md.append(f"- {fmt}")
        md.append("")

        return "\n".join(md)


# =============================================================================
# DOMAIN ASSESSOR
# =============================================================================

class DomainAssessor:
    """Computes domain-specific readiness metrics for a dataset.

    Works on tabular data (CSV, Parquet, DataFrame), NPZ archives, NetCDF,
    FASTA, and PDB files. Returns a structured dict of universal +
    domain-specific metrics that can be displayed or serialized to JSON.
    """

    # Required fields per domain; each entry: (canonical_name, [aliases])
    HPC_REQUIRED = [
        ("walltime",   ["walltime", "runtime", "elapsed", "duration", "cpu_time",
                        "time_elapsed", "batch_time_elapsed", "time_limit"]),
        ("node_count", ["node_count", "nodes", "num_nodes", "nnodes", "ncores", "cores",
                        "nodes_required", "num_nodes_required"]),
        ("job_id",     ["job_id", "jobid", "id", "job", "jobname"]),
        ("queue",      ["queue", "partition", "qos", "account", "project"]),
    ]
    CLIMATE_REQUIRED = [
        ("temperature",   ["tas", "t2m", "temperature", "temp", "air_temperature"]),
        ("precipitation", ["pr", "precipitation", "precip", "rain"]),
        ("latitude",      ["lat", "latitude", "grid_center_lat"]),
        ("longitude",     ["lon", "longitude", "grid_center_lon"]),
        ("time",          ["time", "date", "datetime"]),
    ]
    MATERIALS_REQUIRED = [
        ("formula",          ["formula", "composition", "compound", "material", "auid"]),
        ("formation_energy", ["formation_energy", "energy_above_hull", "e_form",
                              "delta_e", "formation_enthalpy", "enthalpy_formation_atom"]),
        ("band_gap",         ["band_gap", "bandgap", "bg", "gap", "Egap"]),
        ("structure",        ["lattice", "spacegroup", "space_group", "crystal", "cif",
                              "Pearson_symbol_orig"]),
    ]
    LIFE_SCIENCES_REQUIRED = [
        ("sequence",   ["sequence", "seq", "protein_sequence", "aa_seq"]),
        ("identifier", ["id", "protein_id", "uniprot", "pdb_id", "gene_id"]),
        ("label",      ["function", "label", "class", "target", "go_term"]),
        ("msa",        ["msa", "alignment", "a3m"]),
    ]
    GEOSPATIAL_REQUIRED = [
        ("h3_index",  ["h3_index", "h3", "hex_id", "grid_id", "cell_id"]),
        ("latitude",  ["lat", "latitude", "y"]),
        ("longitude", ["lon", "longitude", "x"]),
        ("value",     ["value", "pm25", "aqi", "smoke", "concentration"]),
    ]

    def __init__(self, path: str, domain: Optional[str] = None):
        self.path = Path(path)
        self._domain_hint = domain
        self._df: Optional[Any] = None   # pandas DataFrame if loaded
        self._npz: Optional[Dict] = None  # dict of arrays if NPZ

    def assess(self) -> Dict[str, Any]:
        """Run full assessment. Returns metrics dict."""
        self._load()
        domain = self._domain_hint or self._infer_domain()
        universal = self._universal_metrics(domain)
        specific = self._domain_metrics(domain)
        level, gaps = self._readiness_level(domain, universal, specific)
        return {
            "path": str(self.path),
            "domain": domain,
            "readiness_level": level,
            "readiness_name": MaturityLevel(level).name,
            "universal": universal,
            "domain_specific": specific,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self):
        ext = self.path.suffix.lower()
        try:
            import pandas as pd
            if ext == ".csv":
                self._df = pd.read_csv(self.path, nrows=100_000)
            elif ext == ".parquet":
                self._df = pd.read_parquet(self.path)
            elif ext == ".tsv":
                self._df = pd.read_csv(self.path, sep="\t", nrows=100_000)
            elif ext == ".npz":
                import numpy as np
                raw = np.load(self.path, allow_pickle=False)
                try:
                    self._npz = dict(raw)
                except ValueError:
                    raw.close()
                    raw = np.load(self.path, allow_pickle=True)
                    self._npz = dict(raw)
                raw.close()
                rows = {}
                for k, v in self._npz.items():
                    if v.ndim == 1 and v.dtype.kind in "fiu":
                        rows[k] = v
                if rows:
                    self._df = pd.DataFrame(rows)
                elif not rows:
                    # Check for object array of dicts (e.g. RAPS raps_jobs.npz)
                    for k, v in self._npz.items():
                        if v.ndim == 1 and v.dtype.kind == 'O' and len(v) > 0:
                            first = v[0]
                            if isinstance(first, dict):
                                try:
                                    self._df = pd.DataFrame(
                                        list(v[:100_000]),
                                        columns=list(first.keys())
                                    )
                                except Exception:
                                    pass
                                break
            elif ext in [".h5", ".hdf5"]:
                try:
                    import h5py
                    rows = {}
                    with h5py.File(self.path, 'r') as f:
                        def _collect(name, obj):
                            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype.kind in "fiu":
                                rows[name] = obj[:]
                        f.visititems(_collect)
                    if rows:
                        self._df = pd.DataFrame(rows)
                except ImportError:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Domain inference
    # ------------------------------------------------------------------

    def _infer_domain(self) -> str:
        ext = self.path.suffix.lower()
        ext_map = {
            ".nc": "climate", ".nc4": "climate", ".grib": "climate",
            ".cif": "materials", ".poscar": "materials",
            ".fasta": "life_sciences", ".fastq": "life_sciences",
            ".fa": "life_sciences", ".fq": "life_sciences",
            ".pdb": "life_sciences", ".mmcif": "life_sciences",
            ".bp": "fusion",
            ".shp": "geospatial", ".geojson": "geospatial",
        }
        if ext in ext_map:
            return ext_map[ext]

        if self._df is not None:
            cols = [c.lower() for c in self._df.columns]
            col_text = " ".join(cols)
            scores: Dict[str, float] = {}
            for domain, patterns in DOMAIN_FILE_PATTERNS.items():
                score = sum(
                    sum(1 for c in cols if kw.lower() in c) + (0.5 if kw.lower() in col_text else 0)
                    for kw in patterns["keywords"]
                )
                scores[domain.value] = score
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best

        return "general"

    # ------------------------------------------------------------------
    # Universal metrics (any domain, tabular or array)
    # ------------------------------------------------------------------

    def _universal_metrics(self, domain: str = "general") -> Dict[str, Any]:
        m: Dict[str, Any] = {}

        if self._df is not None:
            df = self._df
            n = len(df)
            if n == 0:
                return {"rows": 0, "columns": len(df.columns)}

            m["rows"] = n
            m["columns"] = len(df.columns)

            # Completeness
            total_cells = n * len(df.columns)
            null_cells = int(df.isnull().sum().sum())
            m["completeness"] = round(1 - null_cells / total_cells, 4) if total_cells > 0 else 1.0
            null_frac = df.isnull().mean()
            worst_null = null_frac[null_frac > 0].sort_values(ascending=False).head(5)
            m["null_by_column"] = {col: round(v, 4) for col, v in worst_null.items()}

            # Duplicate ratio
            try:
                m["duplicate_ratio"] = round(df.duplicated().sum() / n, 4)
            except Exception:
                m["duplicate_ratio"] = None

            # Outlier ratio (IQR method on numeric columns)
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                Q1 = numeric.quantile(0.25)
                Q3 = numeric.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = ((numeric < Q1 - 1.5 * IQR) | (numeric > Q3 + 1.5 * IQR))
                m["outlier_ratio"] = round(float(outlier_mask.any(axis=1).sum() / n), 4)
                col_outlier = (outlier_mask.sum() / n).sort_values(ascending=False).head(5)
                m["outlier_by_column"] = {col: round(v, 4) for col, v in col_outlier.items() if v > 0}
            else:
                m["outlier_ratio"] = None

            # Class imbalance — label/target columns
            label_cols = [c for c in df.columns
                          if any(k in c.lower() for k in ("label", "target", "class"))]
            if label_cols:
                col = label_cols[0]
                vc = df[col].value_counts(normalize=True)
                if len(vc) >= 2:
                    m["class_imbalance_ratio"] = round(float(vc.iloc[0] / vc.iloc[-1]), 3)
                    m["class_imbalance_col"] = col

            # Feature correlation — flag highly correlated pairs (|r| > 0.95)
            if not numeric.empty and numeric.shape[1] >= 2 and n >= 10:
                try:
                    corr = numeric.corr().abs()
                    upper = corr.where(
                        __import__('numpy').triu(
                            __import__('numpy').ones(corr.shape, dtype=bool), k=1
                        )
                    )
                    high_corr_pairs = [
                        (c, r, round(float(upper.loc[r, c]), 4))
                        for c in upper.columns
                        for r in upper.index
                        if upper.loc[r, c] is not None and not __import__('numpy').isnan(upper.loc[r, c])
                        and upper.loc[r, c] > 0.95
                    ][:10]
                    m["high_correlation_pairs"] = len(high_corr_pairs)
                    if high_corr_pairs:
                        m["high_correlation_examples"] = [
                            f"{a}↔{b} ({v})" for a, b, v in high_corr_pairs[:3]
                        ]
                except Exception:
                    pass

            # Dimensionality ratio (features / samples)
            m["dimensionality_ratio"] = round(len(df.columns) / n, 6) if n > 0 else None

            # k-anonymity on quasi-identifier columns
            qi_candidates = {
                "hpc": ["partition", "queue", "node_count", "user", "account", "project"],
                "life_sciences": ["experimental_method", "resolution", "organism"],
                "general": [],
            }
            qi_cols = [
                c for c in df.columns
                if any(q in c.lower() for q in qi_candidates.get(domain, []) + qi_candidates["general"])
            ]
            if qi_cols:
                try:
                    k = int(df.groupby(qi_cols).size().min())
                    m["k_anonymity"] = k
                    m["k_anonymity_cols"] = qi_cols
                except Exception:
                    pass

        elif self._npz is not None:
            import numpy as np
            arrays = {k: v for k, v in self._npz.items()
                      if isinstance(v, np.ndarray) and v.dtype.kind in "fiu"}
            if arrays:
                total = sum(a.size for a in arrays.values())
                nan_count = sum(
                    int(np.isnan(a).sum()) if a.dtype.kind == 'f' else 0
                    for a in arrays.values()
                )
                first = next(iter(arrays.values()))
                m["rows"] = first.shape[0] if first.ndim >= 1 else 0
                m["arrays"] = len(self._npz)
                m["completeness"] = round(1 - nan_count / total, 4) if total > 0 else 1.0
                m["duplicate_ratio"] = None
                m["outlier_ratio"] = None

        return m

    # ------------------------------------------------------------------
    # Domain-specific metrics
    # ------------------------------------------------------------------

    def _domain_metrics(self, domain: str) -> Dict[str, Any]:
        if domain == "hpc":
            return self._hpc_metrics()
        elif domain == "climate":
            return self._climate_metrics()
        elif domain == "materials":
            return self._materials_metrics()
        elif domain == "life_sciences":
            return self._life_sciences_metrics()
        elif domain == "geospatial":
            return self._geospatial_metrics()
        return {}

    def _find_col(self, aliases: List[str]) -> Optional[str]:
        """Return the first column matching any alias (case-insensitive)."""
        if self._df is None:
            return None
        cols_lower = {c.lower(): c for c in self._df.columns}
        for alias in aliases:
            if alias.lower() in cols_lower:
                return cols_lower[alias.lower()]
        return None

    def _hpc_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"required_fields": {}}
        if self._df is None:
            return m

        df = self._df
        n = len(df)

        # Required field coverage
        for canon, aliases in self.HPC_REQUIRED:
            col = self._find_col(aliases)
            if col:
                non_null = int(df[col].notna().sum())
                m["required_fields"][canon] = {
                    "column": col,
                    "coverage": round(non_null / n, 4) if n > 0 else 0,
                    "present": True,
                }
            else:
                m["required_fields"][canon] = {"present": False}

        # Walltime validity (> 0)
        wt_col = self._find_col(["walltime", "runtime", "elapsed", "time_elapsed",
                                  "batch_time_elapsed", "duration", "time_limit"])
        if wt_col and df[wt_col].dtype.kind in "fiu":
            valid = int((df[wt_col] > 0).sum())
            m["runtime_valid_ratio"] = round(valid / n, 4) if n > 0 else 0

        # Node count sanity (> 0)
        nc_col = self._find_col(["node_count", "nodes", "num_nodes", "nnodes"])
        if nc_col and df[nc_col].dtype.kind in "fiu":
            valid = int((df[nc_col] > 0).sum())
            m["node_count_valid_ratio"] = round(valid / n, 4) if n > 0 else 0

        # Feature coverage completeness: fraction of non-null across all columns
        m["feature_coverage"] = round(df.notna().mean().mean(), 4)

        # Temporal drift: split timeline in half, compare queue/node distributions
        time_col = self._find_col(["submit", "start", "start_time", "submit_time", "begin",
                                    "time_start", "batch_time_start"])
        if time_col:
            try:
                import pandas as pd
                times = pd.to_datetime(df[time_col], unit='s', errors='coerce')
                if times.isna().all():
                    times = pd.to_datetime(df[time_col], errors='coerce')
                times = times.dropna()
                if len(times) > 10:
                    m["temporal_range"] = f"{times.min().date()} to {times.max().date()}"
                    m["temporal_days"] = (times.max() - times.min()).days
                    # Drift: compare node_count distribution between first/second half
                    if nc_col and df[nc_col].dtype.kind in "fiu":
                        idx = times.index
                        mid = len(idx) // 2
                        first_half = df.loc[idx[:mid], nc_col].dropna()
                        second_half = df.loc[idx[mid:], nc_col].dropna()
                        if len(first_half) > 0 and len(second_half) > 0:
                            drift = abs(float(first_half.mean()) - float(second_half.mean()))
                            m["temporal_drift_node_count"] = round(drift, 3)
            except Exception:
                pass

        # Utilization traces
        trace_cols = [c for c in df.columns
                      if any(t in c.lower() for t in ("cpu_trace", "gpu_trace", "ntx", "nrx"))]
        m["utilization_traces"] = trace_cols

        # Schema alignment score: fraction of RAPS canonical fields found
        raps_canonical = ["walltime", "node_count", "job_id", "queue",
                          "submit", "start", "end", "exit_code", "user"]
        cols_lower = {c.lower() for c in df.columns}
        found = sum(
            1 for canon in raps_canonical
            if any(alias in cols_lower for alias in [canon, canon.replace("_", ""), canon + "_time"])
        )
        m["schema_alignment_score"] = round(found / len(raps_canonical), 3)

        return m

    def _climate_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"required_fields": {}}

        if self.path.suffix.lower() in [".nc", ".nc4", ".grib"]:
            try:
                import xarray as xr
                import numpy as np
                ds = xr.open_dataset(self.path)
                variables = list(ds.data_vars)
                coords = list(ds.coords)
                all_names = set(variables + coords)

                for canon, aliases in self.CLIMATE_REQUIRED:
                    found = next((a for a in aliases if a in all_names), None)
                    m["required_fields"][canon] = {"present": found is not None, "column": found}

                # Temporal continuity: detect gaps in time steps
                for tname in ("time", "TIME", "t"):
                    if tname in ds.coords:
                        t = ds[tname]
                        tvals = t.values
                        m["temporal_range"] = f"{str(tvals[0])[:10]} to {str(tvals[-1])[:10]}"
                        m["temporal_steps"] = int(t.size)
                        # Detect gaps: check if steps are uniform
                        if len(tvals) > 2:
                            try:
                                import pandas as pd
                                tdiffs = pd.Series(tvals).diff().dropna()
                                expected = tdiffs.mode().iloc[0]
                                gap_count = int((tdiffs != expected).sum())
                                m["temporal_gap_steps"] = gap_count
                                m["temporal_continuity"] = round(1 - gap_count / len(tdiffs), 4)
                            except Exception:
                                pass
                        break

                # Spatial grid alignment: check regularity of lat/lon spacing
                for lname in ("lat", "latitude"):
                    if lname in ds.coords:
                        lats = ds[lname].values
                        if len(lats) > 1:
                            m["lat_resolution_deg"] = round(abs(float(lats[1] - lats[0])), 4)
                            m["lat_range"] = f"{float(lats.min()):.1f} to {float(lats.max()):.1f}"
                            # Regularity: std of diffs should be near zero
                            diffs = np.diff(lats)
                            if len(diffs) > 1:
                                m["lat_grid_regularity"] = round(
                                    1 - float(np.std(diffs)) / (abs(float(np.mean(diffs))) + 1e-9), 4
                                )
                        break
                for lname in ("lon", "longitude"):
                    if lname in ds.coords:
                        lons = ds[lname].values
                        if len(lons) > 1:
                            m["lon_resolution_deg"] = round(abs(float(lons[1] - lons[0])), 4)
                            diffs = np.diff(lons)
                            if len(diffs) > 1:
                                m["lon_grid_regularity"] = round(
                                    1 - float(np.std(diffs)) / (abs(float(np.mean(diffs))) + 1e-9), 4
                                )
                        break

                # CF convention compliance: check key attributes on variables
                cf_compliant = 0
                cf_total = 0
                for var in variables:
                    cf_total += 1
                    attrs = ds[var].attrs
                    has_units = "units" in attrs
                    has_long_name = "long_name" in attrs or "standard_name" in attrs
                    if has_units and has_long_name:
                        cf_compliant += 1
                m["cf_compliance_ratio"] = round(cf_compliant / cf_total, 4) if cf_total > 0 else None

                # Check time calendar
                if "time" in ds.coords:
                    calendar = ds["time"].attrs.get("calendar", ds.coords["time"].encoding.get("calendar"))
                    m["time_calendar"] = calendar or "unknown"

                # NaN fraction per variable
                nan_fracs = {}
                for var in variables:
                    frac = float(np.isnan(ds[var].values.astype(float)).mean())
                    if frac > 0:
                        nan_fracs[var] = round(frac, 4)
                if nan_fracs:
                    m["nan_fraction_by_variable"] = nan_fracs

                # Normalization leakage hint: look for normalize_mean/std siblings
                siblings = list(self.path.parent.glob("normalize_*.npz"))
                split_files = list(self.path.parent.glob("*train*.npz")) + \
                              list(self.path.parent.glob("*val*.npz"))
                if siblings and split_files:
                    train_stats = any("train" in s.name for s in siblings)
                    m["normalization_leakage_risk"] = not train_stats
                    m["normalization_stats_files"] = [s.name for s in siblings]

                m["variables"] = variables
                ds.close()
                return m
            except Exception:
                pass

        # Fallback: tabular
        if self._df is not None:
            for canon, aliases in self.CLIMATE_REQUIRED:
                col = self._find_col(aliases)
                m["required_fields"][canon] = {"present": col is not None, "column": col}

        return m

    def _materials_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"required_fields": {}}
        if self._df is None:
            return m

        df = self._df
        n = len(df)

        for canon, aliases in self.MATERIALS_REQUIRED:
            col = self._find_col(aliases)
            if col:
                non_null = int(df[col].notna().sum())
                m["required_fields"][canon] = {
                    "present": True,
                    "column": col,
                    "coverage": round(non_null / n, 4),
                }
            else:
                m["required_fields"][canon] = {"present": False}

        # Property coverage: fraction of samples with any target property
        target_cols = [c for c in df.columns
                       if any(k in c.lower() for k in ("formation", "band_gap", "energy",
                                                        "enthalpy", "modulus", "conductivity"))]
        if target_cols:
            coverage = float(df[target_cols].notna().any(axis=1).mean())
            m["property_coverage"] = round(coverage, 4)
            m["property_columns"] = target_cols[:5]

        # Composition feature richness
        numeric_cols = df.select_dtypes(include="number").columns
        m["numeric_feature_count"] = len(numeric_cols)
        m["total_feature_count"] = len(df.columns)

        magpie_like = [c for c in df.columns
                       if any(k in c.lower() for k in ("magpie", "mendeleev", "atomicweight",
                                                        "electronegativity", "valence", "nvalence",
                                                        "nunfilled", "gsmagmom"))]
        m["composition_features_count"] = len(magpie_like)

        # Composition diversity: unique elements across formulas
        formula_col = self._find_col(["formula", "compound", "composition", "auid"])
        if formula_col:
            m["unique_compositions"] = int(df[formula_col].nunique())

        # Dimensionality warning for small datasets
        if n < len(numeric_cols):
            m["curse_of_dimensionality"] = True
            m["samples_to_features_ratio"] = round(n / max(len(numeric_cols), 1), 3)

        return m

    def _life_sciences_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"required_fields": {}}
        ext = self.path.suffix.lower()

        if ext in [".fasta", ".fastq", ".fa", ".fq"]:
            lengths, ambiguous_count, total_chars = [], 0, 0
            ambiguous_chars = set("NXBZJ?-")
            with open(self.path, 'r') as f:
                current_seq: List[str] = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>') or line.startswith('@'):
                        if current_seq:
                            seq = "".join(current_seq)
                            lengths.append(len(seq))
                            ambiguous_count += sum(1 for c in seq if c.upper() in ambiguous_chars)
                            total_chars += len(seq)
                        current_seq = []
                    elif line and not line.startswith('+') and not line.startswith('#'):
                        current_seq.append(line)
                if current_seq:
                    seq = "".join(current_seq)
                    lengths.append(len(seq))
                    ambiguous_count += sum(1 for c in seq if c.upper() in ambiguous_chars)
                    total_chars += len(seq)

            if lengths:
                m["sequence_count"] = len(lengths)
                m["avg_length"] = round(sum(lengths) / len(lengths), 1)
                m["min_length"] = min(lengths)
                m["max_length"] = max(lengths)
                # Ambiguous residue ratio (proxy for data quality)
                m["ambiguous_ratio"] = round(ambiguous_count / total_chars, 4) if total_chars > 0 else 0
                m["required_fields"]["sequence"] = {"present": True, "coverage": 1.0}

            # MSA depth: look for .a3m / .sto files alongside
            msa_files = (
                list(self.path.parent.glob("*.a3m")) +
                list(self.path.parent.glob("*.sto")) +
                list(self.path.parent.glob("*.aln"))
            )
            if msa_files:
                # Count sequences in first MSA file
                try:
                    with open(msa_files[0], 'r') as f:
                        msa_depth = sum(1 for l in f if l.startswith('>'))
                    m["msa_depth"] = msa_depth
                    m["msa_file"] = msa_files[0].name
                    m["required_fields"]["msa"] = {"present": True, "coverage": 1.0}
                except Exception:
                    m["required_fields"]["msa"] = {"present": True}
            else:
                m["required_fields"]["msa"] = {"present": False}

            return m

        if ext in [".pdb", ".mmcif", ".cif"]:
            atom_count, residue_ids, chain_ids = 0, set(), set()
            r_factor, method = None, None
            is_pdb_fmt = ext == ".pdb"
            with open(self.path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_count += 1
                        try:
                            if is_pdb_fmt:
                                chain_ids.add(line[21])
                                residue_ids.add((line[21], line[22:26].strip()))
                            else:
                                # mmCIF: fields are whitespace-separated
                                parts = line.split()
                                if len(parts) > 7:
                                    chain_ids.add(parts[6])
                                    residue_ids.add((parts[6], parts[8]))
                        except IndexError:
                            pass
                    # R-factor from REMARK 3
                    elif line.startswith("REMARK   3") and "R VALUE" in line and r_factor is None:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p in (":", "VALUE") and i + 1 < len(parts):
                                try:
                                    r_factor = float(parts[i + 1])
                                    break
                                except ValueError:
                                    pass
                    # Experimental method
                    elif line.startswith("EXPDTA") and method is None:
                        method = line[6:].strip().split(";")[0].strip()

            m["atom_count"] = atom_count
            m["residue_count"] = len(residue_ids)
            m["chain_count"] = len(chain_ids)
            if r_factor is not None:
                m["r_factor"] = r_factor
                m["structure_resolution_flag"] = "good" if r_factor < 0.25 else "poor"
            if method:
                m["experimental_method"] = method
            m["required_fields"]["sequence"] = {"present": atom_count > 0, "coverage": 1.0}
            # Check for MSA alongside the structure file
            msa_files = (
                list(self.path.parent.glob("*.a3m")) +
                list(self.path.parent.glob("*.sto")) +
                list(self.path.parent.glob("*.aln"))
            )
            m["required_fields"]["msa"] = {"present": len(msa_files) > 0}
            if msa_files:
                m["msa_file"] = msa_files[0].name
            m["required_fields"]["identifier"] = {"present": False}
            m["required_fields"]["label"] = {"present": False}
            return m

        # Tabular
        if self._df is not None:
            df = self._df
            n = len(df)
            for canon, aliases in self.LIFE_SCIENCES_REQUIRED:
                col = self._find_col(aliases)
                if col:
                    non_null = int(df[col].notna().sum())
                    m["required_fields"][canon] = {
                        "present": True, "column": col,
                        "coverage": round(non_null / n, 4),
                    }
                else:
                    m["required_fields"][canon] = {"present": False}

            # Experimental method bias: check if method column has imbalanced distribution
            method_col = self._find_col(["experimental_method", "method", "technique"])
            if method_col:
                vc = df[method_col].value_counts(normalize=True)
                m["method_distribution"] = {str(k): round(float(v), 3) for k, v in vc.head(5).items()}

        return m

    def _geospatial_metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"required_fields": {}}
        if self._df is None:
            return m

        df = self._df
        n = len(df)

        for canon, aliases in self.GEOSPATIAL_REQUIRED:
            col = self._find_col(aliases)
            if col:
                non_null = int(df[col].notna().sum())
                m["required_fields"][canon] = {
                    "present": True, "column": col,
                    "coverage": round(non_null / n, 4),
                }
            else:
                m["required_fields"][canon] = {"present": False}

        h3_col = self._find_col(["h3_index", "h3", "hex_id"])
        if h3_col:
            try:
                import h3
                sample_val = df[h3_col].dropna().iloc[0]
                m["h3_resolution"] = h3.get_resolution(int(sample_val))
            except Exception:
                pass

        return m

    # ------------------------------------------------------------------
    # Readiness level + gap analysis
    # ------------------------------------------------------------------

    def _readiness_level(self, domain: str, universal: Dict, specific: Dict) -> tuple:
        level = 1
        gaps: List[str] = []

        completeness = universal.get("completeness")
        dup_ratio = universal.get("duplicate_ratio") or 0
        outlier_ratio = universal.get("outlier_ratio") or 0

        # Level 2: CLEANED
        has_data = (
            universal.get("rows", 0) > 0
            or universal.get("arrays", 0) > 0
            or specific.get("sequence_count", 0) > 0   # FASTA
            or specific.get("atom_count", 0) > 0       # PDB
        )
        if has_data:
            level = 2

        cleaning_gaps = []
        if completeness is not None and completeness < 0.95:
            cleaning_gaps.append(f"completeness {completeness:.1%} < 95% — handle missing values")
        if dup_ratio > 0.01:
            cleaning_gaps.append(f"duplicate_ratio {dup_ratio:.1%} > 1% — deduplicate")
        if outlier_ratio is not None and outlier_ratio > 0.05:
            cleaning_gaps.append(f"outlier_ratio {outlier_ratio:.1%} > 5% — investigate outliers")
        if universal.get("k_anonymity") is not None and universal["k_anonymity"] < 2:
            cleaning_gaps.append(f"k_anonymity={universal['k_anonymity']} — records are uniquely identifiable")

        if has_data and not cleaning_gaps:
            level = max(level, 2)
        else:
            gaps.extend(cleaning_gaps)

        # Level 3: LABELED
        has_label = False
        if self._df is not None:
            label_kws = ("label", "target", "class", "formation_energy", "band_gap",
                         "enthalpy", "go_term", "function")
            has_label = any(any(k in c.lower() for k in label_kws) for c in self._df.columns)
        if domain == "life_sciences" and specific.get("atom_count", 0) > 0:
            # PDB/mmCIF structures: coordinates are the implicit label for structure prediction
            # but only count as LABELED if they also have a functional annotation
            has_label = specific.get("r_factor") is not None or specific.get("experimental_method") is not None

        if has_label and not cleaning_gaps:
            level = max(level, 3)
        elif not has_label and level >= 2 and not cleaning_gaps:
            gaps.append("no target/label column — define a prediction target")

        # Level 4: FEATURE_ENGINEERED — domain required fields complete
        req = specific.get("required_fields", {})
        missing_required = [k for k, v in req.items() if not v.get("present")]
        low_coverage = [
            f"{k} ({v['coverage']:.0%})"
            for k, v in req.items()
            if v.get("present") and v.get("coverage", 1) < 0.9
        ]
        cf_ok = specific.get("cf_compliance_ratio", 1.0) or 1.0

        domain_ready = not missing_required and not low_coverage and cf_ok >= 0.8
        if domain_ready and has_label and not cleaning_gaps:
            level = max(level, 4)
        else:
            if missing_required:
                gaps.append(f"missing required {domain} fields: {', '.join(missing_required)}")
            if low_coverage:
                gaps.append(f"low field coverage: {', '.join(low_coverage)}")
            if cf_ok < 0.8:
                gaps.append(f"CF compliance {cf_ok:.0%} < 80% — add units/standard_name attributes")
            if specific.get("normalization_leakage_risk"):
                gaps.append("normalization stats not prefixed 'train_' — potential leakage risk")
            if specific.get("curse_of_dimensionality"):
                gaps.append(f"more features than samples (ratio={specific.get('samples_to_features_ratio')}) "
                             "— consider dimensionality reduction")

        # Level 5: FULLY_READY — AI-format output AND all previous levels passed
        ext = self.path.suffix.lower()
        if ext in (".npz", ".h5", ".hdf5", ".zarr", ".tfrecord", ".pth") and level == 4:
            level = 5
        elif level == 4:
            gaps.append("save to AI-ready format (NPZ/HDF5/Zarr) with train/val/test split")

        return level, gaps


def run_assess(path: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Assess a dataset and return structured readiness metrics."""
    assessor = DomainAssessor(path, domain=domain)
    return assessor.assess()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_dataset(file_path: str) -> DataCard:
    """Analyze a dataset and return a DataCard"""
    analyzer = DataAnalyzer(file_path)
    return analyzer.analyze()


def get_recommendations(file_path: str) -> Dict[str, Any]:
    """Analyze a dataset and return pipeline recommendations"""
    data_card = analyze_dataset(file_path)
    recommender = PipelineRecommender(data_card)
    return recommender.get_recommendations()


def generate_report(file_path: str) -> str:
    """Generate a full analysis report in markdown"""
    data_card = analyze_dataset(file_path)
    recommender = PipelineRecommender(data_card)

    output = []
    output.append("=" * 60)
    output.append("REDI - Data Readiness Analysis")
    output.append("=" * 60 + "\n")
    output.append(data_card.to_markdown())
    output.append("\n" + "=" * 60 + "\n")
    output.append(recommender.get_recommendations_markdown())

    return "\n".join(output)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python assess.py <file_path>")
        print("\nAnalyzes a dataset and generates a Data Card with pipeline recommendations.")
        sys.exit(1)

    print(generate_report(sys.argv[1]))
