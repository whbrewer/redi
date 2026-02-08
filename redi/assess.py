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
            "target_level": MaturityLevel.FULLY_AI_READY.name,
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
            MaturityLevel.FULLY_AI_READY: len(all_steps)
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
