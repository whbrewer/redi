# REDI - Readiness Engine for Data Integration

A modular pipeline for transforming raw scientific data into AI-ready datasets.

## Installation

```bash
# Install from source (editable mode for development)
cd /path/to/drai
pip install -e .

# Install with domain-specific dependencies
pip install -e ".[climate]"      # Climate data support (xarray, xesmf, zarr)
pip install -e ".[materials]"    # Materials science (pymatgen, matminer)
pip install -e ".[life_sciences]" # Life sciences (biopython)
pip install -e ".[hpc]"          # HPC data (scikit-learn)
pip install -e ".[fusion]"       # Fusion energy (h5py)

# Install with all dependencies
pip install -e ".[all]"

# Install with specific output format support
pip install -e ".[parquet]"      # Parquet output support
pip install -e ".[hdf5]"         # HDF5 output support
pip install -e ".[torch]"        # PyTorch tensor support
```

### Climate Domain Setup (ESMF/xesmf)

The climate domain uses [xesmf](https://xesmf.readthedocs.io/) for regridding, which requires [ESMF](https://earthsystemmodeling.org/) (Earth System Modeling Framework). ESMF must be built from source and the Python bindings installed manually.

**Option 1: Using Conda (Easiest)**

```bash
conda create -n redi python=3.12
conda activate redi
conda install -c conda-forge esmpy xesmf xarray netcdf4 zarr
pip install -e .
```

**Option 2: Build ESMF from Source**

1. Clone and build ESMF:

```bash
git clone https://github.com/esmf-org/esmf.git
cd esmf
make
```

See the [ESMF User's Guide](http://earthsystemmodeling.org/docs/nightly/develop/ESMF_usrdoc/node10.html) for detailed build instructions and compiler requirements.

2. Install the Python bindings (esmpy):

```bash
# Set ESMFMKFILE to point to your built ESMF
export ESMFMKFILE=/path/to/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk

# Install esmpy from the ESMF source tree
pip install /path/to/esmf/src/addon/esmpy/

# Install xesmf
pip install xesmf
```

3. Configure your virtualenv to set ESMFMKFILE automatically:

Add to your venv's `bin/activate` script (before the final `hash -r` line):

```bash
# ESMF configuration for climate processing (xesmf/esmpy)
export ESMFMKFILE=/path/to/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk
```

And add to the `deactivate()` function:

```bash
unset ESMFMKFILE
```

Now ESMF will be configured automatically when you activate the venv.

**Verify Installation**

```bash
source /path/to/venv/bin/activate
python -c "import esmpy; import xesmf; print('ESMF/xesmf ready!')"
```

## Quick Start

### Run Full Pipeline

```bash
# Auto-detect domain and process
redi data.csv -o ./output

# Specify domain explicitly
redi data.nc --domain climate -o ./output

# Run specific stages only
redi data.csv --stages ingest,preprocess,transform
```

### Analyze Dataset Only

```bash
# Generate Data Card and recommendations (no processing)
redi data.csv --analyze

# Or use the analyze module directly
python -m redi.analyze data.csv
```

## Pipeline Stages

| Stage          | Description                                     |
|----------------|-------------------------------------------------|
| **ingest**     | Load data, detect format, stream large files    |
| **preprocess** | Clean, handle missing values, remove duplicates |
| **transform**  | Feature engineering, normalization, encoding    |
| **structure**  | Convert to tensors, graphs, or sequences        |
| **output**     | Train/val/test split, shard, save               |

## Supported Domains

| Domain            | File Types         | Key Tools          |
|-------------------|--------------------|--------------------|
| **climate**       | NetCDF, GRIB, Zarr | xarray, xesmf      |
| **materials**     | CIF, POSCAR, CSV   | pymatgen, matminer |
| **life_sciences** | FASTA, PDB, VCF    | biopython          |
| **fusion**        | HDF5, ADIOS        | h5py, MDSplus      |
| **hpc**           | CSV, Parquet       | pandas, sklearn    |

## CLI Options

```
redi <input> [options]

Arguments:
  input                   Input file path

Options:
  -d, --domain DOMAIN     Scientific domain (auto-detected if not specified)
                          Choices: climate, materials, life_sciences, fusion, hpc, general
  -o, --output DIR        Output directory (default: ./output)
  -s, --stages STAGES     Comma-separated list of stages to run
  -a, --analyze           Only analyze and generate Data Card (no processing)
  -v, --verbose           Verbose output
```

## Module Usage

### Analyze a Dataset

```python
from analyze import analyze_dataset, get_recommendations, generate_report

# Get a DataCard object
data_card = analyze_dataset("data.csv")
print(f"Domain: {data_card.detected_domain}")
print(f"Readiness Level: {data_card.readiness_level}")
print(f"Features: {data_card.overview.num_features}")

# Get pipeline recommendations
recs = get_recommendations("data.csv")
print(f"Recommended tools: {recs['recommended_tools']}")

# Generate full markdown report
report = generate_report("data.csv")
print(report)
```

### Run Pipeline Programmatically

```python
from cli import run_pipeline

ctx = run_pipeline(
    input_path="data.csv",
    output_dir="./output",
    domain="hpc",  # or None for auto-detect
    stages=["ingest", "preprocess", "transform", "structure", "output"]
)

print(f"Completed: {[s.value for s in ctx.completed_stages]}")
print(f"Output files: {ctx.metadata.get('saved_files', [])}")
```

### Use Individual Pipeline Steps

```python
from pipeline import Pipeline, create_context
from pipeline.ingest import get_ingest_steps
from pipeline.preprocess import get_preprocess_steps

# Create context
ctx = create_context("data.csv", "./output", "hpc")

# Build custom pipeline
pipeline = Pipeline("hpc")
for step in get_ingest_steps():
    pipeline.add_step(step)
for step in get_preprocess_steps("hpc"):
    pipeline.add_step(step)

# Execute
ctx = pipeline.execute(ctx)
```

## Data Readiness Levels

| Level | Name               | Description                                        |
|-------|--------------------|----------------------------------------------------|
| 1     | RAW                | Unprocessed, as originally collected               |
| 2     | CLEANED            | Missing values handled, outliers addressed         |
| 3     | LABELED            | Target variables defined, annotations added        |
| 4     | FEATURE_ENGINEERED | Domain features extracted, transformations applied |
| 5     | FULLY_AI_READY     | Sharded, normalized, in optimized format           |

## Output Formats

- **NPZ** - NumPy compressed arrays (general purpose)
- **HDF5** - Hierarchical data (large datasets, metadata)
- **Parquet** - Columnar format (tabular data)
- **Zarr** - Chunked arrays (climate data)
- **PTH** - PyTorch tensors (deep learning)

## Project Structure

```
redi/
├── cli.py              # Command-line interface
├── analyze.py          # Data Card generation and analysis
├── pipeline/
│   ├── __init__.py     # Pipeline, PipelineStep, PipelineContext
│   ├── ingest.py       # Format detection, loading
│   ├── preprocess.py   # Cleaning, deduplication
│   ├── transform.py    # Normalization, encoding
│   ├── structure.py    # Tensorization, graphs
│   └── output.py       # Splitting, sharding, saving
└── domains/
    ├── __init__.py     # Domain detection
    ├── climate.py      # xarray/xesmf processing
    ├── materials.py    # pymatgen/matminer features
    ├── life_sciences.py # BioPython, PII handling
    ├── fusion.py       # Plasma signal processing
    └── hpc.py          # Job accounting processing
```

## Example: HPC Job Data

```bash
# Process Aurora job accounting data
redi /path/to/hpc/ANL-ALCF-DJC-AURORA.csv -o ./hpc_output

# Output:
# hpc_output/
# ├── train.npz    (80% of data)
# ├── val.npz      (10% of data)
# ├── test.npz     (10% of data)
# └── metadata.json
```

## Testing

REDI includes unit tests and smoke tests for all supported domains.

### Install Test Dependencies

```bash
pip install -e ".[dev]"
# or just pytest
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=redi --cov-report=term-missing
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_analyze.py tests/test_pipeline.py -v

# Smoke tests by domain
pytest tests/test_smoke_hpc.py -v           # HPC domain
pytest tests/test_smoke_climate.py -v       # Climate domain
pytest tests/test_smoke_materials.py -v     # Materials/AFLOW domain
pytest tests/test_smoke_life_sciences.py -v # Bio/OpenFold domain
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures (sample data generators)
├── test_analyze.py          # Unit tests for DataAnalyzer and recommendations
├── test_pipeline.py         # Unit tests for Pipeline core classes
├── test_smoke_hpc.py        # HPC domain end-to-end tests
├── test_smoke_climate.py    # Climate/NetCDF end-to-end tests
├── test_smoke_materials.py  # Materials science end-to-end tests
└── test_smoke_life_sciences.py  # Life sciences/OpenFold end-to-end tests
```

## References

- [DRAI Framework Paper](https://doi.org/10.1145/3750720.3757282)
- [Data Cards Paper](https://doi.org/10.1145/3531146.3533231)
