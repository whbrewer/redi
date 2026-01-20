# REDI - Readiness Engine for Data Integration

A modular pipeline for transforming raw scientific data into AI-ready datasets.

REDI operates in three modes:
- **Orchestrator** (`redi run`) - Run the full pipeline on your data
- **Sidecar** (`redi sidecar`) - Observe existing workflows (Snakemake, Nextflow)
- **Discovery** (`redi discover`) - Analyze unknown datasets and propose pipelines

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

## Claude Code Skill Installation

REDI includes a Claude Code skill that lets you invoke it with `/redi` in Claude Code conversations.

### Install the Skill

Claude Code skills are stored in `.claude/skills/<skill-name>/SKILL.md` within your project. To install:

```bash
# From the redi directory
mkdir -p .claude/skills/redi
cp SKILL.md .claude/skills/redi/SKILL.md
```

Or create a symlink to keep the skill in sync with the source:

```bash
mkdir -p .claude/skills/redi
ln -s "$(pwd)/SKILL.md" .claude/skills/redi/SKILL.md
```

### Skill File Format

The skill file requires YAML frontmatter with `name` and `description`:

```markdown
---
name: redi
description: REDI transforms raw scientific data into AI-ready datasets...
---

# REDI - Readiness Engine for Data Integration
...
```

### Usage

In a Claude Code conversation, type `/redi` to activate the skill. Claude will then use the REDI CLI to process your datasets.

## Quick Start

### Orchestrator Mode (Run Pipeline)

```bash
# Auto-detect domain and process
redi run data.csv -o ./output

# Specify domain explicitly
redi run data.nc --domain climate -o ./output

# Run specific stages only
redi run data.csv --stages ingest,preprocess,transform

# Backward compatible - 'run' is the default
redi data.csv -o ./output
```

### Sidecar Mode (Observe Workflows)

```bash
# Observe a Snakemake workflow
redi sidecar --workflow Snakefile

# Specify workflow type
redi sidecar --workflow pipeline.nf --type nextflow

# Use YAML config
redi sidecar -c sidecar_config.yaml
```

### Discovery Mode (Analyze Unknown Data)

```bash
# Analyze a dataset and propose pipeline
redi discover /path/to/unknown/data/

# Analyze with output directory
redi discover dataset.csv -o ./analysis

# Non-interactive mode
redi discover /data --no-interactive

# Use YAML config
redi discover -c discover_config.yaml
```

### Multi-File Processing

REDI supports processing multiple files that will be concatenated together:

```bash
# Glob pattern (quote to prevent shell expansion)
redi run "/path/to/data/*.nc" -d climate -o ./output

# Multiple explicit files
redi run file1.nc file2.nc file3.nc -d climate -o ./output

# Bracket patterns for ranges
redi run "/path/to/temp_18[5-9]*.nc" -d climate -o ./output
```

For climate/NetCDF data, files are concatenated along the time dimension using xarray. For CSV/Parquet, files are concatenated row-wise.

## YAML Workflow Configs

For reproducible runs, place options in a YAML file and pass it directly to the CLI.

### Run Mode Config

```yaml
# run_config.yaml
input_path: /path/to/data.nc
domain: climate
output_dir: ./output
output_format: npz
```

```yaml
# Multi-file with glob pattern
input_path: "/path/to/data/*.nc"
domain: climate
output_dir: ./output
```

```yaml
# Explicit file list
input_paths:
  - /path/to/data_1850.nc
  - /path/to/data_1860.nc
  - /path/to/data_1870.nc
domain: climate
output_dir: ./output
```

```bash
redi run config.yaml
# or just:
redi config.yaml
```

### Sidecar Mode Config

```yaml
# sidecar_config.yaml
workflow: ./Snakefile
workflow_type: snakemake
output_dir: ./redi_output

# Override automatic stage mapping
stage_mapping:
  ingest:
    - download
  preprocess:
    - regrid
    - clean
  output:
    - all

# Data Card settings
data_card:
  enabled: true
  metadata:
    project: "Climate Data Processing"
```

```bash
redi sidecar -c sidecar_config.yaml
```

### Discovery Mode Config

```yaml
# discover_config.yaml
input: /path/to/unknown/dataset
output_dir: ./discovery_output

inspection:
  max_files: 100
  include_patterns:
    - "**/*.csv"
    - "**/*.npy"
    - "**/*.nc"

inference:
  domain_hints:
    physics:
      - adc
      - detector
      - event

proposal:
  target_level: 5
  preferred_format: npz
```

```bash
redi discover -c discover_config.yaml
```

See `examples/` for complete configuration examples.

## CLI Reference

### Main Commands

```
redi <command> [options]

Commands:
  run        Run the REDI pipeline (orchestrator mode) - default
  sidecar    Observe existing workflows (sidecar mode)
  discover   Analyze unknown datasets and propose pipelines

For command-specific help:
  redi run --help
  redi sidecar --help
  redi discover --help
```

### redi run

```
redi run <input> [input2 ...] [options]

Arguments:
  input                   Input file path(s), glob pattern, or YAML config

Options:
  -d, --domain DOMAIN     Scientific domain (auto-detected if not specified)
                          Choices: climate, materials, life_sciences, fusion, hpc, general
  -o, --output DIR        Output directory (default: ./output)
  -s, --stages STAGES     Comma-separated list of stages to run
  -f, --format FORMAT     Output format: npz, parquet, csv, hdf5, zarr (default: npz)
  -v, --verbose           Verbose output
  -c, --config FILE       YAML configuration file path
  --style STYLE           Output style: rich (default), simple, or json
  --fair                  Run FAIR principles assessment
  --catalog               Export to data catalog after processing
```

### redi sidecar

```
redi sidecar [options]

Options:
  -w, --workflow FILE     Path to workflow file (Snakefile, *.nf, etc.)
  -t, --type TYPE         Workflow type: auto, snakemake, nextflow, bash
  -o, --output DIR        Output directory (default: ./output)
  --watch                 Watch for changes and re-run
  -c, --config FILE       YAML configuration file
  --style STYLE           Output style: rich, simple, json
```

### redi discover

```
redi discover [input] [options]

Arguments:
  input                   Path to data file or directory

Options:
  -o, --output DIR        Output directory (default: ./output)
  --no-interactive        Don't prompt for confirmation
  -c, --config FILE       YAML configuration file
  --style STYLE           Output style: rich, simple, json
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

## REDI Modes Explained

### Orchestrator Mode (`redi run`)

Use when you want REDI to own the data processing pipeline:
- Processes data through all stages (ingest → preprocess → transform → structure → output)
- Auto-detects or uses specified domain
- Generates Data Card and metadata
- Best for: ad-hoc processing, domains without existing pipelines

### Sidecar Mode (`redi sidecar`)

Use when you have an existing workflow and want consistent data readiness tracking:
- Observes Snakemake, Nextflow, or other workflows
- Maps workflow rules to REDI pipeline stages
- Generates Data Card without modifying your workflow
- Best for: established domain pipelines (ClimaX, OpenFold, etc.)

### Discovery Mode (`redi discover`)

Use when you encounter an unknown dataset:
- Inspects files and gathers metadata
- Infers data modality, domain, and characteristics
- Proposes a preprocessing pipeline
- Best for: new datasets, unfamiliar domains

## Module Usage

### Analyze a Dataset

```python
from redi.analyze import analyze_dataset, get_recommendations, generate_report

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
from redi.cli import run_pipeline

# Single file
ctx = run_pipeline(
    input_path="data.csv",
    output_dir="./output",
    domain="hpc",  # or None for auto-detect
    stages=["ingest", "preprocess", "transform", "structure", "output"]
)

# Multiple files
ctx = run_pipeline(
    input_paths=["data_1850.nc", "data_1860.nc", "data_1870.nc"],
    output_dir="./output",
    domain="climate"
)

print(f"Completed: {[s.value for s in ctx.completed_stages]}")
print(f"Output files: {ctx.metadata.get('saved_files', [])}")
```

### Use Individual Pipeline Steps

```python
from redi.pipeline import Pipeline, create_context
from redi.pipeline.ingest import get_ingest_steps
from redi.pipeline.preprocess import get_preprocess_steps

# Create context (single file)
ctx = create_context("data.csv", "./output", "hpc")

# Create context (multiple files)
ctx = create_context(
    output_dir="./output",
    domain="climate",
    input_paths=["data_1850.nc", "data_1860.nc", "data_1870.nc"]
)

# Build custom pipeline
pipeline = Pipeline("climate")
for step in get_ingest_steps():
    pipeline.add_step(step)
for step in get_preprocess_steps("climate"):
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
├── cli.py              # Command-line interface (run, sidecar, discover)
├── analyze.py          # Data Card generation and analysis
├── config.py           # YAML configuration handling
├── pipeline/
│   ├── __init__.py     # Pipeline, PipelineStep, PipelineContext
│   ├── ingest.py       # Format detection, loading
│   ├── preprocess.py   # Cleaning, deduplication
│   ├── transform.py    # Normalization, encoding
│   ├── structure.py    # Tensorization, graphs
│   └── output.py       # Splitting, sharding, saving
├── domains/
│   ├── __init__.py     # Domain detection
│   ├── climate.py      # xarray/xesmf processing
│   ├── materials.py    # pymatgen/matminer features
│   ├── life_sciences.py # BioPython, PII handling
│   ├── fusion.py       # Plasma signal processing
│   └── hpc.py          # Job accounting processing
├── fair/               # FAIR principles assessment
├── catalog/            # Data catalog integrations
└── examples/           # YAML configuration examples
    ├── sidecar_config.yaml
    ├── sidecar_climax_example.yaml
    ├── discover_config.yaml
    └── discover_hep_example.yaml
```

## Example: HPC Job Data

```bash
# Process Aurora job accounting data
redi run /path/to/hpc/ANL-ALCF-DJC-AURORA.csv -o ./hpc_output

# Output:
# hpc_output/
# ├── train.npz    (80% of data)
# ├── val.npz      (10% of data)
# ├── test.npz     (10% of data)
# ├── DATA_CARD.md
# └── metadata.json
```

## Example: Sidecar with ClimaX

```bash
# Observe ClimaX data preprocessing workflow
redi sidecar -w /path/to/ClimaX/snakemake_configs/TaiESM1/Snakefile -o ./climax_redi

# Output:
# climax_redi/
# └── sidecar_report.json (stage mapping, rule analysis)
```

## Example: Discover Unknown Dataset

```bash
# Analyze a dataset from Zenodo
redi discover /path/to/downloaded/sphenix_data -o ./analysis

# Output:
# analysis/
# ├── discovery_report.json
# └── DISCOVERY_REPORT.md
```

## Demo Script

Interactive demo showcasing REDI capabilities with sample or real data.

### Run with Sample Data

```bash
python demo.py --domain hpc       # HPC job accounting demo
python demo.py --domain climate   # Climate/NetCDF demo
python demo.py --domain materials # Materials science demo
python demo.py --domain bio       # Life sciences demo
python demo.py --all              # Run all domain demos
```

### Run with Real Data

```bash
# HPC job accounting data
python demo.py -i /path/to/ANL-ALCF-DJC-AURORA.csv -d hpc -o ./output --keep

# Climate NetCDF data
python demo.py -i /path/to/climate.nc -d climate -o ./output --keep

# Materials science data (AFLOW-style CSV or CIF files)
python demo.py -i /path/to/materials.csv -d materials -o ./output --keep

# Life sciences data (FASTA, PDB, or CSV with sequence data)
python demo.py -i /path/to/proteins.fasta -d bio -o ./output --keep
```

### Output Format Options

```bash
python demo.py -d hpc --format npz      # NumPy compressed (default)
python demo.py -d hpc --format parquet  # Apache Parquet
python demo.py -d hpc --format csv      # CSV files
python demo.py -d hpc --format hdf5     # HDF5 format
python demo.py -d hpc --format zarr     # Zarr arrays (climate)
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
