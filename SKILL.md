---
name: redi
description: REDI (Readiness Engine for Data Integration) transforms raw scientific data into AI-ready datasets. Use when the user wants to analyze a dataset for AI/ML readiness, generate a Data Card, get preprocessing recommendations, or process data through the full pipeline. Supports climate, materials science, life sciences, fusion, and HPC domains.
---

# REDI - Readiness Engine for Data Integration

REDI transforms raw scientific data into AI-ready datasets. Use when the user wants to:
- Analyze a dataset for AI/ML readiness
- Generate a Data Card documenting a dataset
- Get preprocessing recommendations
- Process data through the full pipeline
- Understand what steps are needed to make data AI-ready

Supports climate, materials science, life sciences, fusion, and HPC domains.

## Usage

When the user provides a dataset path, run the REDI CLI:

```bash
# Full pipeline (auto-detects domain)
redi /path/to/data.csv -o ./output

# Analysis only (generates Data Card)
redi --analyze /path/to/data.csv

# Specific domain
redi /path/to/data.nc --domain climate -o ./output
```

## Quick Start

```bash
# Activate the environment
source /opt/venvs/drai/bin/activate

# Analyze a dataset (generates Data Card)
redi --analyze data.csv

# Run full pipeline
redi data.csv -o ./output

# Specify domain explicitly
redi data.csv --domain hpc -o ./output

# Run specific stages only
redi data.csv --stages ingest,preprocess,transform

# Process multiple files (glob pattern)
redi "/path/to/data/*.nc" -d climate -o ./output

# Process multiple files (explicit list)
redi file1.nc file2.nc file3.nc -d climate -o ./output
```

## CLI Options

```
redi <input> [input2 ...] [options]

Arguments:
  input                   Input file path(s) or glob pattern (e.g., "*.nc")

Options:
  -d, --domain DOMAIN     Domain: climate, materials, life_sciences, fusion, hpc, general
  -o, --output DIR        Output directory (default: ./output)
  -s, --stages STAGES     Comma-separated stages to run
  -f, --format FORMAT     Output format: npz, parquet, csv, hdf5, zarr
  -a, --analyze           Only analyze and generate Data Card
  -v, --verbose           Verbose output
  -c, --config FILE       YAML configuration file
```

## Domains

| Domain | Detected By | Key Tools | Output Formats |
|--------|-------------|-----------|----------------|
| climate | .nc, .grib, temperature/lat/lon | xarray, xesmf | NPZ, Zarr |
| materials | .cif, .poscar, formula/lattice | pymatgen | HDF5, NPZ |
| life_sciences | .fasta, .pdb, sequence/protein | biopython | HDF5 |
| fusion | plasma/disruption/tokamak | h5py | HDF5, NPZ |
| hpc | job/node/walltime/queue | pandas, sklearn | NPZ, Parquet |

## Pipeline Stages

1. **ingest** - Load data, detect format, extract metadata
2. **preprocess** - Clean, deduplicate, handle missing values
3. **transform** - Normalize, encode categoricals, engineer features
4. **structure** - Convert to tensors/graphs/sequences
5. **output** - Train/val/test split, shard, save

## Readiness Levels

| Level | Name | Description |
|-------|------|-------------|
| 1 | RAW | Unprocessed, as collected |
| 2 | CLEANED | Missing values handled, quality-checked |
| 3 | LABELED | Target variables defined |
| 4 | FEATURE_ENGINEERED | Domain features extracted |
| 5 | FULLY_AI_READY | Sharded, normalized, optimized format |

## Project Structure

```
redi/
├── cli.py              # CLI entry point (redi command)
├── analyze.py          # Data Card generation
├── pipeline/           # Pipeline stage modules
│   ├── ingest.py
│   ├── preprocess.py
│   ├── transform.py
│   ├── structure.py
│   └── output.py
├── domains/            # Domain-specific processors
│   ├── climate.py
│   ├── materials.py
│   ├── life_sciences.py
│   ├── fusion.py
│   └── hpc.py
└── tests/              # Unit and smoke tests
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run domain-specific tests
pytest tests/test_smoke_hpc.py -v
pytest tests/test_smoke_climate.py -v
pytest tests/test_smoke_materials.py -v
pytest tests/test_smoke_life_sciences.py -v
```

## Climate Domain Setup

The climate domain requires ESMF for regridding:

```bash
# Option 1: Conda (easiest)
conda install -c conda-forge esmpy xesmf

# Option 2: From source (ESMF already built at /home/w1b/drai/climate/esmf)
export ESMFMKFILE=/home/w1b/drai/climate/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk
pip install /home/w1b/drai/climate/esmf/src/addon/esmpy/
pip install xesmf
```

Note: The drai virtualenv auto-sets ESMFMKFILE on activation.
