---
name: redi
description: REDI (Readiness Engine for Data Integration) transforms raw scientific data into AI-ready datasets. Use when the user wants to run data pipelines, observe existing workflows, or discover/analyze unknown datasets. Supports climate, materials science, life sciences, fusion, and HPC domains.
---

# REDI - Readiness Engine for Data Integration

REDI transforms raw scientific data into AI-ready datasets. It operates in three modes:

- **run** - Run the REDI pipeline (orchestrator mode) - transforms data through the full pipeline
- **analyze** - Observe existing workflows (Snakemake, Nextflow, bash) and generate Data Cards
- **discover** - Analyze unknown datasets and propose pipelines

Use when the user wants to:
- Transform data through the AI-readiness pipeline
- Observe/document an existing workflow
- Analyze an unknown dataset and get recommendations
- Generate a Data Card documenting a dataset

Supports climate, materials science, life sciences, fusion, and HPC domains.

## Quick Start

```bash
# Activate the environment
source /opt/venvs/drai/bin/activate

# Run pipeline (default mode)
redi run data.csv -o ./output
redi data.csv -o ./output              # 'run' is the default

# Observe an existing workflow
redi analyze --workflow Snakefile

# Analyze unknown data and propose pipeline
redi discover /path/to/data/
```

## Commands

### redi run

Run the REDI pipeline (orchestrator mode). This is the default command.

```bash
redi run <input> [options]
redi <input> [options]                 # 'run' is implicit

# Examples
redi run data.csv                      # Auto-detect domain, run full pipeline
redi run data.csv --domain hpc         # Specify domain
redi run data.csv -o ./processed       # Custom output directory
redi run data.csv --stages ingest,preprocess  # Run specific stages
redi run "data_*.nc"                   # Process multiple files (glob)
redi run data.npy --array              # Optimize for array data
```

**Options:**
```
  input                   Input file path(s) or glob pattern
  -d, --domain DOMAIN     Domain: climate, materials, life_sciences, fusion, hpc, general
  -o, --output DIR        Output directory
  -s, --stages STAGES     Comma-separated stages to run
  -f, --format FORMAT     Output format: npz, parquet, csv, hdf5, zarr
  --array                 Optimize pipeline for array data (skip DataFrame steps)
  --normalize METHOD      Normalization for arrays: minmax, zscore
  -v, --verbose           Verbose output
  --style STYLE           Output style: rich, simple, json
  -c, --config FILE       YAML configuration file
```

### redi analyze

Observe existing workflows and generate Data Cards without modifying the workflow.

```bash
redi analyze --workflow <path> [options]

# Examples
redi analyze --workflow Snakefile
redi analyze --workflow pipeline.nf --type nextflow
redi analyze --workflow Snakefile --watch    # Watch for changes
redi analyze -c analyze_config.yaml
```

**Options:**
```
  -w, --workflow FILE     Path to workflow file (Snakefile, *.nf, etc.)
  -t, --type TYPE         Workflow type: auto, snakemake, nextflow, bash
  -o, --output DIR        Output directory
  --watch                 Watch for changes and re-run
  --style STYLE           Output style: rich, simple, json
  -c, --config FILE       YAML configuration file
```

### redi discover

Analyze unknown datasets and propose pipelines.

```bash
redi discover <path> [options]

# Examples
redi discover /path/to/data/
redi discover dataset.csv -o ./analysis
redi discover /data --no-interactive
redi discover -c discover_config.yaml
```

**Options:**
```
  input                   Path to data file or directory
  -o, --output DIR        Output directory
  --style STYLE           Output style: rich, simple, json
  --no-interactive        Don't prompt for confirmation
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
├── cli.py              # CLI entry point (run, analyze, discover commands)
├── datacard.py         # Data Card data model (enums, dataclasses)
├── assess.py           # Data analysis and pipeline recommendations
├── analyze.py          # Backwards-compat shim (re-exports datacard + assess)
├── config.py           # Configuration management
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
├── dataloaders/        # Custom data loaders
│   ├── zarr_loader.py
│   └── sst_binary.py
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

# Option 2: From source (ESMF already built at /opt/drai/esmf)
export ESMFMKFILE=/opt/drai/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk
pip install /opt/drai/esmf/src/addon/esmpy/
pip install xesmf
```

Note: The drai virtualenv auto-sets ESMFMKFILE on activation.
