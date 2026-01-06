# REDI - Readiness Engine for Data Integration

## Description

REDI transforms raw scientific data into AI-ready datasets. Use when the user wants to:
- Analyze a dataset for AI/ML readiness
- Generate a Data Card documenting a dataset
- Get preprocessing recommendations
- Process data through the full pipeline
- Understand what steps are needed to make data AI-ready

Supports climate, materials science, life sciences, fusion, and HPC domains.

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
```

## Demo Script

Interactive demo showcasing all capabilities:

```bash
# Run with sample data
python demo.py                          # HPC demo (default)
python demo.py --domain climate         # Climate demo
python demo.py --domain materials       # Materials science demo
python demo.py --domain bio             # Life sciences demo
python demo.py --all                    # All domains

# Run with real data
python demo.py -i /path/to/data.csv -d hpc -o ./output --keep

# Change output format
python demo.py --format parquet         # Parquet output
python demo.py --format csv             # CSV output
```

## CLI Options

```
redi <input> [options]

Arguments:
  input                   Input file path

Options:
  -d, --domain DOMAIN     Domain: climate, materials, life_sciences, fusion, hpc, general
  -o, --output DIR        Output directory (default: ./output)
  -s, --stages STAGES     Comma-separated stages to run
  -a, --analyze           Only analyze and generate Data Card
  -v, --verbose           Verbose output
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
├── demo.py             # Interactive demo script
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
