# REDI - Readiness Engine for Data Integration

## Description

REDI transforms raw scientific data into AI-ready datasets. Use when the user wants to:
- Analyze a dataset for AI/ML readiness
- Generate a Data Card documenting a dataset
- Get preprocessing recommendations
- Process data through the full pipeline
- Understand what steps are needed to make data AI-ready

Supports climate, materials science, life sciences, fusion, and HPC domains.

## Usage

### Analyze a dataset
```bash
cd /path/to/redis
source /opt/venvs/drai/bin/activate
python analyze.py <file_path>
```

### Run full pipeline
```bash
python cli.py <file_path> -o ./output
```

### Specify domain
```bash
python cli.py <file_path> --domain hpc -o ./output
```

### Run specific stages
```bash
python cli.py <file_path> --stages ingest,preprocess,transform
```

## Domains

| Domain | Detected By | Output |
|--------|-------------|--------|
| climate | .nc, .grib, temperature/lat/lon columns | NPZ, Zarr |
| materials | .cif, .poscar, formula/lattice columns | HDF5, NPZ |
| life_sciences | .fasta, .pdb, sequence/protein columns | HDF5 |
| fusion | plasma/disruption/tokamak columns | HDF5 |
| hpc | job/node/walltime/queue columns | NPZ, Parquet |

## Pipeline Stages

1. **ingest** - Load and detect format
2. **preprocess** - Clean, deduplicate, handle missing
3. **transform** - Normalize, encode, engineer features
4. **structure** - Convert to tensors/graphs
5. **output** - Split, shard, save

## Readiness Levels

| Level | Name | Description |
|-------|------|-------------|
| 1 | RAW | Unprocessed |
| 2 | CLEANED | Quality-checked |
| 3 | LABELED | Targets defined |
| 4 | FEATURE_ENGINEERED | Domain features extracted |
| 5 | FULLY_AI_READY | Sharded, optimized format |

## Files

- `cli.py` - Main entry point
- `analyze.py` - Data Card generation (can run standalone)
- `pipeline/` - Pipeline stage modules
- `domains/` - Domain-specific processors
