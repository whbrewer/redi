# Drossfire

**Data Readiness for AI** - A modular pipeline for transforming raw scientific data into AI-ready datasets.

The name "Drossfire" evokes refining metal by burning away impurities (dross) to reveal pure, AI-ready data.

## Installation

```bash
# Activate the virtual environment
source /opt/venvs/drai/bin/activate

# Navigate to drossfire
cd /home/w1b/drai/drossfire
```

## Quick Start

### Run Full Pipeline

```bash
# Auto-detect domain and process
python cli.py data.csv -o ./output

# Specify domain explicitly
python cli.py data.nc --domain climate -o ./output

# Run specific stages only
python cli.py data.csv --stages ingest,preprocess,transform
```

### Analyze Dataset Only

```bash
# Generate Data Card and recommendations (no processing)
python cli.py data.csv --analyze

# Or use analyze.py directly
python analyze.py data.csv
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
python cli.py <input> [options]

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
drossfire/
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
python cli.py /home/w1b/drai/hpc/ANL-ALCF-DJC-AURORA_20250127_20251031.csv -o ./hpc_output

# Output:
# hpc_output/
# ├── train.npz    (80% of data)
# ├── val.npz      (10% of data)
# ├── test.npz     (10% of data)
# └── metadata.json
```

## References

- [DRAI Framework Paper](https://doi.org/10.1145/3750720.3757282)
- [Data Cards Paper](https://doi.org/10.1145/3531146.3533231)
