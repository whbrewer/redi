# Quick Start

## Installation

REDI requires Python 3.9 or later. Install the core package:

```bash
pip install -e .
```

Install with optional domain-specific dependencies:

```bash
# All scientific domains
pip install -e ".[all-domains]"

# Individual domains
pip install -e ".[climate]"
pip install -e ".[materials]"
pip install -e ".[life_sciences]"
pip install -e ".[fusion]"
pip install -e ".[hpc]"
pip install -e ".[geospatial]"

# Output format support
pip install -e ".[parquet]"
pip install -e ".[hdf5]"
```

## Activating the bundled environment

If you are using the pre-built `drai` virtual environment:

```bash
source /opt/venvs/drai/bin/activate
```

The environment automatically sets `ESMFMKFILE` for climate domain regridding.

## Try it now — no data required

`redi demo` downloads a small public dataset and runs the full pipeline so you
can see REDI working before you have your own data:

```bash
redi demo                          # UCI Wine Quality (general domain, ~84 KB)
redi demo --domain hpc             # Synthetic HPC job telemetry
redi demo --domain climate         # NOAA GHCND daily weather station data
redi demo --domain geospatial      # EPA Annual AQI by County (2023)
redi demo --domain materials       # Synthetic crystal structure dataset
redi demo --list                   # Show all available demos
```

Output lands in `./demo_output/<domain>/` by default.

## Running the pipeline

The most common usage is `redi run`. The `run` subcommand is the default, so
these two forms are equivalent:

```bash
redi run data.csv -o ./output
redi data.csv -o ./output
```

**Auto-detect domain and run the full pipeline:**

```bash
redi run data.csv
```

**Specify the scientific domain:**

```bash
redi run data.csv --domain hpc
```

**Custom output directory:**

```bash
redi run data.csv -o ./processed
```

**Run specific pipeline stages only:**

```bash
redi run data.csv --stages ingest,preprocess
```

**Process multiple files (glob pattern):**

```bash
redi run "data_*.nc"
```

**Optimize for array data:**

```bash
redi run data.npy --array --normalize minmax
```

## Downloading climate data

REDI can download data from remote sources before processing.

CMIP6 via ESGF:

```bash
redi run --download --source esgf \
    --variable tas --model TaiESM1 --experiment historical \
    -o ./cmip6_output
```

ERA5 via Copernicus CDS:

```bash
redi run --download --source cds \
    --variable 2m_temperature --year 2020 \
    -o ./era5_output
```

> **Note:** CDS access requires an API key. Register at
> https://cds.climate.copernicus.eu/ and configure `~/.cdsapirc`.

## OpenFold protein structure datasets

The `life_sciences` domain recognises OpenFold dataset layouts and runs the
full IPTSO feature-engineering pipeline. Two output modes are available:

NPZ mode (default) — per-protein feature tensors:

```bash
redi run /path/to/openfold_data/ --domain life_sciences -o ./openfold_ready
redi run /path/to/openfold_data/ -c examples/openfold_config.yaml
```

OpenFoldDataModule-compatible layout (drop-in replacement):

```bash
redi run /path/to/openfold_data/ --domain life_sciences \
    --format openfold_native -o ./openfold_native
```

## sPHENIX TPC (nuclear physics)

Process sPHENIX outer TPC data for NeuralCompression_v3:

```bash
redi run /path/to/sphenix/outer/ --domain nuclear_physics -o ./sphenix_ready
redi run /path/to/sphenix/outer/ -c examples/sphenix_config.yaml
```

> **Note:** The sPHENIX pipeline applies zero-suppression and `log2`
> normalization. When loading with `DatasetTPC`, set `log: false` in the
> model config.

## Observing an existing workflow

```bash
redi inspect --workflow Snakefile
redi inspect --workflow pipeline.nf --type nextflow
redi inspect --workflow Snakefile --watch   # watch for changes
```

## Analyzing an unknown dataset

```bash
redi discover /path/to/data/
redi discover dataset.csv -o ./analysis
redi discover /data --no-interactive
```

## Validating REDI output

```bash
redi validate output/myrun_a3f2b1c -c config.yaml
redi validate output/myrun_a3f2b1c --ground-truth /path/to/truth.pt
```

## Using a YAML configuration file

All options can be supplied via a YAML config file:

```bash
redi run -c my_pipeline.yaml
```

CLI flags override corresponding YAML keys. For the full list of supported
configuration keys see [CLI Reference](cli.md).
