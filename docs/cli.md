# CLI Reference

REDI exposes eight subcommands: `run`, `inspect`, `discover`, `validate`, `assess`,
`install`, `plugins`, and `demo`. The `run` subcommand is the default — passing a
file directly without a subcommand is equivalent to `redi run <file>`.

The tables below mirror the live `argparse` definitions (`redi <command> --help`),
so run that command if you suspect this page is out of date.

```
usage: redi [-h] [--version]
            {run,inspect,discover,validate,assess,install,plugins,demo} ...
```

| Command | Purpose |
|---|---|
| `demo` | Download a sample dataset and run the pipeline end-to-end |
| `run` | Run the REDI pipeline (orchestrator mode) — default |
| `inspect` | Inspect data files or observe existing workflows |
| `discover` | Analyze unknown datasets and propose pipelines |
| `validate` | Validate REDI output against ground truth |
| `assess` | Assess AI-readiness with domain-specific metrics |
| `install` | Install a REDI plugin from a URL, local path, or PyPI package |
| `plugins` | List installed REDI plugins |

```bash
redi run data.csv                     # Run pipeline on data
redi data.csv                         # Same as above (run is default)
redi inspect data.csv                 # Inspect a data file
redi inspect Snakefile                # Observe Snakemake workflow
redi discover /path/to/data/          # Analyze unknown dataset
redi validate output/ -c config.yaml  # Validate against ground truth
```

## redi run

```
usage: redi run [-h]
                [-d {climate,materials,life_sciences,fusion,hpc,geospatial,nuclear_physics,combustion,general}]
                [-o OUTPUT] [--unique] [--hash-content] [-s STAGES]
                [-f {npz,parquet,csv,hdf5,zarr,pt}] [--shards N] [--array]
                [--normalize {minmax,zscore}] [--provenance] [-v] [--verify]
                [--raps-jobs] [--start START] [--end END]
                [--style {rich,simple,json}] [-c CONFIG] [--download]
                [--source {esgf,cds}] [--variable VARIABLE] [--model MODEL]
                [--experiment EXPERIMENT] [--frequency FREQUENCY]
                [--year YEAR] [-p] [-w WORKERS]
                [--backend {local,gnu,mpi,dragon}] [--batch-size BATCH_SIZE]
                [--generate-only] [--distribution {round_robin,balanced}]
                [--dragon-mode {subprocess,inprocess}]
                [--dragon-nodes DRAGON_NODES]
                [--dragon-launcher {local,slurm}] [--slurm-nodes SLURM_NODES]
                [--slurm-cores SLURM_CORES] [--slurm-time SLURM_TIME]
                [--slurm-account SLURM_ACCOUNT]
                [--slurm-partition SLURM_PARTITION]
                [--slurm-output SLURM_OUTPUT]
                [input ...]
```

**Positional**

| Argument | Description |
|---|---|
| `input` | Input file path(s) or glob pattern |

**Options**

| Flag | Description |
|---|---|
| `-d, --domain` | Scientific domain: `climate`, `materials`, `life_sciences`, `fusion`, `hpc`, `geospatial`, `nuclear_physics`, `combustion`, `general` (auto-detected if not specified) |
| `-o, --output` | Output directory (disables hash-based naming) |
| `--unique` | Append timestamp to output dir for unique runs |
| `--hash-content` | Hash file content (not just path+size) for output dir naming |
| `-s, --stages` | Comma-separated stages to run: `ingest`, `preprocess`, `transform`, `structure`, `output`. Prerequisites are auto-included (e.g. `-s preprocess` runs `ingest,preprocess`) |
| `-f, --format` | Output format: `npz`, `parquet`, `csv`, `hdf5`, `zarr`, `pt` |
| `--shards N` | Number of output shards per split (e.g. match your training node count) |
| `--array` | Optimize pipeline for array data (skip DataFrame-specific steps) |
| `--normalize` | Normalization method for array data: `minmax`, `zscore` |
| `--provenance` | Enable FlowCept provenance tracking and generate `PROVENANCE_CARD.md` |
| `-v, --verbose` | Verbose output |
| `--verify` | Run verification checks after each pipeline step |
| `--raps-jobs` | Extract RAPS Job objects from HPC telemetry (HPC domain only) |
| `--start` | Filter jobs from this datetime, inclusive (ISO format `2017-10-03T00:00` or `YYYYMMDD`). Overrides `start` key in YAML config |
| `--end` | Filter jobs until this datetime, exclusive. Overrides `end` key in YAML config |
| `--style` | Output style: `rich`, `simple`, `json` |
| `-c, --config` | YAML configuration file path |

**Download options** — download data from remote sources (ESGF for CMIP6, CDS for ERA5)

| Flag | Description |
|---|---|
| `--download` | Download data before processing |
| `--source` | Data source: `esgf` (CMIP6) or `cds` (ERA5) |
| `--variable` | Variable to download (e.g. `tas`, `pr`, `t2m`) |
| `--model` | Climate model/source_id (e.g. `TaiESM1`, `MPI-ESM1-2-HR`) |
| `--experiment` | Experiment ID (e.g. `historical`, `ssp585`) |
| `--frequency` | Time frequency (e.g. `6hr`, `day`, `mon`) |
| `--year` | Year to download (for CDS/ERA5) |

**Parallel execution**

| Flag | Description |
|---|---|
| `-p, --parallel` | Enable parallel execution across input files |
| `-w, --workers` | Number of parallel workers (default: auto-detect CPU count) |
| `--backend` | Parallel backend: `local` (GNU parallel, no SLURM), `gnu` (SLURM + GNU parallel), `mpi` (SLURM + mpi4py), `dragon` (Dragon HPC distributed runtime) |
| `--batch-size` | Files per worker batch (default: 4) |
| `--generate-only` | Generate parallel scripts without executing (for SLURM) |
| `--distribution` | File distribution strategy: `round_robin` (default), `balanced` |

**Dragon options** (for `--backend dragon`)

| Flag | Description |
|---|---|
| `--dragon-mode` | `subprocess`: each worker calls `redi run` as a subprocess; `inprocess`: pipeline runs inside the Dragon worker (experimental) |
| `--dragon-nodes` | Number of nodes for multi-node Dragon launch (default: 1) |
| `--dragon-launcher` | `local`: call `dragon` directly; `slurm`: wrap in a SLURM batch script via `--generate-only` |

**SLURM options** (for `--backend gnu`, `mpi`, or `dragon`)

| Flag | Description |
|---|---|
| `--slurm-nodes` | Number of compute nodes (default: 4) |
| `--slurm-cores` | CPU cores per node (default: 56) |
| `--slurm-time` | Time limit `HH:MM:SS` (default: `0:30:00`) |
| `--slurm-account` | SLURM account/allocation |
| `--slurm-partition` | SLURM partition (default: `batch`) |
| `--slurm-output` | Final output directory (for copying from burst buffer) |

```bash
redi run data.csv                      # Auto-detect and run full pipeline
redi run data.csv --domain hpc         # Specify domain
redi run data.csv -o ./processed       # Custom output directory
redi run data.csv --stages ingest,preprocess  # Run specific stages
redi run "data_*.nc"                   # Process multiple files (glob)

# Download examples (climate domain)
redi run --download --source esgf --variable tas --model TaiESM1
redi run --download --source cds --variable 2m_temperature --year 2020
```

## redi inspect

```
usage: redi inspect [-h] [-w WORKFLOW_FLAG]
                    [-t {auto,snakemake,nextflow,bash}] [-o OUTPUT] [--unique]
                    [--data-dir DATA_DIR] [--watch]
                    [--style {rich,simple,json}] [-c CONFIG]
                    [input]
```

| Flag | Description |
|---|---|
| `input` | Path to data file, workflow file, or directory |
| `-w, --workflow` | Path to workflow file (overrides positional input) |
| `-t, --type` | Workflow type override: `auto`, `snakemake`, `nextflow`, `bash` (forces workflow mode) |
| `-o, --output` | Output directory (disables hash-based naming) |
| `--unique` | Append timestamp to output dir for unique runs |
| `--data-dir` | Directory containing data files to scan for statistics |
| `--watch` | Watch for changes and re-run |
| `--style` | Output style: `rich`, `simple`, `json` |
| `-c, --config` | YAML configuration file |

```bash
redi inspect data.csv                       # Inspect a data file
redi inspect /path/to/data.nc               # Inspect a NetCDF file
redi inspect Snakefile                      # Auto-detect and observe workflow
redi inspect pipeline.nf --type nextflow    # Observe Nextflow workflow
redi inspect --workflow Snakefile --watch   # Watch for changes
redi inspect -c inspect_config.yaml
```

## redi discover

```
usage: redi discover [-h] [-o OUTPUT] [--unique] [--style {rich,simple,json}]
                     [--no-interactive] [-c CONFIG]
                     [input]
```

| Flag | Description |
|---|---|
| `input` | Path to data file or directory |
| `-o, --output` | Output directory (disables hash-based naming) |
| `--unique` | Append timestamp to output dir for unique runs |
| `--style` | Output style: `rich`, `simple`, `json` |
| `--no-interactive` | Don't prompt for confirmation |
| `-c, --config` | YAML configuration file |

Output files written to the output directory:

* `discovery_report.json` — full inspection/inference/proposal data
* `DISCOVERY_REPORT.md` — human-readable domain-level summary
* `processing_plan.md` — per-file discoveries and preprocessing suggestions

```bash
redi discover /path/to/data/
redi discover dataset.csv -o ./analysis
redi discover /data --no-interactive
redi discover -c discover_config.yaml
```

## redi validate

```
usage: redi validate [-h] [-g GROUND_TRUTH] [--gt-variable GT_VARIABLE]
                     [-c CONFIG] [-o OUTPUT] [--aidrin] [--per-feature]
                     [--samples [N]] [--style {rich,simple,json}] [-v]
                     input
```

| Flag | Description |
|---|---|
| `input` | Path to REDI output directory or a direct output file (npz, npy, parquet, csv) |
| `-g, --ground-truth` | Path to ground truth file (overrides config) |
| `--gt-variable` | Variable name to extract from NetCDF ground truth (default: first variable) |
| `-c, --config` | Validation YAML config file |
| `-o, --output` | Output directory for validation report |
| `--aidrin` | Run AIDRIN metric comparison (slow) |
| `--per-feature` | Per-feature similarity breakdown |
| `--samples [N]` | Show N random job samples side-by-side (default 10) |
| `--style` | Output style: `rich`, `simple`, `json` |
| `-v, --verbose` | Verbose output |

```bash
redi validate output/usa_0_data_5c79b84 -c examples/nasa_power_validate.yaml
redi validate output/usa_0_data_5c79b84 --ground-truth /path/to/truth.pt
redi validate output/usa_0_data_5c79b84 -c config.yaml --aidrin
redi validate output/usa_0_data_5c79b84 -c config.yaml --style json
```

## redi assess

```
usage: redi assess [-h]
                   [-d {climate,materials,life_sciences,fusion,hpc,geospatial,general}]
                   [-o OUTPUT] [--style {rich,simple,json}] [-c CONFIG]
                   [input]
```

| Flag | Description |
|---|---|
| `input` | Path to data file |
| `-d, --domain` | Domain (auto-detected if omitted) |
| `-o, --output` | Save JSON report to this path |
| `--style` | Output style: `rich`, `simple`, `json` (default: `rich`) |
| `-c, --config` | YAML config file (supports multi-file comparison) |

```bash
redi assess data.csv                       # auto-detect domain
redi assess output/frontier/raps_jobs.npz --domain hpc
redi assess /opt/drai/ClimaX/TaiESM1/tas.nc
redi assess /opt/drai/aflow/aflow_ml_ready.csv --domain materials
redi assess --style json data.parquet      # JSON output
```

## redi install

```
usage: redi install [-h] [--upgrade] target
```

| Flag | Description |
|---|---|
| `target` | pip-compatible install target |
| `--upgrade` | Upgrade if already installed |

```bash
redi install git+https://github.com/org/redi-xgc1
redi install ./redi-xgc1            # local directory
redi install redi-xgc1==1.0.0       # PyPI package
redi install ./plugin.tar.gz --upgrade
```

## redi plugins

```
usage: redi plugins [-h] [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output as JSON |

## redi demo

```
usage: redi demo [-h] [-d {general,hpc,climate,geospatial,materials}]
                 [-o OUTPUT] [--list] [--style {rich,simple,json}]
```

| Flag | Description |
|---|---|
| `-d, --domain` | Domain to demo (default: `general`) |
| `-o, --output` | Output directory (default: `./demo_output/<domain>`) |
| `--list` | List available demos and exit |
| `--style` | Output style: `rich`, `simple`, `json` |

Downloads a small public dataset (or generates synthetic data) for the chosen
domain, runs the full REDI pipeline, and prints the resulting Data Card.

| Domain | Dataset |
|---|---|
| `general` (default) | UCI Wine Quality (~84 KB CSV) |
| `hpc` | Synthetic HPC job log (5,000 jobs) |
| `climate` | NOAA GHCND Honolulu station (~400 KB CSV) |
| `geospatial` | EPA Annual AQI by County 2023 (~250 KB CSV) |
| `materials` | Synthetic crystal structure dataset |

```bash
redi demo                          # Run general demo (UCI Wine Quality)
redi demo --domain hpc             # Run HPC telemetry demo
redi demo --domain climate -o ./out
```
