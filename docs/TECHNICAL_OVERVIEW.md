# REDI Technical Overview

## What It Does

REDI (Readiness Engine for Data Integration) is a modular pipeline framework that transforms raw scientific data into AI-ready datasets. It handles the full lifecycle: ingesting heterogeneous formats, cleaning, feature engineering, structuring for ML, and outputting optimized splits. It tracks data maturity across a 5-level readiness taxonomy (RAW → CLEANED → LABELED → FEATURE_ENGINEERED → FULLY_READY).

## Four Operating Modes

| Mode | Purpose | Entry Point |
|---|---|---|
| **`redi run`** | Execute the full transformation pipeline on data | `cli.py:_handle_run_command()` |
| **`redi inspect`** | Observe existing workflows (Snakemake/Nextflow) and generate Data Cards without modifying anything | `cli.py:_handle_inspect_command()` |
| **`redi discover`** | Analyze unknown datasets, auto-detect domain, and propose a pipeline | `cli.py:_handle_discover_command()` |
| **`redi validate`** | Compare REDI output against ground-truth data with statistical similarity metrics | `cli.py:_handle_validate_command()` |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    CLI (cli.py)                   │
│  Argument parsing, domain detection, output dir  │
│  generation, visual progress, style (rich/json)  │
└────────────────────┬─────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Pipeline Orchestrator  │
        │  pipeline/__init__.py   │
        │                         │
        │  PipelineContext (data   │
        │  + metadata flows       │
        │  through all stages)    │
        └────────────┬────────────┘
                     │
    ┌────────┬───────┼───────┬──────────┐
    ▼        ▼       ▼       ▼          ▼
 INGEST  PREPROC  TRANSFORM  STRUCT   OUTPUT
```

### Core Abstractions (`redi/pipeline/__init__.py`)

- **`PipelineStage`** — Enum of the 5 stages: `INGEST → PREPROCESS → TRANSFORM → STRUCTURE → OUTPUT`
- **`PipelineContext`** — Mutable dataclass that flows through every step. Carries:
  - `data` (the actual DataFrame/xarray/array), `metadata` (dict), `input_path(s)`, `output_dir`, `domain`, `config`
  - Pipeline state: `current_stage`, `completed_stages`, `errors`, `verification_results`
  - Optional Flowcept handle for provenance tracking
- **`PipelineStep`** — Base class. Each step implements `execute(ctx) → ctx`. Optional `validate()` (pre-check) and `verify()` (post-check). Steps are auto-instrumented with Flowcept for lineage tracking via `__init_subclass__`.
- **`Pipeline`** — Orchestrator that sequences steps, handles stage transitions, runs verification, and manages Flowcept lifecycle.

### Pipeline Stages in Detail

**1. Ingest** (`pipeline/ingest.py`, ~911 lines)
- `DetectFormat` sniffs file type from extension + content (CSV, JSON, Parquet, NetCDF, HDF5, NPY/NPZ, SICKLE formats)
- Format-specific loaders: `LoadCSV`, `LoadJSON`, `LoadParquet`, `LoadNetCDF`, `LoadNPY`, `LoadNPZ`, `LoadHDF5`, `LoadSICKLE`
- Multi-file support: glob expansion, concatenation of DataFrames, validation that all files share the same format

**2. Preprocess** (`pipeline/preprocess.py`, ~592 lines)
- Generic steps: `HandleMissingValues` (drop/fill/mean/median/mode), `RemoveDuplicates`, `ValidateDataTypes`, `HandleOutliers` (IQR/zscore → clip/remove/flag), `AnonymizePII`
- Array-mode steps: `NormalizeArray`, `ComputeArrayStats` (skip DataFrame logic for large N-d arrays)

**3. Transform** (`pipeline/transform.py`, ~370 lines)
- `NormalizeNumeric` (zscore/minmax/robust), `EncodeCategorical` (one-hot/label/frequency), `EngineerFeatures` (custom expressions), `AutoFeatureEngineering` (datetime extraction, log transforms), `SelectFeatures` (variance/correlation filtering)

**4. Structure** (`pipeline/structure.py`, ~360 lines)
- `ToNumPyArrays` — DataFrame → `{X, y, feature_names}`
- `ToTorchTensors` — NumPy → PyTorch (CPU/GPU)
- `CreateSequences` — Sliding-window for time series
- `CreateGraphStructure` / `CreateTorchGeometricData` — For GNN inputs

**5. Output** (`pipeline/output.py`, ~469 lines)
- `TrainValTestSplit` (random/stratified, configurable ratios)
- `ShardData` (split large datasets into smaller files)
- Format writers: `SaveNPZ`, `SaveHDF5`, `SaveParquet`, `SaveCSV`, `SaveZarr`, `SaveTFRecord`, `SaveJSON`
- `SaveMetadata` — Data Card + pipeline log

## Domain System

Each domain (`redi/domains/<name>.py`) exports `get_ingest_steps()`, `get_preprocess_steps()`, `get_transform_steps()`, `get_structure_steps()`, and `get_output_steps()` that return domain-specific `PipelineStep` instances.

Domain auto-detection (`domains/__init__.py`) works in two passes:
1. **File extension** — `.nc` → climate, `.cif` → materials, `.fasta` → life sciences, etc.
2. **Column keyword scoring** — Counts keyword hits across columns (e.g., "temperature", "lat", "lon" → climate)

| Domain | Key Libraries | Specialized Steps |
|---|---|---|
| **Climate** (~740 lines) | xarray, xesmf, netCDF4 | `LoadClimateData`, `RegridClimate` (xESMF regridding), `NormalizeClimate` (per-var/per-level), `SelectVariables`, `SelectPressureLevels`, `SaveClimate` (Zarr/NPZ with coords) |
| **Materials** (~456 lines) | pymatgen, matminer | `LoadStructureFile` (CIF/POSCAR), `CleanMaterialsData`, `FeaturizeCompositions` (Magpie features), `ToGraphStructure` (crystal → graph) |
| **Life Sciences** (~377 lines) | biopython | `LoadFASTA`, `AnonymizePII` (hash/remove/mask), `FilterSequences`, `EncodeSequences` (integer encoding with padding), `ComputeSequenceFeatures` |
| **Fusion** (~456 lines) | h5py | `LoadFusionHDF5`, `AlignSignals` (common timebase), `HandleShotBoundaries`, `EngineerPlasmaFeatures` (Beta_N, Greenwald fraction), `EncodeDisruptionLabels` |
| **HPC** (~631 lines) | pandas, sklearn | `LoadHPCData`, `CleanHPCData`, `ParseTimestamps`, `EngineerHPCFeatures` (wait_time, core_hours), `EncodeHPCCategoricals` |
| **Geospatial** (~551 lines) | h3, geopandas | `LoadGeospatialData`, `HexifyData` (lat/lon → H3 hexagons), `AggregateH3Cells`, `TemporalAggregation`, `SpatialSplit` |

### HPC Job Mapper (`hpc_job_mapper.py`, ~1,689 lines)

A standalone subsystem for extracting structured RAPS Job objects from raw HPC telemetry. Uses a multi-phase column-mapping strategy (exact match → token overlap → fuzzy matching) with confidence scores, then derives missing fields and normalizes timestamps.

## Data Flow (run mode)

```
User Input (file path + optional config)
  │
  ├─ expand_input_paths()     ← glob expansion, multi-file support
  ├─ detect_domain()          ← auto-detect from extension + column keywords
  ├─ generate_output_dir()    ← deterministic hash-based dir naming
  │
  ▼
PipelineContext created
  │
  ├─ Domain module selected   ← get_domain_pipeline(domain)
  ├─ Steps assembled          ← domain.get_{stage}_steps() for each stage
  │                             (or generic steps for "general" domain)
  │
  ▼
Pipeline.execute(ctx)
  │
  ├─ For each step:
  │    1. step.validate(ctx)  ← pre-check
  │    2. step.execute(ctx)   ← transform data, update metadata
  │    3. step.verify(ctx)    ← post-check (if --verify enabled)
  │    4. Flowcept capture    ← automatic provenance (if enabled)
  │
  ▼
Output directory contains:
  ├─ train/val/test splits (NPZ, HDF5, Parquet, etc.)
  ├─ data_card.json           ← 8-section metadata document
  ├─ metadata.json            ← pipeline run details
  └─ repro.sh                 ← reproducibility script
```

## Data Acquisition

Two built-in downloaders (`redi/downloaders/`) can fetch data before processing:

- **ESGF** (`esgf.py`) — CMIP6 climate model data. Searches ESGF Solr nodes, validates checksums, supports parallel downloads with ThreadPoolExecutor.
- **CDS** (`cds.py`) — ERA5 reanalysis from Copernicus. Wraps the `cdsapi` library. Requires API key in `~/.cdsapirc`.

Both implement a `BaseDownloader` abstract class with `search()` and `download()` methods and are registered via `@register_downloader("name")`.

## Custom Data Loaders

`redi/dataloaders/` provides loaders for specialized array formats via a `BaseDataLoader` abstract class:

- **GESTS** — GPU-based Extreme-Scale Turbulence Simulations
- **SST-Binary** — Stably Stratified Turbulence (raw binary floats)
- **Zarr** — Chunked N-dimensional arrays

Factory: `get_dataloader(path, format_type="auto")` returns the appropriate loader.

## Supporting Modules

| Module | Purpose |
|---|---|
| **`datacard.py`** | Data Card data model: `MaturityLevel` enum (1-5), `DataCard` dataclass with 8 sections (overview, provenance, characteristics, sensitive attrs, transformations, annotations, limitations, readiness). Serializes to JSON/Markdown. |
| **`assess.py`** | `DataAnalyzer` (inspects datasets → Data Cards) and `PipelineRecommender` (suggests processing steps per domain) |
| **`config.py`** | YAML config loading, deep merge with defaults, validation. Covers regrid, normalize, split, shard, and ClimaX output settings. |
| **`repro.py`** | Reproducibility: logs every command to `repro.sh` with full environment metadata (host, user, Python version, git commit hash, cwd). |
| **`provenance_card.py`** | Lightweight provenance helpers backed by Flowcept reports. Generates human-readable provenance artifacts from pipeline execution data. |
| **`inspect/`** | `snapshot.py` (execution context capture), `quality.py` (data quality metrics), `suggestions.py` (improvement recommendations), `transforms.py` (transformation tracking), `snakemake_log.py` and `nextflow_parser.py` (workflow parsers for inspect mode) |
| **`discover/`** | Three-phase discovery engine: inspect files → infer domain/modality → propose pipeline. Entry point for `redi discover`. |

## Configuration

YAML-driven (`config.py`). Key config sections:

```yaml
domain: climate
output_dir: ./output
output_format: npz
variables:
  select: [tas, pr]
  pressure_levels: [500, 850]
regrid:
  enabled: true
  target_resolution: 1.0
  method: bilinear
normalize:
  method: zscore          # zscore | minmax
  per_variable: true
split:
  method: random          # random | temporal
  ratios: [0.7, 0.15, 0.15]
shard:
  enabled: false
  samples_per_shard: 10000
```

## Testing

Tests live in `tests/` with domain-specific smoke tests (`test_smoke_hpc.py`, `test_smoke_climate.py`, etc.) and unit tests for the pipeline, config, and analysis modules. Run with `pytest tests/ -v`.

## Key Design Decisions

- **Context-passing architecture** — A single `PipelineContext` object is mutated and returned by each step, carrying both data and metadata through the entire pipeline.
- **Domain as a plugin** — Each domain is a module exporting step-factory functions. Adding a new domain means adding one file under `domains/` and registering its keywords/extensions.
- **Lazy imports** — Domain-specific dependencies (xarray, pymatgen, biopython) are imported inside functions, not at module level, so the core framework stays lightweight.
- **Flowcept auto-instrumentation** — Every `PipelineStep` subclass gets automatic provenance wrapping via `__init_subclass__`, with zero changes to step implementations.
- **Array mode** — The `--array` flag skips DataFrame-centric steps for high-dimensional data (e.g., climate grids), using direct NumPy normalization instead.
