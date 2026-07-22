# REDI — Claude Project Instructions

## What REDI Is

REDI (Readiness Engine for Data Integration) is a modular pipeline framework that transforms raw scientific data into AI-ready datasets. It supports six scientific domains (climate, materials, life sciences, fusion, HPC, geospatial) and operates in three modes:

- **`redi run`** — Execute the full transformation pipeline on data files
- **`redi analyze`** — Observe existing workflows (Snakemake/Nextflow) and generate Data Cards without modifying anything
- **`redi discover`** — Analyze unknown datasets, auto-detect the domain, and propose a pipeline

The CLI entry point is `redi/cli.py`. The virtual environment is at `/opt/venvs/drai/bin/activate`. Run tests with `pytest tests/ -v`.

## Architecture

The pipeline has 5 sequential stages. A `PipelineContext` dataclass flows through every step, carrying `data` (DataFrame, xarray Dataset, NumPy array, or dict), `metadata` (accumulated dict), paths, domain, config, and error state.

```
INGEST → PREPROCESS → TRANSFORM → STRUCTURE → OUTPUT
```

- **Ingest** (`pipeline/ingest.py`): Format detection and loading (CSV, JSON, Parquet, NetCDF, HDF5, NPY/NPZ, SICKLE). Multi-file concatenation.
- **Preprocess** (`pipeline/preprocess.py`): Missing values, duplicates, outliers, PII anonymization, array normalization.
- **Transform** (`pipeline/transform.py`): Numeric normalization, categorical encoding, feature engineering/selection.
- **Structure** (`pipeline/structure.py`): Convert to NumPy arrays, PyTorch tensors, time-series sequences, or graph structures.
- **Output** (`pipeline/output.py`): Train/val/test split, sharding, format writers (NPZ, HDF5, Parquet, CSV, Zarr, TFRecord), metadata/Data Card export.

Each domain module (`redi/domains/<name>.py`) provides specialized step implementations via factory functions: `get_ingest_steps()`, `get_preprocess_steps()`, etc. Domain auto-detection uses file extensions first, then keyword scoring on column names (`domains/__init__.py`).

### Key Supporting Modules

- `datacard.py` — Data Card data model with 5-level maturity taxonomy (RAW → FULLY_READY) and 8-section metadata structure. Serializes to JSON/Markdown.
- `assess.py` — `DataAnalyzer` (dataset → Data Card) and `PipelineRecommender` (domain-aware pipeline suggestions).
- `config.py` — YAML config loading with deep merge against defaults and validation.
- `repro.py` — Logs every command to `repro.sh` with full environment metadata for reproducibility.
- `inspect/` — Snapshot capture, quality checks (NaN/outlier/distribution shift), transformation parameter detection, data leakage detection, Snakemake/Nextflow log parsing.
- `downloaders/` — ESGF (CMIP6) and CDS (ERA5) data acquisition with `BaseDownloader` ABC.
- `dataloaders/` — GESTS, SST-Binary, and Zarr loaders with `BaseDataLoader` ABC.
- `domains/hpc/` — HPC domain package (`job_mapper.py`, `power_qc.py`, `qc_validation.py`, `sacct_loader.py`, `swf_loader.py`). `job_mapper.py` does RAPS-compatible job extraction from HPC telemetry with 4-phase fuzzy column matching.

## Key Terminology

- **PipelineContext** (`ctx`): The mutable state object passed through all steps. Holds `ctx.data`, `ctx.metadata`, `ctx.errors`, `ctx.config`, paths, and stage tracking.
- **PipelineStep**: Base class for all pipeline steps. Subclasses implement `execute(ctx) → ctx`. Optional `validate(ctx)` and `verify(ctx)`.
- **Domain**: One of climate, materials, life_sciences, fusion, hpc, geospatial, general. Determines which specialized steps are loaded.
- **MaturityLevel**: 5-level readiness taxonomy: RAW (1), CLEANED (2), LABELED (3), FEATURE_ENGINEERED (4), FULLY_READY (5).
- **Data Card**: Structured metadata document with 8 sections (overview, provenance, characteristics, sensitive attributes, transformations, annotations, limitations, readiness).
- **Flowcept**: Optional provenance tracking. Steps are auto-instrumented via `PipelineStep.__init_subclass__`.
- **RAPS Job**: Standardized HPC job representation used by ExaDigiT. The `hpc/job_mapper.py` auto-maps arbitrary HPC telemetry to this format.
- **Array mode** (`--array`): Skips DataFrame-centric steps for high-dimensional data; uses direct NumPy normalization.

## Coding Conventions

**Step structure.** Every `PipelineStep` subclass must set class attributes `name`, `stage`, and `description`. Call `super().__init__()`. The `execute()` method is required; `validate()` and `verify()` are optional. Always return `ctx`.

**Error handling.** Steps must not raise exceptions that halt the pipeline. Catch errors and record them with `ctx.add_error(f"Error in step: {str(e)}")`. For missing optional dependencies, use `ctx.add_error(missing_dependency_error("package_name"))`.

**Imports.** Group as stdlib → third-party → local (relative). Optional dependencies (xarray, pymatgen, torch, etc.) are imported lazily inside `execute()` methods, wrapped in `try/except ImportError`. Domain factory functions import from `..pipeline` inside the function body to avoid circular imports.

**Metadata tracking.** Steps record what they did via `ctx.metadata["key"] = {...}` with snake_case keys. Include before/after values and parameters used.

**Naming.** Classes: PascalCase, verb-noun (`LoadHPCData`, `NormalizeNumeric`). Functions/methods: snake_case. Private: leading underscore. Factory functions: `get_<stage>_steps()` returning `[step_instances]`.

**Type hints.** Used on all function parameters and return types. `Optional[T]` for nullable, `Any` for flexible data containers.

**Docstrings.** Google-style. One-liner for simple functions; Args/Returns sections for complex ones.

**Logging.** `logger = logging.getLogger(__name__)` at module level. Use `logger.info()` for high-level progress, `logger.debug()` for detail.

**Tests.** pytest with `@pytest.fixture` for setup, class-based grouping (`class TestClimateSmoke`), `pytest.importorskip()`/`pytest.skip()` for optional dependencies. Fixtures use `tmp_path` for temp directories.

**Config access.** Steps read config via `ctx.config.get("section", {}).get("key", default)`.

## Current Development Status

The codebase is **production-ready**. All three CLI modes, all 5 pipeline stages, all 6 domain modules, downloaders, inspect subsystem, and the HPC job mapper are fully implemented — no stubs or placeholder code.

**Test coverage exists for:** pipeline mechanics, YAML config, data analysis, array loaders, Nextflow parser, HPC job mapper, and domain smoke tests for climate, materials, life sciences, and HPC. **Missing test coverage:** fusion domain, geospatial domain, downloaders, repro module, inspect subsystem.

**Planned but unstarted:** Intelligent sampling (`pipeline/sampling.py`) — a new stage between PREPROCESS and TRANSFORM with 5 sampling strategies for large datasets. Design is documented in `IMPLEMENTATION_PLAN_INTELLIGENT_SAMPLING.md`.

**Minor TODOs:** `pipeline/__init__.py` line 78 (`with_flowcept` default needs rework) and line 335 (stage completion logic question in `execute_stage()`). Neither blocks functionality.

When working on this codebase, follow existing patterns closely. New pipeline steps should match the template in the conventions section. New domain modules must export `get_ingest_steps()` through `get_output_steps()` and register keywords/extensions in `domains/__init__.py`.
