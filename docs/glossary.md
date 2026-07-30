# Glossary

**PipelineContext**
: The central mutable dataclass (`ctx`) that flows through every
pipeline step. It carries `ctx.data` (the actual data object),
`ctx.metadata` (accumulated facts), input/output paths, domain,
config, error list, and pipeline-state bookkeeping.
See [`redi.pipeline.PipelineContext`](../redi/pipeline/__init__.py).

**PipelineStep**
: Base class for every processing step. Subclasses set class
attributes `name`, `stage`, and `description`, and implement
`execute(ctx) → ctx`. Optional `validate(ctx)` and
`verify(ctx)` hooks provide pre- and post-checks.
See [`redi.pipeline.PipelineStep`](../redi/pipeline/__init__.py).

**PipelineStage**
: Enum of the five sequential stages: `INGEST`, `PREPROCESS`,
`TRANSFORM`, `STRUCTURE`, `OUTPUT`.
See [`redi.pipeline.PipelineStage`](../redi/pipeline/__init__.py).

**Domain**
: One of the eight supported scientific areas: `climate`,
`materials`, `life_sciences`, `fusion`, `hpc`,
`geospatial`, `nuclear_physics`, `general`. Determines
which specialised `PipelineStep` implementations are loaded.
See [`redi.datacard.ScientificDomain`](../redi/datacard.py).

**MaturityLevel**
: Five-level data readiness taxonomy: `RAW` (1), `CLEANED` (2),
`LABELED` (3), `FEATURE_ENGINEERED` (4), `FULLY_READY` (5).
See [`redi.datacard.MaturityLevel`](../redi/datacard.py) and
[Data Readiness Levels](concepts/readiness-levels.md).

**Data Card**
: Structured metadata document with eight sections (overview,
provenance, characteristics, sensitive attributes, transformations,
annotations, limitations, readiness) written to `data_card.json`
in the output directory after each pipeline run.
See [`redi.datacard.DataCard`](../redi/datacard.py).

**Array mode**
: Pipeline mode enabled with `--array` that skips DataFrame-centric
steps and applies direct NumPy normalisation. Intended for
high-dimensional data such as climate grids or simulation outputs.

**Flowcept**
: Optional provenance-tracking library. When enabled (`provenance:
true` in config), every `PipelineStep` execution is recorded as
a lineage task via automatic `__init_subclass__` instrumentation —
no changes to step implementations required.

**RAPS Job**
: Standardised HPC job representation used by the ExaDigiT framework.
The [`redi.domains.hpc.job_mapper`](../redi/domains/hpc/job_mapper.py) module auto-maps arbitrary
HPC telemetry columns to this format using a four-phase fuzzy column
matching strategy.

**HPC Job Mapper**
: Standalone subsystem in [`redi.domains.hpc.job_mapper`](../redi/domains/hpc/job_mapper.py) for
extracting structured RAPS Job objects from raw HPC telemetry.
Uses exact match → token overlap → fuzzy matching with confidence
scores, then derives missing fields and normalises timestamps.
Activated with the `--raps-jobs` CLI flag.

**ClimaX output**
: Output format compatible with the ClimaX climate foundation model.
Enabled via `climax_output.enabled: true` in the YAML config.
Produces yearly shards, coordinate arrays (`lat.npy`,
`lon.npy`), and normalisation statistics.

**ESGF**
: Earth System Grid Federation — federated data infrastructure for
CMIP6 climate model data. REDI can download from ESGF via
`--download --source esgf`. See [`redi.downloaders.esgf`](../redi/downloaders/esgf.py).

**CDS**
: Copernicus Climate Data Store — source of ERA5 reanalysis data.
REDI can download from CDS via `--download --source cds`.
Requires an API key in `~/.cdsapirc`.
See [`redi.downloaders.cds`](../redi/downloaders/cds.py).

**OpenFold**
: A specialisation of the `life_sciences` domain for protein structure
datasets. Activated automatically when the input directory contains
`fasta/`, `pdb/`, and `alignments/` subdirectories. Runs the
full IPTSO feature-engineering pipeline and writes either per-protein
NPZ files (`output_format: npz`) or a directory layout compatible
with `OpenFoldDataModule` (`output_format: openfold_native`).
See [`redi.domains.openfold`](../redi/domains/openfold.py).

**sPHENIX**
: The sPHENIX Time Projection Chamber (TPC) outer detector dataset,
processed under the `nuclear_physics` domain. Input is a directory
of `uint16` wedge `.npy` files (shape 192 × 249 × 16) with
`train.txt` / `test.txt` manifests. REDI applies zero-suppression
and `log2(ADC + 1)` normalization, writing float32 `.npy` output
compatible with NeuralCompression_v3's `DatasetTPC` loader.
See [`redi.domains.sphenix`](../redi/domains/sphenix.py).

**NeuralCompression_v3**
: An AI compression framework developed for high-energy physics detector
data. REDI's `nuclear_physics` (sPHENIX) pipeline produces output
that can be loaded directly via
`DatasetTPC(data_root=..., split='train', dimension=2)`. Set
`log: false` in the model config since REDI has already applied
`log2` normalization.

**repro.sh**
: Shell script written to every output directory by [`redi.repro`](../redi/repro.py).
Records the exact `redi` invocation with full environment metadata
(host, user, Python version, working directory) for reproducibility.
