# REDI Parallel Execution

This module enables parallel processing of multiple files across CPU cores and HPC cluster nodes.

## Overview

REDI supports two types of parallelism:

| Type | Description | Status |
|------|-------------|--------|
| **Type 1** | Multiple files distributed across workers/nodes | Implemented |
| **Type 2** | Single large file chunked across workers | Future |

Three backends are available:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `local` | GNU parallel on a single machine | Workstation / login node |
| `gnu` | SLURM + GNU parallel across nodes | HPC, no inter-rank communication needed |
| `mpi` | SLURM + mpi4py across nodes | HPC, unified output with allreduce/gather |

> **Note:** `--backend slurm` is a deprecated alias for `--backend gnu`.

## Requirements

### Local Backend

**GNU Parallel** must be installed:

```bash
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel

# RHEL/CentOS
sudo yum install parallel
```

### GNU / MPI Backends (HPC)

- SLURM workload manager
- GNU parallel module for the `gnu` backend (`module load parallel`)
- `mpi4py` for the `mpi` backend: `pip install mpi4py`
- Python environment with REDI installed on all nodes

## Quick Start

### Local Parallel Processing

```bash
# Process all NetCDF files in parallel (auto-detect CPU cores)
redi run data/*.nc --parallel

# Specify number of workers
redi run data/*.nc --parallel --workers 4

# Use balanced distribution (by file size)
redi run data/*.nc --parallel --distribution balanced
```

### MPI Runner (Recommended for HPC)

The MPI runner is the primary interface for multi-node parallel execution. Launch it directly with `srun`:

```bash
# Point at a directory — rank 0 discovers files, all ranks process in parallel
srun python -m redi.parallel.mpi_runner \
    --input /data/nc/ --output /path/to/output --domain climate --merge

# Limit to first N files
srun python -m redi.parallel.mpi_runner \
    --input /data/nc/ --output /path/out --n-files 16 --shards 4 --merge

# Use a pre-built file list
srun python -m redi.parallel.mpi_runner \
    --filelist files.txt --output /path/out --merge
```

After processing, rank 0:
- Aggregates global statistics via `allreduce`
- Gathers per-file results via `gather`
- Merges per-file NPZ outputs into N shard files (if `--merge`)
- Writes `metadata.json` and `datacard.json`

### MPI Backend via CLI (generates SLURM scripts)

```bash
# Generate SLURM scripts and submit
redi run data/*.nc --parallel --backend mpi --slurm-account myacct

# Generate scripts only (review before submitting)
redi run data/*.nc --parallel --backend mpi --slurm-account myacct --generate-only
```

### GNU Backend via CLI

```bash
# Generate SLURM scripts and submit (GNU parallel across nodes)
redi run data/*.nc --parallel --backend gnu --slurm-account myacct

# Full configuration
redi run data/*.nc --parallel --backend gnu \
    --slurm-account myacct \
    --slurm-nodes 4 \
    --slurm-cores 56 \
    --slurm-time 1:00:00 \
    --slurm-partition batch \
    --generate-only
```

## CLI Options

### Parallel Execution Group

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --parallel` | - | Enable parallel execution |
| `-w, --workers` | auto | Number of parallel workers |
| `--backend` | local | Backend: `local`, `gnu`, or `mpi` |
| `--batch-size` | 4 | Files per worker batch |
| `--distribution` | round_robin | Distribution: `round_robin` or `balanced` |
| `--generate-only` | - | Generate scripts without executing |

### SLURM Options (gnu and mpi backends)

| Option | Default | Description |
|--------|---------|-------------|
| `--slurm-nodes` | 4 | Number of compute nodes |
| `--slurm-cores` | 56 | CPU cores per node (gnu backend) |
| `--slurm-time` | 0:30:00 | Time limit (HH:MM:SS) |
| `--slurm-account` | - | SLURM account/allocation |
| `--slurm-partition` | batch | SLURM partition |
| `--slurm-output` | - | Final output directory |

### MPI Runner Options (`python -m redi.parallel.mpi_runner`)

| Option | Description |
|--------|-------------|
| `--input DIR` | Input directory or glob pattern (mutually exclusive with `--filelist`) |
| `--filelist FILE` | Text file with one input path per line |
| `--output DIR` | Base output directory (required) |
| `--domain DOMAIN` | Domain override passed to `redi run` |
| `--format FORMAT` | Output format: `npz`, `hdf5`, `zarr` (default: `npz`) |
| `--stages STAGES` | Comma-separated pipeline stages |
| `--n-files N` | Limit to first N files when using `--input` |
| `--glob PATTERN` | Glob pattern when `--input` is a directory |
| `--merge` | Gather per-rank results on rank 0 and write unified outputs |
| `--shards N` | Number of output shard files to write (default: 1) |
| `--style STYLE` | redi output style: `simple`, `rich`, `json` (default: `simple`) |

## YAML Configuration

```yaml
# pipeline.yaml
input_path: "/data/*.nc"
output_dir: ./output
domain: climate

parallel:
  enabled: true
  backend: mpi             # local | gnu | mpi
  workers: auto            # auto | integer
  batch_size: 4
  distribution: round_robin  # round_robin | balanced

  slurm:
    nodes: 4
    cores_per_node: 56
    time_limit: "0:30:00"
    account: "myacct"
    partition: batch
    final_output: "/lustre/project/results"
    constraint: nvme
    modules:
      - miniforge3/23.11.0-0
      - parallel
    conda_env: conda-redi
```

## Distribution Strategies

### Round Robin (default)

Files are distributed cyclically across workers:

```
Worker 0: files 0, 3, 6, 9, ...
Worker 1: files 1, 4, 7, 10, ...
Worker 2: files 2, 5, 8, 11, ...
```

Best for: Files of similar sizes.

### Balanced

Files are assigned to minimize total size disparity across workers:

```
Worker 0: large_file.nc, tiny_file.nc     (total: ~500MB)
Worker 1: medium1.nc, medium2.nc          (total: ~500MB)
```

Best for: Files with varying sizes.

## Generated Scripts

### MPI Backend

When using `--backend mpi`, two files are created in `<output>/mpi_scripts/`:

```
mpi_scripts/
├── multinodejob.sh     # SLURM batch script (uses srun to launch mpi_runner)
└── inputdatalist.txt   # Flat file list consumed by mpi_runner rank 0
```

Submit manually:
```bash
cd output/mpi_scripts
sbatch multinodejob.sh
```

### GNU Backend

When using `--backend gnu`, four files are created in `<output>/parallel_scripts/`:

```
parallel_scripts/
├── multinodejob.sh     # SLURM batch submission script
├── driver.sh           # Per-node work distributor (AWK + GNU parallel)
├── payload.sh          # Single job executor
└── inputdatalist.txt   # List of input files
```

Submit manually:
```bash
cd output/parallel_scripts
sbatch multinodejob.sh
```

Monitor:
```bash
squeue -u $USER
tail -f redi-parallel-*.out
```

## Architecture

```
redi/parallel/
├── __init__.py           # Exports: ParallelExecutor, FileDistributor, etc.
├── executor.py           # ParallelExecutor, ParallelConfig, ParallelResult
├── distribution.py       # File distribution strategies
├── mpi_runner.py         # MPI entry point (python -m redi.parallel.mpi_runner)
├── chunking.py           # Type 2 chunking interface (future)
├── backends/
│   ├── __init__.py       # Backend registry (local, gnu, mpi)
│   ├── base.py           # Abstract ParallelBackend
│   ├── local.py          # LocalBackend (GNU parallel, single machine)
│   ├── gnu.py            # GnuBackend (SLURM + GNU parallel across nodes)
│   └── mpi.py            # MpiBackend (SLURM + mpi4py)
└── README.md             # This file
```

### Key Classes

**ParallelExecutor**: Main orchestrator
```python
from redi.parallel import ParallelExecutor

executor = ParallelExecutor(
    backend="local",
    workers=8,
    batch_size=4,
)
result = executor.execute([Path("file1.nc"), Path("file2.nc")])
```

**FileDistributor**: Distribution strategies
```python
from redi.parallel import FileDistributor

batches = FileDistributor.round_robin(files, num_workers=4)
batches = FileDistributor.balanced(files, num_workers=4)
batches = FileDistributor.batched(files, batch_size=4)
```

**ParallelResult**: Execution result
```python
result = executor.execute(files)
print(f"Success: {result.success}")
print(f"Completed: {result.completed}/{result.total_files}")
print(f"Failed: {result.failed}")
print(f"Errors: {result.errors}")
```

## Execution Flow

### MPI Backend

```
SLURM Job (multinodejob.sh)
    │
    ▼
srun python -m redi.parallel.mpi_runner --filelist ... --merge
    │
    ├── Rank 0: discovers/broadcasts file list
    │
    ├── All ranks: round-robin file assignment (rank i processes files i, i+size, ...)
    │       │
    │       ▼
    │   redi run <file> -o per_rank/rankNNNN/<stem>/  (subprocess, no MPI init)
    │
    ├── All ranks: allreduce → global sample counts, success/failure totals
    │
    └── Rank 0 (--merge):
            gather() ← per-rank results
            merge per-file NPZ → shard_NNNN.npz
            write metadata.json + datacard.json
```

### GNU Backend

```
SLURM Job (multinodejob.sh)
    │
    ▼
srun launches 1 task per node
    │
    ├── Node 0: driver.sh
    │       AWK: NR % 4 == 0  → files 0, 4, 8, ...
    │       parallel -j56 payload.sh
    │       redi run <files> -o /mnt/bb/
    │
    ├── Node 1: driver.sh
    │       AWK: NR % 4 == 1  → files 1, 5, 9, ...
    │       ...
    │
    └── Results copied to final output directory
```

## Type 2 Parallelism (Future)

The `chunking.py` module provides the interface for single-file parallelism:

```python
from redi.parallel.chunking import ChunkStrategy, ChunkSpec

class TimeChunkStrategy(ChunkStrategy):
    """Split NetCDF by time dimension."""

    def chunk(self, file_path: Path, num_chunks: int) -> List[ChunkSpec]:
        pass  # Future: split by time via xarray

    def merge(self, chunk_results: List[Path], output_path: Path) -> Path:
        pass  # Future: concatenate along time dimension
```

Planned strategies: `TimeChunkStrategy`, `RowChunkStrategy`, `SpatialChunkStrategy`.

## Performance Tips

1. **MPI vs GNU**: Use `mpi` when you need a unified merged output (single `datacard.json`, merged NPZ shards). Use `gnu` when outputs can remain per-file and you don't need inter-rank communication.

2. **Batch size**: Set `--batch-size` to balance overhead vs. parallelism.
   - Small files: Use larger batch size (8–16)
   - Large files: Use smaller batch size (1–4)

3. **Distribution**: Use `--distribution balanced` when file sizes vary significantly.

4. **Burst buffer**: Use node-local NVMe (`/mnt/bb/$USER`) for fast I/O, then copy results to shared storage.

## Troubleshooting

### "mpi4py not installed"

```bash
MPICC=cc pip install --no-binary mpi4py mpi4py
```

### "GNU parallel is not installed"

Install GNU parallel (see Requirements above).

### "sbatch is not available"

Use `--generate-only` to create scripts for transfer to an HPC cluster.

### Jobs fail with "conda: command not found"

Ensure the SLURM modules list includes your conda module:
```yaml
slurm:
  modules:
    - miniforge3/23.11.0-0
    - parallel
```

## Testing

```bash
pytest tests/test_parallel.py -v
```

## Files Reference

| File | Purpose |
|------|---------|
| `executor.py` | `ParallelExecutor`, `ParallelConfig`, `ParallelResult` |
| `distribution.py` | `FileDistributor` (round_robin, balanced, batched) |
| `mpi_runner.py` | MPI entry point (`python -m redi.parallel.mpi_runner`) |
| `chunking.py` | `ChunkStrategy`, `ChunkSpec` (Type 2 stubs) |
| `backends/base.py` | Abstract `ParallelBackend` |
| `backends/local.py` | `LocalBackend` (GNU parallel, single machine) |
| `backends/gnu.py` | `GnuBackend` (SLURM + GNU parallel across nodes) |
| `backends/mpi.py` | `MpiBackend` (SLURM + mpi4py) |
