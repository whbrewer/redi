# REDI Parallel Execution

This module enables parallel processing of multiple files across CPU cores and HPC cluster nodes using GNU parallel.

## Overview

REDI supports two types of parallelism:

| Type | Description | Status |
|------|-------------|--------|
| **Type 1** | Multiple files distributed across workers/nodes | ✅ Implemented |
| **Type 2** | Single large file chunked across workers | 🔮 Future |

## Requirements

### Local Execution

**GNU Parallel** must be installed:

```bash
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel

# RHEL/CentOS
sudo yum install parallel
```

Verify installation:
```bash
parallel --version
```

### HPC/SLURM Execution

- SLURM workload manager
- GNU parallel module (typically `module load parallel`)
- Conda environment with REDI installed

## Quick Start

### Local Parallel Processing

```bash
# Process all NetCDF files in parallel (auto-detect CPU cores)
redi data/*.nc --parallel

# Specify number of workers
redi data/*.nc --parallel --workers 8

# Use balanced distribution (by file size)
redi data/*.nc --parallel --distribution balanced
```

### SLURM Cluster Processing

```bash
# Generate SLURM scripts without submitting
redi data/*.nc --parallel --backend slurm --slurm-account myacct --generate-only

# Generate and submit job
redi data/*.nc --parallel --backend slurm --slurm-account myacct

# Full SLURM configuration
redi data/*.nc --parallel --backend slurm \
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
| `--backend` | local | Backend: `local` or `slurm` |
| `--batch-size` | 4 | Files per worker batch |
| `--distribution` | round_robin | Distribution: `round_robin` or `balanced` |
| `--generate-only` | - | Generate scripts without executing |

### SLURM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--slurm-nodes` | 4 | Number of compute nodes |
| `--slurm-cores` | 56 | CPU cores per node |
| `--slurm-time` | 0:30:00 | Time limit (HH:MM:SS) |
| `--slurm-account` | - | SLURM account/allocation |
| `--slurm-partition` | batch | SLURM partition |
| `--slurm-output` | - | Final output directory |

## YAML Configuration

```yaml
# pipeline.yaml
input_path: "/data/*.nc"
output_dir: ./output
domain: climate

parallel:
  enabled: true
  backend: local           # local | slurm
  workers: auto            # auto | integer
  batch_size: 4
  distribution: round_robin  # round_robin | balanced

  slurm:
    nodes: 4
    cores_per_node: 56
    time_limit: "0:30:00"
    account: "myacct"
    partition: batch
    burst_buffer: "/mnt/bb/$USER"
    final_output: "/lustre/project/results"
    constraint: nvme
    modules:
      - miniforge3/23.11.0-0
      - parallel
    conda_env: conda-redi
```

Run with:
```bash
redi pipeline.yaml
```

## Distribution Strategies

### Round Robin (default)

Files are distributed cyclically across workers, matching the AWK modulo approach:

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

## Generated SLURM Scripts

When using `--backend slurm --generate-only`, four files are created:

```
parallel_scripts/
├── multinodejob.sh     # SLURM batch submission script
├── driver.sh           # Per-node work distributor
├── payload.sh          # Single job executor
└── inputdatalist.txt   # List of input files
```

### Submit manually:

```bash
cd output/parallel_scripts
sbatch multinodejob.sh
```

### Monitor job:

```bash
squeue -u $USER
tail -f redi-parallel-*.out
```

## Architecture

```
redi/parallel/
├── __init__.py           # Exports: ParallelExecutor, FileDistributor, etc.
├── executor.py           # Main ParallelExecutor class
├── distribution.py       # File distribution strategies
├── chunking.py           # Type 2 chunking interface (future)
├── backends/
│   ├── __init__.py       # Backend registry
│   ├── base.py           # Abstract ParallelBackend
│   ├── local.py          # LocalBackend (GNU parallel)
│   └── slurm.py          # SlurmBackend (script generation)
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

# Round-robin across 4 workers
batches = FileDistributor.round_robin(files, num_workers=4)

# Balance by file size
batches = FileDistributor.balanced(files, num_workers=4)

# Fixed-size batches
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

## Multi-Node Execution Flow

```
SLURM Job (multinodejob.sh)
    │
    ▼
srun launches 1 task per node
    │
    ├── Node 0: driver.sh
    │       │
    │       ▼
    │   AWK: NR % 4 == 0 (selects files 0, 4, 8, ...)
    │       │
    │       ▼
    │   parallel -j56 payload.sh (56 concurrent jobs)
    │       │
    │       ▼
    │   redi --style=simple file1.nc file2.nc ... -o /mnt/bb/
    │
    ├── Node 1: driver.sh
    │       │
    │       ▼
    │   AWK: NR % 4 == 1 (selects files 1, 5, 9, ...)
    │       ...
    │
    └── (Nodes 2, 3 similar)

    ▼
Results copied to final output directory
```

## Type 2 Parallelism (Future)

The `chunking.py` module provides the interface for single-file parallelism:

```python
from redi.parallel.chunking import ChunkStrategy, ChunkSpec

class TimeChunkStrategy(ChunkStrategy):
    """Split NetCDF by time dimension."""

    def chunk(self, file_path: Path, num_chunks: int) -> List[ChunkSpec]:
        # Future: Use xarray to split by time
        pass

    def merge(self, chunk_results: List[Path], output_path: Path) -> Path:
        # Future: Concatenate results along time dimension
        pass
```

Planned chunking strategies:
- **TimeChunkStrategy**: Split NetCDF/Zarr by time dimension
- **RowChunkStrategy**: Split tabular data (CSV, Parquet) by row ranges
- **SpatialChunkStrategy**: Split gridded data by lat/lon regions

## Performance Tips

1. **Batch size**: Set `--batch-size` to balance overhead vs. parallelism
   - Small files: Use larger batch size (8-16)
   - Large files: Use smaller batch size (1-4)

2. **Distribution**: Use `--distribution balanced` when file sizes vary significantly

3. **SLURM burst buffer**: Use node-local NVMe storage (`/mnt/bb/$USER`) for fast I/O, then copy results to shared storage

4. **Workers vs. cores**: For I/O-bound tasks, workers can exceed physical cores. For CPU-bound tasks, match workers to cores.

## Troubleshooting

### "GNU parallel is not installed"

Install GNU parallel (see Requirements section above).

### "sbatch is not available"

You're trying to use `--backend slurm` on a non-SLURM system. Use `--generate-only` to create scripts for transfer to an HPC cluster.

### Jobs fail with "conda: command not found"

Ensure the SLURM modules list includes your conda module:
```yaml
slurm:
  modules:
    - miniforge3/23.11.0-0  # Adjust for your system
    - parallel
```

### Output files missing

Check the burst buffer path and ensure the final copy step completed:
```bash
ls /mnt/bb/$USER/
```

## Testing

Run the parallel module tests:

```bash
pytest tests/test_parallel.py -v
```

## Files Reference

| File | Purpose |
|------|---------|
| `executor.py` | ParallelExecutor, ParallelConfig, ParallelResult |
| `distribution.py` | FileDistributor (round_robin, balanced, batched) |
| `chunking.py` | ChunkStrategy, ChunkSpec (Type 2 stubs) |
| `backends/base.py` | Abstract ParallelBackend |
| `backends/local.py` | LocalBackend (GNU parallel subprocess) |
| `backends/slurm.py` | SlurmBackend (script generation) |
