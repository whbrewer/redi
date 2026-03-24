# REDI I/O Format Benchmark Results

Benchmarks read throughput (samples/sec) across file formats, simulating
a multi-worker DataLoader at training time. Run on an exclusive Andes node
(256 GB RAM) on Lustre (Orion).

Script: `benchmarks/redi_io_benchmark.py`

---

## Life Sciences (OpenFold)

**Setup:** 5,000 proteins packed from per-protein NPZ files
(`/lustre/orion/gen053/world-shared/openfold/redi_output/train/`).
Access pattern: random protein reads. Metric: proteins/sec.

| Format | w=1 | w=2 | w=4 | w=8 |
|--------|----:|----:|----:|----:|
| NPZ (per-protein files) | 334 | 649 | 1,124 | 1,546 |
| LMDB (compressed) | 737 | 961 | 1,025 | 883 |
| HDF5 (all proteins, one file) | **2,933** | **3,993** | **4,100** | **3,001** |

**Job:** 3245764 / 3245871 — 2026-03-24

### Analysis

- **HDF5 wins** because all 5,000 proteins are in one file — a single `open()`
  amortized across all reads. No per-sample metadata ops on Lustre MDS.

- **NPZ scales near-linearly** with workers because each worker opens independent
  files. But each open/close is a metadata op, so throughput begins to saturate
  the MDS around w=8.

- **LMDB peaks at w=4 then drops** — LMDB uses a single reader/writer lock; at
  w=8 readers start contending on the lock even in read-only mode.

- **Key insight:** at w=1, HDF5 is 9x faster than NPZ. This is the real-world
  gap between "many small files" and "one consolidated file" on Lustre. At scale
  (100 GPU training job), each GPU has its own DataLoader workers hitting Lustre
  simultaneously — the MDS contention from 500K individual NPZ files would be
  severe.

### Recommendation

For life sciences (OpenFold-style) workloads: **HDF5** with all proteins in a
single file per split (train.h5, val.h5, test.h5). LMDB is a viable alternative
(used by Open Catalyst / fairchem) but shows reader lock contention at high
worker counts. Per-protein NPZ is the worst option for Lustre at scale despite
being the simplest to produce.

---

## Climate (ClimaX) — PENDING

**Setup:** 59 GB single NPZ (`/lustre/orion/scratch/whbrewer/stf053/ClimaX/data.npz`),
shape (240898, 181, 360) float32. Access pattern: random timestep window reads
(window=10). Formats: NPZ vs HDF5 (chunked, gzip) vs Zarr (chunked, lz4).

*Results pending job 3245871.*

---

## Fusion (XGC1) — PENDING

**Setup:** 5 GB HDF5
(`/lustre/orion/stf053/proj-shared/ketan/redi-all-data-processed/xgc.3d.00002.bp.outdir/data.h5`),
28 variables, shape (32, 1,277,797) float64. Access pattern: random plane reads
(window=1). Formats: HDF5 vs Zarr (lz4).

*Results pending job 3245871.*
