# REDI Pipeline Stage Timing Benchmarks — SC26 Paper Plan

## Objective

Run the REDI pipeline on four scientific AI-readiness cases, collect per-stage timings
(ingest / preprocess / transform / structure / output), and produce a horizontal stacked
bar chart for the paper.

---

## Cases

| # | Case | Domain | Input Format | Sample Size | Data Path |
|---|------|--------|-------------|-------------|-----------|
| 1 | **OpenFold** | life_sciences | mmCIF protein structures | 5 000 files | `/lustre/orion/gen053/world-shared/openfold/data/pdb_mmcif/mmcif_files/` |
| 2 | **HydraGNN / OC22** | materials | LMDB (structure→energy) | full dataset (89 G) | `/lustre/orion/gen053/world-shared/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/` |
| 3 | **XGC1** | fusion | ADIOS2 `.bp` checkpoints | 5 files | `/lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data/n560fr_ITER_PFPO_W_Ne/xgc.3d.000{02,04,06,08,10}.bp` |
| 4 | **ClimaX / TaiESM1** | climate | NetCDF 10-yr chunks | 5 files | `/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature/` |

---

## Provenance / Timing Strategy

Provenance and Flowcept tracking is enabled by setting `provenance: true` in the
per-run YAML config.  Each pipeline step prints a line like:

```
✅ step_name (1234ms)
```

The benchmark runner scripts parse these with the regex `(\w+)\s+.*?\((\d+)ms\)`,
aggregate into the five canonical REDI stages, and write a JSON result file.

---

## Deliverables

```
benchmarks/
├── configs/
│   ├── bench_openfold.yaml
│   ├── bench_oc22.yaml
│   ├── bench_xgc1.yaml
│   └── bench_climax.yaml
├── runners/
│   ├── run_openfold_bench.py
│   ├── run_oc22_bench.py
│   ├── run_xgc1_bench.py
│   └── run_climax_bench.py
├── jobs/
│   ├── job_openfold.sh
│   ├── job_oc22.sh
│   ├── job_xgc1.sh
│   └── job_climax.sh
├── results/
│   ├── bench_openfold.json     (generated)
│   ├── bench_oc22.json         (generated)
│   ├── bench_xgc1.json         (generated)
│   └── bench_climax.json       (generated)
└── plot_stage_timings.py
```

---

## Phase 1 — Investigation & Domain Support (current blocker)

Before writing runners, verify:

### 1a. OC22 / LMDB support

REDI's materials domain currently handles CIF/POSCAR crystal files.  OC22 data is
stored as LMDB key-value stores (atom positions + energies).  **Action needed:**

- Open one LMDB entry to see the schema (atom types, positions, energy, forces).
- Determine if the existing `load_crystal_structure` step can be pointed at LMDB or
  if a new `LoadLMDB` pipeline step is required.
- Likely need a thin LMDB loader in `redi/pipeline/ingest.py` that yields dicts
  compatible with the materials transform steps.

### 1b. XGC1 / ADIOS2 support

The fusion domain declares `LoadADIOS` in `redi/domains/fusion.py`.  **Action needed:**

- Confirm `adios2` Python bindings are installed in the drai venv.
- Run a quick test read of `xgc.3d.00002.bp` to see what variables are present and
  that the loader returns the expected context metadata.

### 1c. OpenFold mmCIF at scale

The existing timing benchmark processes a single FASTA (1 protein).  For 5 000 mmCIF
files we need batch processing.  **Action needed:**

- Check if passing a directory (glob) to `redi run` with `--domain life_sciences`
  already batches proteins, or if we need `max_proteins` config key.
- Confirm output shapes and throughput metric (proteins/s or amino-acids/s).

---

## Phase 2 — YAML Configs

Each config sets:
- `input_path` / `input_dir`
- `output_dir`
- `domain`
- `provenance: true`
- `output_format: npz`
- Domain-specific options (regrid target, normalization, etc.)

### bench_openfold.yaml (sketch)
```yaml
domain: life_sciences
provenance: true
output_format: hdf5
max_proteins: 5000
num_workers: 8
```

### bench_oc22.yaml (sketch)
```yaml
domain: materials
provenance: true
output_format: npz
# lmdb_path: set at runtime
```

### bench_xgc1.yaml (sketch)
```yaml
domain: fusion
provenance: true
output_format: npz
```

### bench_climax.yaml (sketch)
```yaml
domain: climate
provenance: true
output_format: npz
climax_output: true
regrid_target: 128x256
```

---

## Phase 3 — Benchmark Runner Scripts

Each runner follows the pattern established in
`scripts/benchmarks/bench_climate_timing.py`:

1. Build a temporary YAML config pointing at the sample data.
2. Invoke `redi run -c <config.yaml>` as a subprocess.
3. Parse per-step timings from stdout with `STEP_RE`.
4. Map steps → 5 stages via a `STAGE_MAP`.
5. Write JSON result to `benchmarks/results/bench_<case>.json`.

**Stage maps** (to be filled in during phase-1 investigation):

| Step | Ingest | Preprocess | Transform | Structure | Output |
|------|--------|------------|-----------|-----------|--------|
| OpenFold | detect_format, load_mmcif | filter_sequences, anonymize_pii | encode_sequences, sequence_features | prepare_transformer, train_val_test_split | save_hdf5, write_checksums, save_metadata |
| OC22 | detect_format, load_lmdb | normalize_structures | featurize_atoms | structure_to_graph, train_val_test_split | save_npz, save_metadata |
| XGC1 | detect_format, load_adios | filter_plasma | normalize_signals | fusion_to_tensor, train_val_test_split | save_npz, save_metadata |
| ClimaX | load_climate | regrid_climate | normalize_climate, select_variables, compute_anomalies | climate_to_numpy, train_val_test_split | save_climate, save_metadata |

---

## Phase 4 — SLURM Job Scripts

Each job script:
- Runs on 1 node (no GPU needed for preprocessing)
- Sets `#SBATCH --time=02:00:00` (adjust after first test run)
- Activates `~/venv/redi/bin/activate`
- Calls the runner script
- Writes stdout/stderr to `benchmarks/jobs/logs/`

Template:
```bash
#!/bin/bash
#SBATCH -A stf053
#SBATCH -J redi_bench_<case>
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o benchmarks/jobs/logs/bench_<case>_%j.out

source ~/venv/redi/bin/activate
python benchmarks/runners/run_<case>_bench.py
```

---

## Phase 5 — Aggregation & Plotting

`benchmarks/plot_stage_timings.py`:

1. Load the four JSON result files.
2. Extract `stages_ms` for each case.
3. Build a horizontal stacked bar chart:
   - X-axis: time (seconds)
   - Y-axis: the 4 cases (OpenFold, OC22, XGC1, ClimaX)
   - Segments: ingest (blue), preprocess (orange), transform (green), structure (red), output (purple)
   - Legend showing stage colors
4. Save as `benchmarks/results/stage_timings.pdf` and `.png`.

**Normalization options** (TBD):
- Raw wall-clock time (seconds)
- Time per input GB (s/GiB)
- Time per output sample (ms/sample)
- Fraction of total (percentage stacked bar)

All four modes can be implemented with a `--normalize` flag.

---

## Known Risks / Open Questions

| Risk | Mitigation |
|------|-----------|
| LMDB not natively supported by materials domain | Implement `LoadLMDB` step or pre-extract subset to CIF |
| ADIOS2 bindings not installed | Test with `python -c "import adios2"` first |
| OpenFold batch mode may not exist | Use `--stages ingest,preprocess,transform,structure,output` with glob input |
| ClimaX on Lustre vs local paths | Check if regridding ESMF works on Frontier without env setup |
| Long wall-clock for OC22 (89G) | Run in two phases: estimate from 1G sample, then full |
| Provenance adds overhead | Report both with/without flowcept if overhead is significant |

---

## Order of Execution

1. [ ] Phase 1 — Investigation (LMDB schema, ADIOS2 test, OpenFold batch mode)
2. [ ] Phase 2 — Write 4 YAML configs
3. [ ] Phase 3 — Write 4 runner scripts
4. [ ] Phase 4 — Write 4 SLURM job scripts
5. [ ] Phase 5 — Write aggregation + plot script
6. [ ] Submit jobs, collect results
7. [ ] Run `plot_stage_timings.py`, iterate on normalization

---

## File Counts & Data Sizes

| Case | Files used | Approx input size |
|------|-----------|------------------|
| OpenFold | 5 000 mmCIF | ~5–20 GB |
| OC22 | full LMDB | 89 GB |
| XGC1 | 5 .bp dirs | ~varies, each bp dir is parallel I/O |
| ClimaX | 5 .nc files | ~5×30 GB = ~150 GB |
