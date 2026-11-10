# Tutorial: making atomistic data AI-ready with REDI (BeF2 example)

This walks through turning a raw table of DFT/AIMD frames into training-ready arrays
with REDI, using the BeF2 PBE-D3 dataset as the example. By the end you will have
train/val/test files that are periodic-aware, split by trajectory, and described by
`index.json` and `metadata.json`. The pipeline takes 10–40 seconds, depending mostly on
how quickly the Parquet file is read from Lustre.

## The dataset

BeF2 is 1,998 frames of Be₆₇F₁₃₄ (201 atoms each) with positions, forces, energy and
cell computed at the PBE-D3 level in CP2K. It is stored as one Parquet file with **one
row per atom** (401,598 rows, 23 columns):

```
/lustre/orion/stf249/proj-shared/hackathon/BeF2/bef2_pbe-d3.parquet
```

| Column | Meaning |
|--------|---------|
| `source_file`, `group` | where the frame came from in the original HDF5 store |
| `frame_id`, `frame_idx`, `atom_idx` | structure id, index within its source file, atom order |
| `frame_label` | e.g. `mol-0013-0000000155` — trajectory `mol-0013`, step 155 |
| `species` | `Be` or `F` |
| `x`, `y`, `z` | Cartesian positions (Å) |
| `fx`, `fy`, `fz` | forces (eV/Å) |
| `energy` | total energy of the frame (eV), repeated on every atom of the frame |
| `cell_00` … `cell_22` | the frame's 3×3 cell |

The file is a valid Parquet file with flat primitive columns, so it also loads directly
with `pandas`, `pyarrow` or Hugging Face `datasets`. It is **not** training-ready:
training code wants one record per structure, atomic numbers, a periodic neighbour
graph, and a split that does not leak.

## What "AI-ready" means for this kind of data

| Problem in the raw table | What REDI does |
|--------------------------|----------------|
| One row per atom, structure info repeated | Groups rows into 1,998 structures; checks energy and cell are constant within each |
| Element symbols | Maps species to atomic numbers (Be=4, F=9) |
| Positions unwrapped (289k atoms outside the cell) | Wraps positions into the cell |
| No graph | Builds a 6 Å radius graph over periodic images (≤100 neighbours per atom) |
| Frames in a trajectory are nearly identical | Splits by trajectory, not at random, so train and test do not share near-duplicates |
| Large force values | Reports the max \|F\| per structure; optionally drops structures above a threshold |
| Total energies of about −115,000 eV | Fits per-element reference energies and stores `energy_referenced` |
| No record of what was done | Writes `index.json`, `metadata.json` (with a file/record manifest), `redi_report.md` and `repro.sh` |

## 1. Set up

You need Python ≥ 3.9 and a virtual environment containing REDI and the atomistics plugin:

```bash
# from the directory that holds the redi and redi-atomistics checkouts
pip install -e ./redi
git clone git@code.ornl.gov:drai/redi-atomistics.git   # the plugin
pip install -e "./redi-atomistics[table]"     # adds pandas + pyarrow for Parquet/CSV input
pip install torch torch_geometric             # only needed for -f pt output
```

On OLCF systems the interpreter for a project may be registered with `afx env which`.

Check the plugin is visible:

```bash
redi plugins
```

You should see `atomistic` (and `hydragnn`, an alias of the same pipeline, kept from the
plugin's earlier name `redi-hydragnn`).

## 2. Run the pipeline

```bash
cd /lustre/orion/stf249/proj-shared/hackathon/BeF2
redi run bef2_pbe-d3.parquet -d atomistic --style simple -o ./atomistic_out
```

Always pass `-o`. Without it REDI chooses a hashed output directory name.

Expected output (the time at the end is 10–40 s; reading the input is the slow part):

```
Domain: atomistic
Pipeline: 9 steps
  [OK] load_atom_table
  [OK] filter_records
  [OK] flag_force_outliers
  [OK] reference_energies
  [OK] build_graphs
  [OK] assemble_pyg_data
  [OK] save_output
Done: 7 ok, 0 warnings, 0 errors
```

Nine steps are loaded and seven run. `load_ocp_lmdb` and `load_atom_h5` skip themselves
because the input is a table, not an LMDB shard or an HDF5 store. Each step decides whether it applies from its input, so the
same `-d atomistic` command also handles OC22/OC20 LMDB data.

What the steps do:

| Step | Stage | What it does here |
|------|-------|-------------------|
| `load_atom_table` | ingest | Reads the Parquet, groups rows by `frame_id`, maps species to Z, drops structures whose energy or cell varies across atoms (none here) |
| `filter_records` | preprocess | Drops structures with NaN/Inf energy or forces, or zero atoms (none here) |
| `flag_force_outliers` | preprocess | Records max \|F\| per structure and summarizes the distribution |
| `reference_energies` | transform | Least-squares per-element reference energies |
| `build_graphs` | transform | Wraps positions into the cell, builds the periodic radius graph |
| `assemble_pyg_data` | structure | Builds PyG `Data` objects (only used for `-f pt`) |
| `save_output` | output | Splits by trajectory group and writes the files below |

## 3. What you get

```
atomistic_out/
├── train.npz   1,636 structures
├── val.npz       198 structures
├── test.npz      164 structures
├── index.json              splits, which trajectories went where, data checks
├── metadata.json           run record + manifest of every output file
├── redi_report.md     data card
└── repro.sh                the command that produced this directory
```

Each NPZ holds flat per-atom arrays and per-structure arrays:

| Array | Shape | Notes |
|-------|-------|-------|
| `atomic_numbers`, `tags` | (total_atoms,) | |
| `positions`, `forces` | (total_atoms, 3) | positions wrapped into the cell; forces in the original eV/Å |
| `natoms` | (n_structures,) | 201 for every BeF2 structure |
| `split_index` | (n_structures,) | cumulative atom offsets |
| `cell` | (n_structures, 3, 3) | rows are lattice vectors |
| `energy`, `energy_referenced`, `energy_per_atom` | (n_structures,) | total, reference-subtracted, per atom |
| `max_force`, `frame_id`, `group`, `sid` | (n_structures,) | for inspection and traceability |

Read one structure back:

```python
import numpy as np

train = np.load("atomistic_out/train.npz")
offsets = np.concatenate([[0], np.cumsum(train["natoms"])])

i = 0
atoms = slice(offsets[i], offsets[i + 1])
pos    = train["positions"][atoms]        # (201, 3)
forces = train["forces"][atoms]           # (201, 3)
z      = train["atomic_numbers"][atoms]   # 4 = Be, 9 = F
cell   = train["cell"][i]                 # (3, 3)
energy = train["energy"][i]
```

## 4. Check the result before training

Do not just trust the exit code. Look at what REDI recorded.

**`index.json`** — the split and the data checks:

```python
import json
idx = json.load(open("atomistic_out/index.json"))
idx["splits"]                  # {'train': 1636, 'val': 198, 'test': 164}
idx["split_groups"]            # which trajectories are in each split
idx["atomistic"]["n_atoms_outside_cell"]    # raw atoms outside the cell, before wrapping
idx["atomistic"]["force_outliers"]          # max |F| percentiles, counts above 10/20/30, top 10
idx["atomistic"]["energy_reference"]        # per-element references, 'identifiable'
```

**`metadata.json`** — the `outputs` section lists every file with its format, size,
SHA-256 and array shapes/dtypes, and `n_records` (structures) for each split, so you can
confirm nothing was lost:

```python
meta = json.load(open("atomistic_out/metadata.json"))
for f in meta["outputs"]["files"]:
    print(f["path"], f["format"], f.get("n_records"))
```

**`metadata.json` → `pipeline_steps`** — what each step changed and how long it took.
Every step is listed in order with its status, start/end time and duration; steps that did
not apply (here `load_ocp_lmdb`) are recorded as `skipped`. Steps also report their own
numbers under `changes`, and `metadata_added` / `metadata_removed` list the pipeline
state a step created or discarded:

```python
for s in meta["pipeline_steps"]:
    print(f'{s["order"]}. {s["name"]:22} {s["status"]:8} {s["seconds"]:7.2f}s', s.get("changes", ""))
print(meta["pipeline_timing"])   # started_at, ended_at, total_seconds
```

For BeF2 this records, for example (one run; times vary with file caching):

| Step | Seconds | What it recorded |
|------|--------:|------------------|
| `load_ocp_lmdb` | 0.00 | skipped, not applicable to this input |
| `load_atom_table` | 25.6 | 401,598 rows → 1,998 structures; 0 dropped for inconsistent energy/cell; 289,419 atoms outside the cell; Be→4, F→9 |
| `filter_records` | 0.02 | 1,998 in, 1,998 out, 0 dropped |
| `flag_force_outliers` | 0.02 | max \|F\| 41.8; 229 structures above 20; none dropped (no `max_force` set) |
| `reference_energies` | 0.02 | per-element references fitted; not identifiable (one composition) |
| `build_graphs` | 5.4 | 27,099,979 edges; periodic; 289,419 atoms wrapped into the cell; forces unchanged |
| `assemble_pyg_data` | 8.4 | 1,998 structures, 401,598 atoms (PyG objects, only used by `-f pt`) |
| `save_output` | 0.5 | 1,636 / 198 / 164 split by group |

Rerunning with a different setting (say `max_force: 30.0`) shows up directly: `flag_force_outliers`
then reports 36 dropped and every later step reports 1,962 structures.

Things worth knowing about BeF2 specifically:

- **Splits are by trajectory.** 27 trajectories (`mol-boot` plus `mol-0000` … `mol-0028`
  with gaps): 20 in train, 3 in val, 4 in test. Each trajectory is in exactly one split.
  Nearly identical consecutive MD frames therefore cannot appear on both sides of the
  train/test boundary. Split sizes are approximate (80/10/10 by structure count) because
  whole trajectories are assigned.
- **Forces include large values.** Max \|F\| is 41.8 eV/Å; 229 structures have a force
  above 20 and 36 above 30. REDI reports them but keeps them. To drop, set `max_force`
  (next section). Decide whether those frames are real high-energy configurations or
  SCF failures before dropping them.
- **Energy references cannot be fitted per element.** Every frame is Be₆₇F₁₃₄, so the
  fit is a single constant shift, and `index.json` says `identifiable: false`. A model
  trained on this data alone will not generalize to other Be:F ratios.
- **Units are declared, not checked.** REDI assumes eV and Å (recorded under
  `atomistic.units`). Confirm this against your source; the values here (about
  −574 eV/atom) are consistent with CP2K in eV.
- **Normalize in the trainer.** Forces are kept raw. REDI's OC22 path z-scores forces per
  structure, which is usually wrong for training a potential, so it is off for tables.
  Turn it on with `normalize_forces: true` only if your model expects it.

## 5. Options

Put options in a YAML config and pass it with `-c`:

```yaml
# bef2.yaml
domain: atomistic
input_path: /lustre/orion/stf249/proj-shared/hackathon/BeF2/bef2_pbe-d3.parquet
output_format: npz        # or pt
max_force: 30.0           # drop structures with any |F| above this (eV/Å)
```

```bash
redi run -c bef2.yaml --style simple -o ./atomistic_out
```

With `max_force: 30.0`, 36 structures are dropped and 1,962 remain (the split becomes
1,615 / 184 / 163).

| Option | Default | Effect |
|--------|---------|--------|
| `output_format` / `-f` | `npz` | `npz` (flat arrays), `parquet` (one row per structure, per-atom list columns), `pt` (one PyG `Data` file per structure), `json` (HydraGNN native) |
| `max_force` | none | Drop structures whose max per-atom \|F\| exceeds this |
| `pbc` | `true` if a cell is present | Wrap positions and build the periodic graph |
| `normalize_forces` | `false` | Per-structure z-score of forces |
| `energy_reference` | `fit` | `none` skips the reference-energy fit |
| `group_column` | derived from `frame_label` | Column that defines trajectories for the split |
| `columns` | see below | Rename input columns |
| `energy_unit`, `length_unit` | `eV`, `Angstrom` | Recorded in `index.json`; no conversion |
| `manifest_checksums` | `true` | Set `false` to skip SHA-256 in `metadata.json` for very large outputs |
| `num_workers` | auto | Parallel graph building |
| `radius_cutoff` | `6.0` | Graph radius in Å. The BeF2 paper's HIP-NN uses 7.5 Å (about 132 neighbours per atom here) |
| `max_neighbours` | `100` | Cap on edges per atom; raise it with `radius_cutoff` (7.5 Å needs about 160 or more to avoid truncation) |
| `h5_datasets` | see HDF5 section | Rename datasets when reading HDF5 stores |

`-f pt` writes one file per structure (1,998 files, about 890 MB for BeF2) and records
each directory as a single grouped entry in `metadata.json`. It needs `torch` and
`torch_geometric`. Each `Data` object carries `cell`, `pbc` and `edge_shifts`.

## 6. Adapting this to your own data

Your table needs one row per atom with, at minimum: a structure id, element (symbol or
atomic number), x/y/z, fx/fy/fz, and a total energy. A cell (`cell_00` … `cell_22`) is
optional but needed for periodic graphs. CSV works as well as Parquet.

If your columns have different names, map them:

```yaml
domain: atomistic
input_path: /path/to/my_frames.parquet
columns:
  frame: structure_id
  species: element
  energy: E_total
  label: run_name          # optional; used to derive trajectory groups
group_column: run_name     # or name the split-group column directly
```

For the split to be meaningful, give REDI a trajectory (or any grouping of correlated
structures). By default the group is `frame_label` with its trailing `-<digits>` removed
(`mol-0013-0000000155` → `mol-0013`). If there is no label or group column, REDI falls
back to a random 80/10/10 split by structure, which leaks for MD data.

If your data is OC22/OC20 LMDB shards instead, use the same command on the `.lmdb`
file or a directory of them; the table steps skip themselves.

### HDF5 stores

The Parquet file above was exported from an HDF5 store (hippynn "h5store" layout: one
group per composition, with per-frame `coordinates`, `forces`, `energy`, `cell`, `_id`
and a per-composition `species`). REDI reads that layout directly, so you can skip the
export. Pass the files (or a directory of them); `load_atom_h5` replaces
`load_atom_table` and every later step is the same:

```bash
pip install -e "./redi-atomistics[h5]"        # adds h5py
redi run -d atomistic --style simple -o ./atomistic_out \
    /opt/data/redi/BeF2/hipnn/pbe-d3/h5store/*.h5
```

Expected output (about 8 s on a workstation; the h5 files total 18.9 MB):

```
ReferenceEnergies: compositions have rank 1 < 2 elements; per-element references are not identifiable (fit is a constant shift)

  BuildGraphs: 1,998 structures, num_workers=auto
Input: 30 files (18.9 MB)
Domain: atomistic
Pipeline: 9 steps
Processing...
  [OK] load_atom_h5
  [OK] filter_records
  [OK] flag_force_outliers
  [OK] reference_energies
  BuildGraphs: 1,998 structures, 27,099,979 edges  (3.3s)
  [OK] build_graphs
  [OK] assemble_pyg_data
  [OK] save_output
Done: 7 ok, 0 warnings, 0 errors
```

The `ReferenceEnergies` notice is expected: every frame is Be₆₇F₁₃₄, so the two
per-element references cannot be told apart and the fit is only a constant shift (see
section 4). The output directory has the same layout and the same 1,636 / 198 / 164
split as the Parquet run; `index.json` names `h5store` as the dataset and lists
`n_files: 30`. The edge count depends on `radius_cutoff` (6 Å here; 7.5 Å gives about
53.2 million).

This gives the same 1,998 structures. The split group is the `_id` label with its
trailing step removed, as for `frame_label`. Dataset names can be remapped with
`h5_datasets:` in the config (keys: `coordinates`, `forces`, `energy`, `species`,
`cell`, `label`).

## 7. Optional: describe the dataset for sharing

`make_croissant.py` in the BeF2 directory writes a Croissant JSON-LD description of the
Parquet file that `mlcroissant` can load and validate. It leaves `license`, `version`,
`datePublished` and `citeAs` unset, and the URL is a placeholder. Fill those in before
publishing. REDI's `metadata.json` can be passed on to SetGo for FAIR/data-card
enrichment (see the redi skill for the hand-off).

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Error loading LMDB … not an LMDB file` | You used an older REDI/plugin; table input needs `redi-atomistics` ≥ 0.3 and `.parquet`/`.csv` extension |
| `Error loading LMDB … not an LMDB file` on `.h5` input | The plugin predates the HDF5 reader (`load_atom_h5`); update `redi-atomistics` and install its `h5` extra |
| `LoadAtomTable: missing columns [...]` | Column names differ; add a `columns:` mapping |
| `LoadAtomTable: unknown element symbols` | Species values are not element symbols; use symbols or atomic numbers |
| Everything lands in train, or val/test are empty | Too few trajectory groups; check `split_groups` in `index.json` and `group_column` |
| `-s output` skips steps | `--stages` only auto-includes `ingest`; list every stage you want, or omit `-s` |
| `.pt` output fails with an import error | Install `torch` and `torch_geometric` |
| `metadata.json` says a record count is missing | Arrays in the file have different lengths and the step did not report a count; the atomistics plugin does report them |
