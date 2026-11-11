"""
Atomistic REDI pipeline steps (HydraGNN-ready output).

Steps run in sequence (each skips itself when not applicable to the input):

  LoadOCPLMDB           — load Open Catalyst (OC20/OC22) LMDB shard(s) from file or directory
  LoadAtomTable         — load a per-atom Parquet/CSV table, group rows into structures
  LoadAtomH5            — load frames from HDF5 store(s) (group per composition)
  FilterRecords         — drop structures with NaN energy/forces or zero atoms
  FlagForceOutliers     — report max |F| per structure, optionally drop outliers (table input)
  ReferenceEnergies     — fit per-element reference energies (table input)
  BuildGraphs           — normalize fields + build cutoff-radius graph edges
  AssemblePyGData       — assemble per-structure PyG Data objects
  SaveOutput            — write NPZ / PT output + index.json

The classes were named after OC22 (LoadOC22LMDB, BuildOC22Graphs, AssembleOC22PyGData)
before this domain handled other data (and SaveOutput was SaveHydraGNNOutput);
those names remain as aliases at the end of this module.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RADIUS_CUTOFF  = 6.0    # Angstroms — standard for OC22/HydraGNN
MAX_NEIGHBOURS = 100    # max edges per node


# ---------------------------------------------------------------------------
# Minimal base step — avoids importing redi core at module load time
# ---------------------------------------------------------------------------

class _AtomisticStep:
    """Minimal PipelineStep compatible with REDI's executor."""

    stage       = "ingest"
    description = ""

    def validate(self, ctx) -> bool:
        return True

    def execute(self, ctx):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Attribute access that works for both plain objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return vars(obj).get(key, default)
    except TypeError:
        return getattr(obj, key, default)


def _report_changes(ctx, step_name: str, **fields) -> None:
    """
    Record what a step did (counts in/out, what was dropped or rewritten) in
    ctx.metadata["step_changes"][step_name]. REDI copies it into the step's entry in
    metadata.json under pipeline_steps, next to the step's duration.
    """
    ctx.metadata.setdefault("step_changes", {})[step_name] = fields


def _detect_split_from_path(path: Path) -> Optional[str]:
    """Return 'train', 'val', or 'test' if found in path components."""
    for part in list(path.parts) + [path.name]:
        part = part.lower()
        if "train" in part:
            return "train"
        if "val" in part:
            return "val"
        if "test" in part:
            return "test"
    return None


_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {sym: z for z, sym in enumerate(_ELEMENTS, start=1)}

TABLE_SUFFIXES = (".parquet", ".pq", ".csv")

# Default column names for LoadAtomTable (override via ctx.config["columns"])
DEFAULT_TABLE_COLUMNS = {
    "frame":       "frame_id",
    "atom":        "atom_idx",
    "species":     "species",
    "x": "x", "y": "y", "z": "z",
    "fx": "fx", "fy": "fy", "fz": "fz",
    "energy":      "energy",
    "cell_prefix": "cell_",       # cell_00 .. cell_22, rows are lattice vectors
    "label":       "frame_label",
}

# Trailing step number of a frame label: "mol-0013-0000000155" -> "mol-0013"
_LABEL_STEP_RE = re.compile(r"^(.*)-\d+$")

# The 27 cell-image offsets; index 13 is (0, 0, 0)
_IMAGE_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64)
_ZERO_SHIFT   = 13


H5_SUFFIXES = (".h5", ".hdf5", ".hdf")

# ctx.metadata["loader"] values whose records come from flat-frame sources (table, HDF5);
# the table-only steps (force outliers, reference energies) run for these.
_FRAME_LOADERS = ("atom_table", "hdf5")


def _is_table_input(path) -> bool:
    return Path(path).suffix.lower() in TABLE_SUFFIXES


def _h5_files(path) -> List[Path]:
    """HDF5 files at path: the file itself, or the sorted *.h5 files in a directory."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in H5_SUFFIXES)
    return [path] if path.suffix.lower() in H5_SUFFIXES else []


def _is_h5_input(path) -> bool:
    return bool(_h5_files(path))


def _h5_inputs(ctx) -> List[Path]:
    """All HDF5 files for a run: ctx.input_paths (multi-file runs), else ctx.input_path."""
    paths = [Path(p) for p in (getattr(ctx, "input_paths", None) or [])]
    files = sorted({f for p in paths for f in _h5_files(p)}) if paths else []
    return files or _h5_files(ctx.input_path)


def _cap_neighbours(src, dst, dist, n_nodes: int, max_neighbours: int):
    """Keep at most max_neighbours shortest edges per source node. Returns the kept mask."""
    keep = np.ones(len(src), dtype=bool)
    if max_neighbours < len(src):
        keep[:] = False
        count = np.zeros(n_nodes, dtype=np.int32)
        for idx in np.argsort(dist):
            n = src[idx]
            if count[n] < max_neighbours:
                keep[idx] = True
                count[n] += 1
    return keep


def _radius_graph_numpy(pos: np.ndarray, r: float, max_neighbours: int):
    """Build radius graph using scipy cKDTree. Returns (edge_index [2,E], edge_attr [E,1])."""
    from scipy.spatial import cKDTree
    tree  = cKDTree(pos)
    pairs = tree.query_pairs(r=r, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    src  = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst  = np.concatenate([pairs[:, 1], pairs[:, 0]])
    diff = pos[dst] - pos[src]
    dist = np.linalg.norm(diff, axis=1).astype(np.float32)

    keep = _cap_neighbours(src, dst, dist, len(pos), max_neighbours)
    src, dst, dist = src[keep], dst[keep], dist[keep]

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr  = dist[:, None]
    return edge_index, edge_attr


def _wrap_positions(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap Cartesian positions into the cell (rows of ``cell`` are lattice vectors)."""
    frac = np.linalg.solve(cell.T, pos.T).T
    frac -= np.floor(frac)
    return frac @ cell


def _radius_graph_pbc(pos: np.ndarray, cell: np.ndarray, r: float, max_neighbours: int):
    """
    Periodic radius graph over the 27 nearest cell images. ``pos`` must already be
    wrapped into the cell; any cell shape (orthorhombic or triclinic) is supported.

    Returns (edge_index [2,E], edge_attr [E,1], edge_shifts [E,3]) where
    edge_shifts is the integer image offset of dst relative to src, so the edge
    vector is pos[dst] + edge_shifts @ cell - pos[src].
    """
    from scipy.spatial import cKDTree
    n = len(pos)

    # Perpendicular cell widths: one shell of images is enough only if r <= the smallest
    widths = abs(np.linalg.det(cell)) / np.linalg.norm(
        np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
    )
    if r > widths.min():
        logger.warning(
            "Radius cutoff %.2f exceeds smallest cell width %.2f; some periodic "
            "neighbours will be missed", r, widths.min(),
        )

    images = (pos[None, :, :] + (_IMAGE_SHIFTS @ cell)[:, None, :]).reshape(-1, 3)
    pairs  = cKDTree(pos).sparse_distance_matrix(cKDTree(images), r, output_type="ndarray")

    src       = pairs["i"].astype(np.int64)
    dst       = (pairs["j"] % n).astype(np.int64)
    shift_idx = pairs["j"] // n
    dist      = pairs["v"].astype(np.float32)

    not_self = ~((src == dst) & (shift_idx == _ZERO_SHIFT))
    src, dst, shift_idx, dist = src[not_self], dst[not_self], shift_idx[not_self], dist[not_self]

    order = np.lexsort((shift_idx, dst, src))   # deterministic edge order
    src, dst, shift_idx, dist = src[order], dst[order], shift_idx[order], dist[order]

    keep = _cap_neighbours(src, dst, dist, n, max_neighbours)
    src, dst, shift_idx, dist = src[keep], dst[keep], shift_idx[keep], dist[keep]

    edge_index  = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr   = dist[:, None]
    edge_shifts = _IMAGE_SHIFTS[shift_idx].astype(np.float32)
    return edge_index, edge_attr, edge_shifts


def _process_one_structure(obj, options: Optional[dict] = None) -> Optional[dict]:
    """
    Normalize a single structure and build its radius graph.
    Used as the worker function for ProcessPoolExecutor in BuildGraphs.
    Returns a normalized dict, or None if the structure should be skipped.

    options (from ctx.metadata["structure_options"]; defaults keep OC22 behaviour):
      pbc              — wrap positions into the cell and build a periodic graph
                         (needs a cell; default False: centre positions, open graph)
      normalize_forces — per-structure z-score of forces (default True)
      radius_cutoff    — graph radius in Angstrom (default RADIUS_CUTOFF)
      max_neighbours   — max edges per node (default MAX_NEIGHBOURS)
      passthrough      — extra record fields copied to the output when present
    """
    options = options or {}
    radius_cutoff  = float(options.get("radius_cutoff", RADIUS_CUTOFF))
    max_neighbours = int(options.get("max_neighbours", MAX_NEIGHBOURS))
    try:
        pos    = np.array(_get(obj, "pos"),            dtype=np.float32)
        z      = np.array(_get(obj, "atomic_numbers"), dtype=np.int32)
        forces = np.array(_get(obj, "force"),          dtype=np.float32)
        energy = float(_get(obj, "y"))
        natoms = int(_get(obj, "natoms"))

        tags_raw = _get(obj, "tags")
        tags = (np.array(tags_raw, dtype=np.int32)
                if tags_raw is not None
                else np.zeros(natoms, dtype=np.int32))

        cell_raw   = _get(obj, "cell")
        periodic   = bool(options.get("pbc", False)) and cell_raw is not None
        e_per_atom = np.float32(energy / natoms)

        if options.get("normalize_forces", True):
            f_std      = forces.std()
            forces_out = (forces - forces.mean()) / (f_std + 1e-8)
        else:
            forces_out = forces

        edge_shifts = None
        if periodic:
            cell    = np.array(cell_raw, dtype=np.float64).reshape(3, 3)
            pos_out = _wrap_positions(pos.astype(np.float64), cell).astype(np.float32)
            n_wrapped = int((np.abs(pos_out - pos) > 1e-2).any(axis=1).sum())
            edge_index, edge_attr, edge_shifts = _radius_graph_pbc(
                pos_out.astype(np.float64), cell, radius_cutoff, max_neighbours
            )
        else:
            pos_out = pos - pos.mean(axis=0)
            edge_index, edge_attr = _radius_graph_numpy(
                pos_out, radius_cutoff, max_neighbours
            )

        record: Dict[str, Any] = {
            "atomic_numbers":  z,
            "pos":             pos_out,
            "forces":          forces_out.astype(np.float32),
            "energy_per_atom": e_per_atom,
            "natoms":          natoms,
            "tags":            tags,
            "sid":             str(_get(obj, "sid", "")),
            "edge_index":      edge_index,
            "edge_attr":       edge_attr,
        }
        if edge_shifts is not None:
            record["edge_shifts"] = edge_shifts
            record["n_wrapped"]   = n_wrapped

        for key in options.get("passthrough", ()):
            value = _get(obj, key)
            if value is not None:
                record[key] = value

        if cell_raw is not None:
            record["cell"] = np.array(cell_raw, dtype=np.float32)

        fixed_raw = _get(obj, "fixed")
        if fixed_raw is not None:
            record["fixed"] = np.array(fixed_raw, dtype=np.int32)

        return record

    except Exception as exc:
        logger.warning("Skipping structure: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 1: LoadOCPLMDB
# ---------------------------------------------------------------------------

class LoadOCPLMDB(_AtomisticStep):
    """
    Load OC22 (or OC20) structures from an LMDB shard file or a directory
    of shards.

    When given a directory, iterates all *.lmdb files in sorted order.
    Respects ``max_samples`` from ctx.config to cap the total records loaded.

    Sets:
      ctx.metadata["records"]          — list of raw PyG/pickle objects
      ctx.metadata["dataset"]          — "oc22" or "oc20"
      ctx.metadata["split"]            — detected split ("train"/"val"/"test"/None)
      ctx.metadata["lmdb_path"]        — resolved path string
      ctx.metadata["n_structures_raw"] — number of records loaded
    """

    name        = "load_ocp_lmdb"
    stage       = "ingest"
    description = "Load OC22/OC20 structures from LMDB shard(s)"

    def __init__(self, dataset: str = "oc22"):
        self.dataset = dataset.lower()

    def validate(self, ctx) -> bool:
        return not _is_table_input(ctx.input_path) and not _h5_inputs(ctx)

    def execute(self, ctx):
        try:
            import lmdb

            path        = Path(ctx.input_path)
            max_samples = ctx.config.get("max_samples") if ctx.config else None

            # Collect shard paths
            if path.is_dir():
                shards = sorted(
                    p for p in path.iterdir()
                    if p.suffix == ".lmdb" and not p.name.endswith("-lock")
                )
                if not shards:
                    # Maybe it's an LMDB subdir (contains data.mdb)
                    shards = [path]
            else:
                shards = [path]

            if not shards:
                ctx.add_error(f"No .lmdb shards found in {path}")
                return ctx

            records = []
            for shard in shards:
                subdir = shard.is_dir()
                env = lmdb.open(
                    str(shard), subdir=subdir, readonly=True,
                    lock=False, meminit=False, map_size=int(1e12),
                )
                txn    = env.begin()
                cursor = txn.cursor()
                cursor.first()
                while True:
                    records.append(pickle.loads(cursor.value()))
                    if max_samples and len(records) >= max_samples:
                        txn.abort()
                        env.close()
                        break
                    if not cursor.next():
                        txn.abort()
                        env.close()
                        break

                if max_samples and len(records) >= max_samples:
                    break

            split = _detect_split_from_path(path)

            ctx.metadata["records"]          = records
            ctx.metadata["dataset"]          = self.dataset
            ctx.metadata["split"]            = split
            ctx.metadata["lmdb_path"]        = str(path)
            ctx.metadata["n_structures_raw"] = len(records)

            _report_changes(ctx, self.name, dataset=self.dataset, shards=len(shards),
                            structures_loaded=len(records), split_from_path=split)
            logger.info(
                "Loaded %d structures from %s (dataset=%s, split=%s, shards=%d)",
                len(records), path.name, self.dataset, split, len(shards),
            )

        except ImportError:
            msg = ("lmdb is required for LoadOCPLMDB. "
                   "Install with: pip install 'redi[atomistic]'")
            logger.error(msg)
            ctx.add_error(msg)
        except Exception as exc:
            logger.error("Error loading LMDB: %s", exc)
            ctx.add_error(f"Error loading LMDB: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 1b: LoadAtomTable
# ---------------------------------------------------------------------------

class LoadAtomTable(_AtomisticStep):
    """
    Load a per-atom table (Parquet or CSV, one row per atom) and group its rows
    into structures — e.g. DFT/AIMD frames exported from CP2K or VASP.

    Expected columns (rename via ctx.config["columns"], see DEFAULT_TABLE_COLUMNS):
      frame_id            structure identifier
      atom_idx            (optional) atom order within a structure
      species             element symbol or atomic number
      x, y, z             Cartesian positions
      fx, fy, fz          forces
      energy              total energy, repeated on every atom of a structure
      cell_00 .. cell_22  (optional) 3x3 cell, rows are lattice vectors
      frame_label         (optional) used to derive the split group

    The split group is ctx.config["group_column"] if given, otherwise frame_label
    with its trailing "-<step>" removed ("mol-0013-0000000155" -> "mol-0013"), so
    correlated frames from one trajectory never straddle train/val/test.

    Structures whose energy or cell is not constant across their atoms are dropped.

    Other ctx.config keys: pbc (default True when a cell is present),
    normalize_forces (default False), energy_unit / length_unit (default eV / Angstrom;
    recorded in index.json, not converted).

    Sets:
      ctx.metadata["records"]             — list of per-structure records
      ctx.metadata["structure_options"]   — options for BuildGraphs
      ctx.metadata["split_by"]            — "group" when groups are available
      ctx.metadata["atomistic_summary"]   — data checks, written to index.json
    """

    name        = "load_atom_table"
    stage       = "ingest"
    description = "Load per-atom Parquet/CSV table and group rows into structures"

    def validate(self, ctx) -> bool:
        return _is_table_input(ctx.input_path)

    def execute(self, ctx):
        try:
            import pandas as pd

            cfg  = ctx.config or {}
            cols = {**DEFAULT_TABLE_COLUMNS, **(cfg.get("columns") or {})}
            path = Path(ctx.input_path)
            df   = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

            required = [cols[k] for k in ("frame", "species", "x", "y", "z", "fx", "fy", "fz", "energy")]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                ctx.add_error(f"LoadAtomTable: missing columns {missing}")
                return ctx

            sort_cols = [cols["frame"]] + ([cols["atom"]] if cols["atom"] in df.columns else [])
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
            frame_ids, starts, counts = np.unique(
                df[cols["frame"]].to_numpy(), return_index=True, return_counts=True
            )

            species = df[cols["species"]]
            if pd.api.types.is_numeric_dtype(species):
                z = species.to_numpy(dtype=np.int32)
            else:
                symbols = species.astype(str).str.strip().str.capitalize()
                unknown = sorted(set(symbols) - set(ATOMIC_NUMBERS))
                if unknown:
                    ctx.add_error(f"LoadAtomTable: unknown element symbols {unknown}")
                    return ctx
                z = symbols.map(ATOMIC_NUMBERS).to_numpy(dtype=np.int32)

            pos    = df[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=np.float64)
            forces = df[[cols["fx"], cols["fy"], cols["fz"]]].to_numpy(dtype=np.float64)
            energy = df[cols["energy"]].to_numpy(dtype=np.float64)

            cell_cols = [f"{cols['cell_prefix']}{i}{j}" for i in range(3) for j in range(3)]
            has_cell  = all(c in df.columns for c in cell_cols)
            cells     = df[cell_cols].to_numpy(dtype=np.float64) if has_cell else None

            labels = df[cols["label"]].astype(str).to_numpy() if cols["label"] in df.columns else None
            group_col = cfg.get("group_column")
            if group_col:
                groups = df[group_col].astype(str).to_numpy()
            elif labels is not None:
                groups = np.array([_LABEL_STEP_RE.sub(r"\1", s) for s in labels])
            else:
                groups = None

            # Per-structure consistency: energy and cell must be constant across atoms
            ends = starts + counts
            def _varies(values):
                v = values if values.ndim == 2 else values[:, None]
                spread = np.maximum.reduceat(v, starts, axis=0) - np.minimum.reduceat(v, starts, axis=0)
                scale  = np.maximum(np.abs(v[starts]), 1.0)
                return (spread > 1e-8 * scale).any(axis=1)
            inconsistent = _varies(energy)
            if has_cell:
                inconsistent |= _varies(cells)

            records, n_outside = [], 0
            for k, (fid, s, e) in enumerate(zip(frame_ids, starts, ends)):
                if inconsistent[k]:
                    continue
                cell = cells[s].reshape(3, 3) if has_cell else None
                if cell is not None:
                    frac = np.linalg.solve(cell.T, pos[s:e].T).T
                    n_outside += int(((frac < 0) | (frac >= 1)).any(axis=1).sum())
                records.append(SimpleNamespace(
                    sid            = labels[s] if labels is not None else str(fid),
                    frame_id       = fid.item() if hasattr(fid, "item") else fid,
                    natoms         = int(e - s),
                    atomic_numbers = z[s:e],
                    pos            = pos[s:e],
                    force          = forces[s:e],
                    y              = float(energy[s]),
                    energy         = float(energy[s]),
                    cell           = cell,
                    tags           = None,
                    group          = groups[s] if groups is not None else None,
                ))

            net_force = np.abs(np.add.reduceat(forces, starts, axis=0)).max() if len(starts) else 0.0
            symbols_present = {_ELEMENTS[i - 1]: int(n) for i, n in zip(*np.unique(z, return_counts=True))}

            ctx.metadata["records"]          = records
            ctx.metadata["loader"]           = "atom_table"
            ctx.metadata["dataset"]          = path.stem
            ctx.metadata["split"]            = None
            ctx.metadata["split_by"]         = "group" if groups is not None else None
            ctx.metadata["n_structures_raw"] = len(records)
            ctx.metadata["structure_options"] = {
                "pbc":              bool(cfg.get("pbc", True)) and has_cell,
                "normalize_forces": bool(cfg.get("normalize_forces", False)),
                "passthrough":      ["frame_id", "group", "energy", "energy_referenced", "max_force"],
            }
            ctx.metadata["atomistic_summary"] = {
                "source":                 str(path),
                "n_atoms":                int(len(df)),
                "n_structures":           len(records),
                "natoms_min":             int(counts.min()) if len(counts) else 0,
                "natoms_max":             int(counts.max()) if len(counts) else 0,
                "species_counts":         symbols_present,
                "n_inconsistent_dropped": int(inconsistent.sum()),
                "n_atoms_outside_cell":   n_outside,
                "max_net_force":          float(net_force),
                "has_cell":               has_cell,
                "n_groups":               int(len(set(groups))) if groups is not None else 0,
                "units": {
                    "energy": cfg.get("energy_unit", "eV"),
                    "length": cfg.get("length_unit", "Angstrom"),
                    "force":  f"{cfg.get('energy_unit', 'eV')}/{cfg.get('length_unit', 'Angstrom')}",
                    "note":   "declared, not converted or verified",
                },
            }

            _report_changes(
                ctx, self.name,
                rows_read=int(len(df)),
                structures_built=len(records),
                structures_dropped_inconsistent=int(inconsistent.sum()),
                atoms_outside_cell=n_outside,
                species_to_atomic_number={sym: ATOMIC_NUMBERS[sym] for sym in symbols_present},
                units_assumed=ctx.metadata["atomistic_summary"]["units"]["note"],
            )
            logger.info(
                "Loaded %d structures (%d atoms) from %s; %d inconsistent dropped, "
                "%d atoms outside cell",
                len(records), len(df), path.name, int(inconsistent.sum()), n_outside,
            )

        except Exception as exc:
            logger.error("Error loading atom table: %s", exc)
            ctx.add_error(f"Error loading atom table: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 1c: LoadAtomH5
# ---------------------------------------------------------------------------

class LoadAtomH5(_AtomisticStep):
    """
    Load frames from HDF5 stores in the hippynn/ANI "h5store" layout: one group per
    composition, each holding per-frame datasets. Input is a file or a directory of
    *.h5 / *.hdf5 files (all files are read, in sorted order).

    Datasets per group (rename via ctx.config["h5_datasets"]):
      coordinates   (n_frames, n_atoms, 3)
      forces        (n_frames, n_atoms, 3)
      energy        (n_frames,)
      species       (n_atoms,) or (n_frames, n_atoms); element symbols or atomic numbers
      cell          (n_frames, 3, 3), optional (also accepts (3, 3) or (n_frames, 3))
      _id           (n_frames,), optional frame label, used for the split group

    Records have the same shape as LoadAtomTable's, so the same downstream steps apply.
    The split group is the _id label with its trailing "-<step>" removed (as in
    LoadAtomTable); without _id it is the HDF5 group name.

    Other ctx.config keys: pbc, normalize_forces, energy_unit, length_unit, max_samples.
    """

    name        = "load_atom_h5"
    stage       = "ingest"
    description = "Load frames from HDF5 store(s) (group per composition)"

    DEFAULT_DATASETS = {
        "coordinates": "coordinates", "forces": "forces", "energy": "energy",
        "species": "species", "cell": "cell", "label": "_id",
    }

    def validate(self, ctx) -> bool:
        return bool(_h5_inputs(ctx))

    @staticmethod
    def _atomic_numbers(raw: np.ndarray) -> np.ndarray:
        if np.issubdtype(raw.dtype, np.number):
            return raw.astype(np.int32)
        symbols = np.char.capitalize(np.char.strip(raw.astype(str)))
        unknown = sorted(set(symbols.ravel().tolist()) - set(ATOMIC_NUMBERS))
        if unknown:
            raise ValueError(f"unknown element symbols {unknown}")
        lut = np.vectorize(ATOMIC_NUMBERS.__getitem__, otypes=[np.int32])
        return lut(symbols)

    def execute(self, ctx):
        try:
            import h5py

            cfg  = ctx.config or {}
            ds   = {**self.DEFAULT_DATASETS, **(cfg.get("h5_datasets") or {})}
            path = Path(ctx.input_path)
            files = _h5_inputs(ctx)
            max_samples = cfg.get("max_samples")
            has_cell_any = False

            records, n_outside, n_nonfinite, n_groups_read = [], 0, 0, 0
            for f in files:
                with h5py.File(f, "r") as h:
                    for gname, g in h.items():
                        if not isinstance(g, h5py.Group) or ds["coordinates"] not in g:
                            continue
                        missing = [ds[k] for k in ("forces", "energy", "species") if ds[k] not in g]
                        if missing:
                            ctx.add_error(f"LoadAtomH5: {f.name}/{gname} missing datasets {missing}")
                            return ctx
                        n_groups_read += 1
                        pos    = g[ds["coordinates"]][()].astype(np.float64)
                        forces = g[ds["forces"]][()].astype(np.float64)
                        energy = g[ds["energy"]][()].astype(np.float64).reshape(-1)
                        n_frames, n_atoms = pos.shape[0], pos.shape[1]
                        if forces.shape != pos.shape or energy.shape[0] != n_frames:
                            ctx.add_error(
                                f"LoadAtomH5: {f.name}/{gname} shape mismatch: coordinates "
                                f"{pos.shape}, forces {forces.shape}, energy {energy.shape}")
                            return ctx
                        z = self._atomic_numbers(g[ds["species"]][()])
                        if z.ndim == 1:
                            z = np.broadcast_to(z, (n_frames, n_atoms))
                        if z.shape != (n_frames, n_atoms):
                            ctx.add_error(f"LoadAtomH5: {f.name}/{gname} species shape {z.shape}")
                            return ctx

                        cells = None
                        if ds["cell"] in g:
                            c = g[ds["cell"]][()].astype(np.float64)
                            if c.shape == (3, 3):
                                c = np.broadcast_to(c, (n_frames, 3, 3))
                            elif c.shape == (n_frames, 3):
                                c = np.stack([np.diag(r) for r in c])
                            if c.shape != (n_frames, 3, 3):
                                ctx.add_error(f"LoadAtomH5: {f.name}/{gname} cell shape {c.shape}")
                                return ctx
                            cells, has_cell_any = c, True

                        labels = None
                        if ds["label"] in g:
                            raw = g[ds["label"]][()]
                            labels = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])

                        for i in range(n_frames):
                            if not (np.isfinite(energy[i]) and np.isfinite(pos[i]).all()
                                    and np.isfinite(forces[i]).all()):
                                n_nonfinite += 1
                                continue
                            cell = cells[i] if cells is not None else None
                            if cell is not None:
                                frac = np.linalg.solve(cell.T, pos[i].T).T
                                n_outside += int(((frac < 0) | (frac >= 1)).any(axis=1).sum())
                            sid = labels[i] if labels is not None else f"{f.stem}/{gname}/{i}"
                            records.append(SimpleNamespace(
                                sid            = sid,
                                frame_id       = sid,
                                natoms         = int(n_atoms),
                                atomic_numbers = np.ascontiguousarray(z[i]),
                                pos            = pos[i],
                                force          = forces[i],
                                y              = float(energy[i]),
                                energy         = float(energy[i]),
                                cell           = cell,
                                tags           = None,
                                group          = _LABEL_STEP_RE.sub(r"\1", sid) if labels is not None else gname,
                            ))
                            if max_samples and len(records) >= max_samples:
                                break
                        if max_samples and len(records) >= max_samples:
                            break
                if max_samples and len(records) >= max_samples:
                    break

            if not records:
                ctx.add_error(f"LoadAtomH5: no frames found in {path} "
                              f"(expected groups with a '{ds['coordinates']}' dataset)")
                return ctx

            # A multi-file run is described by its directory, not by the first file
            src = files[0].parent if len(files) > 1 else path
            counts = np.array([r.natoms for r in records])
            zs, zc = np.unique(np.concatenate([r.atomic_numbers for r in records]), return_counts=True)
            symbols_present = {_ELEMENTS[int(i) - 1]: int(n) for i, n in zip(zs, zc)}
            net_force = max(float(np.abs(r.force.sum(axis=0)).max()) for r in records)
            groups = {r.group for r in records}

            ctx.metadata["records"]          = records
            ctx.metadata["loader"]           = "hdf5"
            ctx.metadata["dataset"]          = src.stem if src.is_file() else src.name
            ctx.metadata["split"]            = None
            ctx.metadata["split_by"]         = "group"
            ctx.metadata["n_structures_raw"] = len(records)
            ctx.metadata["structure_options"] = {
                "pbc":              bool(cfg.get("pbc", True)) and has_cell_any,
                "normalize_forces": bool(cfg.get("normalize_forces", False)),
                "passthrough":      ["frame_id", "group", "energy", "energy_referenced", "max_force"],
            }
            ctx.metadata["atomistic_summary"] = {
                "source":                 str(src),
                "n_files":                len(files),
                "n_atoms":                int(counts.sum()),
                "n_structures":           len(records),
                "natoms_min":             int(counts.min()),
                "natoms_max":             int(counts.max()),
                "species_counts":         symbols_present,
                "n_nonfinite_dropped":    n_nonfinite,
                "n_atoms_outside_cell":   n_outside,
                "max_net_force":          net_force,
                "has_cell":               has_cell_any,
                "n_groups":               len(groups),
                "units": {
                    "energy": cfg.get("energy_unit", "eV"),
                    "length": cfg.get("length_unit", "Angstrom"),
                    "force":  f"{cfg.get('energy_unit', 'eV')}/{cfg.get('length_unit', 'Angstrom')}",
                    "note":   "declared, not converted or verified",
                },
            }
            _report_changes(
                ctx, self.name,
                files_read=len(files), h5_groups_read=n_groups_read,
                structures_built=len(records),
                structures_dropped_nonfinite=n_nonfinite,
                atoms_outside_cell=n_outside,
                species_to_atomic_number={sym: ATOMIC_NUMBERS[sym] for sym in symbols_present},
                units_assumed=ctx.metadata["atomistic_summary"]["units"]["note"],
            )
            logger.info("Loaded %d structures from %d HDF5 file(s) at %s; %d atoms outside cell",
                        len(records), len(files), path, n_outside)

        except ImportError:
            msg = "h5py is required for LoadAtomH5. Install with: pip install 'redi[hdf5]'"
            logger.error(msg)
            ctx.add_error(msg)
        except Exception as exc:
            logger.error("Error loading HDF5: %s", exc)
            ctx.add_error(f"Error loading HDF5: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 2: FilterRecords
# ---------------------------------------------------------------------------

class FilterRecords(_AtomisticStep):
    """
    Drop records that have missing, NaN, or Inf energy/forces, or zero atoms.

    Reads  ctx.metadata["records"]
    Writes ctx.metadata["records"]  (filtered)
           ctx.metadata["n_valid"]
           ctx.metadata["n_dropped"]
    """

    name        = "filter_records"
    stage       = "preprocess"
    description = "Drop records with NaN energy/forces or zero atoms"

    def validate(self, ctx) -> bool:
        return "records" in ctx.metadata

    def execute(self, ctx):
        try:
            records = ctx.metadata["records"]
            valid, n_dropped = [], 0

            for obj in records:
                try:
                    energy = _get(obj, "y")
                    natoms = _get(obj, "natoms")
                    force  = _get(obj, "force")

                    if energy is None or natoms is None or force is None:
                        n_dropped += 1; continue
                    if int(natoms) == 0:
                        n_dropped += 1; continue
                    if not np.isfinite(float(energy)):
                        n_dropped += 1; continue
                    if not np.all(np.isfinite(np.array(force, dtype=np.float32))):
                        n_dropped += 1; continue

                    valid.append(obj)
                except Exception:
                    n_dropped += 1

            ctx.metadata["records"]   = valid
            ctx.metadata["n_valid"]   = len(valid)
            ctx.metadata["n_dropped"] = n_dropped
            _report_changes(ctx, self.name, records_in=len(records), records_out=len(valid),
                            dropped=n_dropped, criteria="non-finite energy or forces, or zero atoms")

            if n_dropped:
                logger.info("FilterRecords: dropped %d invalid structures", n_dropped)
            logger.info("FilterRecords: %d valid structures", len(valid))

        except Exception as exc:
            logger.error("Error in FilterRecords: %s", exc)
            ctx.add_error(f"Error in FilterRecords: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 2b: FlagForceOutliers
# ---------------------------------------------------------------------------

class FlagForceOutliers(_AtomisticStep):
    """
    Record the max per-atom |F| of each structure and summarize the distribution.
    Structures above ctx.config["max_force"] are dropped only if that key is set;
    otherwise outliers are reported in index.json for inspection.

    Runs for LoadAtomTable input only.
    """

    name        = "flag_force_outliers"
    stage       = "preprocess"
    description = "Report per-structure max |F|; drop above max_force if configured"

    REPORT_THRESHOLDS = (10.0, 20.0, 30.0)

    def validate(self, ctx) -> bool:
        return ctx.metadata.get("loader") in _FRAME_LOADERS and "records" in ctx.metadata

    def execute(self, ctx):
        try:
            records   = ctx.metadata["records"]
            max_force = (ctx.config or {}).get("max_force")

            fmax = np.array([np.linalg.norm(np.asarray(r.force), axis=1).max() for r in records])
            for r, f in zip(records, fmax):
                r.max_force = float(f)

            n_dropped = 0
            if max_force is not None:
                kept = [r for r in records if r.max_force <= float(max_force)]
                n_dropped = len(records) - len(kept)
                ctx.metadata["records"] = kept

            top = np.argsort(fmax)[::-1][:10]
            summary = ctx.metadata.setdefault("atomistic_summary", {})
            summary["force_outliers"] = {
                "max_force_per_structure": {
                    "p50": float(np.percentile(fmax, 50)) if len(fmax) else None,
                    "p99": float(np.percentile(fmax, 99)) if len(fmax) else None,
                    "max": float(fmax.max()) if len(fmax) else None,
                },
                "n_above": {str(t): int((fmax > t).sum()) for t in self.REPORT_THRESHOLDS},
                "top": [{"sid": records[i].sid, "max_force": float(fmax[i])} for i in top],
                "max_force_threshold": max_force,
                "n_dropped": n_dropped,
            }
            above = int((fmax > self.REPORT_THRESHOLDS[1]).sum())
            _report_changes(
                ctx, self.name, records_in=len(records), records_out=len(records) - n_dropped,
                dropped=n_dropped, max_force_threshold=max_force,
                max_force_overall=float(fmax.max()) if len(fmax) else None,
                **{f"structures_above_{self.REPORT_THRESHOLDS[1]:g}": above},
                note="outliers are reported, and dropped only when max_force is set",
            )
            logger.info(
                "FlagForceOutliers: max |F| %.2f; %d structures above %s; %d dropped",
                fmax.max() if len(fmax) else 0.0,
                int((fmax > self.REPORT_THRESHOLDS[1]).sum()), self.REPORT_THRESHOLDS[1], n_dropped,
            )

        except Exception as exc:
            logger.error("Error in FlagForceOutliers: %s", exc)
            ctx.add_error(f"Error in FlagForceOutliers: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 2c: ReferenceEnergies
# ---------------------------------------------------------------------------

class ReferenceEnergies(_AtomisticStep):
    """
    Fit per-element reference energies by least squares, E ≈ Σ_el n_el · μ_el,
    and store energy_referenced = E − Σ n_el · μ_el on each structure.

    If every structure has the same composition (or compositions are linearly
    dependent), the μ_el are not identifiable: the min-norm fit reduces to a
    constant shift, and index.json marks the references as not identifiable.

    Runs for LoadAtomTable input unless ctx.config["energy_reference"] == "none".
    """

    name        = "reference_energies"
    stage       = "transform"
    description = "Fit per-element reference energies; store referenced energies"

    def validate(self, ctx) -> bool:
        return (
            ctx.metadata.get("loader") in _FRAME_LOADERS
            and "records" in ctx.metadata
            and (ctx.config or {}).get("energy_reference", "fit") != "none"
        )

    def execute(self, ctx):
        try:
            records  = ctx.metadata["records"]
            elements = np.unique(np.concatenate([np.asarray(r.atomic_numbers) for r in records]))
            C = np.array([[np.count_nonzero(np.asarray(r.atomic_numbers) == el) for el in elements]
                          for r in records], dtype=np.float64)
            E = np.array([r.y for r in records], dtype=np.float64)

            mu, *_ = np.linalg.lstsq(C, E, rcond=None)
            rank   = int(np.linalg.matrix_rank(C))
            e_ref  = E - C @ mu
            for r, e in zip(records, e_ref):
                r.energy_referenced = float(e)

            identifiable = rank == len(elements)
            if not identifiable:
                logger.warning(
                    "ReferenceEnergies: compositions have rank %d < %d elements; per-element "
                    "references are not identifiable (fit is a constant shift)",
                    rank, len(elements),
                )

            natoms = C.sum(axis=1)
            summary = ctx.metadata.setdefault("atomistic_summary", {})
            summary["energy_reference"] = {
                "method":       "linear_fit",
                "per_element":  {_ELEMENTS[int(el) - 1]: float(m) for el, m in zip(elements, mu)},
                "identifiable": identifiable,
                "composition_rank": rank,
                "referenced_energy_per_atom_std": float(np.std(e_ref / natoms)) if len(E) else None,
            }
            _report_changes(
                ctx, self.name, adds="energy_referenced (total energy minus fitted per-element references)",
                per_element_reference=summary["energy_reference"]["per_element"],
                identifiable=identifiable, structures=len(records),
            )

        except Exception as exc:
            logger.error("Error in ReferenceEnergies: %s", exc)
            ctx.add_error(f"Error in ReferenceEnergies: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 3: BuildGraphs
# ---------------------------------------------------------------------------

class BuildGraphs(_AtomisticStep):
    """
    Per-structure normalization and radius graph construction.

    For each structure:
      - Center atomic positions (subtract mean)
      - Compute per-atom energy
      - Z-score normalize forces
      - Build cutoff-radius graph edges (default r=6 Å, max_neighbours=100;
        override with ctx.config radius_cutoff / max_neighbours)

    Uses ``num_workers`` from ctx.config (or constructor) for parallel
    radius graph computation via ProcessPoolExecutor.

    Reads  ctx.metadata["records"]     (raw PyG/pickle objects)
    Writes ctx.metadata["normalized"]  (list of normalized dicts with edge_index/edge_attr)
    """

    name        = "build_graphs"
    stage       = "transform"
    description = "Normalize fields and build cutoff-radius graph edges (parallel)"

    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers

    def validate(self, ctx) -> bool:
        return "records" in ctx.metadata

    def execute(self, ctx):
        import sys
        import time as _time
        try:
            records     = ctx.metadata["records"]
            cfg         = ctx.config or {}
            radius      = float(cfg.get("radius_cutoff", RADIUS_CUTOFF))
            max_nbrs    = int(cfg.get("max_neighbours", MAX_NEIGHBOURS))
            if radius <= 0 or max_nbrs < 1:
                ctx.add_error(f"BuildGraphs: radius_cutoff must be > 0 and max_neighbours >= 1 "
                              f"(got {radius}, {max_nbrs})")
                return ctx
            options     = {**(ctx.metadata.get("structure_options") or {}),
                           "radius_cutoff": radius, "max_neighbours": max_nbrs}
            num_workers = (ctx.config.get("num_workers") if ctx.config else None) or self.num_workers
            n           = len(records)

            print(f"\n  BuildGraphs: {n:,} structures, num_workers={num_workers or 'auto'}",
                  flush=True, file=sys.stderr)

            t0 = _time.perf_counter()
            if num_workers == 1 or n < 100:
                # Serial — avoids ProcessPoolExecutor overhead for small workloads
                normalized = []
                for obj in records:
                    result = _process_one_structure(obj, options)
                    if result is not None:
                        normalized.append(result)
            else:
                normalized = [None] * n
                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = {pool.submit(_process_one_structure, obj, options): i
                               for i, obj in enumerate(records)}
                    for future in as_completed(futures):
                        idx    = futures[future]
                        result = future.result()
                        normalized[idx] = result
                normalized = [r for r in normalized if r is not None]

            elapsed = _time.perf_counter() - t0
            n_edges = sum(r["edge_index"].shape[1] for r in normalized)
            print(
                f"  BuildGraphs: {len(normalized):,} structures, "
                f"{n_edges:,} edges  ({elapsed:.1f}s)",
                flush=True, file=sys.stderr,
            )

            ctx.metadata["normalized"] = normalized
            periodic = any("edge_shifts" in r for r in normalized)
            _report_changes(
                ctx, self.name, records_in=n, records_out=len(normalized), dropped=n - len(normalized),
                edges=int(n_edges), radius_cutoff=radius, max_neighbours=max_nbrs,
                periodic=periodic,
                positions=("wrapped into the cell" if periodic else "centred on the structure mean"),
                atoms_wrapped=int(sum(r.get("n_wrapped", 0) for r in normalized)),
                forces=("z-scored per structure" if options.get("normalize_forces", True)
                        else "unchanged"),
            )

        except Exception as exc:
            logger.error("Error in BuildGraphs: %s", exc)
            ctx.add_error(f"Error in BuildGraphs: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 4: AssemblePyGData
# ---------------------------------------------------------------------------

class AssemblePyGData(_AtomisticStep):
    """
    Assemble per-structure PyG Data objects from the normalized records.

    Node features x = [Z/100, tag]  shape [N, 2]
    Graph target  y = [energy_per_atom]
    Edge index and attributes from BuildGraphs.

    Reads  ctx.metadata["normalized"]
    Writes ctx.metadata["pyg_data"]  (list of torch_geometric.data.Data)
    """

    name        = "assemble_pyg_data"
    stage       = "structure"
    description = "Assemble PyG Data objects with node features, edges, targets"

    def validate(self, ctx) -> bool:
        return "normalized" in ctx.metadata

    def execute(self, ctx):
        try:
            import torch
            from torch_geometric.data import Data

            records      = ctx.metadata["normalized"]
            data_objects = []

            for r in records:
                N   = r["natoms"]
                Z   = r["atomic_numbers"].astype(np.float32) / 100.0
                tag = r["tags"].astype(np.float32)
                x   = np.stack([Z, tag], axis=1)   # [N, 2]

                data = Data(
                    x          = torch.as_tensor(x,              dtype=torch.float32),
                    pos        = torch.as_tensor(r["pos"],        dtype=torch.float32),
                    y          = torch.tensor([r["energy_per_atom"]], dtype=torch.float32),
                    edge_index = torch.as_tensor(r["edge_index"], dtype=torch.long),
                    edge_attr  = torch.as_tensor(r["edge_attr"],  dtype=torch.float32),
                    force      = torch.as_tensor(r["forces"],     dtype=torch.float32),
                    natoms     = torch.tensor([N],                dtype=torch.long),
                )
                data.sid = r["sid"]
                if "edge_shifts" in r:
                    data.cell        = torch.as_tensor(r["cell"], dtype=torch.float32)
                    data.pbc         = torch.tensor([True, True, True])
                    data.edge_shifts = torch.as_tensor(r["edge_shifts"], dtype=torch.float32)
                if "energy" in r:
                    data.energy = torch.tensor([r["energy"]], dtype=torch.float64)
                if "energy_referenced" in r:
                    data.energy_referenced = torch.tensor([r["energy_referenced"]], dtype=torch.float64)
                data_objects.append(data)

            ctx.metadata["pyg_data"]     = data_objects
            ctx.metadata["n_atoms_total"] = sum(int(d.natoms[0]) for d in data_objects)
            _report_changes(ctx, self.name, structures=len(data_objects),
                            atoms=ctx.metadata["n_atoms_total"])
            logger.info(
                "AssemblePyGData: %d Data objects, %d atoms",
                len(data_objects), ctx.metadata["n_atoms_total"],
            )

        except ImportError:
            msg = ("torch and torch_geometric are required for AssemblePyGData.")
            logger.error(msg)
            ctx.add_error(msg)
        except Exception as exc:
            logger.error("Error in AssemblePyGData: %s", exc)
            ctx.add_error(f"Error in AssemblePyGData: {exc}")

        return ctx


# ---------------------------------------------------------------------------
# Step 5: SaveOutput
# ---------------------------------------------------------------------------

class SaveOutput(_AtomisticStep):
    """
    Write output in the requested format and always write index.json.

    FORMAT "npz" (default)
    ~~~~~~~~~~~~~~~~~~~~~~
    Consolidated arrays per split: train.npz / val.npz / test.npz
    Arrays: atomic_numbers, positions, forces, energy_per_atom, natoms,
            split_index, tags

    FORMAT "parquet"
    ~~~~~~~~~~~~~~~~
    One Parquet file per split: train.parquet / val.parquet / test.parquet
    One row per structure. Per-atom fields are list columns (atomic_numbers,
    tags; positions and forces as lists of 3-vectors); per-structure fields
    (energy_per_atom, natoms, cell, energy, ...) are scalar columns.

    FORMAT "pt"
    ~~~~~~~~~~~
    One PyTorch Geometric Data object per structure.
    Files written to <output_dir>/<split>/graphdata_NNNNN.pt

    Split detection
    ~~~~~~~~~~~~~~~
    Uses ctx.metadata["split"] (set by LoadOCPLMDB). If None, uses 80/10/10.
    """

    name        = "save_output"
    stage       = "output"
    description = "Write HydraGNN-ready NPZ/Parquet/PT output + index.json"

    FORMATS = ("npz", "json", "parquet", "pt")

    def __init__(self, output_format: str = "npz", rng_seed: int = 2024):
        self.output_format = output_format.lower()
        self.rng_seed      = rng_seed

    def validate(self, ctx) -> bool:
        return "normalized" in ctx.metadata

    def execute(self, ctx):
        try:
            records   = ctx.metadata["normalized"]
            if self.output_format not in self.FORMATS:
                ctx.add_error(
                    f"output_format={self.output_format!r} is not supported by the "
                    f"atomistic domain; use one of {', '.join(self.FORMATS)}"
                )
                return ctx
            if (self.output_format == "pt"
                    and len(ctx.metadata.get("pyg_data", [])) != len(records)):
                ctx.add_error(
                    "output_format='pt' needs one PyG Data object per structure from "
                    "AssemblePyGData, which requires torch and torch_geometric"
                )
                return ctx
            dataset   = ctx.metadata.get("dataset", "oc22")
            split_key = ctx.metadata.get("split")
            out_root  = Path(ctx.output_dir or "output")
            out_root.mkdir(parents=True, exist_ok=True)

            group_split = (
                split_key is None
                and ctx.metadata.get("split_by") == "group"
                and all("group" in r for r in records)
            )
            if split_key is not None:
                splits_map = {split_key: list(range(len(records)))}
            elif group_split:
                splits_map = self._make_group_splits([r["group"] for r in records])
            else:
                splits_map = self._make_splits(len(records))

            features = ["atomic_numbers", "positions", "forces",
                        "energy_per_atom", "natoms", "split_index", "tags"]
            if self.output_format == "pt":
                self._save_pt(ctx.metadata.get("pyg_data", []), splits_map, out_root)
            elif self.output_format == "parquet":
                features = self._save_parquet(records, splits_map, out_root) or features
            else:
                features = self._save_npz(records, splits_map, out_root) or features

            index = {
                "plugin":        "redi-atomistics",
                "version":       "0.4.0",
                "dataset":       dataset,
                "output_format": self.output_format,
                "n_structures":  len(records),
                "splits": {split: len(idxs) for split, idxs in splits_map.items()},
                "features": features,
            }
            if self.output_format == "json":
                index["hydragnn_native"] = True    # same arrays as npz, flagged for HydraGNN's loader
            if group_split:
                index["split_method"] = "group"
                index["split_groups"] = {
                    split: sorted({records[i]["group"] for i in idxs})
                    for split, idxs in splits_map.items()
                }
            if "atomistic_summary" in ctx.metadata:
                index["atomistic"] = ctx.metadata["atomistic_summary"]
            index_path = out_root / "index.json"
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

            ctx.metadata["hydragnn_index_path"] = str(index_path)
            ctx.metadata["n_structures_out"]    = len(records)
            # Structures per output path, for REDI's metadata.json output manifest
            # (NPZ arrays mix per-atom and per-structure lengths, so it can't infer them)
            out_name = {"pt": "{}/*.pt", "parquet": "{}.parquet"}.get(self.output_format, "{}.npz")
            ctx.metadata["output_records"] = {
                out_name.format(split): len(idxs)
                for split, idxs in splits_map.items() if idxs
            }
            _report_changes(
                ctx, self.name, output_format=self.output_format, structures=len(records),
                splits={split: len(idxs) for split, idxs in splits_map.items()},
                split_method=("by group" if group_split else "preset from path" if split_key
                              else "random 80/10/10"),
            )
            logger.info("Wrote index.json (%d structures)", len(records))

            # The structures are on disk now; drop the in-memory copies so REDI's
            # metadata.json does not serialize every array as text
            for key in ("records", "normalized", "pyg_data"):
                ctx.metadata.pop(key, None)

        except Exception as exc:
            logger.error("Error in SaveOutput: %s", exc)
            ctx.add_error(f"Error in SaveOutput: {exc}")

        return ctx

    def _make_splits(self, n: int, ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[int]]:
        rng     = np.random.default_rng(self.rng_seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_train = int(round(ratios[0] * n))
        n_val   = int(round(ratios[1] * n))
        return {
            "train": indices[:n_train].tolist(),
            "val":   indices[n_train:n_train + n_val].tolist(),
            "test":  indices[n_train + n_val:].tolist(),
        }

    def _make_group_splits(self, groups, ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[int]]:
        """
        Assign whole groups (e.g. trajectories) to splits so correlated structures
        never straddle train/val/test. Groups are shuffled, then filled into train
        until it reaches its share of structures, then val, then test.
        """
        groups = np.asarray(groups)
        uniq   = np.unique(groups)
        np.random.default_rng(self.rng_seed).shuffle(uniq)

        bounds = np.cumsum(ratios)[:2] * len(groups)
        splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        members = {}
        filled  = 0
        for g in uniq:
            idx  = np.flatnonzero(groups == g).tolist()
            name = "train" if filled < bounds[0] else "val" if filled < bounds[1] else "test"
            splits[name].extend(idx)
            members.setdefault(name, []).append((len(idx), g, idx))
            filled += len(idx)

        # With enough groups, never leave val/test empty: move train's smallest group over
        for name in ("test", "val"):
            if not splits[name] and len(members.get("train", [])) > 1:
                size, g, idx = min(members["train"], key=lambda m: m[0])
                members["train"].remove((size, g, idx))
                splits["train"] = [i for i in splits["train"] if groups[i] != g]
                splits[name] = idx

        return {name: sorted(idxs) for name, idxs in splits.items()}

    # Optional per-structure fields written to NPZ when every record has them
    _OPTIONAL_ARRAYS = {
        "cell":              ("cell",              np.float32),
        "energy":            ("energy",            np.float64),
        "energy_referenced": ("energy_referenced", np.float64),
        "max_force":         ("max_force",         np.float32),
        "frame_id":          ("frame_id",          None),
        "group":             ("group",             str),
        "sid":               ("sid",               str),
    }

    def _assemble_arrays(self, records, indices):
        subset = [records[i] for i in indices]
        arrays = {
            "atomic_numbers":  np.concatenate([r["atomic_numbers"] for r in subset]),
            "positions":       np.concatenate([r["pos"]            for r in subset]),
            "forces":          np.concatenate([r["forces"]         for r in subset]),
            "energy_per_atom": np.array([r["energy_per_atom"]      for r in subset], dtype=np.float32),
            "natoms":          np.array([r["natoms"]               for r in subset], dtype=np.int32),
            "split_index":     np.cumsum([0] + [r["natoms"] for r in subset[:-1]], dtype=np.int32),
            "tags":            np.concatenate([r["tags"]            for r in subset]),
        }
        # Extra fields only for LoadAtomTable records (they pass "energy" through),
        # so OC22 output is unchanged
        if all("energy" in r for r in subset):
            for out_name, (key, dtype) in self._OPTIONAL_ARRAYS.items():
                if all(key in r for r in subset):
                    arrays[out_name] = np.array([r[key] for r in subset], dtype=dtype)
        return arrays

    def _save_npz(self, records, splits_map, out_root) -> Optional[List[str]]:
        """Write one NPZ per split; return the array names written."""
        features = None
        for split, indices in splits_map.items():
            if not indices:
                continue
            arrays   = self._assemble_arrays(records, indices)
            out_path = out_root / f"{split}.npz"
            np.savez_compressed(str(out_path), **arrays)
            features = features or list(arrays)
            logger.info("Wrote %s (%d structures)", out_path.name, len(indices))
        return features

    def _save_parquet(self, records, splits_map, out_root) -> Optional[List[str]]:
        """Write one Parquet file per split (one row per structure); return the column names."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("output_format='parquet' requires pyarrow.")

        def per_atom(subset, key, width=None):
            # list<value> (or list<fixed_size_list<value, width>>) with one entry per structure
            arrs    = [np.asarray(r[key]) for r in subset]
            values  = pa.array(np.concatenate([a.reshape(-1) for a in arrs]))
            if width:
                values = pa.FixedSizeListArray.from_arrays(values, width)
            offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64)
            return pa.LargeListArray.from_arrays(pa.array(offsets), values)

        features = None
        for split, indices in splits_map.items():
            if not indices:
                continue
            subset  = [records[i] for i in indices]
            columns = {
                "atomic_numbers":  per_atom(subset, "atomic_numbers"),
                "positions":       per_atom(subset, "pos", width=3),
                "forces":          per_atom(subset, "forces", width=3),
                "tags":            per_atom(subset, "tags"),
                "energy_per_atom": pa.array(np.array([r["energy_per_atom"] for r in subset], dtype=np.float32)),
                "natoms":          pa.array(np.array([r["natoms"] for r in subset], dtype=np.int32)),
            }
            # Same optional per-structure fields as the NPZ output
            if all("energy" in r for r in subset):
                for out_name, (key, dtype) in self._OPTIONAL_ARRAYS.items():
                    if not all(key in r for r in subset):
                        continue
                    if key == "cell":
                        flat = pa.array(np.stack([np.asarray(r["cell"], dtype=np.float32).reshape(9)
                                                  for r in subset]).reshape(-1))
                        columns[out_name] = pa.FixedSizeListArray.from_arrays(
                            pa.FixedSizeListArray.from_arrays(flat, 3), 3)
                    else:
                        vals = np.array([r[key] for r in subset], dtype=dtype)
                        columns[out_name] = pa.array(vals.tolist() if vals.dtype.kind == "U" else vals)
            table    = pa.table(columns)
            out_path = out_root / f"{split}.parquet"
            pq.write_table(table, str(out_path), compression="zstd")
            features = features or table.column_names
            logger.info("Wrote %s (%d structures)", out_path.name, len(indices))
        return features

    def _save_pt(self, data_objects, splits_map, out_root):
        try:
            import torch
        except ImportError:
            raise ImportError("output_format='pt' requires torch and torch_geometric.")

        for split, indices in splits_map.items():
            split_dir = out_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for file_idx, rec_idx in enumerate(indices):
                out_path = split_dir / f"graphdata_{file_idx:05d}.pt"
                import torch as _torch
                _torch.save(data_objects[rec_idx], str(out_path))
            logger.info("Wrote %d PT files to %s/", len(indices), split)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (classes were named after OC22 / HydraGNN)
# ---------------------------------------------------------------------------

LoadOC22LMDB        = LoadOCPLMDB
BuildOC22Graphs     = BuildGraphs
AssembleOC22PyGData = AssemblePyGData
SaveHydraGNNOutput  = SaveOutput


# ---------------------------------------------------------------------------
# Domain interface
# ---------------------------------------------------------------------------

NAME        = "atomistic"
DOMAIN      = "atomistic"
DESCRIPTION = "Atomistic structures (OC22/OC20 LMDB, per-atom Parquet/CSV, HDF5 h5store) → HydraGNN-ready NPZ/PT files"
VERSION     = "0.4.0"


def get_ingest_steps(dataset: str = "oc22"):
    return [LoadOCPLMDB(dataset=dataset), LoadAtomTable(), LoadAtomH5()]


def get_preprocess_steps():
    return [FilterRecords(), FlagForceOutliers()]


def get_transform_steps(num_workers: int = None):
    return [ReferenceEnergies(), BuildGraphs(num_workers=num_workers)]


def get_structure_steps():
    return [AssemblePyGData()]


def get_output_steps(output_format: str = "npz"):
    return [SaveOutput(output_format=output_format)]


def get_full_pipeline(output_format: str = "npz", dataset: str = "oc22"):
    return (
        get_ingest_steps(dataset=dataset)
        + get_preprocess_steps()
        + get_transform_steps()
        + get_structure_steps()
        + get_output_steps(output_format=output_format)
    )
