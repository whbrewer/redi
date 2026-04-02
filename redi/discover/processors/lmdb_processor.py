"""
redi/discover/processors/lmdb_processor.py

Processor for LMDB files used by Open Catalyst Project (OC20/OC22) and HydraGNN.

Each LMDB entry is a pickled object — typically an ASE Atoms object (OC20 raw)
or a PyTorch Geometric Data object (HydraGNN preprocessed).  We sample up to
SAMPLE_SIZE entries to characterise the dataset without reading the whole file.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from redi.discover.processors import (
    BaseProcessor,
    FileDiscovery,
    FileReadError,
    FileParseError,
    PreprocessingStep,
)

SAMPLE_SIZE = 200  # entries to inspect


class LMDBProcessor(BaseProcessor):
    """Processor for LMDB files (OC20/OC22/HydraGNN)."""

    extension = ".lmdb"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            import lmdb
        except ImportError as exc:
            raise FileReadError(
                path=path,
                file_type="LMDB",
                reason="lmdb Python package not available (pip install lmdb)",
                cause=exc,
            ) from exc

        try:
            env = lmdb.open(
                str(path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        except Exception as exc:
            raise FileReadError(
                path=path,
                file_type="LMDB",
                reason=f"Could not open LMDB: {exc}",
                cause=exc,
            ) from exc

        try:
            with env.begin() as txn:
                total_entries = env.stat()["entries"]
                sample = _read_sample(txn, min(SAMPLE_SIZE, total_entries))
        finally:
            env.close()

        if not sample:
            raise FileParseError(
                path=path,
                file_type="LMDB",
                reason="LMDB is empty or contains no decodeable entries",
            )

        schema = _infer_schema(sample)
        metadata = {
            "total_entries": total_entries,
            "sampled_entries": len(sample),
            "schema": schema,
        }

        steps = _build_preprocessing_steps(schema, total_entries)

        return FileDiscovery(
            file_path=str(path),
            file_type="LMDB",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_sample(txn, n: int) -> list[Any]:
    """Read up to *n* entries from an open LMDB transaction."""
    sample = []
    cursor = txn.cursor()
    for i, (_, value) in enumerate(cursor.iternext_dup() if False else cursor.iternext(keys=True, values=True)):
        if i >= n:
            break
        # Skip integer-encoded length keys (some LMDB writers store a "length" entry)
        if len(value) <= 8:
            continue
        try:
            obj = pickle.loads(value)
            sample.append(obj)
        except Exception:
            continue
    return sample


def _infer_schema(sample: list[Any]) -> dict:
    """Infer dataset schema from a list of sampled entries."""
    schema: dict = {
        "entry_type": "unknown",
        "fields": {},
        "atom_counts": [],
        "element_sets": [],
    }

    for obj in sample:
        obj_type = type(obj).__name__
        schema["entry_type"] = obj_type

        if obj_type == "Atoms":
            # ASE Atoms object (OC20 raw format)
            _characterise_ase_atoms(obj, schema)
        elif obj_type == "Data":
            # PyTorch Geometric Data (HydraGNN preprocessed)
            _characterise_pyg_data(obj, schema)
        elif isinstance(obj, dict):
            schema["entry_type"] = "dict"
            for k, v in obj.items():
                if k not in schema["fields"]:
                    schema["fields"][k] = _type_summary(v)
        else:
            # Try attribute inspection
            for attr in ("pos", "x", "z", "energy", "forces", "cell", "natoms"):
                val = getattr(obj, attr, None)
                if val is not None and attr not in schema["fields"]:
                    schema["fields"][attr] = _type_summary(val)

    # Summarise atom count distribution
    if schema["atom_counts"]:
        counts = schema["atom_counts"]
        schema["atom_count_stats"] = {
            "min": int(min(counts)),
            "max": int(max(counts)),
            "mean": round(sum(counts) / len(counts), 1),
        }
    # Unique elements seen
    if schema["element_sets"]:
        all_elements: set = set()
        for s in schema["element_sets"]:
            all_elements.update(s)
        schema["unique_elements"] = sorted(all_elements)

    # Clean up intermediate lists
    del schema["atom_counts"]
    del schema["element_sets"]

    return schema


def _characterise_ase_atoms(atoms: Any, schema: dict) -> None:
    """Extract info from an ASE Atoms object."""
    import numpy as np

    schema["fields"].setdefault("positions", f"float64 ({len(atoms)}, 3)")
    schema["fields"].setdefault("atomic_numbers", f"int ({len(atoms)},)")
    schema["fields"].setdefault("cell", "float64 (3, 3)")
    schema["fields"].setdefault("pbc", "bool (3,)")

    schema["atom_counts"].append(len(atoms))

    try:
        elements = set(atoms.get_chemical_symbols())
        schema["element_sets"].append(elements)
    except Exception:
        pass

    # OC20-specific info stored in atoms.info
    info = getattr(atoms, "info", {}) or {}
    for key in ("energy", "forces", "stress", "fid", "sid", "ad_idx", "bulk_id"):
        if key in info and key not in schema["fields"]:
            schema["fields"][key] = _type_summary(info[key])

    # Calculator results
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        results = getattr(calc, "results", {}) or {}
        for key in ("energy", "forces", "stress"):
            if key in results and key not in schema["fields"]:
                schema["fields"][key] = _type_summary(results[key])


def _characterise_pyg_data(data: Any, schema: dict) -> None:
    """Extract info from a PyTorch Geometric Data object."""
    for attr in data.keys() if hasattr(data, "keys") else []:
        val = getattr(data, attr, None)
        if val is not None and attr not in schema["fields"]:
            schema["fields"][attr] = _type_summary(val)

    if hasattr(data, "pos") and data.pos is not None:
        schema["atom_counts"].append(data.pos.shape[0])

    if hasattr(data, "z") and data.z is not None:
        try:
            schema["element_sets"].append(set(data.z.tolist()))
        except Exception:
            pass


def _type_summary(val: Any) -> str:
    """Short type+shape description of a value."""
    t = type(val).__name__
    if hasattr(val, "shape"):
        return f"{t} {list(val.shape)}"
    if hasattr(val, "__len__"):
        return f"{t} len={len(val)}"
    return f"{t} = {val!r}"[:60]


def _build_preprocessing_steps(schema: dict, total_entries: int) -> list[PreprocessingStep]:
    steps = []
    entry_type = schema.get("entry_type", "unknown")
    fields = schema.get("fields", {})
    stats = schema.get("atom_count_stats", {})
    elements = schema.get("unique_elements", [])

    if entry_type == "Atoms":
        steps.append(PreprocessingStep(
            name="Convert to graph representation",
            rationale=(
                "Raw OC20 entries are ASE Atoms objects — GNN models require "
                "a graph with node features (atomic numbers) and edge features (distances)"
            ),
            suggested_method=(
                "Build radius graph: connect atoms within cutoff radius (~6 Å); "
                "node features = one-hot atomic number; edge features = interatomic distances"
            ),
        ))
    elif entry_type == "Data":
        steps.append(PreprocessingStep(
            name="Verify graph connectivity",
            rationale="Preprocessed PyG Data objects may use fixed cutoffs that need validation",
            suggested_method="Check edge_index is populated; recompute radius graph if cutoff changed",
        ))

    if "energy" in fields or "y" in fields:
        steps.append(PreprocessingStep(
            name="Normalize energy target",
            rationale=(
                "DFT energies are extensive (scale with system size) and span a wide range "
                "across different adsorbate/catalyst combinations"
            ),
            suggested_method=(
                "Subtract per-element reference energies (linear atomization energy); "
                "then Z-score normalize residuals"
            ),
        ))

    if "forces" in fields:
        steps.append(PreprocessingStep(
            name="Normalize forces",
            rationale="Force magnitudes vary widely across configurations; normalizing stabilizes training",
            suggested_method="Z-score normalize forces using dataset mean/std",
        ))

    if stats:
        min_n, max_n = stats.get("min", 0), stats.get("max", 0)
        if max_n - min_n > 5:
            steps.append(PreprocessingStep(
                name="Handle variable system sizes",
                rationale=(
                    f"Atom counts range {min_n}–{max_n} — "
                    "batching requires padding or dynamic batching"
                ),
                suggested_method=(
                    "Use PyG DataLoader with dynamic batching (batch by total atom count); "
                    "avoid fixed-size padding"
                ),
            ))

    steps.append(PreprocessingStep(
        name="Train/val/test split by system ID",
        rationale=(
            "OC20 defines in-distribution (ID) and out-of-distribution (OOD) splits "
            "by catalyst composition and adsorbate identity to test generalization"
        ),
        suggested_method=(
            "Use official OC20 split files (train/val_id/val_ood_ads/val_ood_cat); "
            "do NOT split randomly — this would leak catalyst identity across sets"
        ),
    ))

    return steps
