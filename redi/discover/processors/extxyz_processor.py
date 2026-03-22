"""
redi/discover/processors/extxyz_processor.py

Processor for extended XYZ (.extxyz) files, including xz-compressed variants
(.extxyz.xz) as produced by the Open Catalyst Project (OC20/OC22) and
commonly used in atomistic machine learning datasets.

Reads a sample of frames to characterize system sizes, chemical species,
available properties (energy, forces, stress), and lattice parameters.
"""

from __future__ import annotations

import io
import os
from collections import Counter
from pathlib import Path

from redi.discover.processors import (
    BaseProcessor,
    FileDiscovery,
    FileReadError,
    FileParseError,
    PreprocessingStep,
)

SAMPLE_FRAMES = 50   # frames to inspect
MAX_LINES_SCAN = 200_000  # stop scanning after this many lines (avoids reading full large file)


class ExtXYZProcessor(BaseProcessor):
    """Processor for extended XYZ files (plain or xz-compressed)."""

    extension = ".extxyz"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            frames, total_frames_est = _read_sample(path)
        except (OSError, IOError) as exc:
            raise FileReadError(path=path, file_type="extXYZ", reason=str(exc), cause=exc) from exc
        except Exception as exc:
            raise FileParseError(path=path, file_type="extXYZ", reason=str(exc), cause=exc) from exc

        if not frames:
            raise FileParseError(path=path, file_type="extXYZ",
                                  reason="No parseable frames found")

        metadata = _summarise_frames(frames, total_frames_est)
        steps = _build_preprocessing_steps(metadata)

        return FileDiscovery(
            file_path=str(path),
            file_type="extXYZ",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )


class CompressedExtXYZProcessor(BaseProcessor):
    """Processor for xz-compressed extended XYZ files (.extxyz.xz or .xz)."""

    extension = ".xz"

    def analyze(self, path: Path) -> FileDiscovery:
        # Only handle .extxyz.xz; delegate other .xz files to generic
        stem = path.stem  # e.g. "0.extxyz"
        if not stem.endswith(".extxyz"):
            from redi.discover.processors import GenericProcessor
            return GenericProcessor().analyze(path)

        try:
            frames, total_frames_est = _read_sample(path, compressed=True)
        except (OSError, IOError) as exc:
            raise FileReadError(path=path, file_type="extXYZ (xz)", reason=str(exc), cause=exc) from exc
        except Exception as exc:
            raise FileParseError(path=path, file_type="extXYZ (xz)", reason=str(exc), cause=exc) from exc

        if not frames:
            raise FileParseError(path=path, file_type="extXYZ (xz)",
                                  reason="No parseable frames found")

        metadata = _summarise_frames(frames, total_frames_est)
        steps = _build_preprocessing_steps(metadata)

        return FileDiscovery(
            file_path=str(path),
            file_type="extXYZ (xz)",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _open_file(path: Path, compressed: bool = False):
    """Return a text-mode file-like object for the given path."""
    if compressed:
        import lzma
        return lzma.open(str(path), "rt")
    return open(path, "r")


def _read_sample(path: Path, compressed: bool = False):
    """Read up to SAMPLE_FRAMES frames from an extXYZ file.

    Returns (list_of_frame_dicts, estimated_total_frames).
    """
    frames = []
    line_count = 0

    with _open_file(path, compressed) as fh:
        while len(frames) < SAMPLE_FRAMES and line_count < MAX_LINES_SCAN:
            frame = _read_one_frame(fh)
            if frame is None:
                break
            frames.append(frame)
            line_count += frame["natoms"] + 2  # natoms line + comment + atom lines

    # Estimate total frames from compressed-size ratio or line scan
    total_est = len(frames)
    if frames:
        avg_atoms = sum(f["natoms"] for f in frames) / len(frames)
        avg_lines_per_frame = avg_atoms + 2
        if compressed:
            # Estimate by decompressed size vs lines read
            compressed_size = path.stat().st_size
            # Rough ratio: extxyz compresses ~10:1 for atomistic data
            est_lines = compressed_size * 10 / 80  # ~80 bytes/line
        else:
            est_lines = path.stat().st_size / 80
        total_est = max(len(frames), int(est_lines / avg_lines_per_frame))

    return frames, total_est


def _read_one_frame(fh) -> dict | None:
    """Parse a single extXYZ frame from an open file handle."""
    # Line 1: number of atoms
    line1 = fh.readline()
    if not line1:
        return None
    line1 = line1.strip()
    if not line1.isdigit():
        return None
    natoms = int(line1)

    # Line 2: comment / properties line
    comment = fh.readline().rstrip("\n")
    props = _parse_comment(comment)

    # Atom lines
    species = []
    for _ in range(natoms):
        atom_line = fh.readline()
        if not atom_line:
            break
        parts = atom_line.split()
        if parts:
            species.append(parts[0])

    frame = {
        "natoms": natoms,
        "species": species,
        "properties": props,
    }
    return frame


def _parse_comment(comment: str) -> dict:
    """Parse key=value pairs from an extXYZ comment line."""
    props: dict = {}
    # Simple tokenizer: handles key="..." and key=value
    import re
    for m in re.finditer(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)', comment):
        key, val = m.group(1), m.group(2).strip('"')
        props[key] = val
    return props


def _summarise_frames(frames: list[dict], total_est: int) -> dict:
    """Build a metadata summary from sampled frames."""
    atom_counts = [f["natoms"] for f in frames]
    all_species: Counter = Counter()
    for f in frames:
        all_species.update(f["species"])

    # Collect scalar properties (energy etc.)
    property_keys: set = set()
    for f in frames:
        property_keys.update(f["properties"].keys())

    # Energy values
    energies = []
    for f in frames:
        e = f["properties"].get("energy") or f["properties"].get("Energy")
        if e is not None:
            try:
                energies.append(float(e))
            except ValueError:
                pass

    # Detect property columns from Properties= field
    col_info = {}
    sample_props_str = frames[0]["properties"].get("Properties", "") if frames else ""
    if sample_props_str:
        col_info = _parse_properties_field(sample_props_str)

    metadata = {
        "estimated_frames": total_est,
        "sampled_frames": len(frames),
        "atom_count_stats": {
            "min": min(atom_counts),
            "max": max(atom_counts),
            "mean": round(sum(atom_counts) / len(atom_counts), 1),
        },
        "unique_elements": sorted(all_species.keys()),
        "element_counts": dict(all_species.most_common(20)),
        "frame_property_keys": sorted(property_keys),
        "column_properties": col_info,
    }

    if energies:
        metadata["energy_stats"] = {
            "min": round(min(energies), 4),
            "max": round(max(energies), 4),
            "mean": round(sum(energies) / len(energies), 4),
        }

    # Detect whether forces are present
    metadata["has_forces"] = "forces" in col_info or any(
        "force" in k.lower() for k in property_keys
    )
    metadata["has_energy"] = bool(energies) or "energy" in {k.lower() for k in property_keys}
    metadata["has_lattice"] = "Lattice" in property_keys or any(
        f["properties"].get("Lattice") for f in frames
    )

    return metadata


def _parse_properties_field(props_str: str) -> dict:
    """Parse extXYZ Properties= field like 'species:S:1:pos:R:3:forces:R:3'."""
    result = {}
    parts = props_str.split(":")
    i = 0
    while i + 2 < len(parts):
        name, dtype, count = parts[i], parts[i + 1], parts[i + 2]
        result[name] = {"dtype": dtype, "ncols": count}
        i += 3
    return result


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------

def _build_preprocessing_steps(metadata: dict) -> list[PreprocessingStep]:
    steps = []
    stats = metadata.get("atom_count_stats", {})
    elements = metadata.get("unique_elements", [])

    steps.append(PreprocessingStep(
        name="Build radius graph",
        rationale=(
            "extXYZ stores atomic positions and a unit cell but no graph connectivity. "
            "GNN models require explicit edges between neighboring atoms."
        ),
        suggested_method=(
            "Connect atoms within a cutoff radius (typically 6 Å for OC20); "
            "node features = atomic number embeddings; "
            "edge features = pairwise distances (and optionally direction vectors)"
        ),
    ))

    if metadata.get("has_energy"):
        steps.append(PreprocessingStep(
            name="Subtract reference energies",
            rationale=(
                "DFT total energies are extensive and element-dependent. "
                "Raw energies are not comparable across different compositions."
            ),
            suggested_method=(
                "Fit per-element linear reference energies on the training set; "
                "subtract to get adsorption energies. OC20 provides reference values "
                "in the dataset metadata."
            ),
        ))

    if metadata.get("has_forces"):
        steps.append(PreprocessingStep(
            name="Normalize forces",
            rationale="Force magnitudes vary widely across frames; normalizing stabilizes training",
            suggested_method="Z-score normalize forces using training-set mean/std per Cartesian component",
        ))

    if stats.get("max", 0) - stats.get("min", 0) > 10:
        steps.append(PreprocessingStep(
            name="Dynamic batching by atom count",
            rationale=(
                f"System sizes range {stats['min']}–{stats['max']} atoms. "
                "Fixed-size batches waste memory on small systems."
            ),
            suggested_method=(
                "Batch by total atom budget (e.g. 1000 atoms/batch) using PyG's "
                "DynamicBatchSampler"
            ),
        ))

    steps.append(PreprocessingStep(
        name="Split by system identity",
        rationale=(
            "OC20 defines in-distribution (ID) and out-of-distribution (OOD) splits "
            "based on catalyst composition and adsorbate identity"
        ),
        suggested_method=(
            "Use official OC20 split metadata (system_id in .txt files); "
            "never split randomly — this leaks catalyst identity across train/val/test"
        ),
    ))

    return steps
