"""
REDI Combustion Domain Module

Domain-specific preprocessing for AMReX block-structured AMR combustion data.
Supports AMReX plotfile format (HyperCLaw-V1.1) with multiple refinement levels.

Tools: numpy (native AMReX reader, no external dependencies required)
"""

import json
import logging
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# AMReX plotfile signature
AMREX_VERSION_PREFIX = "HyperCLaw"

# Known combustion variables and their physical units
COMBUSTION_VARIABLES = {
    "x_velocity": {"units": "m/s", "category": "velocity"},
    "y_velocity": {"units": "m/s", "category": "velocity"},
    "z_velocity": {"units": "m/s", "category": "velocity"},
    "density": {"units": "kg/m^3", "category": "thermodynamic"},
    "rhoh": {"units": "J/m^3", "category": "thermodynamic"},
    "temp": {"units": "K", "category": "thermodynamic"},
    "rhoRT": {"units": "Pa", "category": "thermodynamic"},
    "RhoRT": {"units": "Pa", "category": "thermodynamic"},
    "divu": {"units": "1/s", "category": "derived"},
    "gradpx": {"units": "Pa/m", "category": "derived"},
    "gradpy": {"units": "Pa/m", "category": "derived"},
    "gradpz": {"units": "Pa/m", "category": "derived"},
    "avg_pressure": {"units": "Pa", "category": "thermodynamic"},
    "mag_vort": {"units": "1/s", "category": "derived"},
}


# ---------------------------------------------------------------------------
# AMReX plotfile reader
# ---------------------------------------------------------------------------

def _parse_box_string(box_str: str) -> Tuple[List[int], List[int]]:
    """Parse AMReX box string like '((0,0) (63,127) (1,1))' → (lo, hi)."""
    nums = [int(x) for x in re.findall(r"-?\d+", box_str)]
    dim = len(nums) // 3
    lo = nums[:dim]
    hi = nums[dim:2 * dim]
    return lo, hi


def _read_plotfile_header(plotfile_dir: Path) -> dict:
    """Parse the main AMReX plotfile Header file."""
    with open(plotfile_dir / "Header") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    idx = 0
    version = lines[idx].strip(); idx += 1
    ncomp = int(lines[idx].strip()); idx += 1
    var_names = []
    for _ in range(ncomp):
        var_names.append(lines[idx].strip()); idx += 1
    spacedim = int(lines[idx].strip()); idx += 1
    time = float(lines[idx].strip()); idx += 1
    finest_level = int(lines[idx].strip()); idx += 1
    num_levels = finest_level + 1

    prob_lo = [float(x) for x in lines[idx].split()]; idx += 1
    prob_hi = [float(x) for x in lines[idx].split()]; idx += 1

    # Refinement ratios (one per level boundary, finest_level values on one line)
    ref_ratios = [int(x) for x in lines[idx].split()]; idx += 1

    # All domain boxes on a single line: "((lo) (hi) (typ)) ((lo) (hi) (typ)) ..."
    boxes_line = lines[idx]; idx += 1
    domain_boxes = []
    for m in re.finditer(r"\(\([^)]+\)[^)]*\([^)]+\)[^)]*\([^)]+\)\)", boxes_line):
        lo, hi = _parse_box_string(m.group(0))
        domain_boxes.append((lo, hi))
    # Fallback: if all boxes not found, try newline-per-box (old format)
    if not domain_boxes:
        lo, hi = _parse_box_string(boxes_line)
        domain_boxes.append((lo, hi))
        for _ in range(num_levels - 1):
            lo, hi = _parse_box_string(lines[idx]); idx += 1
            domain_boxes.append((lo, hi))

    # Steps per level (integers, all on one line)
    _steps = lines[idx].split(); idx += 1

    # Cell sizes per level (spacedim values per line)
    dx_per_level = []
    for _ in range(num_levels):
        dx = [float(x) for x in lines[idx].split()]; idx += 1
        dx_per_level.append(dx)

    # Coordinate system + boundary padding (skip)
    # Remaining lines describe per-level grids but we read those from Cell_H

    return {
        "version": version,
        "ncomp": ncomp,
        "var_names": var_names,
        "spacedim": spacedim,
        "time": time,
        "finest_level": finest_level,
        "num_levels": num_levels,
        "prob_lo": prob_lo,
        "prob_hi": prob_hi,
        "ref_ratios": ref_ratios,
        "domain_boxes": domain_boxes,
        "dx_per_level": dx_per_level,
    }


def _read_level_header(level_dir: Path) -> List[Tuple[str, int]]:
    """
    Parse Level_N/Cell_H to get list of (filename, byte_offset) per FAB.
    Returns list in FAB order.
    """
    with open(level_dir / "Cell_H") as f:
        content = f.read()

    entries = []
    for match in re.finditer(r"FabOnDisk:\s+(\S+)\s+(\d+)", content):
        fname = match.group(1)
        offset = int(match.group(2))
        entries.append((fname, offset))
    return entries


def _read_fab(
    level_dir: Path, filename: str, byte_offset: int, spacedim: Optional[int] = None
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Read one AMReX FAB from a binary Cell_D file.

    Args:
        spacedim: known spatial dimensionality from the plotfile Header, used
            to disambiguate the box parse (tried first, before falling back).

    Returns:
        data:  float64 array of shape (ncomp, *spatial_dims), C-order
        lo:    cell index lower bound
        hi:    cell index upper bound
    """
    filepath = level_dir / filename
    with open(filepath, "rb") as f:
        f.seek(byte_offset)
        # Read ASCII header line (terminated by '\n')
        header_bytes = b""
        while True:
            ch = f.read(1)
            if not ch or ch == b"\n":
                break
            header_bytes += ch
        header_line = header_bytes.decode("ascii", errors="ignore").strip()
        data_start = f.tell()

        # Parse box bounds from header
        # Format: FAB ((...)) (((lo0,lo1) (hi0,hi1) (t0,t1))) ncomp
        nums = [int(x) for x in re.findall(r"-?\d+", header_line)]

        # The FAB header contains: precision/format ints, then box ints, then ncomp
        # Box is 3*dim ints (lo, hi, type), preceded by format descriptors
        # Strategy: parse ncomp (last number) and box (last 3*dim before ncomp)
        ncomp = nums[-1]

        # Infer spacedim from number of remaining ints before ncomp
        # Typical: format section has 18 ints (two format descriptors of 9 each)
        # Then 3*dim ints for the box
        # Total pattern: [18 format ints] [3*dim box ints] [ncomp]
        remaining = nums[:-1]  # exclude ncomp
        # Try the known spacedim first (if given), then fall back to the other option.
        dims_to_try = [d for d in (spacedim, 2, 3) if d in (2, 3)]
        dims_to_try = list(dict.fromkeys(dims_to_try))  # dedupe, preserve order
        for dim in dims_to_try:
            if len(remaining) < 3 * dim:
                continue
            box_ints = remaining[-(3 * dim):]
            lo = box_ints[:dim]
            hi = box_ints[dim:2 * dim]
            nx = [hi[i] - lo[i] + 1 for i in range(dim)]
            if any(n <= 0 for n in nx):
                # Wrong dimensionality guess produced an invalid box; try the next.
                continue
            ncells = 1
            for n in nx:
                ncells *= n
            total_expected = ncomp * ncells * 8  # float64

            # Read and verify size matches (re-seek since a prior failed
            # attempt may have consumed bytes from the wrong dimensionality)
            f.seek(data_start)
            raw = f.read(total_expected)
            if len(raw) == total_expected:
                arr = np.frombuffer(raw, dtype=np.float64)
                # AMReX stores in Fortran (column-major) order per component
                # Reshape to (ncomp, nx[0], nx[1], ...) in Fortran order
                shape_fortran = tuple(nx) + (ncomp,)
                arr_f = arr.reshape(shape_fortran, order="F")
                # Transpose to (ncomp, *spatial) in C order
                axes = (dim,) + tuple(range(dim))
                data = np.ascontiguousarray(arr_f.transpose(axes))
                return data, lo, hi

        raise ValueError(f"Could not determine FAB dimensions from header: {header_line!r}")


def _load_amrex_plotfile(plotfile_dir: Path) -> dict:
    """
    Load all data from an AMReX plotfile directory.

    Returns dict with:
      - "header": parsed header info
      - "levels": list of level dicts, each with:
          - "boxes": list of (lo, hi) cell index tuples
          - "data": dict mapping var_name → list of per-box arrays
          - "dx": cell spacing (list of floats)
    """
    hdr = _read_plotfile_header(plotfile_dir)
    logger.info(
        f"AMReX plotfile: {hdr['num_levels']} levels, {hdr['ncomp']} vars, "
        f"t={hdr['time']:.6g}, domain {hdr['prob_lo']}→{hdr['prob_hi']}"
    )

    levels = []
    for lev in range(hdr["num_levels"]):
        level_dir = plotfile_dir / f"Level_{lev}"
        if not level_dir.is_dir():
            logger.warning(f"Level directory not found: {level_dir}")
            continue

        fab_entries = _read_level_header(level_dir)
        if not fab_entries:
            logger.warning(f"No FabOnDisk entries found in {level_dir}/Cell_H")
            continue

        boxes = []
        # data[var_name] = list of per-box arrays (ncomp dim is squeezed per var)
        var_data: Dict[str, List[np.ndarray]] = {v: [] for v in hdr["var_names"]}

        for fname, offset in fab_entries:
            try:
                fab_arr, lo, hi = _read_fab(level_dir, fname, offset, spacedim=hdr["spacedim"])
                # fab_arr shape: (ncomp, nx, ny[, nz])
                boxes.append((lo, hi))
                for i, var in enumerate(hdr["var_names"]):
                    var_data[var].append(fab_arr[i])
            except Exception as e:
                logger.warning(f"  Skipping FAB {fname}@{offset}: {e}")

        levels.append({
            "level": lev,
            "dx": hdr["dx_per_level"][lev],
            "boxes": boxes,
            "data": var_data,
        })
        nboxes = len(boxes)
        ncells = sum((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) for lo, hi in boxes)
        logger.info(f"  Level {lev}: {nboxes} boxes, ~{ncells:,} cells, dx={hdr['dx_per_level'][lev]}")

    return {"header": hdr, "levels": levels}


def _composite_to_uniform(amr_data: dict, target_level: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Composite all AMR levels into a single uniform grid at the finest level.

    Coarser-level cells are up-sampled (replicated) to fill fine-level cells
    that are not covered by finer data, following standard AMR convention.
    Works for any spatial dimensionality (2D or 3D) using the box's own
    number of axes rather than hard-coding x/y.

    Returns dict mapping var_name → array at finest resolution, shape
    (nx, ny) for 2D or (nx, ny, nz) for 3D data.
    """
    hdr = amr_data["header"]
    levels = amr_data["levels"]
    if not levels:
        return {}

    finest = target_level if target_level is not None else hdr["finest_level"]
    dim = hdr["spacedim"]

    # Determine finest domain shape from domain box
    dom_lo, dom_hi = hdr["domain_boxes"][finest]
    shape = tuple(dom_hi[d] - dom_lo[d] + 1 for d in range(dim))

    var_names = hdr["var_names"]
    composite = {v: np.full(shape, np.nan, dtype=np.float64) for v in var_names}

    # Fill from coarsest to finest so finer data overwrites coarser
    for lev_data in levels:
        lev = lev_data["level"]
        if lev > finest:
            continue
        # Refinement factor from this level to target level
        ratio = 1
        for r in range(lev, finest):
            ratio *= hdr["ref_ratios"][r] if r < len(hdr["ref_ratios"]) else 2

        for box_idx, (lo, hi) in enumerate(lev_data["boxes"]):
            fine_lo = [lo[d] * ratio for d in range(dim)]
            fine_hi = [(hi[d] + 1) * ratio - 1 for d in range(dim)]

            # Clamp to domain
            starts = [max(fine_lo[d] - dom_lo[d], 0) for d in range(dim)]
            stops = [min(fine_hi[d] - dom_lo[d] + 1, shape[d]) for d in range(dim)]
            if any(stops[d] <= starts[d] for d in range(dim)):
                continue

            target_slices = tuple(slice(starts[d], stops[d]) for d in range(dim))
            source_slices = tuple(slice(0, stops[d] - starts[d]) for d in range(dim))

            for var in var_names:
                box_arr = lev_data["data"][var][box_idx]  # shape (bx0, bx1, ...)
                upsampled = box_arr
                if ratio > 1:
                    for d in range(dim):
                        upsampled = np.repeat(upsampled, ratio, axis=d)
                composite[var][target_slices] = upsampled[source_slices]

    return composite


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_ingest_steps():
    """AMReX combustion ingest steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.ingest import DetectFormat

    class LoadAMReXPlotfile(PipelineStep):
        """Load an AMReX block-structured AMR plotfile directory."""

        name = "load_amrex_plotfile"
        stage = PipelineStage.INGEST
        description = "Load AMReX plotfile with multi-level AMR data"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "amrex-plotfile"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            plotfile_dir = ctx.input_path
            try:
                amr_data = _load_amrex_plotfile(plotfile_dir)
                ctx.data = amr_data
                hdr = amr_data["header"]

                ctx.metadata.update({
                    "amrex_version": hdr["version"],
                    "num_levels": hdr["num_levels"],
                    "finest_level": hdr["finest_level"],
                    "ref_ratios": hdr["ref_ratios"],
                    "var_names": hdr["var_names"],
                    "ncomp": hdr["ncomp"],
                    "spacedim": hdr["spacedim"],
                    "time": hdr["time"],
                    "prob_lo": hdr["prob_lo"],
                    "prob_hi": hdr["prob_hi"],
                    "dx_per_level": hdr["dx_per_level"],
                    "num_boxes_per_level": [len(lv["boxes"]) for lv in amr_data["levels"]],
                })

                # Species mass fractions
                species = [v for v in hdr["var_names"] if v.startswith("Y(")]
                ctx.metadata["species"] = species

                logger.info(
                    f"Loaded AMReX plotfile: {hdr['num_levels']} levels, "
                    f"{hdr['ncomp']} vars, species={species}"
                )

            except Exception as e:
                ctx.add_error(f"Error loading AMReX plotfile: {e}")

            return ctx

    return [DetectFormat(), LoadAMReXPlotfile()]


def get_preprocess_steps():
    """AMReX combustion preprocess steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class ClipSpeciesFractions(PipelineStep):
        """Clip species mass fractions to physical range [0, 1]."""

        name = "clip_species_fractions"
        stage = PipelineStage.PREPROCESS
        description = "Clip Y(species) mass fractions to [0, 1]"

        def validate(self, ctx: PipelineContext) -> bool:
            return isinstance(ctx.data, dict) and "levels" in ctx.data

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            species = ctx.metadata.get("species", [])
            if not species:
                return ctx
            try:
                for lev_data in ctx.data["levels"]:
                    for sp in species:
                        if sp in lev_data["data"]:
                            for i, arr in enumerate(lev_data["data"][sp]):
                                lev_data["data"][sp][i] = np.clip(arr, 0.0, 1.0)
                logger.info(f"Clipped {len(species)} species fractions to [0, 1]")
            except Exception as e:
                ctx.add_error(f"Error clipping species fractions: {e}")
            return ctx

    class ValidatePhysics(PipelineStep):
        """Validate basic physical consistency of combustion fields."""

        name = "validate_physics"
        stage = PipelineStage.PREPROCESS
        description = "Check physical plausibility of combustion fields"

        def validate(self, ctx: PipelineContext) -> bool:
            return isinstance(ctx.data, dict) and "levels" in ctx.data

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                warnings = []
                for lev_data in ctx.data["levels"]:
                    lev = lev_data["level"]
                    for var in ("density", "temp"):
                        if var in lev_data["data"]:
                            arrs = lev_data["data"][var]
                            vals = np.concatenate([a.ravel() for a in arrs])
                            if np.any(vals <= 0):
                                warnings.append(f"Level {lev}: non-positive values in {var}")
                if warnings:
                    ctx.metadata["physics_warnings"] = warnings
                    for w in warnings:
                        logger.warning(f"  Physics: {w}")
                else:
                    logger.info("Physics validation passed")
            except Exception as e:
                logger.warning(f"Physics validation failed: {e}")
            return ctx

    return [ClipSpeciesFractions(), ValidatePhysics()]


def get_transform_steps():
    """AMReX combustion transform steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class DeriveCombustionFeatures(PipelineStep):
        """Derive combustion-specific fields from base variables."""

        name = "derive_combustion_features"
        stage = PipelineStage.TRANSFORM
        description = "Derive vorticity, mixture fraction, heat release proxy"

        def validate(self, ctx: PipelineContext) -> bool:
            return isinstance(ctx.data, dict) and "levels" in ctx.data

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                derive_cfg = ctx.config.get("combustion", {}).get("derive", {})
                do_vorticity = derive_cfg.get("vorticity", True)
                do_fuel_proxy = derive_cfg.get("fuel_proxy", True)

                derived_names = []
                for lev_data in ctx.data["levels"]:
                    vnames = ctx.data["header"]["var_names"]
                    dx = lev_data["dx"]

                    for box_idx in range(len(lev_data["boxes"])):
                        # Mixture fraction: Z = Y(fuel) / (Y(fuel) + Y(oxidizer))
                        # Using O2 as oxidizer proxy; fuel = 1 - Y(O2) - Y(N2)
                        if do_fuel_proxy and "Y(O2)" in lev_data["data"] and "Y(N2)" in lev_data["data"]:
                            yo2 = lev_data["data"]["Y(O2)"][box_idx]
                            yn2 = lev_data["data"]["Y(N2)"][box_idx]
                            y_fuel = np.clip(1.0 - yo2 - yn2, 0.0, 1.0)
                            if "Y(fuel_proxy)" not in lev_data["data"]:
                                lev_data["data"]["Y(fuel_proxy)"] = []
                            lev_data["data"]["Y(fuel_proxy)"].append(y_fuel)
                            if "Y(fuel_proxy)" not in derived_names:
                                derived_names.append("Y(fuel_proxy)")

                        # Vorticity magnitude (2D: ∂v/∂x - ∂u/∂y)
                        if do_vorticity and "x_velocity" in lev_data["data"] and \
                                "y_velocity" in lev_data["data"] and \
                                len(dx) >= 2:
                            u = lev_data["data"]["x_velocity"][box_idx]
                            v = lev_data["data"]["y_velocity"][box_idx]
                            if u.ndim == 2 and v.ndim == 2:
                                dvdx = np.gradient(v, dx[0], axis=0)
                                dudy = np.gradient(u, dx[1], axis=1)
                                vort = dvdx - dudy
                                if "vorticity" not in lev_data["data"]:
                                    lev_data["data"]["vorticity"] = []
                                lev_data["data"]["vorticity"].append(vort)
                                if "vorticity" not in derived_names:
                                    derived_names.append("vorticity")

                if derived_names:
                    ctx.metadata["derived_fields"] = derived_names
                    # Register derived fields so compositor picks them up
                    ctx.data["header"]["var_names"] = (
                        ctx.data["header"]["var_names"] + derived_names
                    )
                    logger.info(f"Derived combustion fields: {derived_names}")
            except Exception as e:
                ctx.add_error(f"Error deriving combustion features: {e}")
            return ctx

    class NormalizeCombustionFields(PipelineStep):
        """Normalize combustion fields by reference physical scales."""

        name = "normalize_combustion_fields"
        stage = PipelineStage.TRANSFORM
        description = "Normalize fields by physical reference scales"

        # Reference scales for common combustion variables
        REF_SCALES = {
            "x_velocity": 10.0,        # m/s (laminar flame speed ~0.4, allow up to 10)
            "y_velocity": 10.0,
            "z_velocity": 10.0,
            "density": 1.2,            # kg/m^3 (air at STP)
            "temp": 2000.0,            # K (adiabatic flame temperature ~2000 K)
            "avg_pressure": 101325.0,  # Pa (1 atm)
            "mag_vort": 1000.0,        # 1/s
            "divu": 1000.0,            # 1/s
        }

        def validate(self, ctx: PipelineContext) -> bool:
            return isinstance(ctx.data, dict) and "levels" in ctx.data

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                norm_cfg = ctx.config.get("combustion", {}).get("normalize", {})
                if not norm_cfg.get("enabled", True):
                    logger.info("Normalization disabled by config")
                    return ctx

                # Merge config scale overrides with defaults
                scales = dict(self.REF_SCALES)
                scales.update(norm_cfg.get("scales", {}))

                normalized = []
                for lev_data in ctx.data["levels"]:
                    for var, scale in scales.items():
                        if var in lev_data["data"]:
                            lev_data["data"][var] = [
                                arr / scale for arr in lev_data["data"][var]
                            ]
                            if var not in normalized:
                                normalized.append(var)
                ctx.metadata["normalization"] = {
                    "method": "reference_scale",
                    "scales": scales,
                    "normalized_vars": normalized,
                }
                logger.info(f"Normalized {len(normalized)} combustion fields")
            except Exception as e:
                ctx.add_error(f"Error normalizing fields: {e}")
            return ctx

    return [DeriveCombustionFeatures(), NormalizeCombustionFields()]


def get_structure_steps():
    """AMReX combustion structure steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CompositeAMRLevels(PipelineStep):
        """
        Composite all AMR levels into a uniform grid at the finest resolution.

        Produces a single 2D array per variable for ML consumption.
        Skipped when combustion.structure_mode is 'per_level'.
        """

        name = "composite_amr_levels"
        stage = PipelineStage.STRUCTURE
        description = "Composite AMR levels to uniform finest-level grid"

        def validate(self, ctx: PipelineContext) -> bool:
            if not (isinstance(ctx.data, dict) and "levels" in ctx.data):
                return False
            mode = ctx.config.get("combustion", {}).get("structure_mode", "composite")
            return mode != "per_level"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                hdr = ctx.data["header"]
                comb_cfg = ctx.config.get("combustion", {})
                target_level = comb_cfg.get("target_level", None)
                composite = _composite_to_uniform(ctx.data, target_level=target_level)

                if not composite:
                    ctx.add_error("Compositing produced no data")
                    return ctx

                # Stack all variables into a single (nvar, *spatial) array
                var_names = list(composite.keys())
                arrays = [composite[v] for v in var_names]
                X = np.stack(arrays, axis=0).astype(np.float32)
                nan_count = np.isnan(X).sum()
                if nan_count > 0:
                    logger.warning(f"  {nan_count:,} NaN cells in composite (uncovered AMR regions)")
                    X = np.nan_to_num(X, nan=0.0)

                dim = hdr["spacedim"]
                dom_lo, dom_hi = hdr["domain_boxes"][hdr["finest_level"]]
                grid_shape = [dom_hi[d] - dom_lo[d] + 1 for d in range(dim)]
                dx_fine = hdr["dx_per_level"][hdr["finest_level"]]

                # Coordinate arrays, one per spatial axis
                axis_names = ["x_coords", "y_coords", "z_coords"]
                coords = {
                    axis_names[d]: (np.arange(grid_shape[d]) + 0.5) * dx_fine[d] + hdr["prob_lo"][d]
                    for d in range(dim)
                }

                ctx.data = {
                    "X": X,
                    "var_names": var_names,
                    "time": hdr["time"],
                    "spacedim": hdr["spacedim"],
                    "finest_level": hdr["finest_level"],
                    "num_levels": hdr["num_levels"],
                    "ref_ratios": hdr["ref_ratios"],
                    "prob_lo": hdr["prob_lo"],
                    "prob_hi": hdr["prob_hi"],
                    **coords,
                }
                ctx.metadata["output_shape"] = list(X.shape)
                ctx.metadata["var_names"] = var_names
                grid_str = "×".join(str(n) for n in grid_shape)
                logger.info(
                    f"Composited to uniform grid: {X.shape} "
                    f"({len(var_names)} vars × {grid_str} cells)"
                )
            except Exception as e:
                ctx.add_error(f"Error compositing AMR levels: {e}")
            return ctx

    class PackPerLevelArrays(PipelineStep):
        """
        Alternative to compositing: keep each AMR level as a separate array.

        Useful for multi-resolution ML models (U-Net, FNO with AMR).
        Runs only when CompositeAMRLevels is skipped (data still a dict with 'levels').
        """

        name = "pack_per_level_arrays"
        stage = PipelineStage.STRUCTURE
        description = "Pack each AMR level into a separate stacked array"

        def validate(self, ctx: PipelineContext) -> bool:
            if not (isinstance(ctx.data, dict) and "levels" in ctx.data):
                return False
            mode = ctx.config.get("combustion", {}).get("structure_mode", "composite")
            return mode == "per_level"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                hdr = ctx.data["header"]
                var_names = hdr["var_names"]
                packed_levels = {}

                for lev_data in ctx.data["levels"]:
                    lev = lev_data["level"]
                    # Stack boxes for each variable into a list of arrays
                    level_vars = {}
                    for var in var_names:
                        if var in lev_data["data"] and lev_data["data"][var]:
                            level_vars[var] = lev_data["data"][var]  # list of box arrays
                    packed_levels[f"level_{lev}"] = {
                        "data": level_vars,
                        "boxes": lev_data["boxes"],
                        "dx": lev_data["dx"],
                    }

                ctx.data = {
                    "levels": packed_levels,
                    "var_names": var_names,
                    "time": hdr["time"],
                    "spacedim": hdr["spacedim"],
                    "num_levels": hdr["num_levels"],
                    "ref_ratios": hdr["ref_ratios"],
                    "prob_lo": hdr["prob_lo"],
                    "prob_hi": hdr["prob_hi"],
                }
                ctx.metadata["structure_mode"] = "per_level"
                ctx.metadata["var_names"] = var_names
                logger.info(f"Packed {len(packed_levels)} AMR levels (per-level mode)")
            except Exception as e:
                ctx.add_error(f"Error packing per-level arrays: {e}")
            return ctx

    return [CompositeAMRLevels(), PackPerLevelArrays()]


def get_output_steps():
    """AMReX combustion output steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.output import WriteChecksums, SaveMetadata

    class SaveCombustionNPZ(PipelineStep):
        """
        Save composited AMR field to a single NPZ file.

        Each plotfile snapshot is one complete sample; no train/val/test split
        is applied here. When processing a time series of plotfiles, callers
        should stack snapshots and split at a higher level.
        """

        name = "save_combustion_npz"
        stage = PipelineStage.OUTPUT
        description = "Save AMR field arrays to NPZ"

        def validate(self, ctx: PipelineContext) -> bool:
            return not ctx.is_multi_file and isinstance(ctx.data, dict) and "X" in ctx.data

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                output_dir = ctx.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)

                save_kwargs = {}
                for key, val in ctx.data.items():
                    if isinstance(val, np.ndarray):
                        save_kwargs[key] = val
                    elif isinstance(val, (list, tuple)):
                        try:
                            save_kwargs[key] = np.array(val)
                        except Exception:
                            save_kwargs[key] = np.array(val, dtype=object)
                    elif isinstance(val, (int, float, str)):
                        save_kwargs[key] = np.array(val)

                out_path = output_dir / "snapshot.npz"
                np.savez_compressed(out_path, **save_kwargs)
                ctx.metadata["output_files"] = [str(out_path)]
                logger.info(f"Saved AMReX snapshot: {out_path} ({out_path.stat().st_size // 1024} KB)")

            except Exception as e:
                ctx.add_error(f"Error saving combustion NPZ: {e}")
            return ctx

    class SavePerLevelNPZ(PipelineStep):
        """
        Save per-level AMR field arrays (structure_mode='per_level') to NPZ.

        Box arrays for a given level/variable are stacked into one array when
        every box has the same shape (the common case for a fixed AMReX grid
        layout); otherwise they're saved as a ragged object array.
        """

        name = "save_per_level_npz"
        stage = PipelineStage.OUTPUT
        description = "Save per-level AMR field arrays to NPZ"

        def validate(self, ctx: PipelineContext) -> bool:
            return (
                not ctx.is_multi_file
                and isinstance(ctx.data, dict)
                and isinstance(ctx.data.get("levels"), dict)
            )

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                output_dir = ctx.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)

                save_kwargs = {}
                for level_key, level_info in ctx.data["levels"].items():
                    level_vars = level_info.get("data", {})
                    boxes = level_info.get("boxes", [])

                    for var, box_arrays in level_vars.items():
                        key = f"{level_key}__{var}"
                        shapes = {arr.shape for arr in box_arrays}
                        if len(shapes) == 1:
                            save_kwargs[key] = np.stack(box_arrays, axis=0)
                        else:
                            obj_arr = np.empty(len(box_arrays), dtype=object)
                            for i, arr in enumerate(box_arrays):
                                obj_arr[i] = arr
                            save_kwargs[key] = obj_arr

                    save_kwargs[f"{level_key}__box_lo"] = np.array([b[0] for b in boxes])
                    save_kwargs[f"{level_key}__box_hi"] = np.array([b[1] for b in boxes])
                    save_kwargs[f"{level_key}__dx"] = np.array(level_info.get("dx", []))

                save_kwargs["var_names"] = np.array(ctx.data.get("var_names", []))
                save_kwargs["time"] = np.array(ctx.data.get("time", 0.0))
                save_kwargs["spacedim"] = np.array(ctx.data.get("spacedim", 0))
                save_kwargs["num_levels"] = np.array(ctx.data.get("num_levels", 0))
                save_kwargs["ref_ratios"] = np.array(ctx.data.get("ref_ratios", []))
                save_kwargs["prob_lo"] = np.array(ctx.data.get("prob_lo", []))
                save_kwargs["prob_hi"] = np.array(ctx.data.get("prob_hi", []))

                out_path = output_dir / "snapshot.npz"
                np.savez_compressed(out_path, **save_kwargs)
                ctx.metadata["output_files"] = [str(out_path)]
                logger.info(f"Saved per-level AMReX snapshot: {out_path} ({out_path.stat().st_size // 1024} KB)")

            except Exception as e:
                ctx.add_error(f"Error saving per-level combustion NPZ: {e}")
            return ctx

    class SaveTimeSeriesHDF5(PipelineStep):
        """
        Consolidate a time series of AMR plotfiles into one HDF5 file with
        full/train/val/test groups.

        Activates only when multiple plotfile paths are passed in (e.g.
        `redi run plt00000 plt00100 ... -d combustion`). Each snapshot is run
        through the domain's own ingest->preprocess->transform->structure
        steps (reused as-is, so field values match single-file processing),
        forced to structure_mode='composite' so every snapshot lands on the
        same fixed-shape uniform grid -- AMReX regrids its refined levels as
        the flame moves (box counts genuinely differ run to run), so raw
        per-box arrays can't be stacked across time; compositing to the
        finest-level grid is what makes a shared tensor shape possible.

        Snapshots are stacked into a single (time, var, *spatial) tensor.
        Splits are chronological, not shuffled: consecutive plotfiles are
        states of one continuously evolving flow, so a random split would
        scatter near-duplicate neighboring timesteps across train/val/test
        and leak information between them.
        """

        name = "save_time_series_hdf5"
        stage = PipelineStage.OUTPUT
        description = "Stack AMR snapshots across time; save full/train/val/test HDF5"

        def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
            super().__init__()
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.is_multi_file and ctx.metadata.get("file_format") == "amrex-plotfile"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import h5py
            except ImportError:
                from ..pipeline import missing_dependency_error
                ctx.add_error(missing_dependency_error("h5py", "hdf5"))
                return ctx

            from ..pipeline import create_context

            try:
                split_cfg = ctx.config.get("split", {})
                if not isinstance(split_cfg, dict):
                    split_cfg = {}
                train_ratio = split_cfg.get("train_ratio", self.train_ratio)
                val_ratio = split_cfg.get("val_ratio", self.val_ratio)
                test_ratio = split_cfg.get("test_ratio", self.test_ratio)

                snapshot_steps = (
                    get_ingest_steps() + get_preprocess_steps()
                    + get_transform_steps() + get_structure_steps()
                )
                # Compositing (not per-box) is required so every snapshot shares
                # one fixed grid shape, regardless of AMR regridding over time.
                snap_combustion_cfg = dict(ctx.config.get("combustion", {}))
                snap_combustion_cfg["structure_mode"] = "composite"

                snapshots = []
                for path in ctx.input_paths:
                    snap_ctx = create_context(
                        input_path=str(path),
                        output_dir=str(ctx.output_dir),
                        domain="combustion",
                        combustion=snap_combustion_cfg,
                    )
                    for step in snapshot_steps:
                        if step.validate(snap_ctx):
                            snap_ctx = step.execute(snap_ctx)
                    if snap_ctx.errors:
                        ctx.add_error(f"Snapshot {path.name} failed: {snap_ctx.errors}")
                        continue
                    if not (isinstance(snap_ctx.data, dict) and "X" in snap_ctx.data):
                        ctx.add_error(f"Snapshot {path.name} produced no composited data")
                        continue
                    snapshots.append(snap_ctx.data)

                if not snapshots:
                    ctx.add_error("No snapshots loaded for time-series consolidation")
                    return ctx

                # Chronological order (physical time), independent of path/glob order
                snapshots.sort(key=lambda s: s.get("time", 0.0))
                n_time = len(snapshots)
                times = np.array([s.get("time", 0.0) for s in snapshots])

                ref_shape = snapshots[0]["X"].shape
                mismatched = [i for i, s in enumerate(snapshots) if s["X"].shape != ref_shape]
                if mismatched:
                    ctx.add_error(
                        f"Composited grid shape differs across snapshots "
                        f"(e.g. domain size changed): {ref_shape} vs "
                        f"{snapshots[mismatched[0]]['X'].shape}"
                    )
                    return ctx

                X_all = np.stack([s["X"] for s in snapshots], axis=0)  # (n_time, nvar, *spatial)

                consolidated: Dict[str, np.ndarray] = {"X": X_all, "times": times}
                for coord_key in ("x_coords", "y_coords", "z_coords"):
                    if coord_key in snapshots[0]:
                        consolidated[coord_key] = snapshots[0][coord_key]

                # Chronological split (no shuffling -- see class docstring)
                train_end = min(int(round(n_time * train_ratio)), n_time)
                val_end = min(max(train_end + int(round(n_time * val_ratio)), train_end), n_time)
                indices = np.arange(n_time)
                split_indices = {
                    "train": indices[:train_end],
                    "val": indices[train_end:val_end],
                    "test": indices[val_end:],
                }
                for split_name, idx in split_indices.items():
                    if len(idx) == 0:
                        logger.warning(f"  Split '{split_name}' has 0 snapshots (only {n_time} snapshots total)")

                def make_group(idx: np.ndarray) -> Dict[str, np.ndarray]:
                    g = {}
                    for k, v in consolidated.items():
                        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_time:
                            g[k] = v[idx]
                        else:
                            g[k] = v
                    return g

                groups = {"full": consolidated}
                groups.update({name: make_group(idx) for name, idx in split_indices.items()})

                output_dir = ctx.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / "dataset.h5"

                with h5py.File(out_path, "w") as f:
                    f.attrs["n_snapshots"] = n_time
                    f.attrs["var_names"] = json.dumps(snapshots[0].get("var_names", []))
                    f.attrs["spacedim"] = snapshots[0].get("spacedim", 0)
                    f.attrs["ref_ratios"] = json.dumps(snapshots[0].get("ref_ratios", []))
                    f.attrs["prob_lo"] = json.dumps(snapshots[0].get("prob_lo", []))
                    f.attrs["prob_hi"] = json.dumps(snapshots[0].get("prob_hi", []))

                    for group_name, group_data in groups.items():
                        grp = f.create_group(group_name)
                        for k, v in group_data.items():
                            if v.size == 0:
                                continue
                            kw = {} if v.ndim == 0 else {"compression": "gzip", "chunks": True}
                            grp.create_dataset(k, data=v, **kw)

                ctx.metadata["output_files"] = [str(out_path)]
                ctx.metadata["time_series"] = {
                    "n_snapshots": n_time,
                    "tensor_shape": list(X_all.shape),
                    "times": times.tolist(),
                    "split": {
                        name: {
                            "n_snapshots": len(idx),
                            "time_range": [float(times[idx[0]]), float(times[idx[-1]])] if len(idx) else None,
                        }
                        for name, idx in split_indices.items()
                    },
                }
                logger.info(
                    f"Saved consolidated time-series HDF5: {out_path} "
                    f"(X: {X_all.shape}; train={len(split_indices['train'])}, "
                    f"val={len(split_indices['val'])}, test={len(split_indices['test'])})"
                )

            except Exception as e:
                ctx.add_error(f"Error consolidating time series: {e}")
            return ctx

    return [SaveCombustionNPZ(), SavePerLevelNPZ(), SaveTimeSeriesHDF5(), WriteChecksums(), SaveMetadata()]


def get_full_pipeline():
    """Complete AMReX combustion preprocessing pipeline."""
    return (
        get_ingest_steps()
        + get_preprocess_steps()
        + get_transform_steps()
        + get_structure_steps()
        + get_output_steps()
    )
