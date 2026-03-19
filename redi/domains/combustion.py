"""
REDI Combustion Domain Module

Domain-specific preprocessing for AMReX block-structured AMR combustion data.
Supports AMReX plotfile format (HyperCLaw-V1.1) with multiple refinement levels.

Tools: numpy (native AMReX reader, no external dependencies required)
"""

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


def _read_fab(level_dir: Path, filename: str, byte_offset: int) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Read one AMReX FAB from a binary Cell_D file.

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
        # We know spacedim from the main header but let's infer from count
        remaining = nums[:-1]  # exclude ncomp
        # Find box by looking at last group of ints
        # Try dim=2 first (3*2=6 box ints), then dim=3
        for dim in (2, 3):
            box_ints = remaining[-(3 * dim):]
            lo = box_ints[:dim]
            hi = box_ints[dim:2 * dim]
            nx = [hi[i] - lo[i] + 1 for i in range(dim)]
            ncells = 1
            for n in nx:
                ncells *= n
            total_expected = ncomp * ncells * 8  # float64

            # Read and verify size matches
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
                fab_arr, lo, hi = _read_fab(level_dir, fname, offset)
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

    Returns dict mapping var_name → 2D array at finest resolution.
    """
    hdr = amr_data["header"]
    levels = amr_data["levels"]
    if not levels:
        return {}

    finest = target_level if target_level is not None else hdr["finest_level"]
    finest_level_data = levels[finest]

    # Determine finest domain size from domain box
    dom_lo, dom_hi = hdr["domain_boxes"][finest]
    nx = dom_hi[0] - dom_lo[0] + 1
    ny = dom_hi[1] - dom_lo[1] + 1 if hdr["spacedim"] >= 2 else 1

    var_names = hdr["var_names"]
    composite = {v: np.full((nx, ny), np.nan, dtype=np.float64) for v in var_names}

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
            for var in var_names:
                box_arr = lev_data["data"][var][box_idx]  # shape (bx, by)
                # Map to fine-grid indices
                fine_lo_x = lo[0] * ratio
                fine_hi_x = (hi[0] + 1) * ratio - 1
                fine_lo_y = lo[1] * ratio if hdr["spacedim"] >= 2 else 0
                fine_hi_y = (hi[1] + 1) * ratio - 1 if hdr["spacedim"] >= 2 else 0

                # Clamp to domain
                x0 = max(fine_lo_x - dom_lo[0], 0)
                x1 = min(fine_hi_x - dom_lo[0] + 1, nx)
                y0 = max(fine_lo_y - dom_lo[1], 0) if hdr["spacedim"] >= 2 else 0
                y1 = min(fine_hi_y - dom_lo[1] + 1, ny) if hdr["spacedim"] >= 2 else 1

                # Up-sample using np.repeat
                bx, by = box_arr.shape if box_arr.ndim >= 2 else (box_arr.shape[0], 1)
                upsampled = np.repeat(np.repeat(box_arr.reshape(bx, by), ratio, axis=0), ratio, axis=1)

                # Slice to target region
                region = upsampled[:x1 - x0, :y1 - y0]
                composite[var][x0:x1, y0:y1] = region

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

                # Stack all variables into a single (C, H, W) array
                var_names = list(composite.keys())
                arrays = [composite[v] for v in var_names]
                # Each array is (nx, ny); stack to (nvar, nx, ny)
                X = np.stack(arrays, axis=0).astype(np.float32)
                nan_count = np.isnan(X).sum()
                if nan_count > 0:
                    logger.warning(f"  {nan_count:,} NaN cells in composite (uncovered AMR regions)")
                    X = np.nan_to_num(X, nan=0.0)

                dom_lo, dom_hi = hdr["domain_boxes"][hdr["finest_level"]]
                nx = dom_hi[0] - dom_lo[0] + 1
                ny = dom_hi[1] - dom_lo[1] + 1 if hdr["spacedim"] >= 2 else 1
                dx_fine = hdr["dx_per_level"][hdr["finest_level"]]

                # Coordinate arrays
                x_coords = (np.arange(nx) + 0.5) * dx_fine[0] + hdr["prob_lo"][0]
                y_coords = (np.arange(ny) + 0.5) * dx_fine[1] + hdr["prob_lo"][1] if hdr["spacedim"] >= 2 else np.array([0.0])

                ctx.data = {
                    "X": X,
                    "var_names": var_names,
                    "x_coords": x_coords,
                    "y_coords": y_coords,
                    "time": hdr["time"],
                    "spacedim": hdr["spacedim"],
                    "finest_level": hdr["finest_level"],
                    "num_levels": hdr["num_levels"],
                    "ref_ratios": hdr["ref_ratios"],
                    "prob_lo": hdr["prob_lo"],
                    "prob_hi": hdr["prob_hi"],
                }
                ctx.metadata["output_shape"] = list(X.shape)
                ctx.metadata["var_names"] = var_names
                logger.info(
                    f"Composited to uniform grid: {X.shape} "
                    f"({len(var_names)} vars × {nx}×{ny} cells)"
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
            return isinstance(ctx.data, dict) and "X" in ctx.data

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

    return [SaveCombustionNPZ(), WriteChecksums(), SaveMetadata()]


def get_full_pipeline():
    """Complete AMReX combustion preprocessing pipeline."""
    return (
        get_ingest_steps()
        + get_preprocess_steps()
        + get_transform_steps()
        + get_structure_steps()
        + get_output_steps()
    )
