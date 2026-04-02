"""
redi/discover/processors/adios2_processor.py

Processor for ADIOS2 BP directories (.bp).
Extracts variable names, shapes, dtypes, step counts, and scalar values.
"""

from __future__ import annotations

from pathlib import Path

from redi.discover.processors import (
    BaseProcessor,
    FileDiscovery,
    FileReadError,
    PreprocessingStep,
)

# Physics variable groups for XGC-style fusion datasets
_ELECTRON_VARS = {"eden", "e_den", "dpot", "epara", "epsi", "etheta", "e_marker_den",
                   "e_mean_weight", "e_weight_variance", "e_T_para", "e_T_perp"}
_ION_VARS = {"iden", "i_den", "iden", "ipara", "ipsi", "itheta", "i_marker_den",
              "i_mean_weight", "i_weight_variance"}
_FIELD_VARS = {"dBphi", "dBpsi", "dBtheta", "pot0", "pot0m", "shpot"}


class ADIOS2Processor(BaseProcessor):
    """Processor for ADIOS2 BP directories."""

    extension = ".bp"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            import adios2
        except ImportError as exc:
            raise FileReadError(
                path=path,
                file_type="ADIOS2 BP",
                reason="adios2 Python package not available",
                cause=exc,
            ) from exc

        try:
            fr = adios2.FileReader(str(path))
        except Exception as exc:
            raise FileReadError(
                path=path,
                file_type="ADIOS2 BP",
                reason=f"Could not open BP file: {exc}",
                cause=exc,
            ) from exc

        try:
            raw_vars = fr.available_variables()
            raw_attrs = fr.available_attributes()
        finally:
            fr.close()

        # Summarise variables
        variables = {}
        scalar_vars = {}
        max_steps = 1
        for name, info in raw_vars.items():
            shape_str = info.get("Shape", "").strip()
            shape = [int(s) for s in shape_str.split(",") if s.strip()] if shape_str else []
            dtype = info.get("Type", "unknown")
            steps = int(info.get("AvailableStepsCount", 1))
            max_steps = max(max_steps, steps)
            variables[name] = {"shape": shape, "dtype": dtype, "steps": steps}
            if not shape:
                scalar_vars[name] = dtype

        # Infer domain hints
        var_names = set(variables.keys())
        has_electron = bool(var_names & _ELECTRON_VARS)
        has_ion = bool(var_names & _ION_VARS)
        has_field = bool(var_names & _FIELD_VARS)

        # Representative grid shape (most common non-scalar shape)
        grid_shapes = [v["shape"] for v in variables.values() if len(v["shape"]) >= 2]
        grid_shape = grid_shapes[0] if grid_shapes else None

        metadata = {
            "variable_count": len(variables),
            "variables": variables,
            "scalar_variables": list(scalar_vars.keys()),
            "attributes": list(raw_attrs.keys()),
            "steps": max_steps,
            "grid_shape": grid_shape,
            "has_electron_fields": has_electron,
            "has_ion_fields": has_ion,
            "has_electromagnetic_fields": has_field,
        }

        steps_list = _build_preprocessing_steps(variables, grid_shape, has_electron, has_ion)

        return FileDiscovery(
            file_path=str(path),
            file_type="ADIOS2 BP",
            file_size_bytes=_bp_size(path),
            metadata=metadata,
            preprocessing_steps=steps_list,
        )


def _bp_size(path: Path) -> int:
    """Sum size of all files inside a BP directory."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def _build_preprocessing_steps(
    variables: dict,
    grid_shape: list | None,
    has_electron: bool,
    has_ion: bool,
) -> list[PreprocessingStep]:
    steps = []

    steps.append(PreprocessingStep(
        name="Extract field arrays",
        rationale=(
            "BP files store all variables as named arrays — ML models require "
            "explicit tensor assembly from selected physics fields"
        ),
        suggested_method=(
            "Use adios2.FileReader to read selected variables by name; "
            "stack into (steps, nphi, nnode) tensors"
        ),
    ))

    if grid_shape and len(grid_shape) == 2:
        nphi, nnode = grid_shape
        steps.append(PreprocessingStep(
            name="Flatten or reshape poloidal grid",
            rationale=(
                f"Fields have shape ({nphi}, {nnode}) — toroidal planes × poloidal nodes. "
                "GNN models need node-level features; CNN models need a 2-D grid layout."
            ),
            suggested_method=(
                "For GNNs: reshape to (nnode, n_features) using mesh connectivity. "
                "For CNNs: interpolate unstructured mesh onto a regular 2-D grid."
            ),
        ))

    if has_electron and has_ion:
        steps.append(PreprocessingStep(
            name="Assemble multi-species feature matrix",
            rationale=(
                "Dataset contains both electron (e_*) and ion (i_*) fields — "
                "co-locating them is required before model input"
            ),
            suggested_method=(
                "Concatenate electron and ion field arrays along the feature axis "
                "at each mesh node"
            ),
        ))

    steps.append(PreprocessingStep(
        name="Normalize physical units",
        rationale=(
            "Fields (density, potential, temperature) span different physical scales "
            "and units — models converge faster on normalised inputs"
        ),
        suggested_method=(
            "Z-score normalise each field independently using per-variable mean/std "
            "computed over all available timesteps"
        ),
    ))

    steps.append(PreprocessingStep(
        name="Define prediction target",
        rationale=(
            "Fusion surrogate models typically predict disruption onset or field "
            "evolution — a target label or next-step field must be specified"
        ),
        suggested_method=(
            "For classification: attach disruption labels from experiment log. "
            "For regression/forecasting: use the next timestep's dpot or eden as target."
        ),
    ))

    return steps
