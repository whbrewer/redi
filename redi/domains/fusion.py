"""
DRAI Fusion Energy Domain Module

Domain-specific preprocessing for fusion/plasma physics data.
Supports tokamak diagnostics and disruption prediction.

Tools: MDSplus, ADIOS, h5py
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ["h5py", "MDSplus", "adios2"]

# Common fusion plasma parameters
PLASMA_PARAMETERS = {
    "magnetic": ["ip", "bt", "bpol", "q95", "li", "kappa", "delta"],
    "thermal": ["te", "ti", "ne", "ni", "pressure", "beta", "betap", "betan"],
    "power": ["pnbi", "pech", "pich", "prad", "ploss", "pohm"],
    "mhd": ["n1", "n2", "n3", "rotating_mode", "locked_mode"],
    "disruption": ["disruption", "disrupt", "vde", "thermal_quench", "current_quench"],
}


def get_ingest_steps():
    """Get fusion-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class LoadFusionHDF5(PipelineStep):
        """Load fusion HDF5 data files"""

        name = "load_fusion_hdf5"
        stage = PipelineStage.INGEST
        description = "Load fusion HDF5 diagnostic data"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "hdf5" and ctx.domain == "fusion"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import h5py
                import numpy as np

                with h5py.File(ctx.input_path, 'r') as f:
                    # Extract structure
                    def get_datasets(group, prefix=""):
                        datasets = {}
                        for key in group.keys():
                            path = f"{prefix}/{key}" if prefix else key
                            if isinstance(group[key], h5py.Dataset):
                                datasets[path] = {
                                    "shape": group[key].shape,
                                    "dtype": str(group[key].dtype)
                                }
                            elif isinstance(group[key], h5py.Group):
                                datasets.update(get_datasets(group[key], path))
                        return datasets

                    structure = get_datasets(f)

                    # Load data into memory
                    data = {}
                    for path in structure.keys():
                        data[path] = f[path][:]

                ctx.data = data
                ctx.metadata["datasets"] = structure
                ctx.metadata["num_signals"] = len(structure)
                logger.info(f"Loaded fusion HDF5: {len(structure)} signals")

            except ImportError:
                ctx.add_error("h5py not installed")
            except Exception as e:
                ctx.add_error(f"Error loading fusion HDF5: {str(e)}")

            return ctx

    class LoadFusionCSV(PipelineStep):
        """Load fusion CSV data (e.g., from D3D, ITER)"""

        name = "load_fusion_csv"
        stage = PipelineStage.INGEST
        description = "Load fusion tabular data"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "csv" and ctx.domain == "fusion"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                df = pd.read_csv(ctx.input_path)
                ctx.data = df

                # Detect shot/pulse column
                shot_keywords = ["shot", "pulse", "discharge"]
                for col in df.columns:
                    if any(kw in col.lower() for kw in shot_keywords):
                        ctx.metadata["shot_column"] = col
                        ctx.metadata["num_shots"] = df[col].nunique()
                        break

                # Detect time column
                time_keywords = ["time", "t_"]
                for col in df.columns:
                    if any(kw in col.lower() for kw in time_keywords):
                        ctx.metadata["time_column"] = col
                        break

                # Detect disruption labels
                for col in df.columns:
                    if any(kw in col.lower() for kw in PLASMA_PARAMETERS["disruption"]):
                        ctx.metadata["disruption_column"] = col
                        break

                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)
                logger.info(f"Loaded fusion CSV: {len(df)} rows")

            except Exception as e:
                ctx.add_error(f"Error loading fusion CSV: {str(e)}")

            return ctx

    return [LoadFusionHDF5(), LoadFusionCSV()]


def get_preprocess_steps():
    """Get fusion-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class AlignSignals(PipelineStep):
        """Align multi-diagnostic signals to common timebase"""

        name = "align_signals"
        stage = PipelineStage.PREPROCESS
        description = "Align signals to common timebase"

        def __init__(self, target_dt: float = 0.001, method: str = "linear"):
            super().__init__()
            self.target_dt = target_dt  # seconds
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                if not isinstance(ctx.data, dict):
                    return ctx

                # Find time arrays
                time_arrays = {}
                for key, arr in ctx.data.items():
                    if "time" in key.lower():
                        time_arrays[key] = arr

                if not time_arrays:
                    logger.warning("No time arrays found for alignment")
                    return ctx

                # Create common timebase
                all_times = np.concatenate(list(time_arrays.values()))
                t_start = np.min(all_times)
                t_end = np.max(all_times)
                common_time = np.arange(t_start, t_end, self.target_dt)

                ctx.data["common_time"] = common_time
                ctx.metadata["aligned_timebase"] = {
                    "dt": self.target_dt,
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "n_points": len(common_time)
                }
                logger.info(f"Created common timebase: {len(common_time)} points")

            except Exception as e:
                ctx.add_error(f"Error aligning signals: {str(e)}")

            return ctx

    class HandleShotBoundaries(PipelineStep):
        """Handle boundaries between plasma shots"""

        name = "handle_shots"
        stage = PipelineStage.PREPROCESS
        description = "Handle shot boundaries in multi-shot data"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                shot_col = ctx.metadata.get("shot_column")
                if not shot_col:
                    return ctx

                # Group by shot
                shots = ctx.data.groupby(shot_col)
                ctx.metadata["shots"] = {
                    "num_shots": shots.ngroups,
                    "samples_per_shot": shots.size().to_dict()
                }
                logger.info(f"Identified {shots.ngroups} shots in data")

            except Exception as e:
                ctx.add_error(f"Error handling shots: {str(e)}")

            return ctx

    return [AlignSignals(), HandleShotBoundaries()]


def get_transform_steps():
    """Get fusion-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class EngineerPlasmaFeatures(PipelineStep):
        """Engineer physics-based plasma features"""

        name = "engineer_plasma_features"
        stage = PipelineStage.TRANSFORM
        description = "Create derived plasma physics features"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                created = []

                # Compute derived quantities if base parameters exist
                cols_lower = {c.lower(): c for c in df.columns}

                # Beta_N = beta * a * B / I_p (normalized beta)
                if "betap" in cols_lower and "ip" in cols_lower:
                    betap_col = cols_lower["betap"]
                    ip_col = cols_lower["ip"]
                    df["derived_betan_proxy"] = df[betap_col] * np.abs(df[ip_col])
                    created.append("derived_betan_proxy")

                # Greenwald fraction
                if "ne" in cols_lower and "ip" in cols_lower:
                    ne_col = cols_lower["ne"]
                    ip_col = cols_lower["ip"]
                    # n_G = I_p / (pi * a^2), approximate
                    df["greenwald_fraction"] = df[ne_col] / (np.abs(df[ip_col]) + 1e-6)
                    created.append("greenwald_fraction")

                # Rate of change features
                time_col = ctx.metadata.get("time_column")
                if time_col:
                    for param_type, params in PLASMA_PARAMETERS.items():
                        for param in params:
                            if param in cols_lower:
                                col = cols_lower[param]
                                df[f"d{param}_dt"] = df[col].diff() / df[time_col].diff()
                                created.append(f"d{param}_dt")

                ctx.data = df
                ctx.metadata["engineered_features"] = created
                logger.info(f"Engineered {len(created)} plasma features")

            except Exception as e:
                ctx.add_error(f"Error engineering features: {str(e)}")

            return ctx

    class LabelDisruptions(PipelineStep):
        """Create disruption warning labels"""

        name = "label_disruptions"
        stage = PipelineStage.TRANSFORM
        description = "Create disruption warning labels"

        def __init__(self, warning_time: float = 0.1):
            super().__init__()
            self.warning_time = warning_time  # seconds before disruption

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                disrupt_col = ctx.metadata.get("disruption_column")
                time_col = ctx.metadata.get("time_column")
                shot_col = ctx.metadata.get("shot_column")

                if not disrupt_col:
                    logger.warning("No disruption column found")
                    return ctx

                # Create warning labels
                df["disruption_warning"] = 0

                if shot_col and time_col:
                    for shot_id in df[shot_col].unique():
                        shot_mask = df[shot_col] == shot_id
                        shot_data = df[shot_mask]

                        if shot_data[disrupt_col].any():
                            # Find disruption time
                            disrupt_time = shot_data[shot_data[disrupt_col] == 1][time_col].min()
                            warning_start = disrupt_time - self.warning_time

                            # Label warning period
                            warning_mask = (
                                (df[shot_col] == shot_id) &
                                (df[time_col] >= warning_start) &
                                (df[time_col] < disrupt_time)
                            )
                            df.loc[warning_mask, "disruption_warning"] = 1

                ctx.data = df
                ctx.metadata["warning_time"] = self.warning_time
                logger.info(f"Created disruption warning labels (t_warn={self.warning_time}s)")

            except Exception as e:
                ctx.add_error(f"Error labeling disruptions: {str(e)}")

            return ctx

    return [EngineerPlasmaFeatures(), LabelDisruptions()]


def get_structure_steps():
    """Get fusion-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CreateSequences(PipelineStep):
        """Create sequences for time series prediction"""

        name = "create_plasma_sequences"
        stage = PipelineStage.STRUCTURE
        description = "Create sequences for disruption prediction"

        def __init__(self, sequence_length: int = 100, prediction_horizon: int = 10):
            super().__init__()
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                shot_col = ctx.metadata.get("shot_column")

                # Get feature columns
                exclude_cols = [shot_col, ctx.metadata.get("time_column"), "disruption_warning"]
                feature_cols = [c for c in df.select_dtypes(include=['number']).columns
                               if c not in exclude_cols]

                sequences = []
                labels = []

                shots = df[shot_col].unique() if shot_col else [None]

                for shot in shots:
                    if shot is not None:
                        shot_data = df[df[shot_col] == shot][feature_cols + ["disruption_warning"]].values
                    else:
                        shot_data = df[feature_cols + ["disruption_warning"]].values

                    for i in range(len(shot_data) - self.sequence_length - self.prediction_horizon):
                        seq = shot_data[i:i + self.sequence_length, :-1]
                        label = shot_data[i + self.sequence_length + self.prediction_horizon - 1, -1]
                        sequences.append(seq)
                        labels.append(label)

                X = np.array(sequences, dtype=np.float32)
                y = np.array(labels, dtype=np.float32)

                ctx.data = {"X": X, "y": y, "feature_names": feature_cols}
                ctx.metadata["sequences"] = {
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                    "n_sequences": len(sequences),
                    "n_features": len(feature_cols)
                }
                logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")

            except Exception as e:
                ctx.add_error(f"Error creating sequences: {str(e)}")

            return ctx

    return [CreateSequences()]


def get_output_steps():
    """Get fusion-specific output steps"""
    from ..pipeline.output import get_output_steps
    return get_output_steps(output_format="hdf5")


def get_full_pipeline():
    """Get the complete fusion preprocessing pipeline"""
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
