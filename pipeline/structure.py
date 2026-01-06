"""
DRAI Pipeline - Structure Stage

Handles tensorization, sequence formatting, and graph construction.
Domain-specific structuring is delegated to domain modules.

Operations:
- Convert to tensors
- Create sequences (for time series, NLP)
- Build graphs (for materials, molecules)
- Create appropriate data structures for ML
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from . import PipelineStep, PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


class ToNumPyArrays(PipelineStep):
    """Convert DataFrame to NumPy arrays"""

    name = "to_numpy"
    stage = PipelineStage.STRUCTURE
    description = "Convert tabular data to NumPy arrays"

    def __init__(
        self,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ):
        super().__init__()
        self.target_column = target_column
        self.feature_columns = feature_columns

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            # Determine features and target
            if self.feature_columns:
                feature_cols = [c for c in self.feature_columns if c in ctx.data.columns]
            else:
                feature_cols = ctx.data.select_dtypes(include=['number']).columns.tolist()
                if self.target_column and self.target_column in feature_cols:
                    feature_cols.remove(self.target_column)

            X = ctx.data[feature_cols].values.astype(np.float32)

            if self.target_column and self.target_column in ctx.data.columns:
                y = ctx.data[self.target_column].values
                ctx.data = {"X": X, "y": y, "feature_names": feature_cols}
            else:
                ctx.data = {"X": X, "feature_names": feature_cols}

            ctx.metadata["numpy_structure"] = {
                "X_shape": X.shape,
                "feature_columns": feature_cols,
                "target_column": self.target_column
            }
            logger.info(f"Converted to NumPy: X shape {X.shape}")

        except Exception as e:
            ctx.add_error(f"Error converting to NumPy: {str(e)}")
        return ctx


class ToTorchTensors(PipelineStep):
    """Convert to PyTorch tensors"""

    name = "to_torch"
    stage = PipelineStage.STRUCTURE
    description = "Convert arrays to PyTorch tensors"

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import torch
            import numpy as np

            if isinstance(ctx.data, dict) and "X" in ctx.data:
                X = torch.tensor(ctx.data["X"], dtype=torch.float32, device=self.device)
                result = {"X": X, "feature_names": ctx.data.get("feature_names", [])}

                if "y" in ctx.data:
                    y = ctx.data["y"]
                    # Determine dtype for y
                    if np.issubdtype(y.dtype, np.integer):
                        y = torch.tensor(y, dtype=torch.long, device=self.device)
                    else:
                        y = torch.tensor(y, dtype=torch.float32, device=self.device)
                    result["y"] = y

                ctx.data = result

            ctx.metadata["torch_device"] = self.device
            logger.info(f"Converted to PyTorch tensors on {self.device}")

        except ImportError:
            ctx.add_error("PyTorch not installed")
        except Exception as e:
            ctx.add_error(f"Error converting to PyTorch: {str(e)}")
        return ctx


class CreateSequences(PipelineStep):
    """Create sequences for time series or sequential data"""

    name = "create_sequences"
    stage = PipelineStage.STRUCTURE
    description = "Create sequences for time series modeling"

    def __init__(
        self,
        sequence_length: int = 10,
        target_column: Optional[str] = None,
        prediction_horizon: int = 1
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.prediction_horizon = prediction_horizon

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np

            if isinstance(ctx.data, pd.DataFrame):
                df = ctx.data
            elif isinstance(ctx.data, dict) and "X" in ctx.data:
                data = ctx.data["X"]
            else:
                return ctx

            if isinstance(ctx.data, pd.DataFrame):
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                data = df[numeric_cols].values

            n_samples = len(data) - self.sequence_length - self.prediction_horizon + 1
            n_features = data.shape[1] if len(data.shape) > 1 else 1

            if n_samples <= 0:
                ctx.add_error("Not enough data for sequence length")
                return ctx

            # Create sequences
            X = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
            y = np.zeros((n_samples, self.prediction_horizon), dtype=np.float32)

            for i in range(n_samples):
                X[i] = data[i:i + self.sequence_length].reshape(self.sequence_length, -1)
                if self.target_column and isinstance(ctx.data, pd.DataFrame):
                    target_idx = df.columns.get_loc(self.target_column)
                    y[i] = data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, target_idx]
                else:
                    y[i] = data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0]

            ctx.data = {"X": X, "y": y}
            ctx.metadata["sequences"] = {
                "sequence_length": self.sequence_length,
                "prediction_horizon": self.prediction_horizon,
                "n_sequences": n_samples,
                "n_features": n_features
            }
            logger.info(f"Created {n_samples} sequences of length {self.sequence_length}")

        except Exception as e:
            ctx.add_error(f"Error creating sequences: {str(e)}")
        return ctx


class CreateGraphStructure(PipelineStep):
    """Create graph structure for GNN-based models"""

    name = "create_graph"
    stage = PipelineStage.STRUCTURE
    description = "Create graph structure (nodes, edges) for GNN"

    def __init__(
        self,
        node_features: Optional[List[str]] = None,
        edge_index_cols: Optional[Tuple[str, str]] = None
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_index_cols = edge_index_cols

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd
            import numpy as np

            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            df = ctx.data

            # Extract node features
            if self.node_features:
                x = df[self.node_features].values.astype(np.float32)
            else:
                x = df.select_dtypes(include=['number']).values.astype(np.float32)

            # Create edge index if specified
            if self.edge_index_cols:
                src_col, dst_col = self.edge_index_cols
                if src_col in df.columns and dst_col in df.columns:
                    edge_index = np.array([
                        df[src_col].values,
                        df[dst_col].values
                    ], dtype=np.int64)
                else:
                    edge_index = None
            else:
                edge_index = None

            ctx.data = {
                "x": x,  # Node features
                "edge_index": edge_index,
                "num_nodes": len(x)
            }

            ctx.metadata["graph_structure"] = {
                "num_nodes": len(x),
                "node_feature_dim": x.shape[1] if len(x.shape) > 1 else 1,
                "has_edges": edge_index is not None
            }
            logger.info(f"Created graph with {len(x)} nodes")

        except Exception as e:
            ctx.add_error(f"Error creating graph structure: {str(e)}")
        return ctx


class CreateTorchGeometricData(PipelineStep):
    """Convert to PyTorch Geometric Data object"""

    name = "to_torch_geometric"
    stage = PipelineStage.STRUCTURE
    description = "Create PyTorch Geometric Data object"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import torch
            from torch_geometric.data import Data

            if not isinstance(ctx.data, dict) or "x" not in ctx.data:
                return ctx

            x = torch.tensor(ctx.data["x"], dtype=torch.float32)

            kwargs = {"x": x}

            if ctx.data.get("edge_index") is not None:
                kwargs["edge_index"] = torch.tensor(ctx.data["edge_index"], dtype=torch.long)

            if "y" in ctx.data:
                kwargs["y"] = torch.tensor(ctx.data["y"])

            ctx.data = Data(**kwargs)
            logger.info("Created PyTorch Geometric Data object")

        except ImportError:
            ctx.add_error("torch_geometric not installed")
        except Exception as e:
            ctx.add_error(f"Error creating PyG Data: {str(e)}")
        return ctx


# Convenience function to get standard structure steps
def get_structure_steps(
    domain: str = "general",
    target_column: Optional[str] = None,
    output_format: str = "numpy"  # "numpy", "torch", "torch_geometric", "sequences"
) -> List[PipelineStep]:
    """Get the standard structure pipeline steps"""
    steps = []

    if output_format == "sequences":
        steps.append(CreateSequences(target_column=target_column))

    if output_format in ["numpy", "torch"]:
        steps.append(ToNumPyArrays(target_column=target_column))

    if output_format == "torch":
        steps.append(ToTorchTensors())

    if output_format == "torch_geometric":
        steps.append(CreateGraphStructure())
        steps.append(CreateTorchGeometricData())

    return steps


def structure(ctx: PipelineContext, **kwargs) -> PipelineContext:
    """Run the structure stage standalone"""
    from . import Pipeline

    pipeline = Pipeline(ctx.domain)
    for step in get_structure_steps(ctx.domain, **kwargs):
        pipeline.add_step(step)

    return pipeline.execute_stage(ctx, PipelineStage.STRUCTURE)
