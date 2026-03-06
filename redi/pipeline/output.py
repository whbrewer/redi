"""
DRAI Pipeline - Output Stage

Handles data sharding, train/val/test splitting, and saving.
Supports multiple output formats optimized for AI training.

Operations:
- Split into train/val/test
- Shard large datasets
- Save in AI-optimized formats (NPZ, HDF5, Parquet, TFRecord, Zarr)
- Save metadata and data cards
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from . import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error

logger = logging.getLogger(__name__)


class TrainValTestSplit(PipelineStep):
    """Split data into train/validation/test sets"""

    name = "train_val_test_split"
    stage = PipelineStage.OUTPUT
    description = "Split data into train/validation/test sets"

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42,
        stratify_column: Optional[str] = None
    ):
        super().__init__()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.stratify_column = stratify_column

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            # Check if split is disabled via config
            split_cfg = ctx.config.get("split", {})
            if isinstance(split_cfg, dict) and not split_cfg.get("enabled", True):
                logger.info("Split disabled via config")
                return ctx

            # Get data arrays - handle both raw arrays and dict format
            if isinstance(ctx.data, np.ndarray):
                # Raw numpy array - wrap in dict format
                X = ctx.data
                y = None
                logger.debug(f"     Input: raw numpy array shape {X.shape}")
            elif isinstance(ctx.data, dict):
                X = ctx.data.get("X")
                y = ctx.data.get("y")
                logger.debug(f"     Input: dict with keys {list(ctx.data.keys())}")
            else:
                ctx.add_error("Expected numpy array or dict with 'X' key for splitting")
                return ctx

            if X is None and isinstance(ctx.data, dict):
                # Named-variable dict (e.g., ADIOS2, NPZ multi-array): split each 1-D+
                # array along axis 0 and pass scalars through unchanged.
                splittable = {k: v for k, v in ctx.data.items()
                              if isinstance(v, np.ndarray) and v.ndim >= 1}
                if not splittable:
                    logger.info("No splittable arrays in data dict, skipping split")
                    return ctx

                n_samples = len(next(iter(splittable.values())))
                logger.debug(f"     Named-variable dict: splitting {len(splittable)} arrays "
                             f"along axis 0, n={n_samples}")

                indices = np.arange(n_samples)
                if self.shuffle:
                    np.random.seed(self.random_seed)
                    np.random.shuffle(indices)

                train_end = int(n_samples * self.train_ratio)
                val_end = train_end + int(n_samples * self.val_ratio)
                train_idx = indices[:train_end]
                val_idx = indices[train_end:val_end]
                test_idx = indices[val_end:]

                splits: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}}
                for k, v in ctx.data.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n_samples:
                        splits["train"][k] = v[train_idx]
                        splits["val"][k]   = v[val_idx]
                        splits["test"][k]  = v[test_idx]
                    else:
                        # Scalars or arrays with a different leading dimension: broadcast
                        for split in splits.values():
                            split[k] = v

                ctx.data = splits
                ctx.metadata["split"] = {
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx),
                    "ratios": {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
                    "shuffle": self.shuffle,
                    "random_seed": self.random_seed,
                }
                logger.info(f"Split {len(splittable)} arrays: "
                            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
                return ctx

            if X is None:
                ctx.add_error("No data to split")
                return ctx

            n_samples = len(X)
            indices = np.arange(n_samples)

            logger.debug(f"     Total samples: {n_samples:,}")
            logger.debug(f"     Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
            logger.debug(f"     Shuffle: {self.shuffle}")
            if self.shuffle:
                logger.debug(f"     Random seed: {self.random_seed}")

            if self.shuffle:
                np.random.seed(self.random_seed)
                np.random.shuffle(indices)

            # Calculate split points
            train_end = int(n_samples * self.train_ratio)
            val_end = train_end + int(n_samples * self.val_ratio)

            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

            logger.debug(f"     Train samples: {len(train_idx):,} ({len(train_idx)/n_samples*100:.1f}%)")
            logger.debug(f"     Val samples: {len(val_idx):,} ({len(val_idx)/n_samples*100:.1f}%)")
            logger.debug(f"     Test samples: {len(test_idx):,} ({len(test_idx)/n_samples*100:.1f}%)")

            splits = {
                "train": {"X": X[train_idx]},
                "val": {"X": X[val_idx]},
                "test": {"X": X[test_idx]}
            }

            if y is not None:
                splits["train"]["y"] = y[train_idx]
                splits["val"]["y"] = y[val_idx]
                splits["test"]["y"] = y[test_idx]
                logger.debug(f"     Target variable included in splits")

            # Preserve feature names
            if "feature_names" in ctx.data:
                for split in splits.values():
                    split["feature_names"] = ctx.data["feature_names"]

            ctx.data = splits
            ctx.metadata["split"] = {
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "ratios": {
                    "train": self.train_ratio,
                    "val": self.val_ratio,
                    "test": self.test_ratio
                },
                "shuffle": self.shuffle,
                "random_seed": self.random_seed
            }
            logger.info(f"Split data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        except Exception as e:
            ctx.add_error(f"Error splitting data: {str(e)}")
        return ctx


class ShardData(PipelineStep):
    """Shard large datasets into smaller files"""

    name = "shard_data"
    stage = PipelineStage.OUTPUT
    description = "Shard data into smaller files"

    def __init__(self, shard_size: int = 10000, output_format: str = "npz"):
        super().__init__()
        self.shard_size = shard_size
        self.output_format = output_format

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            if not isinstance(ctx.data, dict):
                return ctx

            logger.debug(f"     Shard size: {self.shard_size:,}")
            logger.debug(f"     Output format: {self.output_format}")

            # Handle splits or raw data
            if "train" in ctx.data:
                splits = ctx.data
                logger.debug(f"     Processing splits: {list(splits.keys())}")
            else:
                splits = {"data": ctx.data}
                logger.debug(f"     Processing single data block")

            sharded = {}
            total_shards = 0
            for split_name, split_data in splits.items():
                if "X" not in split_data:
                    continue

                X = split_data["X"]
                n_samples = len(X)
                n_shards = (n_samples + self.shard_size - 1) // self.shard_size

                logger.debug(f"     {split_name}: {n_samples:,} samples -> {n_shards} shards")

                shards = []
                for i in range(n_shards):
                    start = i * self.shard_size
                    end = min((i + 1) * self.shard_size, n_samples)

                    shard = {"X": X[start:end]}
                    if "y" in split_data:
                        shard["y"] = split_data["y"][start:end]
                    shards.append(shard)

                sharded[split_name] = {
                    "shards": shards,
                    "feature_names": split_data.get("feature_names", [])
                }
                total_shards += n_shards

            ctx.data = sharded
            ctx.metadata["sharding"] = {
                "shard_size": self.shard_size,
                "output_format": self.output_format
            }
            logger.debug(f"     Total shards created: {total_shards}")
            logger.info(f"Created shards of size {self.shard_size}")

        except Exception as e:
            ctx.add_error(f"Error sharding data: {str(e)}")
        return ctx


class SaveNPZ(PipelineStep):
    """Save data as NPZ files"""

    name = "save_npz"
    stage = PipelineStage.OUTPUT
    description = "Save data as NumPy NPZ files"

    def __init__(self, compress: bool = True):
        super().__init__()
        self.compress = compress

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import numpy as np

            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"     Output directory: {output_dir}")
            logger.debug(f"     Compression: {self.compress}")

            saved_files = []

            if isinstance(ctx.data, dict):
                # Check if sharded
                if any("shards" in v for v in ctx.data.values() if isinstance(v, dict)):
                    logger.debug(f"     Saving sharded data")
                    for split_name, split_data in ctx.data.items():
                        if not isinstance(split_data, dict) or "shards" not in split_data:
                            continue

                        logger.debug(f"     {split_name}: {len(split_data['shards'])} shards")
                        for i, shard in enumerate(split_data["shards"]):
                            file_path = output_dir / f"{split_name}_shard_{i:04d}.npz"
                            if self.compress:
                                np.savez_compressed(file_path, **shard)
                            else:
                                np.savez(file_path, **shard)
                            saved_files.append(str(file_path))
                else:
                    # Save splits directly
                    logger.debug(f"     Saving splits: {list(ctx.data.keys())}")
                    for split_name, split_data in ctx.data.items():
                        if not isinstance(split_data, dict) or "X" not in split_data:
                            continue

                        file_path = output_dir / f"{split_name}.npz"
                        save_data = {k: v for k, v in split_data.items() if k != "feature_names"}
                        if self.compress:
                            np.savez_compressed(file_path, **save_data)
                        else:
                            np.savez(file_path, **save_data)

                        file_size = file_path.stat().st_size
                        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
                        logger.debug(f"     Saved {file_path.name}: {size_str}")
                        saved_files.append(str(file_path))

            ctx.metadata["saved_files"] = saved_files
            ctx.metadata["output_format"] = "npz"
            logger.info(f"Saved {len(saved_files)} NPZ files to {output_dir}")

        except Exception as e:
            ctx.add_error(f"Error saving NPZ: {str(e)}")
        return ctx


class SaveHDF5(PipelineStep):
    """Save data as HDF5 file"""

    name = "save_hdf5"
    stage = PipelineStage.OUTPUT
    description = "Save data as HDF5 file"

    def __init__(self, compression: str = "gzip"):
        super().__init__()
        self.compression = compression

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import h5py
            import numpy as np

            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            file_path = output_dir / "data.h5"

            logger.debug(f"     Output file: {file_path}")
            logger.debug(f"     Compression: {self.compression}")

            with h5py.File(file_path, 'w') as f:
                if isinstance(ctx.data, dict):
                    for split_name, split_data in ctx.data.items():
                        if not isinstance(split_data, dict):
                            continue

                        grp = f.create_group(split_name)
                        logger.debug(f"     Created group: {split_name}")

                        for key, value in split_data.items():
                            if key == "feature_names":
                                grp.attrs["feature_names"] = json.dumps(value)
                            elif key == "shards":
                                for i, shard in enumerate(value):
                                    shard_grp = grp.create_group(f"shard_{i:04d}")
                                    for k, v in shard.items():
                                        shard_grp.create_dataset(k, data=v, compression=self.compression)
                                logger.debug(f"       {len(value)} shards saved")
                            elif hasattr(value, '__array__'):
                                arr = np.asarray(value)
                                # Scalar (0-d) arrays can't use chunk/filter options
                                kw = {} if arr.ndim == 0 else {"compression": self.compression}
                                grp.create_dataset(key, data=arr, **kw)
                                logger.debug(f"       {key}: shape {arr.shape}")

            file_size = file_path.stat().st_size
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
            logger.debug(f"     File size: {size_str}")

            ctx.metadata["saved_files"] = [str(file_path)]
            ctx.metadata["output_format"] = "hdf5"
            logger.info(f"Saved HDF5 to {file_path}")

        except ImportError:
            ctx.add_error(missing_dependency_error("h5py", "hdf5"))
        except Exception as e:
            ctx.add_error(f"Error saving HDF5: {str(e)}")
        return ctx


class SaveParquet(PipelineStep):
    """Save data as Parquet files"""

    name = "save_parquet"
    stage = PipelineStage.OUTPUT
    description = "Save tabular data as Parquet files"

    def __init__(self, compression: str = "snappy"):
        super().__init__()
        self.compression = compression

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import pandas as pd

            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"     Output directory: {output_dir}")
            logger.debug(f"     Compression: {self.compression}")

            saved_files = []

            if isinstance(ctx.data, dict):
                logger.debug(f"     Splits to save: {list(ctx.data.keys())}")
                for split_name, split_data in ctx.data.items():
                    if not isinstance(split_data, dict) or "X" not in split_data:
                        continue

                    X = split_data["X"]
                    feature_names = split_data.get("feature_names", [f"f{i}" for i in range(X.shape[1])])

                    df = pd.DataFrame(X, columns=feature_names)
                    if "y" in split_data:
                        df["target"] = split_data["y"]

                    file_path = output_dir / f"{split_name}.parquet"
                    df.to_parquet(file_path, compression=self.compression)

                    file_size = file_path.stat().st_size
                    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
                    logger.debug(f"     Saved {file_path.name}: {len(df)} rows, {len(df.columns)} columns, {size_str}")
                    saved_files.append(str(file_path))

            ctx.metadata["saved_files"] = saved_files
            ctx.metadata["output_format"] = "parquet"
            logger.info(f"Saved {len(saved_files)} Parquet files to {output_dir}")

        except ImportError:
            ctx.add_error(missing_dependency_error("pyarrow", "parquet"))
        except Exception as e:
            ctx.add_error(f"Error saving Parquet: {str(e)}")
        return ctx


class WriteChecksums(PipelineStep):
    """Write SHA-256 checksums for all saved output files."""

    name = "write_checksums"
    stage = PipelineStage.OUTPUT
    description = "Write SHA-256 checksums for output files"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            saved_files = ctx.metadata.get("saved_files", [])
            if not saved_files:
                return ctx

            output_dir = ctx.output_dir
            checksum_file = output_dir / "checksums.sha256"

            lines = []
            for file_str in saved_files:
                file_path = Path(file_str)
                if not file_path.exists():
                    continue
                digest = _sha256(file_path)
                # Relative path so the file is portable within the output dir
                rel = file_path.relative_to(output_dir)
                lines.append(f"{digest}  {rel}")

            checksum_file.write_text("\n".join(lines) + "\n")
            logger.info(f"Wrote checksums to {checksum_file}")

        except Exception as e:
            ctx.add_error(f"Error writing checksums: {str(e)}")
        return ctx


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, reading in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class SaveMetadata(PipelineStep):
    """Save pipeline metadata and data card"""

    name = "save_metadata"
    stage = PipelineStage.OUTPUT
    description = "Save pipeline metadata as JSON"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"     Output directory: {output_dir}")

            # Save metadata
            metadata_path = output_dir / "metadata.json"
            metadata = {
                "input_file": str(ctx.input_path),
                "domain": ctx.domain,
                "completed_stages": [s.value for s in ctx.completed_stages],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }

            logger.debug(f"     Input file: {ctx.input_path}")
            logger.debug(f"     Domain: {ctx.domain}")
            logger.debug(f"     Completed stages: {len(ctx.completed_stages)}")
            logger.debug(f"     Metadata keys: {list(ctx.metadata.keys())}")
            if ctx.errors:
                logger.debug(f"     Errors recorded: {len(ctx.errors)}")

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            file_size = metadata_path.stat().st_size
            logger.debug(f"     Saved {metadata_path.name}: {file_size} bytes")
            logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            ctx.add_error(f"Error saving metadata: {str(e)}")
        return ctx


# Convenience function to get standard output steps
def get_output_steps(
    output_format: str = "npz",  # "npz", "hdf5", "parquet"
    split: bool = True,
    shard: bool = False,
    shard_size: int = 10000
) -> List[PipelineStep]:
    """Get the standard output pipeline steps"""
    steps = []

    if split:
        steps.append(TrainValTestSplit())

    if shard:
        steps.append(ShardData(shard_size=shard_size, output_format=output_format))

    if output_format == "npz":
        steps.append(SaveNPZ())
    elif output_format == "hdf5":
        steps.append(SaveHDF5())
    elif output_format == "parquet":
        steps.append(SaveParquet())

    steps.append(WriteChecksums())
    steps.append(SaveMetadata())

    return steps


def output(ctx: PipelineContext, **kwargs) -> PipelineContext:
    """Run the output stage standalone"""
    from . import Pipeline

    pipeline = Pipeline(ctx.domain)
    for step in get_output_steps(**kwargs):
        pipeline.add_step(step)

    return pipeline.execute_stage(ctx, PipelineStage.OUTPUT)
