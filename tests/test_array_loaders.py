"""
Unit tests for NPY/NPZ/Manifest loaders and array preprocessing steps.

Tests the new functionality for handling:
- Single and multiple NPY files
- NPZ archive files
- File manifests (train.txt, test.txt style)
- Array normalization (min-max and z-score)
- Array statistics computation
- Train/val/test splitting of raw arrays
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_npy(temp_dir):
    """Create a sample NPY file."""
    arr = np.random.rand(100, 32, 32).astype(np.float32)
    npy_path = temp_dir / "sample.npy"
    np.save(npy_path, arr)
    return npy_path


@pytest.fixture
def sample_npy_multiple(temp_dir):
    """Create multiple NPY files for stacking tests."""
    paths = []
    for i in range(5):
        arr = np.random.rand(10, 8, 8).astype(np.float32)
        path = temp_dir / f"file_{i:02d}.npy"
        np.save(path, arr)
        paths.append(path)
    return paths


@pytest.fixture
def sample_npz(temp_dir):
    """Create a sample NPZ archive file."""
    X = np.random.rand(100, 32, 32).astype(np.float32)
    y = np.random.randint(0, 10, 100)
    npz_path = temp_dir / "sample.npz"
    np.savez_compressed(npz_path, X=X, y=y)
    return npz_path


@pytest.fixture
def sample_manifest(temp_dir):
    """Create a manifest file with NPY paths."""
    # Create NPY files
    for i in range(5):
        arr = np.random.rand(10, 8, 8).astype(np.float32)
        np.save(temp_dir / f"file_{i:02d}.npy", arr)

    # Create manifest
    manifest_path = temp_dir / "train.txt"
    with open(manifest_path, 'w') as f:
        for i in range(5):
            f.write(f"file_{i:02d}.npy\n")
    return manifest_path


class TestLoadNPY:
    """Tests for LoadNPY step."""

    def test_load_single_npy(self, sample_npy, temp_dir):
        """Test loading a single NPY file."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import DetectFormat, LoadNPY

        ctx = create_context(str(sample_npy), str(temp_dir), "general")

        # Detect format
        ctx = DetectFormat().execute(ctx)
        assert ctx.metadata.get("file_format") == "npy"

        # Load NPY
        step = LoadNPY()
        assert step.validate(ctx) is True

        result = step.execute(ctx)
        assert result.data is not None
        assert isinstance(result.data, np.ndarray)
        assert result.data.shape == (100, 32, 32)
        assert result.metadata["array_ndim"] == 3
        assert result.metadata["num_samples"] == 100

    def test_load_multiple_npy(self, sample_npy_multiple, temp_dir):
        """Test loading and stacking multiple NPY files."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadNPY

        ctx = create_context(
            input_path=str(sample_npy_multiple[0]),
            output_dir=str(temp_dir),
            domain="general",
            input_paths=[str(p) for p in sample_npy_multiple]
        )
        ctx.metadata["file_format"] = "npy"

        step = LoadNPY()
        result = step.execute(ctx)

        assert result.data is not None
        assert isinstance(result.data, np.ndarray)
        # 5 files stacked along axis 0, each (10, 8, 8)
        assert result.data.shape == (5, 10, 8, 8)

    def test_load_npy_validation(self, sample_csv, temp_dir):
        """Test that LoadNPY skips non-NPY files."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadNPY

        ctx = create_context(str(sample_csv), str(temp_dir), "general")
        ctx.metadata["file_format"] = "csv"

        step = LoadNPY()
        assert step.validate(ctx) is False


class TestLoadNPZ:
    """Tests for LoadNPZ step."""

    def test_load_npz(self, sample_npz, temp_dir):
        """Test loading NPZ archive."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import DetectFormat, LoadNPZ

        ctx = create_context(str(sample_npz), str(temp_dir), "general")

        # Detect format
        ctx = DetectFormat().execute(ctx)
        assert ctx.metadata.get("file_format") == "npz"

        # Load NPZ
        step = LoadNPZ()
        assert step.validate(ctx) is True

        result = step.execute(ctx)
        assert result.data is not None
        assert isinstance(result.data, dict)
        assert "X" in result.data
        assert "y" in result.data
        assert result.data["X"].shape == (100, 32, 32)
        assert result.data["y"].shape == (100,)
        assert result.metadata["num_samples"] == 100
        assert result.metadata["array_ndim"] == 3

    def test_load_npz_keys_metadata(self, sample_npz, temp_dir):
        """Test NPZ metadata extraction."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadNPZ

        ctx = create_context(str(sample_npz), str(temp_dir), "general")
        ctx.metadata["file_format"] = "npz"

        result = LoadNPZ().execute(ctx)

        assert "npz_keys" in result.metadata
        assert "X" in result.metadata["npz_keys"]
        assert "y" in result.metadata["npz_keys"]
        assert "array_shapes" in result.metadata
        assert result.metadata["array_shapes"]["X"] == (100, 32, 32)


class TestLoadManifest:
    """Tests for LoadManifest step."""

    def test_load_manifest(self, sample_manifest, temp_dir):
        """Test loading files from manifest."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadManifest

        ctx = create_context(str(sample_manifest), str(temp_dir), "general")

        step = LoadManifest()
        assert step.validate(ctx) is True

        result = step.execute(ctx)
        assert result.data is not None
        assert isinstance(result.data, np.ndarray)
        # 5 files stacked, each (10, 8, 8)
        assert result.data.shape == (5, 10, 8, 8)
        assert result.metadata["num_files"] == 5
        assert result.metadata["manifest_file"] == str(sample_manifest)

    def test_manifest_validation_rejects_non_manifest(self, sample_csv, temp_dir):
        """Test that LoadManifest rejects non-manifest txt files."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadManifest

        # Create a text file that's not a manifest
        txt_path = temp_dir / "notes.txt"
        txt_path.write_text("This is just a note, not a manifest.")

        ctx = create_context(str(txt_path), str(temp_dir), "general")

        step = LoadManifest()
        # Should return False because first line doesn't end with .npy/.npz/.h5/.nc
        assert step.validate(ctx) is False


class TestNormalizeArray:
    """Tests for NormalizeArray step."""

    def test_normalize_minmax(self, temp_dir):
        """Test min-max normalization."""
        from redi.pipeline import create_context
        from redi.pipeline.preprocess import NormalizeArray

        ctx = create_context(str(temp_dir / "test.npy"), str(temp_dir), "general")
        ctx.data = np.random.rand(100, 32, 32) * 1000  # Large range

        step = NormalizeArray(method="minmax")
        assert step.validate(ctx) is True

        result = step.execute(ctx)

        assert result.data.min() >= 0
        assert result.data.max() <= 1
        assert result.data.dtype == np.float32
        assert result.metadata["normalization"]["method"] == "minmax"

    def test_normalize_zscore(self, temp_dir):
        """Test z-score normalization."""
        from redi.pipeline import create_context
        from redi.pipeline.preprocess import NormalizeArray

        ctx = create_context(str(temp_dir / "test.npy"), str(temp_dir), "general")
        ctx.data = np.random.rand(100, 32, 32) * 1000 + 500

        step = NormalizeArray(method="zscore")
        result = step.execute(ctx)

        # After z-score, mean should be ~0 and std ~1
        assert abs(result.data.mean()) < 0.01
        assert abs(result.data.std() - 1) < 0.1
        assert result.metadata["normalization"]["method"] == "zscore"

    def test_normalize_dict_format(self, temp_dir):
        """Test normalization with dict format (NPZ style)."""
        from redi.pipeline import create_context
        from redi.pipeline.preprocess import NormalizeArray

        ctx = create_context(str(temp_dir / "test.npz"), str(temp_dir), "general")
        ctx.data = {"X": np.random.rand(100, 32, 32) * 1000, "y": np.array([1, 2, 3])}

        step = NormalizeArray(method="minmax")
        result = step.execute(ctx)

        assert isinstance(result.data, dict)
        assert result.data["X"].min() >= 0
        assert result.data["X"].max() <= 1
        # y should be unchanged
        assert np.array_equal(result.data["y"], np.array([1, 2, 3]))


class TestComputeArrayStats:
    """Tests for ComputeArrayStats step."""

    def test_compute_stats_array(self, temp_dir):
        """Test statistics computation for raw array."""
        from redi.pipeline import create_context
        from redi.pipeline.preprocess import ComputeArrayStats

        ctx = create_context(str(temp_dir / "test.npy"), str(temp_dir), "general")
        ctx.data = np.random.rand(100, 32, 32).astype(np.float32)

        step = ComputeArrayStats()
        assert step.validate(ctx) is True

        result = step.execute(ctx)

        stats = result.metadata["array_stats"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "sparsity" in stats
        assert "shape" in stats
        assert stats["shape"] == [100, 32, 32]

    def test_compute_stats_dict_format(self, temp_dir):
        """Test statistics computation for dict format."""
        from redi.pipeline import create_context
        from redi.pipeline.preprocess import ComputeArrayStats

        ctx = create_context(str(temp_dir / "test.npz"), str(temp_dir), "general")
        ctx.data = {"X": np.random.rand(100, 32, 32), "y": np.random.randint(0, 10, 100)}

        step = ComputeArrayStats()
        result = step.execute(ctx)

        stats = result.metadata["array_stats"]
        assert stats["shape"] == [100, 32, 32]


class TestTrainValTestSplitArray:
    """Tests for TrainValTestSplit with array data."""

    def test_split_raw_array(self, temp_dir):
        """Test splitting a raw numpy array."""
        from redi.pipeline import create_context
        from redi.pipeline.output import TrainValTestSplit

        ctx = create_context(str(temp_dir / "test.npy"), str(temp_dir), "general")
        ctx.data = np.random.rand(100, 32, 32)

        step = TrainValTestSplit()
        result = step.execute(ctx)

        assert "train" in result.data
        assert "val" in result.data
        assert "test" in result.data
        assert result.metadata["split"]["train_size"] == 80
        assert result.metadata["split"]["val_size"] == 10
        assert result.metadata["split"]["test_size"] == 10

    def test_split_dict_with_y(self, temp_dir):
        """Test splitting dict with X and y arrays."""
        from redi.pipeline import create_context
        from redi.pipeline.output import TrainValTestSplit

        ctx = create_context(str(temp_dir / "test.npz"), str(temp_dir), "general")
        ctx.data = {
            "X": np.random.rand(100, 32, 32),
            "y": np.random.randint(0, 10, 100)
        }

        step = TrainValTestSplit()
        result = step.execute(ctx)

        # Check that y is also split
        assert "y" in result.data["train"]
        assert "y" in result.data["val"]
        assert "y" in result.data["test"]

        # Check sizes
        assert len(result.data["train"]["y"]) == 80
        assert len(result.data["val"]["y"]) == 10
        assert len(result.data["test"]["y"]) == 10


class TestValidateInputArray:
    """Tests for ValidateInput with array data."""

    def test_validate_numpy_array(self, sample_npy, temp_dir):
        """Test validation of numpy array."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import ValidateInput

        ctx = create_context(str(sample_npy), str(temp_dir), "general")
        ctx.data = np.random.rand(100, 32, 32)

        step = ValidateInput()
        result = step.execute(ctx)

        assert len(result.errors) == 0

    def test_validate_empty_array_fails(self, temp_dir):
        """Test that empty arrays fail validation."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import ValidateInput

        ctx = create_context(str(temp_dir / "test.npy"), str(temp_dir), "general")
        ctx.data = np.array([])

        step = ValidateInput()
        result = step.execute(ctx)

        assert len(result.errors) > 0
        assert "empty" in result.errors[0].lower()

    def test_validate_dict_arrays(self, temp_dir):
        """Test validation of dict with arrays."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import ValidateInput

        ctx = create_context(str(temp_dir / "test.npz"), str(temp_dir), "general")
        ctx.data = {"X": np.random.rand(100, 32, 32), "y": np.random.randint(0, 10, 100)}

        step = ValidateInput()
        result = step.execute(ctx)

        assert len(result.errors) == 0


class TestEndToEndArrayPipeline:
    """End-to-end tests for array processing pipeline."""

    def test_full_npy_pipeline(self, sample_npy, temp_dir):
        """Test complete pipeline with NPY file."""
        from redi.pipeline import Pipeline, create_context
        from redi.pipeline.ingest import DetectFormat, LoadNPY, ValidateInput
        from redi.pipeline.preprocess import ComputeArrayStats, NormalizeArray
        from redi.pipeline.output import TrainValTestSplit, SaveNPZ, SaveMetadata

        ctx = create_context(str(sample_npy), str(temp_dir / "output"), "general")

        pipeline = Pipeline("general")
        pipeline.add_step(DetectFormat())
        pipeline.add_step(LoadNPY())
        pipeline.add_step(ValidateInput())
        pipeline.add_step(ComputeArrayStats())
        pipeline.add_step(NormalizeArray(method="minmax"))
        pipeline.add_step(TrainValTestSplit())
        pipeline.add_step(SaveNPZ())
        pipeline.add_step(SaveMetadata())

        result = pipeline.execute(ctx)

        # Check no critical errors
        critical_errors = [e for e in result.errors if "missing" not in e.lower()]
        assert len(critical_errors) == 0

        # Check output files exist
        output_dir = temp_dir / "output"
        assert (output_dir / "train.npz").exists()
        assert (output_dir / "val.npz").exists()
        assert (output_dir / "test.npz").exists()
        assert (output_dir / "metadata.json").exists()

        # Verify output content
        train_data = np.load(output_dir / "train.npz")
        assert "X" in train_data
        assert train_data["X"].min() >= 0
        assert train_data["X"].max() <= 1

    def test_full_npz_pipeline(self, sample_npz, temp_dir):
        """Test complete pipeline with NPZ file."""
        from redi.pipeline import Pipeline, create_context
        from redi.pipeline.ingest import DetectFormat, LoadNPZ, ValidateInput
        from redi.pipeline.preprocess import ComputeArrayStats, NormalizeArray
        from redi.pipeline.output import TrainValTestSplit, SaveNPZ

        ctx = create_context(str(sample_npz), str(temp_dir / "output"), "general")

        pipeline = Pipeline("general")
        pipeline.add_step(DetectFormat())
        pipeline.add_step(LoadNPZ())
        pipeline.add_step(ValidateInput())
        pipeline.add_step(ComputeArrayStats())
        pipeline.add_step(NormalizeArray(method="zscore"))
        pipeline.add_step(TrainValTestSplit())
        pipeline.add_step(SaveNPZ())

        result = pipeline.execute(ctx)

        # Check output
        output_dir = temp_dir / "output"
        assert (output_dir / "train.npz").exists()

        # Verify labels are preserved
        train_data = np.load(output_dir / "train.npz")
        assert "y" in train_data

    def test_manifest_pipeline(self, sample_manifest, temp_dir):
        """Test pipeline with manifest input."""
        from redi.pipeline import Pipeline, create_context
        from redi.pipeline.ingest import LoadManifest, ValidateInput
        from redi.pipeline.preprocess import ComputeArrayStats
        from redi.pipeline.output import TrainValTestSplit, SaveNPZ

        ctx = create_context(str(sample_manifest), str(temp_dir / "output"), "general")

        pipeline = Pipeline("general")
        pipeline.add_step(LoadManifest())
        pipeline.add_step(ValidateInput())
        pipeline.add_step(ComputeArrayStats())
        pipeline.add_step(TrainValTestSplit())
        pipeline.add_step(SaveNPZ())

        result = pipeline.execute(ctx)

        assert len([e for e in result.errors if "critical" in e.lower()]) == 0
        assert (temp_dir / "output" / "train.npz").exists()
