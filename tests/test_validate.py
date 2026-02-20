"""Unit tests for redi.validate module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from redi.validate import (
    ValidationResult,
    adjust_shape,
    compute_kl_divergence,
    compute_metrics,
    compute_similarity_metrics,
    load_ground_truth,
    load_redi_output,
    run_validation,
    to_numpy,
)


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_from_list(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_from_numpy(self):
        arr = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = to_numpy(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [4.0, 5.0, 6.0])

    def test_from_torch_tensor(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([7.0, 8.0, 9.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [7.0, 8.0, 9.0])


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(values)
        assert m["count"] == 5
        assert abs(m["mean"] - 3.0) < 1e-5
        assert abs(m["min"] - 1.0) < 1e-5
        assert abs(m["max"] - 5.0) < 1e-5
        assert m["std"] > 0

    def test_single_value(self):
        m = compute_metrics([42.0])
        assert m["count"] == 1
        assert abs(m["mean"] - 42.0) < 1e-5
        assert abs(m["std"]) < 1e-5

    def test_2d_input(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = compute_metrics(values)
        assert m["count"] == 4
        assert abs(m["mean"] - 2.5) < 1e-5


# ---------------------------------------------------------------------------
# compute_kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions(self):
        a = np.random.randn(10000).astype(np.float32)
        kl = compute_kl_divergence(a, a)
        assert abs(kl) < 0.01

    def test_different_distributions(self):
        a = np.random.randn(10000).astype(np.float32)
        b = np.random.randn(10000).astype(np.float32) + 5.0
        kl = compute_kl_divergence(a, b)
        assert kl > 0.1

    def test_constant_distribution(self):
        a = np.ones(100, dtype=np.float32)
        b = np.ones(100, dtype=np.float32)
        kl = compute_kl_divergence(a, b)
        assert kl == 0.0


# ---------------------------------------------------------------------------
# compute_similarity_metrics
# ---------------------------------------------------------------------------

class TestSimilarityMetrics:
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = compute_similarity_metrics(a, a)
        assert abs(sim["mae"]) < 1e-6
        assert abs(sim["rmse"]) < 1e-6
        assert abs(sim["corr"] - 1.0) < 1e-6

    def test_different_arrays(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        sim = compute_similarity_metrics(a, b)
        assert sim["mae"] > 0
        assert sim["rmse"] > 0
        assert sim["corr"] < 0  # negative correlation

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_similarity_metrics(a, b)

    def test_returns_all_keys(self):
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)
        sim = compute_similarity_metrics(a, b)
        assert "mae" in sim
        assert "rmse" in sim
        assert "corr" in sim
        assert "kl" in sim


# ---------------------------------------------------------------------------
# adjust_shape
# ---------------------------------------------------------------------------

class TestAdjustShape:
    def test_exact_match(self):
        data = np.ones((10, 5), dtype=np.float32)
        result = adjust_shape(data, (10, 5))
        assert result.shape == (10, 5)
        np.testing.assert_array_equal(result, data)

    def test_truncation_rows(self):
        data = np.ones((20, 5), dtype=np.float32)
        result = adjust_shape(data, (10, 5))
        assert result.shape == (10, 5)

    def test_padding_rows(self):
        data = np.ones((5, 3), dtype=np.float32)
        result = adjust_shape(data, (10, 3))
        assert result.shape == (10, 3)
        # First 5 rows are 1.0, last 5 are 0.0
        np.testing.assert_array_equal(result[:5], np.ones((5, 3)))
        np.testing.assert_array_equal(result[5:], np.zeros((5, 3)))

    def test_truncation_columns(self):
        data = np.ones((10, 8), dtype=np.float32)
        result = adjust_shape(data, (10, 5))
        assert result.shape == (10, 5)

    def test_padding_columns(self):
        data = np.ones((10, 3), dtype=np.float32)
        result = adjust_shape(data, (10, 5))
        assert result.shape == (10, 5)
        np.testing.assert_array_equal(result[:, :3], np.ones((10, 3)))
        np.testing.assert_array_equal(result[:, 3:], np.zeros((10, 2)))

    def test_3d_reshape(self):
        # Simulate (locations, timesteps, features) target
        data = np.ones((60, 4), dtype=np.float32)
        result = adjust_shape(data, (3, 20, 4))
        assert result.shape == (3, 20, 4)

    def test_3d_truncation_and_reshape(self):
        data = np.ones((100, 4), dtype=np.float32)
        result = adjust_shape(data, (2, 10, 4))
        assert result.shape == (2, 10, 4)

    def test_3d_padding_and_reshape(self):
        data = np.ones((10, 4), dtype=np.float32)
        result = adjust_shape(data, (2, 10, 4))
        assert result.shape == (2, 10, 4)
        # First 10 rows worth of data are 1, remaining 10 are 0
        flat = result.reshape(-1, 4)
        np.testing.assert_array_equal(flat[:10], np.ones((10, 4)))
        np.testing.assert_array_equal(flat[10:], np.zeros((10, 4)))


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_npz_format(self, tmp_path):
        data = np.random.randn(20, 5).astype(np.float32)
        path = tmp_path / "truth.npz"
        np.savez(path, X=data)

        result = load_ground_truth(str(path))
        assert result.shape == (20, 5)
        np.testing.assert_array_almost_equal(result, data)

    def test_npy_format(self, tmp_path):
        data = np.random.randn(15, 3).astype(np.float32)
        path = tmp_path / "truth.npy"
        np.save(path, data)

        result = load_ground_truth(str(path))
        assert result.shape == (15, 3)
        np.testing.assert_array_almost_equal(result, data)

    def test_npz_auto_detect(self, tmp_path):
        data = np.random.randn(10, 4).astype(np.float32)
        path = tmp_path / "data.npz"
        np.savez(path, X=data)

        result = load_ground_truth(str(path))
        assert result.shape == (10, 4)

    def test_explicit_format_override(self, tmp_path):
        data = np.random.randn(10, 4).astype(np.float32)
        path = tmp_path / "data.npz"
        np.savez(path, X=data)

        result = load_ground_truth(str(path), fmt="npz")
        assert result.shape == (10, 4)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_ground_truth("/nonexistent/path/data.npz")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_ground_truth(str(path))

    def test_pt_format(self, tmp_path):
        torch = pytest.importorskip("torch")
        data = torch.randn(10, 4)
        path = tmp_path / "truth.pt"
        torch.save({"X": data}, path)

        result = load_ground_truth(str(path))
        assert result.shape == (10, 4)
        np.testing.assert_array_almost_equal(
            result, data.numpy().astype(np.float32), decimal=5
        )


# ---------------------------------------------------------------------------
# load_redi_output
# ---------------------------------------------------------------------------

class TestLoadRediOutput:
    def test_npz_format(self, tmp_path):
        """Test loading NPZ format REDI output with train/val/test splits."""
        # Create metadata.json
        metadata = {"output_format": "npz"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        # Create split files
        train = np.random.randn(80, 5).astype(np.float32)
        val = np.random.randn(10, 5).astype(np.float32)
        test = np.random.randn(10, 5).astype(np.float32)
        np.savez(tmp_path / "train.npz", X=train)
        np.savez(tmp_path / "val.npz", X=val)
        np.savez(tmp_path / "test.npz", X=test)

        result = load_redi_output(str(tmp_path))
        assert result.shape == (100, 5)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result[:80], train)
        np.testing.assert_array_almost_equal(result[80:90], val)
        np.testing.assert_array_almost_equal(result[90:], test)

    def test_pt_format(self, tmp_path):
        torch = pytest.importorskip("torch")

        metadata = {"output_format": "pt"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        data = torch.randn(100, 5)
        torch.save({"X": data}, tmp_path / "data.pt")

        result = load_redi_output(str(tmp_path))
        assert result.shape == (100, 5)
        np.testing.assert_array_almost_equal(
            result, data.numpy().astype(np.float32), decimal=5
        )

    def test_missing_metadata(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            load_redi_output(str(tmp_path))

    def test_npz_partial_splits(self, tmp_path):
        """Only train.npz exists (no val/test)."""
        metadata = {"output_format": "npz"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        train = np.random.randn(50, 3).astype(np.float32)
        np.savez(tmp_path / "train.npz", X=train)

        result = load_redi_output(str(tmp_path))
        assert result.shape == (50, 3)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_to_dict(self):
        r = ValidationResult(
            redi_dir="/output/test",
            ground_truth_path="/data/truth.pt",
            redi_shape=(100, 5),
            ground_truth_shape=(50, 10, 5),
            aligned=True,
            redi_stats={"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0, "count": 500},
            ground_truth_stats={"mean": 0.1, "std": 0.9, "min": -2.5, "max": 2.5, "count": 2500},
            similarity={"mae": 0.1, "rmse": 0.2, "corr": 0.95, "kl": 0.01},
        )
        d = r.to_dict()
        assert d["redi_dir"] == "/output/test"
        assert d["redi_shape"] == [100, 5]
        assert d["ground_truth_shape"] == [50, 10, 5]
        assert d["aligned"] is True
        assert d["similarity"]["mae"] == 0.1
        assert d["per_feature"] is None
        assert d["aidrin"] is None

    def test_to_json(self):
        r = ValidationResult(
            redi_dir="/output/test",
            ground_truth_path="/data/truth.npz",
            redi_shape=(10, 4),
            ground_truth_shape=(10, 4),
            aligned=False,
            redi_stats={"mean": 0.0},
            ground_truth_stats={"mean": 0.0},
            similarity={"mae": 0.0},
        )
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["redi_dir"] == "/output/test"
        assert parsed["redi_shape"] == [10, 4]


# ---------------------------------------------------------------------------
# run_validation end-to-end
# ---------------------------------------------------------------------------

class TestRunValidation:
    def test_end_to_end_npz(self, tmp_path):
        """End-to-end validation with synthetic NPZ data."""
        # Create REDI output directory
        redi_dir = tmp_path / "redi_output"
        redi_dir.mkdir()
        metadata = {"output_format": "npz"}
        (redi_dir / "metadata.json").write_text(json.dumps(metadata))

        np.random.seed(42)
        data = np.random.randn(100, 5).astype(np.float32)
        np.savez(redi_dir / "train.npz", X=data[:80])
        np.savez(redi_dir / "val.npz", X=data[80:90])
        np.savez(redi_dir / "test.npz", X=data[90:])

        # Create ground truth
        gt_path = tmp_path / "truth.npz"
        gt_data = data + np.random.randn(100, 5).astype(np.float32) * 0.01
        np.savez(gt_path, X=gt_data)

        config = {
            "ground_truth": str(gt_path),
            "format": "npz",
        }

        result = run_validation(str(redi_dir), config)

        assert result.redi_shape == (100, 5)
        assert result.ground_truth_shape == (100, 5)
        assert result.similarity["mae"] < 0.1
        assert result.similarity["rmse"] < 0.1
        assert result.similarity["corr"] > 0.99
        assert result.redi_stats["count"] == 500
        assert result.ground_truth_stats["count"] == 500

    def test_end_to_end_with_shape_alignment(self, tmp_path):
        """Test validation with shape mismatch requiring alignment."""
        redi_dir = tmp_path / "redi_output"
        redi_dir.mkdir()
        metadata = {"output_format": "npz"}
        (redi_dir / "metadata.json").write_text(json.dumps(metadata))

        # REDI output: 200 rows x 5 cols
        np.random.seed(42)
        data = np.random.randn(200, 5).astype(np.float32)
        np.savez(redi_dir / "train.npz", X=data[:160])
        np.savez(redi_dir / "val.npz", X=data[160:180])
        np.savez(redi_dir / "test.npz", X=data[180:])

        # Ground truth: 10x10x5 (3D)
        gt_path = tmp_path / "truth.npz"
        gt_data = np.random.randn(10, 10, 5).astype(np.float32)
        np.savez(gt_path, X=gt_data)

        config = {
            "ground_truth": str(gt_path),
            "alignment": {
                "method": "truncate_pad",
                "target_shape": [10, 10, 5],
            },
        }

        result = run_validation(str(redi_dir), config)

        assert result.redi_shape == (200, 5)
        assert result.ground_truth_shape == (10, 10, 5)
        assert result.aligned is True

    def test_per_feature(self, tmp_path):
        """Test per-feature similarity computation."""
        redi_dir = tmp_path / "redi_output"
        redi_dir.mkdir()
        metadata = {"output_format": "npz"}
        (redi_dir / "metadata.json").write_text(json.dumps(metadata))

        np.random.seed(42)
        data = np.random.randn(50, 3).astype(np.float32)
        np.savez(redi_dir / "train.npz", X=data)

        gt_path = tmp_path / "truth.npz"
        gt_data = data + 0.01
        np.savez(gt_path, X=gt_data)

        config = {"ground_truth": str(gt_path)}
        result = run_validation(str(redi_dir), config, per_feature=True)

        assert result.per_feature is not None
        assert len(result.per_feature) == 3
        for feat in result.per_feature:
            assert "mae" in feat
            assert "rmse" in feat
            assert "corr" in feat
            assert "kl" in feat
            assert "feature_index" in feat

    def test_error_alignment_method(self, tmp_path):
        """Test that alignment method 'error' raises on shape mismatch."""
        redi_dir = tmp_path / "redi_output"
        redi_dir.mkdir()
        metadata = {"output_format": "npz"}
        (redi_dir / "metadata.json").write_text(json.dumps(metadata))

        data = np.random.randn(50, 5).astype(np.float32)
        np.savez(redi_dir / "train.npz", X=data)

        gt_path = tmp_path / "truth.npz"
        gt_data = np.random.randn(30, 5).astype(np.float32)
        np.savez(gt_path, X=gt_data)

        config = {
            "ground_truth": str(gt_path),
            "alignment": {"method": "error"},
        }

        with pytest.raises(ValueError, match="Shape mismatch"):
            run_validation(str(redi_dir), config)

    def test_missing_ground_truth_key(self, tmp_path):
        """Test that missing ground_truth key raises ValueError."""
        with pytest.raises(ValueError, match="ground_truth"):
            run_validation(str(tmp_path), {})
