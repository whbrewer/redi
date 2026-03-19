"""
Tests for the REDI parallel execution module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from redi.parallel import (
    ParallelExecutor,
    ParallelConfig,
    ParallelResult,
    FileDistributor,
    ChunkStrategy,
    ChunkSpec,
)
from redi.parallel.backends import get_backend, LocalBackend, SlurmBackend
from redi.parallel.chunking import TimeChunkStrategy, RowChunkStrategy, get_chunk_strategy


class TestFileDistributor:
    """Tests for file distribution strategies."""

    def test_round_robin_basic(self):
        """Test round-robin distribution with simple inputs."""
        files = [Path(f"file{i}.nc") for i in range(10)]
        result = FileDistributor.round_robin(files, 3)

        assert len(result) == 3
        # Worker 0: files 0, 3, 6, 9
        assert result[0] == [files[0], files[3], files[6], files[9]]
        # Worker 1: files 1, 4, 7
        assert result[1] == [files[1], files[4], files[7]]
        # Worker 2: files 2, 5, 8
        assert result[2] == [files[2], files[5], files[8]]

    def test_round_robin_empty(self):
        """Test round-robin with empty file list."""
        result = FileDistributor.round_robin([], 4)
        assert len(result) == 4
        assert all(len(w) == 0 for w in result)

    def test_round_robin_fewer_files_than_workers(self):
        """Test when there are fewer files than workers."""
        files = [Path(f"file{i}.nc") for i in range(2)]
        result = FileDistributor.round_robin(files, 5)

        assert len(result) == 5
        assert result[0] == [files[0]]
        assert result[1] == [files[1]]
        assert result[2] == []
        assert result[3] == []
        assert result[4] == []

    def test_round_robin_invalid_workers(self):
        """Test round-robin with invalid worker count."""
        files = [Path("file.nc")]
        with pytest.raises(ValueError, match="num_workers must be positive"):
            FileDistributor.round_robin(files, 0)
        with pytest.raises(ValueError, match="num_workers must be positive"):
            FileDistributor.round_robin(files, -1)

    def test_batched_basic(self):
        """Test batching with simple inputs."""
        files = [Path(f"file{i}.nc") for i in range(10)]
        result = FileDistributor.batched(files, 4)

        assert len(result) == 3
        assert result[0] == files[0:4]
        assert result[1] == files[4:8]
        assert result[2] == files[8:10]

    def test_batched_exact_division(self):
        """Test batching when files divide evenly."""
        files = [Path(f"file{i}.nc") for i in range(8)]
        result = FileDistributor.batched(files, 4)

        assert len(result) == 2
        assert result[0] == files[0:4]
        assert result[1] == files[4:8]

    def test_batched_single_batch(self):
        """Test batching when batch_size > num_files."""
        files = [Path(f"file{i}.nc") for i in range(3)]
        result = FileDistributor.batched(files, 10)

        assert len(result) == 1
        assert result[0] == files

    def test_batched_invalid_batch_size(self):
        """Test batching with invalid batch size."""
        files = [Path("file.nc")]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            FileDistributor.batched(files, 0)

    def test_balanced_distribution(self):
        """Test balanced distribution by file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different sizes
            files = []
            for i, size in enumerate([1000, 2000, 3000, 500]):
                path = Path(tmpdir) / f"file{i}.txt"
                path.write_bytes(b"x" * size)
                files.append(path)

            result = FileDistributor.balanced(files, 2)

            # Should balance: one worker gets ~3000, other gets ~3500
            assert len(result) == 2
            assert len(result[0]) + len(result[1]) == 4

    def test_for_nodes(self):
        """Test multi-node distribution."""
        files = [Path(f"file{i}.nc") for i in range(20)]
        result = FileDistributor.for_nodes(files, num_nodes=2, batch_size=3)

        # 2 nodes
        assert len(result) == 2
        # Each node should have batches of up to 3 files
        for node_batches in result:
            for batch in node_batches:
                assert len(batch) <= 3

    def test_write_file_list(self):
        """Test writing file list to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(f"/data/file{i}.nc") for i in range(3)]
            output_path = Path(tmpdir) / "files.txt"

            result = FileDistributor.write_file_list(files, output_path)

            assert result == output_path
            assert output_path.exists()
            content = output_path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3
            assert lines[0] == "/data/file0.nc"


class TestParallelConfig:
    """Tests for ParallelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelConfig()

        assert config.backend == "local"
        assert config.workers is not None  # Should auto-detect
        assert config.batch_size == 4
        assert config.style == "simple"
        assert config.distribution == "round_robin"

    def test_slurm_defaults(self):
        """Test SLURM default configuration."""
        config = ParallelConfig(backend="slurm")

        assert config.slurm_nodes == 4
        assert config.slurm_cores_per_node == 56
        assert config.slurm_time == "0:30:00"
        assert config.slurm_partition == "batch"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParallelConfig(
            backend="slurm",
            workers=8,
            batch_size=2,
            slurm_nodes=10,
            slurm_account="myaccount",
        )

        assert config.backend == "slurm"
        assert config.workers == 8
        assert config.batch_size == 2
        assert config.slurm_nodes == 10
        assert config.slurm_account == "myaccount"


class TestParallelResult:
    """Tests for ParallelResult."""

    def test_successful_result(self):
        """Test successful result."""
        result = ParallelResult(
            success=True,
            total_files=10,
            completed=10,
            failed=0,
        )

        assert result.success
        assert result.total_files == 10
        assert result.completed == 10
        assert result.failed == 0
        assert result.errors == []

    def test_failed_result(self):
        """Test failed result with errors."""
        result = ParallelResult(
            success=False,
            total_files=10,
            completed=7,
            failed=3,
            errors=["Error 1", "Error 2"],
        )

        assert not result.success
        assert result.failed == 3
        assert len(result.errors) == 2


class TestLocalBackend:
    """Tests for LocalBackend."""

    def test_check_available(self):
        """Test checking if GNU parallel is available."""
        backend = LocalBackend(ParallelConfig())
        # Result depends on system, but shouldn't raise
        result = backend.check_available()
        assert isinstance(result, bool)

    def test_generate_scripts(self):
        """Test generating local execution scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ParallelConfig(
                workers=4,
                output_dir=tmpdir,
            )
            backend = LocalBackend(config)

            files = [Path(f"/data/file{i}.nc") for i in range(5)]
            script_dir = Path(tmpdir) / "scripts"

            scripts = backend.generate_scripts(files, script_dir, config)

            assert "run_parallel.sh" in scripts
            assert "input_files.txt" in scripts

            # Check script is executable
            run_script = scripts["run_parallel.sh"]
            assert run_script.exists()
            assert os.access(run_script, os.X_OK)

            # Check file list
            file_list = scripts["input_files.txt"]
            assert file_list.exists()
            content = file_list.read_text()
            assert "/data/file0.nc" in content

    @patch("shutil.which")
    @patch("redi.parallel.backends.local.subprocess.run")
    def test_execute_success(self, mock_run, mock_which):
        """Test successful parallel execution."""
        mock_which.return_value = "/usr/bin/parallel"  # GNU parallel available
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ParallelConfig(workers=2, output_dir=tmpdir)
            backend = LocalBackend(config)

            # Create mock output directories
            for i in range(3):
                (Path(tmpdir) / f"file{i}.out").mkdir()

            files = [Path(f"file{i}.nc") for i in range(3)]
            batches = [[f] for f in files]

            result = backend.execute(batches, config)

            assert result.total_files == 3
            # mock_run should have been called
            mock_run.assert_called_once()


class TestSlurmBackend:
    """Tests for SlurmBackend."""

    def test_generate_scripts(self):
        """Test generating SLURM scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ParallelConfig(
                backend="slurm",
                output_dir=tmpdir,
                slurm_nodes=4,
                slurm_account="testacct",
                slurm_time="1:00:00",
            )
            backend = SlurmBackend(config)

            files = [Path(f"/data/file{i}.nc") for i in range(100)]
            script_dir = Path(tmpdir) / "slurm_scripts"

            scripts = backend.generate_scripts(files, script_dir, config)

            assert "multinodejob.sh" in scripts
            assert "driver.sh" in scripts
            assert "payload.sh" in scripts
            assert "inputdatalist.txt" in scripts

            # Check job script content
            job_script = scripts["multinodejob.sh"]
            content = job_script.read_text()
            assert "#SBATCH -A testacct" in content
            assert "#SBATCH -N 4" in content
            assert "#SBATCH -t 1:00:00" in content

            # Check driver script
            driver_script = scripts["driver.sh"]
            driver_content = driver_script.read_text()
            assert "SLURM_NODEID" in driver_content
            assert "parallel" in driver_content

            # Check payload script
            payload_script = scripts["payload.sh"]
            payload_content = payload_script.read_text()
            assert "redi" in payload_content

            # Check file list
            file_list = scripts["inputdatalist.txt"]
            lines = file_list.read_text().strip().split("\n")
            assert len(lines) == 100


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_get_local_backend(self):
        """Test getting local backend."""
        config = ParallelConfig(backend="local")
        backend = get_backend("local", config)
        assert isinstance(backend, LocalBackend)

    def test_get_slurm_backend(self):
        """Test getting SLURM backend."""
        config = ParallelConfig(backend="slurm")
        backend = get_backend("slurm", config)
        assert isinstance(backend, SlurmBackend)

    def test_get_unknown_backend(self):
        """Test getting unknown backend raises error."""
        config = ParallelConfig()
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown", config)


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_create_executor(self):
        """Test creating an executor."""
        executor = ParallelExecutor(
            backend="local",
            workers=4,
            batch_size=2,
        )

        assert executor.config.backend == "local"
        assert executor.config.workers == 4
        assert executor.config.batch_size == 2

    def test_generate_scripts(self):
        """Test script generation via executor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ParallelExecutor(
                backend="local",
                workers=4,
                output_dir=tmpdir,
            )

            files = [Path(f"/data/file{i}.nc") for i in range(5)]
            scripts = executor.generate_scripts(files)

            assert "run_parallel.sh" in scripts
            assert "input_files.txt" in scripts


class TestChunking:
    """Tests for Type 2 chunking (stubs)."""

    def test_chunk_spec(self):
        """Test ChunkSpec dataclass."""
        spec = ChunkSpec(
            source_path=Path("/data/file.nc"),
            chunk_id=0,
            start=0,
            end=100,
            metadata={"variable": "temperature"},
        )

        assert spec.source_path == Path("/data/file.nc")
        assert spec.chunk_id == 0
        assert spec.metadata["variable"] == "temperature"

    def test_time_chunk_strategy_not_implemented(self):
        """Test TimeChunkStrategy raises NotImplementedError."""
        strategy = TimeChunkStrategy()
        assert strategy.name == "time"

        with pytest.raises(NotImplementedError):
            strategy.can_chunk(Path("file.nc"))

        with pytest.raises(NotImplementedError):
            strategy.chunk(Path("file.nc"), 4)

    def test_row_chunk_strategy_not_implemented(self):
        """Test RowChunkStrategy raises NotImplementedError."""
        strategy = RowChunkStrategy()
        assert strategy.name == "row"

        with pytest.raises(NotImplementedError):
            strategy.can_chunk(Path("file.csv"))

    def test_get_chunk_strategy(self):
        """Test getting chunk strategy by name."""
        time_strategy = get_chunk_strategy("time")
        assert isinstance(time_strategy, TimeChunkStrategy)

        row_strategy = get_chunk_strategy("row")
        assert isinstance(row_strategy, RowChunkStrategy)

    def test_get_unknown_chunk_strategy(self):
        """Test getting unknown chunk strategy raises error."""
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            get_chunk_strategy("unknown")


class TestConfigIntegration:
    """Tests for config.py parallel integration."""

    def test_parallel_defaults(self):
        """Test parallel defaults in config."""
        from redi.config import DEFAULTS

        assert "parallel" in DEFAULTS
        assert DEFAULTS["parallel"]["enabled"] is False
        assert DEFAULTS["parallel"]["backend"] == "local"
        assert DEFAULTS["parallel"]["batch_size"] == 4
        assert "slurm" in DEFAULTS["parallel"]

    def test_parallel_config_validation(self):
        """Test parallel config validation."""
        from redi.config import validate_config

        # Valid config
        config = {
            "parallel": {
                "enabled": True,
                "backend": "local",
                "workers": 4,
                "distribution": "round_robin",
            }
        }
        issues = validate_config(config)
        assert len(issues) == 0

        # Invalid backend
        config = {
            "parallel": {
                "enabled": True,
                "backend": "invalid",
            }
        }
        issues = validate_config(config)
        assert any("backend" in issue for issue in issues)

        # Invalid workers
        config = {
            "parallel": {
                "enabled": True,
                "backend": "local",
                "workers": -1,
            }
        }
        issues = validate_config(config)
        assert any("workers" in issue for issue in issues)
