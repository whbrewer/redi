"""
Smoke tests for HPC domain pipeline.

These tests verify that the complete HPC pipeline runs end-to-end
without errors on sample data.
"""

import pytest
from pathlib import Path


class TestHPCSmoke:
    """Smoke tests for HPC domain."""

    def test_hpc_analyze(self, sample_hpc_csv):
        """Test analyzing HPC job data."""
        from redi.analyze import analyze_dataset, generate_report

        data_card = analyze_dataset(str(sample_hpc_csv))

        assert data_card.detected_domain == "hpc"
        assert data_card.overview.num_records == 5
        assert data_card.readiness_level >= 1

        # Generate report should not raise
        report = generate_report(str(sample_hpc_csv))
        assert "HPC" in report or "hpc" in report

    def test_hpc_full_pipeline(self, sample_hpc_csv, temp_dir):
        """Test running full HPC pipeline."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "hpc_output"
        ctx = run_pipeline(
            input_path=str(sample_hpc_csv),
            output_dir=str(output_dir),
            domain="hpc"
        )

        assert ctx is not None
        # Pipeline should complete (may have some non-fatal errors)
        assert ctx.domain == "hpc"

        # Check that output files were created
        if ctx.metadata.get("saved_files"):
            for f in ctx.metadata["saved_files"]:
                assert Path(f).exists()

    def test_hpc_ingest_only(self, sample_hpc_csv, temp_dir):
        """Test running only ingest stage."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "hpc_ingest"
        ctx = run_pipeline(
            input_path=str(sample_hpc_csv),
            output_dir=str(output_dir),
            domain="hpc",
            stages=["ingest"]
        )

        assert ctx is not None
        assert ctx.data is not None

    def test_hpc_domain_detection(self, sample_hpc_csv):
        """Test that HPC domain is auto-detected."""
        from redi.cli import detect_domain

        domain = detect_domain(Path(sample_hpc_csv))
        assert domain == "hpc"

    def test_hpc_feature_engineering(self, sample_hpc_csv, temp_dir):
        """Test HPC-specific feature engineering."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "hpc_features"
        ctx = run_pipeline(
            input_path=str(sample_hpc_csv),
            output_dir=str(output_dir),
            domain="hpc",
            stages=["ingest", "preprocess", "transform"]
        )

        # Check that timestamp features were engineered
        if ctx.metadata.get("engineered_features"):
            feature_names = ctx.metadata["engineered_features"]
            # Should have time-based features
            assert any("hour" in f or "dayofweek" in f for f in feature_names)


class TestHPCRapsJobs:
    """Smoke tests for RAPS Job extraction pipeline."""

    def test_raps_jobs_pipeline(self, sample_hpc_csv, temp_dir):
        """Test running the RAPS job extraction pipeline end-to-end."""
        from redi.cli import run_pipeline
        import json
        import numpy as np

        output_dir = temp_dir / "raps_output"
        ctx = run_pipeline(
            input_path=str(sample_hpc_csv),
            output_dir=str(output_dir),
            domain="hpc",
            raps_jobs=True,
            style="simple",
        )

        assert ctx is not None

        # Output should be a list of job dicts
        assert isinstance(ctx.data, list)
        assert len(ctx.data) > 0

        # Check that NPZ snapshot was written (primary RAPS input)
        npz_path = output_dir / "raps_jobs.npz"
        assert npz_path.exists()

        data = np.load(str(npz_path), allow_pickle=True)
        assert "jobs" in data
        assert "telemetry_start" in data
        assert "telemetry_end" in data
        assert "start_date" in data
        assert len(data["jobs"]) > 0

        # Verify jobs from the pipeline context
        jobs = ctx.data
        assert len(jobs) > 0

        # Each job should have all canonical fields + RAPS trace metadata
        required_keys = {"id", "name", "nodes_required", "start_time", "end_time",
                         "current_state", "trace_quanta", "downscale"}
        for job in jobs:
            for key in required_keys:
                assert key in job, f"Missing key '{key}' in job dict"

        # Check mapping metadata was saved
        meta_path = output_dir / "raps_mapping_metadata.json"
        assert meta_path.exists()

    def test_raps_jobs_field_values(self, sample_hpc_csv, temp_dir):
        """Test that RAPS job dicts have sensible values."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "raps_values"
        ctx = run_pipeline(
            input_path=str(sample_hpc_csv),
            output_dir=str(output_dir),
            domain="hpc",
            raps_jobs=True,
            style="simple",
        )

        assert isinstance(ctx.data, list)
        for job in ctx.data:
            # nodes_required should be positive
            assert job["nodes_required"] > 0
            # Times should be numeric
            assert isinstance(job["start_time"], (int, float))
            assert isinstance(job["end_time"], (int, float))
            # end_state should be a RAPS state string
            assert isinstance(job["end_state"], str)
            assert job["end_state"] in (
                "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
                "NODE_FAIL", "OUT_OF_MEMORY", "RUNNING", "PENDING",
            )


class TestHPCDataCard:
    """Tests for HPC Data Card generation."""

    def test_hpc_data_card_features(self, sample_hpc_csv):
        """Test that HPC data card has correct features."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_hpc_csv))

        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "JOB_ID" in feature_names
        assert "RUNTIME_SECONDS" in feature_names
        assert "EXIT_CODE" in feature_names

    def test_hpc_recommendations(self, sample_hpc_csv):
        """Test HPC pipeline recommendations."""
        from redi.analyze import get_recommendations

        recs = get_recommendations(str(sample_hpc_csv))

        assert recs["domain"] == "hpc"
        assert "pandas" in recs["recommended_tools"]
        assert any("Parquet" in fmt or "CSV" in fmt for fmt in recs["output_formats"])
