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
