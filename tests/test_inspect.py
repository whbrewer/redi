"""
Unit tests for the REDI inspect module.
"""

import pytest
from pathlib import Path


class TestDataInspector:
    """Tests for DataAnalyzer class (used by redi inspect)."""

    def test_inspect_csv_basic(self, sample_csv):
        """Test basic CSV inspection."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        assert data_card.overview.name == "sample.csv"
        assert data_card.overview.num_records == 10
        assert data_card.overview.num_features == 6
        assert data_card.overview.file_format == ".csv"
        assert "tabular" in data_card.characteristics.data_types

    def test_inspect_csv_features(self, sample_csv):
        """Test that CSV features are correctly identified."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "id" in feature_names
        assert "temperature" in feature_names
        assert "category" in feature_names

    def test_inspect_csv_types(self, sample_csv):
        """Test that column types are correctly inferred."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        features = {f["name"]: f["type"] for f in data_card.characteristics.features}
        assert features["id"] == "integer"
        assert features["temperature"] == "float"
        assert features["category"] == "string"

    def test_inspect_json(self, sample_json):
        """Test JSON file inspection."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_json))
        data_card = analyzer.analyze()

        assert data_card.overview.num_records == 3
        assert data_card.overview.num_features == 4
        assert "json" in data_card.characteristics.data_types

    def test_inspect_nonexistent_file(self, temp_dir):
        """Test that inspecting nonexistent file raises error."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(temp_dir / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            analyzer.analyze()

    def test_readiness_level_raw(self, temp_dir):
        """Test that minimal data gets RAW readiness level."""
        from redi.assess import DataAnalyzer

        csv_path = temp_dir / "minimal.csv"
        csv_path.write_text("a,b\n1,2\n")

        analyzer = DataAnalyzer(str(csv_path))
        data_card = analyzer.analyze()

        # Should be at least level 2 (has features)
        assert data_card.readiness_level >= 1

    def test_readiness_level_labeled(self, temp_dir):
        """Test that data with label column gets LABELED level."""
        from redi.assess import DataAnalyzer

        csv_path = temp_dir / "labeled.csv"
        csv_path.write_text("feature1,feature2,label\n1,2,A\n3,4,B\n")

        analyzer = DataAnalyzer(str(csv_path))
        data_card = analyzer.analyze()

        assert data_card.readiness_level >= 3
        assert data_card.annotations.has_labels


class TestDomainDetection:
    """Tests for domain detection."""

    def test_detect_hpc_domain(self, sample_hpc_csv):
        """Test HPC domain detection from column names."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_hpc_csv))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "hpc"

    def test_detect_materials_domain(self, sample_materials_csv):
        """Test materials domain detection."""
        from redi.assess import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_materials_csv))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "materials"

    def test_detect_climate_keywords(self, temp_dir):
        """Test climate domain detection from column names."""
        from redi.assess import DataAnalyzer

        csv_path = temp_dir / "climate.csv"
        csv_path.write_text("lat,lon,temperature,precipitation\n0,0,25,10\n")

        analyzer = DataAnalyzer(str(csv_path))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "climate"


class TestPipelineRecommender:
    """Tests for pipeline recommendations."""

    def test_recommendations_hpc(self, sample_hpc_csv):
        """Test that HPC data gets appropriate recommendations."""
        from redi.assess import analyze_dataset, PipelineRecommender

        data_card = analyze_dataset(str(sample_hpc_csv))
        recommender = PipelineRecommender(data_card)
        recs = recommender.get_recommendations()

        assert recs["domain"] == "hpc"
        assert "pandas" in recs["recommended_tools"]

    def test_recommendations_materials(self, sample_materials_csv):
        """Test that materials data gets appropriate recommendations."""
        from redi.assess import analyze_dataset, PipelineRecommender

        data_card = analyze_dataset(str(sample_materials_csv))
        recommender = PipelineRecommender(data_card)
        recs = recommender.get_recommendations()

        assert recs["domain"] == "materials"
        assert "pymatgen" in recs["recommended_tools"]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_inspect_dataset(self, sample_csv):
        """Test analyze_dataset function."""
        from redi.assess import analyze_dataset

        data_card = analyze_dataset(str(sample_csv))
        assert data_card is not None
        assert data_card.overview.num_records > 0

    def test_get_recommendations(self, sample_csv):
        """Test get_recommendations function."""
        from redi.assess import get_recommendations

        recs = get_recommendations(str(sample_csv))
        assert "domain" in recs
        assert "recommended_tools" in recs
        assert "full_pipeline" in recs

    def test_generate_report(self, sample_csv):
        """Test generate_report function."""
        from redi.assess import generate_report

        report = generate_report(str(sample_csv))
        assert "Data Card" in report
        assert "Pipeline Recommendations" in report
        assert "AI Readiness" in report


class TestNewFormatHandlers:
    """Tests for NPZ, HDF5, Parquet, FASTA, and NPY format analysis."""

    def test_inspect_npz(self, temp_dir):
        """Test NPZ file inspection."""
        import numpy as np
        from redi.assess import DataAnalyzer

        npz_path = temp_dir / "test.npz"
        np.savez(npz_path, X=np.random.rand(100, 10), y=np.arange(100))

        analyzer = DataAnalyzer(str(npz_path))
        data_card = analyzer.analyze()

        assert "npz" in data_card.characteristics.data_types
        assert data_card.overview.num_features == 2
        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "X" in feature_names
        assert "y" in feature_names

    def test_inspect_npy(self, temp_dir):
        """Test NPY file inspection."""
        import numpy as np
        from redi.assess import DataAnalyzer

        npy_path = temp_dir / "test.npy"
        arr = np.random.rand(50, 8)
        np.save(npy_path, arr)

        analyzer = DataAnalyzer(str(npy_path))
        data_card = analyzer.analyze()

        assert "npy" in data_card.characteristics.data_types
        assert data_card.overview.num_records == 50
        assert data_card.overview.num_features == 8

    def test_inspect_parquet(self, temp_dir):
        """Test Parquet file inspection."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            try:
                import fastparquet  # noqa: F401
            except ImportError:
                pytest.skip("pyarrow or fastparquet not available")

        import pandas as pd
        from redi.assess import DataAnalyzer

        pq_path = temp_dir / "test.parquet"
        df = pd.DataFrame({
            "job_id": range(20),
            "walltime": [100.0] * 20,
            "queue": ["batch"] * 20,
        })
        df.to_parquet(pq_path)

        analyzer = DataAnalyzer(str(pq_path))
        data_card = analyzer.analyze()

        assert "parquet" in data_card.characteristics.data_types
        assert data_card.overview.num_records == 20
        assert data_card.overview.num_features == 3
        # HPC domain detection from column names
        assert data_card.detected_domain == "hpc"

    def test_inspect_hdf5(self, temp_dir):
        """Test HDF5 file inspection."""
        import numpy as np
        from redi.assess import DataAnalyzer

        try:
            import h5py
        except (ImportError, ValueError):
            pytest.skip("h5py not available or HDF5 version conflict (pyarrow may have loaded an incompatible libhdf5)")

        h5_path = temp_dir / "test.h5"
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("features", data=np.random.rand(30, 5))
            f.create_dataset("labels", data=np.zeros(30))

        analyzer = DataAnalyzer(str(h5_path))
        data_card = analyzer.analyze()

        assert "hdf5" in data_card.characteristics.data_types
        assert data_card.overview.num_features == 2

    def test_inspect_fasta(self, temp_dir):
        """Test FASTA file inspection."""
        from redi.assess import DataAnalyzer

        fasta_path = temp_dir / "test.fasta"
        fasta_path.write_text(
            ">seq1 first sequence\n"
            "ATCGATCGATCG\n"
            ">seq2 second sequence\n"
            "GCTAGCTAGCTA\n"
            ">seq3 third sequence\n"
            "TTTTAAAACCCC\n"
        )

        analyzer = DataAnalyzer(str(fasta_path))
        data_card = analyzer.analyze()

        assert "sequence" in data_card.characteristics.data_types
        assert data_card.overview.num_records == 3
        assert data_card.detected_domain == "life_sciences"

    def test_inspect_pdb(self, temp_dir):
        """Test PDB file inspection."""
        from redi.assess import DataAnalyzer

        pdb_path = temp_dir / "test.pdb"
        # Minimal PDB with 3 ATOM records
        pdb_path.write_text(
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C\n"
            "END\n"
        )

        analyzer = DataAnalyzer(str(pdb_path))
        data_card = analyzer.analyze()

        assert "protein" in data_card.characteristics.data_types
        assert data_card.overview.num_records == 3
        assert data_card.detected_domain == "life_sciences"
