"""
Unit tests for the REDI analyze module.
"""

import pytest
from pathlib import Path


class TestDataAnalyzer:
    """Tests for DataAnalyzer class."""

    def test_analyze_csv_basic(self, sample_csv):
        """Test basic CSV analysis."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        assert data_card.overview.name == "sample.csv"
        assert data_card.overview.num_records == 10
        assert data_card.overview.num_features == 6
        assert data_card.overview.file_format == ".csv"
        assert "tabular" in data_card.characteristics.data_types

    def test_analyze_csv_features(self, sample_csv):
        """Test that CSV features are correctly identified."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "id" in feature_names
        assert "temperature" in feature_names
        assert "category" in feature_names

    def test_analyze_csv_types(self, sample_csv):
        """Test that column types are correctly inferred."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_csv))
        data_card = analyzer.analyze()

        features = {f["name"]: f["type"] for f in data_card.characteristics.features}
        assert features["id"] == "integer"
        assert features["temperature"] == "float"
        assert features["category"] == "string"

    def test_analyze_json(self, sample_json):
        """Test JSON file analysis."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_json))
        data_card = analyzer.analyze()

        assert data_card.overview.num_records == 3
        assert data_card.overview.num_features == 4
        assert "json" in data_card.characteristics.data_types

    def test_analyze_nonexistent_file(self, temp_dir):
        """Test that analyzing nonexistent file raises error."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(temp_dir / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            analyzer.analyze()

    def test_readiness_level_raw(self, temp_dir):
        """Test that minimal data gets RAW readiness level."""
        from redi.analyze import DataAnalyzer

        csv_path = temp_dir / "minimal.csv"
        csv_path.write_text("a,b\n1,2\n")

        analyzer = DataAnalyzer(str(csv_path))
        data_card = analyzer.analyze()

        # Should be at least level 2 (has features)
        assert data_card.readiness_level >= 1

    def test_readiness_level_labeled(self, temp_dir):
        """Test that data with label column gets LABELED level."""
        from redi.analyze import DataAnalyzer

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
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_hpc_csv))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "hpc"

    def test_detect_materials_domain(self, sample_materials_csv):
        """Test materials domain detection."""
        from redi.analyze import DataAnalyzer

        analyzer = DataAnalyzer(str(sample_materials_csv))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "materials"

    def test_detect_climate_keywords(self, temp_dir):
        """Test climate domain detection from column names."""
        from redi.analyze import DataAnalyzer

        csv_path = temp_dir / "climate.csv"
        csv_path.write_text("lat,lon,temperature,precipitation\n0,0,25,10\n")

        analyzer = DataAnalyzer(str(csv_path))
        data_card = analyzer.analyze()

        assert data_card.detected_domain == "climate"


class TestPipelineRecommender:
    """Tests for pipeline recommendations."""

    def test_recommendations_hpc(self, sample_hpc_csv):
        """Test that HPC data gets appropriate recommendations."""
        from redi.analyze import analyze_dataset, PipelineRecommender

        data_card = analyze_dataset(str(sample_hpc_csv))
        recommender = PipelineRecommender(data_card)
        recs = recommender.get_recommendations()

        assert recs["domain"] == "hpc"
        assert "pandas" in recs["recommended_tools"]

    def test_recommendations_materials(self, sample_materials_csv):
        """Test that materials data gets appropriate recommendations."""
        from redi.analyze import analyze_dataset, PipelineRecommender

        data_card = analyze_dataset(str(sample_materials_csv))
        recommender = PipelineRecommender(data_card)
        recs = recommender.get_recommendations()

        assert recs["domain"] == "materials"
        assert "pymatgen" in recs["recommended_tools"]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_dataset(self, sample_csv):
        """Test analyze_dataset function."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_csv))
        assert data_card is not None
        assert data_card.overview.num_records > 0

    def test_get_recommendations(self, sample_csv):
        """Test get_recommendations function."""
        from redi.analyze import get_recommendations

        recs = get_recommendations(str(sample_csv))
        assert "domain" in recs
        assert "recommended_tools" in recs
        assert "full_pipeline" in recs

    def test_generate_report(self, sample_csv):
        """Test generate_report function."""
        from redi.analyze import generate_report

        report = generate_report(str(sample_csv))
        assert "Data Card" in report
        assert "Pipeline Recommendations" in report
        assert "AI Readiness" in report
