"""
Unit tests for the REDI catalog module.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestCatalogConfig:
    """Tests for CatalogConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from redi.catalog.base import CatalogConfig

        config = CatalogConfig()
        assert config.host == ""
        assert config.token == ""
        assert config.service_name == "redi"
        assert config.verify_ssl is True
        assert config.timeout == 30
        assert config.extra == {}

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from redi.catalog.base import CatalogConfig

        data = {
            "host": "https://api.example.com",
            "token": "test-token",
            "service_name": "my-service",
            "verify_ssl": False,
            "timeout": 60,
            "repo_id": "user/dataset",
            "private": True,
        }

        config = CatalogConfig.from_dict(data)
        assert config.host == "https://api.example.com"
        assert config.token == "test-token"
        assert config.service_name == "my-service"
        assert config.verify_ssl is False
        assert config.timeout == 60
        assert config.extra["repo_id"] == "user/dataset"
        assert config.extra["private"] is True


class TestCatalogExportResult:
    """Tests for CatalogExportResult."""

    def test_success_result(self):
        """Test creating a success result."""
        from redi.catalog.base import CatalogExportResult

        result = CatalogExportResult(
            success=True,
            catalog_type="huggingface",
            entity_id="user/my-dataset",
            entity_url="https://huggingface.co/datasets/user/my-dataset",
            message="Export successful",
            metadata={"splits": ["train", "val", "test"]},
        )

        assert result.success is True
        assert result.catalog_type == "huggingface"
        assert result.entity_id == "user/my-dataset"
        assert result.timestamp  # Should have auto-generated timestamp

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        from redi.catalog.base import CatalogExportResult

        result = CatalogExportResult(
            success=True,
            catalog_type="huggingface",
            entity_id="user/dataset",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["catalog_type"] == "huggingface"
        assert data["entity_id"] == "user/dataset"
        assert "timestamp" in data


class TestREDIMetadata:
    """Tests for REDIMetadata."""

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        from redi.catalog.base import REDIMetadata

        data = {
            "domain": "climate",
            "input_file": "data/weather.csv",
            "num_rows": 1000,
            "num_columns": 10,
            "feature_columns": ["temp", "pressure", "humidity"],
            "target_column": "rainfall",
            "split": {
                "train_size": 800,
                "val_size": 100,
                "test_size": 100,
            },
            "output_format": "npz",
            "completed_stages": ["ingest", "preprocess", "output"],
        }

        meta = REDIMetadata.from_dict(data)
        assert meta.domain == "climate"
        assert meta.name == "weather"
        assert meta.num_rows == 1000
        assert meta.train_size == 800
        assert meta.val_size == 100
        assert meta.test_size == 100
        assert "temp" in meta.feature_columns

    def test_metadata_from_file(self, temp_dir):
        """Test loading metadata from file."""
        from redi.catalog.base import REDIMetadata

        metadata = {
            "domain": "general",
            "input_file": "sample.csv",
            "num_rows": 100,
            "num_columns": 5,
            "split": {"train_size": 80, "val_size": 10, "test_size": 10},
        }

        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        meta = REDIMetadata.from_file(str(meta_path))
        assert meta.domain == "general"
        assert meta.num_rows == 100
        assert meta.train_size == 80

    def test_metadata_file_not_found(self):
        """Test error when metadata file doesn't exist."""
        from redi.catalog.base import REDIMetadata

        with pytest.raises(FileNotFoundError):
            REDIMetadata.from_file("/nonexistent/metadata.json")

    def test_name_sanitization(self):
        """Test that names are sanitized for catalog compatibility."""
        from redi.catalog.base import REDIMetadata

        data = {
            "input_file": "my data file (v2).csv",
            "domain": "general",
        }

        meta = REDIMetadata.from_dict(data)
        # Name should only contain alphanumeric and underscores
        assert meta.name == "my_data_file__v2_"
        assert all(c.isalnum() or c == "_" for c in meta.name)


class TestCatalogType:
    """Tests for CatalogType enum."""

    def test_catalog_types(self):
        """Test that all expected catalog types exist."""
        from redi.catalog.base import CatalogType

        assert CatalogType.HUGGINGFACE.value == "huggingface"
        assert CatalogType.OPENMETADATA.value == "openmetadata"
        assert CatalogType.MLFLOW.value == "mlflow"
        assert CatalogType.INTAKE.value == "intake"


class TestHuggingFaceExporter:
    """Tests for HuggingFaceExporter."""

    def test_exporter_creation(self):
        """Test creating a HuggingFace exporter."""
        from redi.catalog.huggingface import HuggingFaceExporter
        from redi.catalog.base import CatalogType

        exporter = HuggingFaceExporter()
        assert exporter.catalog_type == CatalogType.HUGGINGFACE

    def test_exporter_configure(self):
        """Test configuring the exporter."""
        from redi.catalog.huggingface import HuggingFaceExporter

        exporter = HuggingFaceExporter()
        exporter.configure(
            token="hf_test_token",
            repo_id="user/dataset",
            private=True,
        )

        assert exporter.config.token == "hf_test_token"
        assert exporter.config.extra["repo_id"] == "user/dataset"
        assert exporter.config.extra["private"] is True

    def test_export_missing_token(self, temp_dir):
        """Test export fails without token."""
        from redi.catalog.huggingface import HuggingFaceExporter

        # Create minimal metadata
        metadata = {
            "domain": "general",
            "input_file": "test.csv",
            "num_rows": 10,
            "split": {"train_size": 8, "val_size": 1, "test_size": 1},
        }
        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        exporter = HuggingFaceExporter()
        exporter.configure(repo_id="test-dataset")

        # Clear any env vars
        with patch.dict("os.environ", {}, clear=True):
            result = exporter.export(str(meta_path))

        assert result.success is False
        assert "token" in result.message.lower()

    def test_export_missing_data_files(self, temp_dir):
        """Test export fails when no data files exist."""
        from redi.catalog.huggingface import HuggingFaceExporter

        metadata = {
            "domain": "general",
            "input_file": "test.csv",
            "num_rows": 10,
            "split": {"train_size": 8, "val_size": 1, "test_size": 1},
        }
        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        exporter = HuggingFaceExporter()
        exporter.configure(token="hf_test", repo_id="user/dataset")

        result = exporter.export(str(meta_path))
        assert result.success is False
        assert "No data files found" in result.message

    def test_auto_prefix_username(self, temp_dir):
        """Test that username is auto-prefixed when not in repo_id."""
        from redi.catalog.huggingface import HuggingFaceExporter

        # Create metadata and data files
        metadata = {
            "domain": "general",
            "input_file": "test.csv",
            "num_rows": 10,
            "output_format": "npz",
            "split": {"train_size": 8, "val_size": 1, "test_size": 1},
        }
        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        # Create NPZ files
        X = np.random.rand(8, 3)
        y = np.random.rand(8)
        np.savez(temp_dir / "train.npz", X=X, y=y)
        np.savez(temp_dir / "val.npz", X=X[:1], y=y[:1])
        np.savez(temp_dir / "test.npz", X=X[:1], y=y[:1])

        exporter = HuggingFaceExporter()
        exporter.configure(token="hf_test", repo_id="my-dataset")  # No username

        # Mock the HuggingFace imports inside the export method
        with patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api_instance = Mock()
            mock_api_instance.whoami.return_value = {"name": "testuser"}
            mock_hf_api.return_value = mock_api_instance

            with patch("datasets.DatasetDict") as mock_dd:
                mock_dataset_dict = Mock()
                mock_dataset_dict.keys.return_value = ["train", "val", "test"]
                mock_dataset_dict.items.return_value = [
                    ("train", Mock(__len__=lambda s: 8)),
                    ("val", Mock(__len__=lambda s: 1)),
                    ("test", Mock(__len__=lambda s: 1)),
                ]
                mock_dd.return_value = mock_dataset_dict

                with patch("datasets.Dataset") as mock_dataset:
                    mock_dataset.from_dict.return_value = Mock(__len__=lambda s: 8)

                    result = exporter.export(str(meta_path))

                    # Check that whoami was called to get username
                    mock_api_instance.whoami.assert_called()

    def test_load_npz(self, temp_dir):
        """Test loading NPZ files."""
        from redi.catalog.huggingface import HuggingFaceExporter

        # Create test NPZ file
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        feature_names = np.array(["feature_a", "feature_b"])

        npz_path = temp_dir / "test.npz"
        np.savez(npz_path, X=X, y=y, feature_names=feature_names)

        exporter = HuggingFaceExporter()
        data = exporter._load_npz(npz_path)

        assert data is not None
        assert "feature_a" in data
        assert "feature_b" in data
        assert "target" in data
        assert len(data["feature_a"]) == 3
        assert data["target"] == [0, 1, 0]

    def test_load_npz_without_feature_names(self, temp_dir):
        """Test loading NPZ files without feature names."""
        from redi.catalog.huggingface import HuggingFaceExporter

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        npz_path = temp_dir / "test.npz"
        np.savez(npz_path, X=X, y=y)

        exporter = HuggingFaceExporter()
        data = exporter._load_npz(npz_path)

        assert data is not None
        assert "feature_0" in data
        assert "feature_1" in data
        assert "target" in data

    def test_generate_dataset_card(self):
        """Test dataset card generation."""
        from redi.catalog.huggingface import HuggingFaceExporter
        from redi.catalog.base import REDIMetadata

        meta = REDIMetadata(
            name="test_dataset",
            description="Test dataset for unit tests",
            domain="climate",
            input_file="test.csv",
            num_rows=1000,
            num_columns=10,
            feature_columns=["temp", "pressure", "humidity"],
            target_column="rainfall",
            train_size=800,
            val_size=100,
            test_size=100,
            completed_stages=["ingest", "normalize", "split"],
        )

        exporter = HuggingFaceExporter()
        card = exporter._generate_dataset_card(meta, "user/test-dataset")

        # Check YAML frontmatter
        assert "---" in card
        assert "license: cc-by-4.0" in card
        assert "redi" in card
        assert "domain:climate" in card

        # Check content
        assert "# test_dataset" in card
        assert "Climate" in card or "climate" in card
        assert "train" in card.lower()
        assert "800" in card
        assert "load_dataset" in card

    def test_get_size_category(self):
        """Test size category calculation."""
        from redi.catalog.huggingface import HuggingFaceExporter

        exporter = HuggingFaceExporter()

        assert exporter._get_size_category(500) == "n<1K"
        assert exporter._get_size_category(5000) == "1K<n<10K"
        assert exporter._get_size_category(50000) == "10K<n<100K"
        assert exporter._get_size_category(500000) == "100K<n<1M"
        assert exporter._get_size_category(5000000) == "1M<n<10M"
        assert exporter._get_size_category(50000000) == "n>10M"


class TestExportToCatalog:
    """Tests for the export_to_catalog convenience function."""

    def test_export_to_catalog_huggingface(self, temp_dir):
        """Test export_to_catalog with huggingface type."""
        from redi.catalog import export_to_catalog

        metadata = {
            "domain": "general",
            "input_file": "test.csv",
            "num_rows": 10,
            "split": {"train_size": 8, "val_size": 1, "test_size": 1},
        }
        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        # This should fail gracefully (no data files)
        result = export_to_catalog(
            str(meta_path),
            catalog_type="huggingface",
            token="hf_test",
            repo_id="user/test",
        )

        assert result.success is False
        assert result.catalog_type == "huggingface"

    def test_list_catalogs(self):
        """Test listing available catalogs."""
        from redi.catalog import list_catalogs

        catalogs = list_catalogs()
        assert "huggingface" in catalogs


class TestHuggingFaceIntegration:
    """Integration tests for HuggingFace export (with mocks)."""

    def test_full_export_flow(self, temp_dir):
        """Test the full export flow with mocked HuggingFace API."""
        from redi.catalog.huggingface import HuggingFaceExporter

        # Create metadata
        metadata = {
            "domain": "climate",
            "input_file": "weather.csv",
            "num_rows": 10,
            "num_columns": 3,
            "feature_columns": ["temp", "pressure"],
            "target_column": "rain",
            "output_format": "npz",
            "split": {"train_size": 8, "val_size": 1, "test_size": 1},
            "completed_stages": ["ingest", "preprocess", "output"],
        }
        meta_path = temp_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        # Create NPZ files
        X = np.random.rand(8, 2)
        y = np.random.rand(8)
        np.savez(temp_dir / "train.npz", X=X, y=y, feature_names=["temp", "pressure"])
        np.savez(temp_dir / "val.npz", X=X[:1], y=y[:1], feature_names=["temp", "pressure"])
        np.savez(temp_dir / "test.npz", X=X[:1], y=y[:1], feature_names=["temp", "pressure"])

        # Mock the HuggingFace imports
        with patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api_instance = Mock()
            mock_api_instance.whoami.return_value = {"name": "testuser"}
            mock_api_instance.upload_file.return_value = None
            mock_hf_api.return_value = mock_api_instance

            with patch("datasets.DatasetDict") as mock_dd:
                mock_dataset_dict_instance = Mock()
                mock_dataset_dict_instance.push_to_hub.return_value = None
                mock_dataset_dict_instance.keys.return_value = ["train", "val", "test"]
                mock_dataset_dict_instance.items.return_value = [
                    ("train", Mock(__len__=lambda s: 8)),
                    ("val", Mock(__len__=lambda s: 1)),
                    ("test", Mock(__len__=lambda s: 1)),
                ]
                mock_dd.return_value = mock_dataset_dict_instance

                with patch("datasets.Dataset") as mock_dataset:
                    mock_dataset.from_dict.return_value = Mock(__len__=lambda s: 8)

                    # Run export
                    exporter = HuggingFaceExporter()
                    exporter.configure(token="hf_test_token", repo_id="testuser/weather-data", private=True)
                    result = exporter.export(str(meta_path))

                    # Verify
                    assert result.success is True
                    assert result.entity_id == "testuser/weather-data"
                    assert "huggingface.co" in result.entity_url
                    assert result.metadata["splits"] == ["train", "val", "test"]

                    # Verify API calls
                    mock_dataset_dict_instance.push_to_hub.assert_called_once()
                    mock_api_instance.upload_file.assert_called_once()
