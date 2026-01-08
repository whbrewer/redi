"""Tests for YAML configuration support."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample YAML config file."""
    config_content = """
domain: climate
output_format: npz
output_dir: ./test_output

regrid:
  target_resolution: 2.0
  method: conservative

normalize:
  method: zscore
  two_pass: true

split:
  method: temporal
  train:
    start: "2000-01-01"
    end: "2015-12-31"
  val:
    start: "2016-01-01"
    end: "2017-12-31"
  test:
    start: "2018-01-01"
    end: "2019-12-31"

climax_output:
  enabled: true
  save_coordinates: true
  save_normalization: true
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """Create a minimal YAML config file."""
    config_content = """
domain: climate
"""
    config_path = tmp_path / "minimal_config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestYAMLConfig:
    """Tests for YAML configuration loading."""

    def test_load_yaml_config(self, sample_config_yaml):
        """Test loading YAML configuration."""
        from redi.config import load_yaml_config

        config = load_yaml_config(str(sample_config_yaml))

        assert config['domain'] == 'climate'
        assert config['regrid']['target_resolution'] == 2.0
        assert config['normalize']['two_pass'] is True
        assert config['split']['method'] == 'temporal'

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file raises error."""
        from redi.config import load_yaml_config

        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_merge_with_defaults(self, sample_config_yaml):
        """Test merging config with defaults."""
        from redi.config import load_yaml_config, merge_with_defaults

        config = load_yaml_config(str(sample_config_yaml))
        merged = merge_with_defaults(config)

        # User values preserved
        assert merged['regrid']['target_resolution'] == 2.0
        assert merged['regrid']['method'] == 'conservative'

        # Defaults applied for missing values
        assert merged['regrid']['lat_range'] == [-90, 90]
        assert merged['regrid']['lon_range'] == [0, 360]
        assert merged['shard']['enabled'] is False

    def test_merge_minimal_config(self, minimal_config_yaml):
        """Test merging minimal config fills in all defaults."""
        from redi.config import load_yaml_config, merge_with_defaults

        config = load_yaml_config(str(minimal_config_yaml))
        merged = merge_with_defaults(config)

        assert merged['domain'] == 'climate'
        assert merged['regrid']['target_resolution'] == 1.0
        assert merged['normalize']['method'] == 'zscore'
        assert merged['split']['method'] == 'random'


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_temporal_split_config_valid(self):
        """Test validation of valid temporal split configuration."""
        from redi.config import validate_config

        valid_config = {
            'split': {
                'method': 'temporal',
                'train': {'start': '2000-01-01', 'end': '2015-12-31'},
                'val': {'start': '2016-01-01', 'end': '2017-12-31'},
                'test': {'start': '2018-01-01', 'end': '2019-12-31'},
            }
        }
        issues = validate_config(valid_config)
        assert len(issues) == 0

    def test_validate_temporal_split_missing_dates(self):
        """Test validation catches missing dates in temporal split."""
        from redi.config import validate_config

        invalid_config = {
            'split': {
                'method': 'temporal',
                'train': {'start': '2000-01-01'},  # Missing end
                'val': {'start': '2016-01-01', 'end': '2017-12-31'},
                'test': {'start': '2018-01-01', 'end': '2019-12-31'},
            }
        }
        issues = validate_config(invalid_config)
        assert len(issues) > 0
        assert 'train' in issues[0]

    def test_validate_invalid_normalization_method(self):
        """Test validation catches invalid normalization method."""
        from redi.config import validate_config

        invalid_config = {
            'normalize': {
                'method': 'invalid_method'
            }
        }
        issues = validate_config(invalid_config)
        assert len(issues) > 0
        assert 'normalization' in issues[0].lower()

    def test_validate_random_split_ratios(self):
        """Test validation of random split ratios."""
        from redi.config import validate_config

        invalid_config = {
            'split': {
                'method': 'random',
                'ratios': {'train': 0.5, 'val': 0.3, 'test': 0.3}  # Sum > 1
            }
        }
        issues = validate_config(invalid_config)
        assert len(issues) > 0
        assert 'ratio' in issues[0].lower()


class TestGenerateExampleConfig:
    """Tests for example config generation."""

    def test_generate_example_config(self):
        """Test generating example configuration."""
        from redi.config import generate_example_config

        example = generate_example_config()

        assert 'domain: climate' in example
        assert 'regrid:' in example
        assert 'normalize:' in example
        assert 'split:' in example
        assert 'climax_output:' in example


class TestStepConfig:
    """Tests for step-specific configuration."""

    def test_get_step_config_regrid(self):
        """Test getting regrid step config."""
        from redi.config import get_step_config

        config = {
            'regrid': {
                'target_resolution': 2.0,
                'method': 'conservative'
            }
        }

        step_config = get_step_config(config, 'regrid_climate')
        assert step_config['target_resolution'] == 2.0
        assert step_config['method'] == 'conservative'

    def test_get_step_config_missing(self):
        """Test getting config for unknown step returns empty dict."""
        from redi.config import get_step_config

        config = {}
        step_config = get_step_config(config, 'unknown_step')
        assert step_config == {}
