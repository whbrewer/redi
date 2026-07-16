"""
REDI Configuration Module

Handles loading, validation, and defaults for YAML configuration files.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS: Dict[str, Any] = {
    'input_path': None,
    'domain': None,  # Auto-detect
    'output_dir': './output',
    'output_format': 'npz',
    'verbose': False,
    'stages': None,  # All stages

    'variables': {
        'select': None,  # All variables
        'pressure_levels': None,  # All levels
    },

    'regrid': {
        'enabled': True,
        'target_resolution': 1.0,
        'method': 'bilinear',
        'lat_range': [-90, 90],
        'lon_range': [0, 360],
    },

    'normalize': {
        'enabled': True,
        'method': 'zscore',
        'two_pass': False,
        'per_variable': True,
        'per_level': False,
        'exclude_variables': [],
    },

    'split': {
        'method': 'random',
        'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'shuffle': True,
        'random_seed': 42,
        'train': {'start': None, 'end': None},
        'val': {'start': None, 'end': None},
        'test': {'start': None, 'end': None},
    },

    'shard': {
        'enabled': False,
        'by': 'year',
        'samples_per_shard': 10000,
    },

    'climax_output': {
        'enabled': False,
        'structure': 'yearly_shards',
        'variable_naming': 'name_level',
        'save_coordinates': True,
        'save_normalization': True,
    },

    'parallel': {
        'enabled': False,
        'backend': 'local',  # local, gnu (SLURM+GNU parallel), mpi (SLURM+mpi4py), dragon
        'workers': None,  # None = auto-detect CPU count
        'batch_size': 4,  # Files per worker batch
        'distribution': 'round_robin',  # round_robin or balanced
        'slurm': {
            'nodes': 4,
            'cores_per_node': 56,
            'time_limit': '0:30:00',
            'account': None,
            'partition': 'batch',
            'burst_buffer': '/mnt/bb/$USER',
            'final_output': None,  # Where to copy results from burst buffer
            'constraint': 'nvme',
            'modules': ['miniforge3/23.11.0-0', 'parallel'],
            'conda_env': 'conda-redi',
        },
        'mpi': {
            'ntasks_per_node': 1,
            'merge': True,           # Gather results and write unified DataCard on rank 0
            'merge_format': 'npz',   # Output format for merged file: npz, hdf5, zarr
        },
        'dragon': {
            'mode': 'subprocess',    # subprocess or inprocess (experimental)
            'nodes': 1,              # number of nodes for multi-node Dragon launch
            'launcher': 'local',     # local (call dragon directly) or slurm
        },
    },
}


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    Recursively merge override dict into base dict (in-place).

    Args:
        base: Base dictionary to merge into
        override: Dictionary with values to override
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default values.

    Missing values in user config will be filled with defaults.
    User values take precedence over defaults.

    Args:
        config: User configuration dictionary

    Returns:
        Merged configuration with all default values filled in
    """
    merged = copy.deepcopy(DEFAULTS)
    _deep_merge(merged, config)
    return merged


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of issues.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error/warning messages (empty if valid)
    """
    issues = []

    # Validate split configuration
    split = config.get('split', {})
    if split.get('method') == 'temporal':
        for split_name in ['train', 'val', 'test']:
            split_cfg = split.get(split_name, {})
            if not split_cfg.get('start') or not split_cfg.get('end'):
                issues.append(
                    f"Temporal split requires 'start' and 'end' dates for '{split_name}'"
                )

    # Validate normalization method
    norm = config.get('normalize', {})
    valid_methods = ['zscore', 'minmax', 'none', None]
    if norm.get('method') not in valid_methods:
        issues.append(
            f"Unknown normalization method: {norm.get('method')}. "
            f"Valid options: {valid_methods}"
        )

    # Validate regrid method
    regrid = config.get('regrid', {})
    valid_regrid = ['bilinear', 'conservative', 'nearest_s2d', 'patch', None]
    if regrid.get('method') not in valid_regrid:
        issues.append(
            f"Unknown regrid method: {regrid.get('method')}. "
            f"Valid options: {valid_regrid}"
        )

    # Validate regrid resolution
    resolution = regrid.get('target_resolution')
    if resolution is not None and (resolution <= 0 or resolution > 10):
        issues.append(
            f"Invalid target_resolution: {resolution}. Must be between 0 and 10 degrees."
        )

    # Validate output format
    valid_formats = ['npz', 'parquet', 'csv', 'hdf5', 'zarr', 'pt', None]
    if config.get('output_format') not in valid_formats:
        issues.append(
            f"Unknown output format: {config.get('output_format')}. "
            f"Valid options: {valid_formats}"
        )

    # Validate domain (built-ins + installed plugins)
    valid_domains = ['climate', 'materials', 'life_sciences', 'fusion', 'hpc', 'geospatial', 'nuclear_physics', 'combustion', 'general', None]
    try:
        from .plugins import get_registry
        valid_domains += [p.domain for p in get_registry().values()]
    except Exception:
        pass
    if config.get('domain') not in valid_domains:
        issues.append(
            f"Unknown domain: {config.get('domain')}. "
            f"Valid options: {valid_domains}"
        )

    # Validate split ratios sum to 1.0 for random splits
    if split.get('method') == 'random':
        ratios = split.get('ratios', {})
        total = sum(ratios.values())
        if abs(total - 1.0) > 0.001:
            issues.append(
                f"Split ratios must sum to 1.0, got {total}"
            )

    # Validate parallel configuration
    parallel = config.get('parallel', {})
    if parallel.get('enabled'):
        # Validate backend
        valid_backends = ['local', 'gnu', 'mpi', 'dragon']
        if parallel.get('backend') not in valid_backends:
            issues.append(
                f"Unknown parallel backend: {parallel.get('backend')}. "
                f"Valid options: {valid_backends}"
            )

        # Validate distribution
        valid_distributions = ['round_robin', 'balanced']
        if parallel.get('distribution') not in valid_distributions:
            issues.append(
                f"Unknown distribution strategy: {parallel.get('distribution')}. "
                f"Valid options: {valid_distributions}"
            )

        # Validate workers
        workers = parallel.get('workers')
        if workers is not None and (not isinstance(workers, int) or workers < 1):
            issues.append(
                f"Invalid workers count: {workers}. Must be a positive integer."
            )

        # Validate batch_size
        batch_size = parallel.get('batch_size')
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            issues.append(
                f"Invalid batch_size: {batch_size}. Must be a positive integer."
            )

        # Validate SLURM config if using a SLURM-based backend
        if parallel.get('backend') in ('gnu', 'mpi'):
            slurm = parallel.get('slurm', {})
            if slurm.get('nodes') and (not isinstance(slurm['nodes'], int) or slurm['nodes'] < 1):
                issues.append(f"Invalid slurm.nodes: {slurm['nodes']}. Must be positive integer.")
            if slurm.get('cores_per_node') and (not isinstance(slurm['cores_per_node'], int) or slurm['cores_per_node'] < 1):
                issues.append(f"Invalid slurm.cores_per_node: {slurm['cores_per_node']}. Must be positive integer.")

        # Validate Dragon config
        if parallel.get('backend') == 'dragon':
            dragon = parallel.get('dragon', {})
            valid_modes = ['subprocess', 'inprocess']
            if dragon.get('mode') and dragon['mode'] not in valid_modes:
                issues.append(
                    f"Invalid dragon.mode: {dragon['mode']}. "
                    f"Valid options: {valid_modes}"
                )
            valid_launchers = ['local', 'slurm']
            if dragon.get('launcher') and dragon['launcher'] not in valid_launchers:
                issues.append(
                    f"Invalid dragon.launcher: {dragon['launcher']}. "
                    f"Valid options: {valid_launchers}"
                )
            if dragon.get('nodes') and (not isinstance(dragon['nodes'], int) or dragon['nodes'] < 1):
                issues.append(f"Invalid dragon.nodes: {dragon['nodes']}. Must be a positive integer.")

    return issues


def get_step_config(config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    """
    Get configuration relevant to a specific pipeline step.

    Args:
        config: Full configuration dictionary
        step_name: Name of the pipeline step

    Returns:
        Configuration dictionary for the step
    """
    step_configs = {
        'regrid_climate': config.get('regrid', {}),
        'normalize_climate': config.get('normalize', {}),
        'compute_norm_stats': config.get('normalize', {}),
        'apply_normalization': config.get('normalize', {}),
        'select_pressure_levels': config.get('variables', {}),
        'select_variables': config.get('variables', {}),
        'temporal_split': config.get('split', {}),
        'train_val_test_split': config.get('split', {}),
        'save_climax': config.get('climax_output', {}),
        'save_climate': {
            'output_format': config.get('output_format', 'npz'),
            **config.get('climax_output', {}),
        },
    }
    return step_configs.get(step_name, {})


def generate_example_config() -> str:
    """
    Generate an example YAML configuration file content.

    Returns:
        YAML string with example configuration
    """
    example = """# REDI Configuration File
# See documentation for full options

# Scientific domain (auto-detected if omitted)
domain: climate

# Input file path (optional when invoking via CLI)
input_path: /path/to/data.nc

# Output settings
output_dir: ./output
output_format: npz  # npz, parquet, csv, hdf5, zarr

# Regridding settings
regrid:
  enabled: true
  target_resolution: 1.0  # degrees
  method: bilinear        # bilinear, conservative, nearest_s2d, patch

# Normalization settings
normalize:
  enabled: true
  method: zscore          # zscore, minmax
  two_pass: true          # compute stats across all data first

# Variable selection
variables:
  select: null            # null = all, or list: [temperature, u_component_of_wind]
  pressure_levels: null   # null = all, or list: [500, 700, 850, 925, 1000]

# Train/val/test split
split:
  method: temporal        # temporal or random
  train:
    start: "1979-01-01"
    end: "2015-12-31"
  val:
    start: "2016-01-01"
    end: "2017-12-31"
  test:
    start: "2018-01-01"
    end: "2020-12-31"

# ClimaX-compatible output format
climax_output:
  enabled: true
  save_coordinates: true     # lat.npy, lon.npy
  save_normalization: true   # normalize_mean.npz, normalize_std.npz
  variable_naming: name_level  # e.g., temperature_500

# Parallel execution settings
parallel:
  enabled: false             # Enable parallel processing across files
  backend: local             # local (GNU parallel, no SLURM), gnu (SLURM+GNU parallel), mpi (SLURM+mpi4py)
  workers: null              # null = auto-detect cores, or integer
  batch_size: 4              # Files per worker batch
  distribution: round_robin  # round_robin or balanced (by file size)
  # SLURM settings (if backend: slurm):
  slurm:
    nodes: 4                 # Number of compute nodes
    cores_per_node: 56       # CPU cores per node
    time_limit: "0:30:00"    # Time limit HH:MM:SS
    account: null            # SLURM account/allocation
    partition: batch         # SLURM partition
    burst_buffer: "/mnt/bb/$USER"  # Fast local storage
    final_output: null       # Where to copy results from burst buffer
    modules:                 # Modules to load
      - miniforge3/23.11.0-0
      - parallel
    conda_env: conda-redi    # Conda environment to activate
"""
    return example
