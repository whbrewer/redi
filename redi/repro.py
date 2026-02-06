"""
Reproducibility logging for REDI.

This module tracks commands run during a session and appends them to a
repro.sh script in the output directory for experiment reproducibility.
"""

import os
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Try to get REDI version
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"


def get_environment_info() -> dict:
    """Gather environment information for reproducibility."""
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "redi_version": __version__,
        "cwd": os.getcwd(),
    }


def _looks_like_path(arg: str) -> bool:
    """Check if an argument looks like a file path.

    Args:
        arg: Command line argument to check

    Returns:
        True if the argument appears to be a file path
    """
    # Skip flags
    if arg.startswith("-"):
        return False

    # Check if it exists on disk (most reliable)
    if os.path.exists(arg):
        return True

    # Check for path-like patterns
    # Contains directory separator
    if "/" in arg or "\\" in arg:
        return True

    # Has a file extension (but not things like "climate" or "hpc")
    if "." in arg:
        parts = arg.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1]) <= 5:
            # Looks like a file extension (.csv, .yaml, .nc, etc.)
            ext = parts[1].lower()
            # Common data/config extensions
            if ext in ("csv", "tsv", "json", "yaml", "yml", "nc", "hdf5", "h5",
                       "npz", "npy", "parquet", "zarr", "grib", "cif", "pdb",
                       "fasta", "txt", "md", "sh"):
                return True

    return False


def _to_absolute_path(arg: str) -> str:
    """Convert a path argument to absolute if it's relative.

    Args:
        arg: Path argument

    Returns:
        Absolute path string
    """
    path = Path(arg)
    if not path.is_absolute():
        return str(path.resolve())
    return arg


def format_command(argv: Optional[List[str]] = None) -> str:
    """Format the command for inclusion in the repro script.

    Converts relative paths to absolute paths for reproducibility.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Shell-safe command string with absolute paths
    """
    if argv is None:
        argv = sys.argv

    processed_args = []
    for arg in argv:
        # Convert relative paths to absolute
        if _looks_like_path(arg):
            arg = _to_absolute_path(arg)
        processed_args.append(shlex.quote(arg))

    return " ".join(processed_args)


def generate_script_header() -> str:
    """Generate the bash script header for repro.sh."""
    env = get_environment_info()

    return f"""#!/usr/bin/env bash
#
# REDI Reproducibility Script
# Generated automatically by REDI v{env['redi_version']}
#
# This script records the commands run to produce the data in this directory.
# Re-running this script should reproduce the same outputs (given the same
# input data and REDI version).
#
# Initial run:
#   Host: {env['hostname']}
#   User: {env['user']}
#   Date: {env['timestamp']}
#   CWD:  {env['cwd']}
#   Python: {env['python_version']}
#

set -euo pipefail

"""


def format_command_entry(argv: Optional[List[str]] = None, comment: Optional[str] = None) -> str:
    """Format a single command entry for the repro script.

    Args:
        argv: Command line arguments (defaults to sys.argv)
        comment: Optional comment to add before the command

    Returns:
        Formatted entry string
    """
    env = get_environment_info()
    cmd = format_command(argv)

    lines = []
    lines.append(f"# [{env['timestamp']}]")
    if comment:
        lines.append(f"# {comment}")
    lines.append(cmd)
    lines.append("")

    return "\n".join(lines)


def log_command(
    output_dir: str,
    argv: Optional[List[str]] = None,
    comment: Optional[str] = None,
    create_if_missing: bool = True,
) -> Path:
    """Log a command to the repro.sh script in the output directory.

    Args:
        output_dir: Output directory where repro.sh will be written
        argv: Command line arguments (defaults to sys.argv)
        comment: Optional comment to add before the command
        create_if_missing: Create the file with header if it doesn't exist

    Returns:
        Path to the repro.sh file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repro_file = output_path / "repro.sh"

    # Create with header if doesn't exist
    if not repro_file.exists() and create_if_missing:
        repro_file.write_text(generate_script_header())
        repro_file.chmod(0o755)  # Make executable

    # Append the command entry
    entry = format_command_entry(argv, comment)
    with open(repro_file, "a") as f:
        f.write(entry)

    return repro_file


def log_pipeline_run(
    output_dir: str,
    input_paths: List[str],
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    array_mode: bool = False,
    normalize_method: Optional[str] = None,
    config_file: Optional[str] = None,
) -> Path:
    """Log a pipeline run command.

    This is a convenience wrapper that logs the pipeline run with
    appropriate context comments.

    Args:
        output_dir: Output directory
        input_paths: Input file paths
        domain: Domain that was used
        stages: Stages that were run
        array_mode: Whether array mode was enabled
        normalize_method: Normalization method used
        config_file: Config file if one was used

    Returns:
        Path to the repro.sh file
    """
    comments = []
    if len(input_paths) > 1:
        comments.append(f"Processing {len(input_paths)} files")
    if domain:
        comments.append(f"Domain: {domain}")
    if stages:
        comments.append(f"Stages: {', '.join(stages)}")
    if array_mode:
        comments.append("Array mode enabled")
    if normalize_method:
        comments.append(f"Normalization: {normalize_method}")
    if config_file:
        comments.append(f"Config: {config_file}")

    comment = " | ".join(comments) if comments else None

    return log_command(output_dir, comment=comment)


def log_sidecar_run(
    output_dir: str,
    workflow_file: str,
    workflow_type: str = "auto",
) -> Path:
    """Log a sidecar run command.

    Args:
        output_dir: Output directory
        workflow_file: Workflow file that was observed
        workflow_type: Type of workflow

    Returns:
        Path to the repro.sh file
    """
    comment = f"Sidecar mode: observing {workflow_file}"
    if workflow_type != "auto":
        comment += f" (type: {workflow_type})"

    return log_command(output_dir, comment=comment)


def log_discover_run(
    output_dir: str,
    input_path: str,
) -> Path:
    """Log a discover run command.

    Args:
        output_dir: Output directory
        input_path: Path that was analyzed

    Returns:
        Path to the repro.sh file
    """
    comment = f"Discover mode: analyzing {input_path}"
    return log_command(output_dir, comment=comment)
