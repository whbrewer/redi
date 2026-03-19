"""
Local GNU parallel backend.

Executes REDI in parallel on the local machine using GNU parallel.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING
import logging

from .base import ParallelBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig, ParallelResult

logger = logging.getLogger(__name__)


class LocalBackend(ParallelBackend):
    """
    Execute REDI in parallel using GNU parallel on local machine.

    Requires GNU parallel to be installed and available in PATH.
    """

    @property
    def name(self) -> str:
        return "local"

    def check_available(self) -> bool:
        """Check if GNU parallel is available."""
        return shutil.which("parallel") is not None

    def execute(
        self,
        file_batches: List[List[Path]],
        config: "ParallelConfig",
    ) -> "ParallelResult":
        """
        Execute REDI on file batches using GNU parallel.

        Args:
            file_batches: List of file batches (each batch is a list of paths).
            config: Parallel configuration.

        Returns:
            ParallelResult with execution status.
        """
        from ..executor import ParallelResult

        if not self.check_available():
            return ParallelResult(
                success=False,
                total_files=sum(len(b) for b in file_batches),
                completed=0,
                failed=sum(len(b) for b in file_batches),
                errors=["GNU parallel is not installed. Install with: apt install parallel"],
            )

        total_files = sum(len(batch) for batch in file_batches)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flatten batches to file list for parallel
        all_files = []
        for batch in file_batches:
            all_files.extend(batch)

        if not all_files:
            return ParallelResult(
                success=True,
                total_files=0,
                completed=0,
                failed=0,
            )

        # Write file list to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            file_list_path = f.name
            for file_path in all_files:
                f.write(f"{file_path}\n")

        try:
            # Build redi command arguments
            redi_base_args = ["redi", "--style", config.style]
            redi_base_args.extend(config.redi_args)

            # Build parallel command
            # Each file gets its own output directory based on filename
            # Format: parallel -j{workers} redi --style=simple {} -o {output_dir}/{/.}.out :::: {file_list}
            parallel_cmd = [
                "parallel",
                f"-j{config.workers}",
                "--bar",  # Show progress bar
                "--halt", "soon,fail=10%",  # Stop if 10% fail
                " ".join(redi_base_args),
                "{}",
                "-o", f"{output_dir}/{{/.}}.out",
                "::::",
                file_list_path,
            ]

            logger.info(f"Running: {' '.join(parallel_cmd)}")

            result = subprocess.run(
                parallel_cmd,
                capture_output=True,
                text=True,
            )

            # Parse results
            errors = []
            if result.returncode != 0:
                errors.append(f"GNU parallel returned code {result.returncode}")
                if result.stderr:
                    errors.append(result.stderr)

            # Count successful outputs
            output_dirs = list(output_dir.glob("*.out"))
            completed = len(output_dirs)
            failed = total_files - completed

            return ParallelResult(
                success=result.returncode == 0,
                total_files=total_files,
                completed=completed,
                failed=failed,
                output_dirs=output_dirs,
                errors=errors,
            )

        finally:
            # Cleanup temp file
            if os.path.exists(file_list_path):
                os.unlink(file_list_path)

    def generate_scripts(
        self,
        input_paths: List[Path],
        script_dir: Path,
        config: "ParallelConfig",
    ) -> Dict[str, Path]:
        """
        Generate a local parallel execution script.

        Args:
            input_paths: List of input file paths.
            script_dir: Directory for generated scripts.
            config: Parallel configuration.

        Returns:
            Dictionary mapping script names to paths.
        """
        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)

        # Write file list
        file_list_path = script_dir / "input_files.txt"
        with open(file_list_path, "w") as f:
            for path in input_paths:
                f.write(f"{path}\n")

        # Build redi args
        redi_args = ["--style", config.style] + config.redi_args
        redi_args_str = " ".join(redi_args)

        # Generate run script
        run_script_content = f"""#!/bin/bash
# REDI Local Parallel Execution Script
# Generated by redi --parallel --generate-only

set -e

# Configuration
WORKERS={config.workers}
OUTPUT_DIR="{config.output_dir}"
FILE_LIST="{file_list_path}"

# Check GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed"
    echo "Install with: apt install parallel (Debian/Ubuntu)"
    echo "           or: brew install parallel (macOS)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run parallel
echo "Processing $(wc -l < "$FILE_LIST") files with $WORKERS workers..."
parallel -j$WORKERS --bar \\
    redi {redi_args_str} {{}} -o "$OUTPUT_DIR/{{/.}}.out" \\
    :::: "$FILE_LIST"

echo "Done! Results in $OUTPUT_DIR"
"""

        run_script_path = script_dir / "run_parallel.sh"
        with open(run_script_path, "w") as f:
            f.write(run_script_content)
        os.chmod(run_script_path, 0o755)

        return {
            "run_parallel.sh": run_script_path,
            "input_files.txt": file_list_path,
        }
