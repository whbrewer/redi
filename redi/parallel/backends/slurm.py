"""
SLURM HPC cluster backend.

Generates SLURM scripts for multi-node parallel execution,
based on the original parallelize/ shell scripts.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
import logging

from .base import ParallelBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig, ParallelResult

logger = logging.getLogger(__name__)


class SlurmBackend(ParallelBackend):
    """
    Generate and optionally submit SLURM jobs for multi-node parallel execution.

    This backend generates three scripts matching the original parallelize/ approach:
        - multinodejob.sh: SLURM batch submission script
        - driver.sh: Per-node work distribution
        - payload.sh: Single job execution wrapper
    """

    @property
    def name(self) -> str:
        return "slurm"

    def check_available(self) -> bool:
        """Check if sbatch is available."""
        return shutil.which("sbatch") is not None

    def execute(
        self,
        file_batches: List[List[Path]],
        config: "ParallelConfig",
    ) -> "ParallelResult":
        """
        Generate scripts and submit SLURM job.

        Args:
            file_batches: List of file batches (used to get total count).
            config: Parallel configuration.

        Returns:
            ParallelResult with execution status.
        """
        from ..executor import ParallelResult

        # Get all files from batches
        all_files = []
        for batch in file_batches:
            all_files.extend(batch)

        total_files = len(all_files)

        if not self.check_available():
            return ParallelResult(
                success=False,
                total_files=total_files,
                completed=0,
                failed=total_files,
                errors=["sbatch is not available. Are you on a SLURM cluster?"],
            )

        # Generate scripts
        script_dir = Path(config.output_dir) / "slurm_scripts"
        scripts = self.generate_scripts(all_files, script_dir, config)

        # Submit job
        job_script = scripts.get("multinodejob.sh")
        if not job_script:
            return ParallelResult(
                success=False,
                total_files=total_files,
                completed=0,
                failed=total_files,
                errors=["Failed to generate SLURM job script"],
            )

        try:
            result = subprocess.run(
                ["sbatch", str(job_script)],
                capture_output=True,
                text=True,
                cwd=script_dir,
            )

            if result.returncode != 0:
                return ParallelResult(
                    success=False,
                    total_files=total_files,
                    completed=0,
                    failed=total_files,
                    errors=[f"sbatch failed: {result.stderr}"],
                    scripts_dir=script_dir,
                )

            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
            logger.info(f"Submitted SLURM job {job_id}")

            return ParallelResult(
                success=True,
                total_files=total_files,
                completed=0,  # Will be completed asynchronously
                failed=0,
                errors=[f"Job submitted: {job_id}. Check status with: squeue -j {job_id}"],
                scripts_dir=script_dir,
            )

        except Exception as e:
            return ParallelResult(
                success=False,
                total_files=total_files,
                completed=0,
                failed=total_files,
                errors=[str(e)],
                scripts_dir=script_dir,
            )

    def generate_scripts(
        self,
        input_paths: List[Path],
        script_dir: Path,
        config: "ParallelConfig",
    ) -> Dict[str, Path]:
        """
        Generate SLURM execution scripts.

        Generates three scripts matching the parallelize/ approach:
            - multinodejob.sh: SLURM batch script
            - driver.sh: Node-level distributor
            - payload.sh: Per-file executor

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
        file_list_path = script_dir / "inputdatalist.txt"
        with open(file_list_path, "w") as f:
            for path in input_paths:
                f.write(f"{path}\n")

        # Generate payload.sh
        payload_path = self._generate_payload(script_dir, config)

        # Generate driver.sh
        driver_path = self._generate_driver(script_dir, config)

        # Generate multinodejob.sh
        job_path = self._generate_job(script_dir, config)

        return {
            "multinodejob.sh": job_path,
            "driver.sh": driver_path,
            "payload.sh": payload_path,
            "inputdatalist.txt": file_list_path,
        }

    def _generate_payload(self, script_dir: Path, config: "ParallelConfig") -> Path:
        """Generate payload.sh - single job execution wrapper."""
        # Build redi args
        redi_args = ["--style", config.style] + config.redi_args
        redi_args_str = " ".join(redi_args)

        # Use burst buffer for output, similar to original
        burst_buffer = config.slurm_burst_buffer

        content = f"""#!/bin/bash
# REDI Payload Script - executes REDI on a batch of files
# Generated by redi --parallel --backend slurm

# Process up to 4 files passed as arguments
# Output goes to burst buffer for fast I/O

redi {redi_args_str} $1 $2 $3 $4 -o {burst_buffer}/$(basename $1).outdir &> {burst_buffer}/$(basename $1).stdoe
"""

        path = script_dir / "payload.sh"
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)

        return path

    def _generate_driver(self, script_dir: Path, config: "ParallelConfig") -> Path:
        """Generate driver.sh - per-node work distribution."""
        cores = config.slurm_cores_per_node
        batch_size = config.batch_size

        content = f"""#!/bin/bash
# REDI Driver Script - distributes work across cores on each node
# Generated by redi --parallel --backend slurm

if [[ -z "${{SLURM_NODEID}}" ]]; then
    echo "Error: \\$SLURM_NODEID not set"
    exit 1
fi
if [[ -z "${{SLURM_NNODES}}" ]]; then
    echo "Error: \\$SLURM_NNODES not set"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Distribute files based on node ID (round-robin)
# NR % NNODE == NODEID selects files for this node
cat $1 | \\
    awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \\
    'NR % NNODE == NODEID' | \\
    parallel -j{cores} -N{batch_size} "$SCRIPT_DIR/payload.sh" {{}}
"""

        path = script_dir / "driver.sh"
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)

        return path

    def _generate_job(self, script_dir: Path, config: "ParallelConfig") -> Path:
        """Generate multinodejob.sh - SLURM batch submission script."""
        # Build SBATCH directives
        sbatch_lines = []

        if config.slurm_account:
            sbatch_lines.append(f"#SBATCH -A {config.slurm_account}")

        sbatch_lines.extend([
            "#SBATCH -J redi-parallel",
            "#SBATCH -o %x-%j.out",
            "#SBATCH -e %x-%j.err",
            f"#SBATCH -t {config.slurm_time}",
            f"#SBATCH -p {config.slurm_partition}",
            f"#SBATCH -N {config.slurm_nodes}",
            "#SBATCH --ntasks-per-node 1",
        ])

        if config.slurm_constraint:
            sbatch_lines.append(f"#SBATCH -C {config.slurm_constraint}")

        sbatch_str = "\n".join(sbatch_lines)

        # Build module loads
        module_loads = "\n".join(
            f"module load {mod}" for mod in config.slurm_modules
        )

        # Build final copy command if specified
        final_copy = ""
        if config.slurm_final_output:
            final_copy = f"\ncp -r {config.slurm_burst_buffer}/* {config.slurm_final_output}/"

        content = f"""#!/bin/bash
# REDI Multi-Node SLURM Job Script
# Generated by redi --parallel --backend slurm
#
# Submit with: sbatch multinodejob.sh
# Check status: squeue -j <job_id>
# View logs: tail -f redi-parallel-<job_id>.out

{sbatch_str}

# Load modules
{module_loads}

# Activate conda environment
conda activate {config.slurm_conda_env}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Run on all nodes
srun --no-kill --ntasks-per-node=1 --wait=0 "$SCRIPT_DIR/driver.sh" "$SCRIPT_DIR/inputdatalist.txt"

# Copy results to final destination
{final_copy}

echo "== Done in $(($SECONDS / 60)) minutes =="
"""

        path = script_dir / "multinodejob.sh"
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)

        return path
