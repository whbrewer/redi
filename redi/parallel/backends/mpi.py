"""
MPI + SLURM backend.

Generates SLURM scripts for multi-node parallel execution using mpi4py.
Unlike the GNU parallel backend, MPI ranks communicate after processing:
rank 0 gathers per-rank metadata, computes aggregate statistics, and writes
a single merged DataCard and (optionally) a merged output file.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING
import logging

from .base import ParallelBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig, ParallelResult

logger = logging.getLogger(__name__)


class MpiBackend(ParallelBackend):
    """
    Generate and optionally submit SLURM jobs for multi-node parallel execution
    using mpi4py.

    Generates two scripts:
        - multinodejob.sh: SLURM batch script (uses srun to launch mpi_runner)
        - inputdatalist.txt: flat file list consumed by mpi_runner rank 0

    Each MPI rank processes its share of files independently, then rank 0
    gathers results, runs an allreduce for global stats, and writes a unified
    DataCard and merged output.
    """

    @property
    def name(self) -> str:
        return "mpi"

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

        script_dir = Path(config.output_dir) / "mpi_scripts"
        scripts = self.generate_scripts(all_files, script_dir, config)

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

            job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
            logger.info(f"Submitted SLURM MPI job {job_id}")

            return ParallelResult(
                success=True,
                total_files=total_files,
                completed=0,  # Completed asynchronously
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
        Generate SLURM + mpi4py execution scripts.

        Args:
            input_paths: List of input file paths.
            script_dir: Directory for generated scripts.
            config: Parallel configuration.

        Returns:
            Dictionary mapping script names to paths.
        """
        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)

        # Write file list — rank 0 of mpi_runner reads this
        file_list_path = script_dir / "inputdatalist.txt"
        with open(file_list_path, "w") as f:
            for path in input_paths:
                f.write(f"{path}\n")

        job_path = self._generate_job(script_dir, config)

        return {
            "multinodejob.sh": job_path,
            "inputdatalist.txt": file_list_path,
        }

    def _generate_job(self, script_dir: Path, config: "ParallelConfig") -> Path:
        """Generate multinodejob.sh - SLURM batch submission script."""
        sbatch_lines = []

        if config.slurm_account:
            sbatch_lines.append(f"#SBATCH -A {config.slurm_account}")

        ntasks_per_node = config.mpi_ntasks_per_node
        total_tasks = config.slurm_nodes * ntasks_per_node

        sbatch_lines.extend([
            "#SBATCH -J redi-mpi",
            "#SBATCH -o %x-%j.out",
            "#SBATCH -e %x-%j.err",
            f"#SBATCH -t {config.slurm_time}",
            f"#SBATCH -p {config.slurm_partition}",
            f"#SBATCH -N {config.slurm_nodes}",
            f"#SBATCH --ntasks-per-node {ntasks_per_node}",
        ])

        if config.slurm_constraint:
            sbatch_lines.append(f"#SBATCH -C {config.slurm_constraint}")

        sbatch_str = "\n".join(sbatch_lines)
        module_loads = "\n".join(
            f"module load {mod}" for mod in config.slurm_modules
        )

        # Build mpi_runner arguments
        redi_args = config.redi_args
        runner_args = [
            f"--filelist $SCRIPT_DIR/inputdatalist.txt",
            f"--output {config.slurm_final_output or config.output_dir}",
        ]
        if config.mpi_merge:
            runner_args.append("--merge")
        if config.mpi_merge_format:
            runner_args.append(f"--format {config.mpi_merge_format}")
        # Pass through any domain/stages from redi_args
        for i, arg in enumerate(redi_args):
            if arg in ("-d", "--domain") and i + 1 < len(redi_args):
                runner_args.append(f"--domain {redi_args[i + 1]}")
            elif arg in ("-s", "--stages") and i + 1 < len(redi_args):
                runner_args.append(f"--stages {redi_args[i + 1]}")
            elif arg in ("-f", "--format") and i + 1 < len(redi_args):
                runner_args.append(f"--format {redi_args[i + 1]}")

        runner_args_str = " \\\n        ".join(runner_args)

        content = f"""#!/bin/bash
# REDI Multi-Node SLURM Job Script (MPI backend)
# Generated by redi --parallel --backend mpi
#
# Submit with: sbatch multinodejob.sh
# Check status: squeue -j <job_id>
#
# Each MPI rank processes its share of files, then rank 0 gathers results,
# computes aggregate statistics, and writes a unified DataCard + merged output.

{sbatch_str}

# Load modules
{module_loads}

# Activate conda environment
conda activate {config.slurm_conda_env}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Launch MPI runner across all {total_tasks} ranks ({config.slurm_nodes} nodes x {ntasks_per_node} tasks/node)
srun python -m redi.parallel.mpi_runner \\
        {runner_args_str}

echo "== Done in $(($SECONDS / 60)) minutes =="
"""
        path = script_dir / "multinodejob.sh"
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)
        return path
