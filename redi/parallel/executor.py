"""
Parallel execution orchestrator for REDI.

Provides the main ParallelExecutor class that coordinates parallel
execution across multiple files using different backends.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .distribution import FileDistributor

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """
    Configuration for parallel execution.

    Attributes:
        backend: Backend type ("local", "gnu", or "mpi").
        workers: Number of parallel workers (None for auto-detect).
        batch_size: Number of files per worker batch.
        output_dir: Base output directory.
        redi_args: Additional arguments to pass to redi CLI.
        style: Output style for redi (default: simple for parallel).
        distribution: Distribution strategy ("round_robin" or "balanced").

        # SLURM options (shared by "gnu" and "mpi" backends)
        slurm_nodes: Number of compute nodes.
        slurm_cores_per_node: CPU cores per node (gnu backend).
        slurm_time: Time limit (HH:MM:SS).
        slurm_account: Account/allocation name.
        slurm_partition: Partition name.
        slurm_burst_buffer: Burst buffer path pattern (gnu backend).
        slurm_modules: Modules to load.
        slurm_conda_env: Conda environment to activate.
        slurm_constraint: Node constraint (e.g., "nvme").
        slurm_final_output: Where to copy results after job completes.

        # MPI-specific options (mpi backend only)
        mpi_ntasks_per_node: MPI tasks per node (default 1).
        mpi_merge: Gather results and write unified DataCard on rank 0.
        mpi_merge_format: Output format for merged file (npz, hdf5, zarr).
    """

    backend: str = "local"
    workers: Optional[int] = None
    batch_size: int = 4
    output_dir: str = "./output"
    redi_args: List[str] = field(default_factory=list)
    style: str = "simple"
    distribution: str = "round_robin"

    # SLURM options (gnu + mpi)
    slurm_nodes: int = 4
    slurm_cores_per_node: int = 56
    slurm_time: str = "0:30:00"
    slurm_account: Optional[str] = None
    slurm_partition: str = "batch"
    slurm_burst_buffer: str = "/mnt/bb/$USER"
    slurm_modules: List[str] = field(
        default_factory=lambda: ["miniforge3/23.11.0-0", "parallel"]
    )
    slurm_conda_env: str = "conda-redi"
    slurm_constraint: Optional[str] = "nvme"
    slurm_final_output: Optional[str] = None

    # MPI-specific options
    mpi_ntasks_per_node: int = 1
    mpi_merge: bool = True
    mpi_merge_format: str = "npz"

    def __post_init__(self):
        if self.workers is None:
            if self.backend == "local":
                self.workers = os.cpu_count() or 4
            elif self.backend == "mpi":
                self.workers = self.slurm_nodes * self.mpi_ntasks_per_node
            else:  # gnu
                self.workers = self.slurm_nodes


@dataclass
class ParallelResult:
    """
    Result of parallel execution.

    Attributes:
        success: Whether all jobs completed successfully.
        total_files: Total number of input files.
        completed: Number of successfully processed files.
        failed: Number of failed files.
        output_dirs: List of output directories created.
        errors: List of error messages.
        scripts_dir: Directory containing generated scripts (SLURM).
    """

    success: bool
    total_files: int
    completed: int
    failed: int
    output_dirs: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scripts_dir: Optional[Path] = None


class ParallelExecutor:
    """
    Orchestrates parallel REDI execution across multiple files.

    Supports multiple backends:
        - local: Uses GNU parallel on the local machine
        - slurm: Generates SLURM scripts for HPC clusters

    Example:
        executor = ParallelExecutor(backend="local", workers=8)
        result = executor.execute([Path("data1.nc"), Path("data2.nc")])
    """

    def __init__(
        self,
        backend: str = "local",
        workers: Optional[int] = None,
        batch_size: int = 4,
        output_dir: str = "./output",
        redi_args: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the parallel executor.

        Args:
            backend: Execution backend ("local" or "slurm").
            workers: Number of parallel workers (auto-detect if None).
            batch_size: Files per worker batch.
            output_dir: Base output directory.
            redi_args: Additional arguments for redi CLI.
            **kwargs: Backend-specific options (slurm_nodes, etc.).
        """
        self.config = ParallelConfig(
            backend=backend,
            workers=workers,
            batch_size=batch_size,
            output_dir=output_dir,
            redi_args=redi_args or [],
            **{k: v for k, v in kwargs.items() if k.startswith("slurm_")},
        )
        self._backend = None

    @property
    def backend(self):
        """Lazy-load the execution backend."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def _create_backend(self):
        """Create the appropriate backend based on config."""
        from .backends import get_backend

        return get_backend(self.config.backend, self.config)

    def execute(self, input_paths: List[Path]) -> ParallelResult:
        """
        Execute REDI in parallel across input files.

        Args:
            input_paths: List of input file paths.

        Returns:
            ParallelResult with execution status.
        """
        if not input_paths:
            return ParallelResult(
                success=True,
                total_files=0,
                completed=0,
                failed=0,
            )

        input_paths = [Path(p) for p in input_paths]
        logger.info(
            f"Starting parallel execution: {len(input_paths)} files, "
            f"backend={self.config.backend}, workers={self.config.workers}"
        )

        # Distribute files
        if self.config.distribution == "balanced":
            file_batches = FileDistributor.balanced(input_paths, self.config.workers)
        else:
            file_batches = FileDistributor.round_robin(input_paths, self.config.workers)

        # Then batch within workers
        all_batches = []
        for worker_files in file_batches:
            batches = FileDistributor.batched(worker_files, self.config.batch_size)
            all_batches.extend(batches)

        return self.backend.execute(all_batches, self.config)

    def generate_scripts(
        self,
        input_paths: List[Path],
        script_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Generate execution scripts without running them.

        Useful for SLURM where users may want to inspect
        or modify scripts before submission.

        Args:
            input_paths: List of input file paths.
            script_dir: Directory for generated scripts.

        Returns:
            Dictionary mapping script names to paths.
        """
        if script_dir is None:
            script_dir = Path(self.config.output_dir) / "parallel_scripts"

        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)

        input_paths = [Path(p) for p in input_paths]

        return self.backend.generate_scripts(input_paths, script_dir, self.config)


def run_parallel(
    input_paths: List[Path],
    backend: str = "local",
    workers: Optional[int] = None,
    batch_size: int = 4,
    output_dir: str = "./output",
    redi_args: Optional[List[str]] = None,
    **kwargs,
) -> ParallelResult:
    """
    Convenience function for parallel execution.

    Args:
        input_paths: List of input file paths.
        backend: Execution backend.
        workers: Number of parallel workers.
        batch_size: Files per worker batch.
        output_dir: Base output directory.
        redi_args: Additional redi CLI arguments.
        **kwargs: Backend-specific options.

    Returns:
        ParallelResult with execution status.
    """
    executor = ParallelExecutor(
        backend=backend,
        workers=workers,
        batch_size=batch_size,
        output_dir=output_dir,
        redi_args=redi_args,
        **kwargs,
    )
    return executor.execute(input_paths)
