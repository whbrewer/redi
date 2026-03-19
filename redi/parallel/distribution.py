"""
File distribution strategies for parallel execution.

Provides different strategies for distributing files across workers,
matching the AWK-based round-robin approach from the original shell scripts
and adding balanced distribution by file size.
"""

from pathlib import Path
from typing import List, Optional
import os


class FileDistributor:
    """Distribute files across workers using various strategies."""

    @staticmethod
    def round_robin(files: List[Path], num_workers: int) -> List[List[Path]]:
        """
        Distribute files round-robin across workers.

        This matches the AWK approach from driver.sh:
            NR % NNODE == NODEID

        Args:
            files: List of file paths to distribute.
            num_workers: Number of workers to distribute across.

        Returns:
            List of file lists, one per worker.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        worker_files: List[List[Path]] = [[] for _ in range(num_workers)]

        for idx, file_path in enumerate(files):
            worker_idx = idx % num_workers
            worker_files[worker_idx].append(file_path)

        return worker_files

    @staticmethod
    def balanced(files: List[Path], num_workers: int) -> List[List[Path]]:
        """
        Distribute files balanced by file size across workers.

        Assigns files to the worker with the least total size,
        resulting in more even workload distribution.

        Args:
            files: List of file paths to distribute.
            num_workers: Number of workers to distribute across.

        Returns:
            List of file lists, one per worker.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        # Get file sizes
        file_sizes = []
        for f in files:
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0  # Unknown size, treat as zero
            file_sizes.append((f, size))

        # Sort by size descending for better packing
        file_sizes.sort(key=lambda x: x[1], reverse=True)

        # Assign to worker with minimum total size
        worker_files: List[List[Path]] = [[] for _ in range(num_workers)]
        worker_sizes: List[int] = [0] * num_workers

        for file_path, size in file_sizes:
            # Find worker with minimum total size
            min_idx = worker_sizes.index(min(worker_sizes))
            worker_files[min_idx].append(file_path)
            worker_sizes[min_idx] += size

        return worker_files

    @staticmethod
    def batched(files: List[Path], batch_size: int) -> List[List[Path]]:
        """
        Group files into fixed-size batches.

        This matches the -N4 argument pattern from payload.sh,
        where multiple files are passed to each worker invocation.

        Args:
            files: List of file paths to batch.
            batch_size: Number of files per batch.

        Returns:
            List of batches, each containing up to batch_size files.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: List[List[Path]] = []

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            batches.append(batch)

        return batches

    @staticmethod
    def for_nodes(
        files: List[Path],
        num_nodes: int,
        batch_size: int = 4,
    ) -> List[List[List[Path]]]:
        """
        Distribute files across nodes, then batch within each node.

        Combines round_robin (for node distribution) with batched
        (for within-node parallel execution). This mirrors the
        multi-node SLURM approach.

        Args:
            files: List of file paths to distribute.
            num_nodes: Number of compute nodes.
            batch_size: Files per worker batch within each node.

        Returns:
            List of nodes, each containing list of batches.
        """
        # First distribute across nodes
        node_files = FileDistributor.round_robin(files, num_nodes)

        # Then batch within each node
        node_batches: List[List[List[Path]]] = []
        for nf in node_files:
            batches = FileDistributor.batched(nf, batch_size)
            node_batches.append(batches)

        return node_batches

    @staticmethod
    def write_file_list(files: List[Path], output_path: Path) -> Path:
        """
        Write a list of files to a text file (one per line).

        This generates input files like inputdatalist.txt.

        Args:
            files: List of file paths.
            output_path: Where to write the file list.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for file_path in files:
                f.write(f"{file_path}\n")

        return output_path
