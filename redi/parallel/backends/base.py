"""
Abstract base class for parallel execution backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..executor import ParallelConfig, ParallelResult


class ParallelBackend(ABC):
    """
    Abstract base class for parallel execution backends.

    Backends handle the actual execution of REDI across multiple
    files, either locally or on a cluster.
    """

    def __init__(self, config: "ParallelConfig"):
        """
        Initialize the backend.

        Args:
            config: Parallel configuration.
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass

    @abstractmethod
    def execute(
        self,
        file_batches: List[List[Path]],
        config: "ParallelConfig",
    ) -> "ParallelResult":
        """
        Execute REDI on file batches.

        Args:
            file_batches: List of file batches to process.
            config: Parallel configuration.

        Returns:
            ParallelResult with execution status.
        """
        pass

    @abstractmethod
    def generate_scripts(
        self,
        input_paths: List[Path],
        script_dir: Path,
        config: "ParallelConfig",
    ) -> Dict[str, Path]:
        """
        Generate execution scripts.

        Args:
            input_paths: List of input file paths.
            script_dir: Directory for generated scripts.
            config: Parallel configuration.

        Returns:
            Dictionary mapping script names to paths.
        """
        pass

    def check_available(self) -> bool:
        """
        Check if this backend is available on the system.

        Returns:
            True if the backend can be used.
        """
        return True
