"""
File chunking strategies for Type 2 parallelism (future).

This module provides the interface for splitting single large files
across multiple workers. The actual implementations will be added
when Type 2 parallelism is implemented.

Type 2 Use Cases:
    - Large NetCDF files split by time dimension
    - Large tabular files split by row ranges
    - Gridded data split by spatial regions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict


@dataclass
class ChunkSpec:
    """
    Specification for a single chunk of a file.

    Attributes:
        source_path: Original file path.
        chunk_id: Unique identifier for this chunk.
        start: Start index/offset (interpretation depends on strategy).
        end: End index/offset (interpretation depends on strategy).
        metadata: Additional chunk-specific metadata.
    """

    source_path: Path
    chunk_id: int
    start: Any  # Could be int, datetime, tuple, etc.
    end: Any
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChunkStrategy(ABC):
    """
    Abstract base class for file chunking strategies.

    Implementations will provide domain-specific chunking logic:
        - TimeChunkStrategy: Split NetCDF by time dimension
        - RowChunkStrategy: Split tabular data by row ranges
        - SpatialChunkStrategy: Split gridded data by lat/lon regions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and configuration."""
        pass

    @abstractmethod
    def can_chunk(self, file_path: Path) -> bool:
        """
        Check if this strategy can chunk the given file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if this strategy can handle the file.
        """
        pass

    @abstractmethod
    def chunk(self, file_path: Path, num_chunks: int) -> List[ChunkSpec]:
        """
        Split a file into chunks for parallel processing.

        Args:
            file_path: Path to the file to chunk.
            num_chunks: Desired number of chunks.

        Returns:
            List of ChunkSpec objects describing each chunk.
        """
        pass

    @abstractmethod
    def extract_chunk(self, spec: ChunkSpec, output_path: Path) -> Path:
        """
        Extract a chunk to a temporary file for processing.

        Args:
            spec: Chunk specification.
            output_path: Where to write the extracted chunk.

        Returns:
            Path to the extracted chunk file.
        """
        pass

    @abstractmethod
    def merge(self, chunk_results: List[Path], output_path: Path) -> Path:
        """
        Merge processed chunks back into a single output.

        Args:
            chunk_results: Paths to processed chunk outputs.
            output_path: Where to write the merged output.

        Returns:
            Path to the merged output file.
        """
        pass


# Placeholder implementations for future Type 2 parallelism


class TimeChunkStrategy(ChunkStrategy):
    """
    Chunk NetCDF files by time dimension.

    Future implementation for climate/weather data.
    """

    @property
    def name(self) -> str:
        return "time"

    def can_chunk(self, file_path: Path) -> bool:
        # Future: Check if file is NetCDF with time dimension
        raise NotImplementedError("TimeChunkStrategy is planned for future implementation")

    def chunk(self, file_path: Path, num_chunks: int) -> List[ChunkSpec]:
        raise NotImplementedError("TimeChunkStrategy is planned for future implementation")

    def extract_chunk(self, spec: ChunkSpec, output_path: Path) -> Path:
        raise NotImplementedError("TimeChunkStrategy is planned for future implementation")

    def merge(self, chunk_results: List[Path], output_path: Path) -> Path:
        raise NotImplementedError("TimeChunkStrategy is planned for future implementation")


class RowChunkStrategy(ChunkStrategy):
    """
    Chunk tabular files by row ranges.

    Future implementation for CSV, Parquet, etc.
    """

    @property
    def name(self) -> str:
        return "row"

    def can_chunk(self, file_path: Path) -> bool:
        # Future: Check if file is tabular format
        raise NotImplementedError("RowChunkStrategy is planned for future implementation")

    def chunk(self, file_path: Path, num_chunks: int) -> List[ChunkSpec]:
        raise NotImplementedError("RowChunkStrategy is planned for future implementation")

    def extract_chunk(self, spec: ChunkSpec, output_path: Path) -> Path:
        raise NotImplementedError("RowChunkStrategy is planned for future implementation")

    def merge(self, chunk_results: List[Path], output_path: Path) -> Path:
        raise NotImplementedError("RowChunkStrategy is planned for future implementation")


# Registry of available chunk strategies
CHUNK_STRATEGIES: Dict[str, type] = {
    "time": TimeChunkStrategy,
    "row": RowChunkStrategy,
}


def get_chunk_strategy(name: str) -> ChunkStrategy:
    """
    Get a chunk strategy by name.

    Args:
        name: Strategy name (e.g., "time", "row").

    Returns:
        Instantiated chunk strategy.

    Raises:
        ValueError: If strategy name is not found.
    """
    if name not in CHUNK_STRATEGIES:
        available = ", ".join(CHUNK_STRATEGIES.keys())
        raise ValueError(f"Unknown chunk strategy: {name}. Available: {available}")

    return CHUNK_STRATEGIES[name]()
