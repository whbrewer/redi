"""
REDI Parallel Execution Module.

Provides parallel execution capabilities for processing multiple files
across cores and nodes using GNU parallel.

Type 1 Parallelism (implemented):
    Multiple files distributed across workers/nodes.

Type 2 Parallelism (future):
    Single large file chunked across workers.
"""

from .executor import ParallelExecutor, ParallelConfig, ParallelResult
from .distribution import FileDistributor
from .chunking import ChunkStrategy, ChunkSpec

__all__ = [
    "ParallelExecutor",
    "ParallelConfig",
    "ParallelResult",
    "FileDistributor",
    "ChunkStrategy",
    "ChunkSpec",
]
