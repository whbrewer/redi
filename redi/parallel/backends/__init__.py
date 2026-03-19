"""
Parallel execution backends.

Provides different backends for parallel execution:
    - local: GNU parallel on local machine
    - slurm: SLURM HPC cluster
"""

from typing import TYPE_CHECKING

from .base import ParallelBackend
from .local import LocalBackend
from .slurm import SlurmBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig

# Registry of available backends
BACKENDS = {
    "local": LocalBackend,
    "slurm": SlurmBackend,
}


def get_backend(name: str, config: "ParallelConfig") -> ParallelBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ("local" or "slurm").
        config: Parallel configuration.

    Returns:
        Configured backend instance.

    Raises:
        ValueError: If backend name is not found.
    """
    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    return BACKENDS[name](config)


__all__ = [
    "ParallelBackend",
    "LocalBackend",
    "SlurmBackend",
    "get_backend",
    "BACKENDS",
]
