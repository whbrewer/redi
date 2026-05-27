"""
Parallel execution backends.

Provides different backends for parallel execution:
    - local:   GNU parallel on local machine (no SLURM)
    - gnu:     SLURM + GNU parallel across nodes
    - mpi:     SLURM + mpi4py across nodes (adds allreduce/gather for unified output)
    - dragon:  Dragon HPC distributed runtime (multi-core and multi-node)
"""

import warnings
from typing import TYPE_CHECKING

from .base import ParallelBackend
from .local import LocalBackend
from .gnu import GnuBackend
from .mpi import MpiBackend
from .dragon import DragonBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig

# Registry of available backends
BACKENDS = {
    "local": LocalBackend,
    "gnu": GnuBackend,
    "mpi": MpiBackend,
    "dragon": DragonBackend,
}

# Deprecated alias
_DEPRECATED = {
    "slurm": ("gnu", GnuBackend),
}


def get_backend(name: str, config: "ParallelConfig") -> ParallelBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ("local", "gnu", or "mpi").
              "slurm" is a deprecated alias for "gnu".
        config: Parallel configuration.

    Returns:
        Configured backend instance.

    Raises:
        ValueError: If backend name is not found.
    """
    if name in _DEPRECATED:
        new_name, cls = _DEPRECATED[name]
        warnings.warn(
            f"Backend '{name}' is deprecated; use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(config)

    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")

    return BACKENDS[name](config)


__all__ = [
    "ParallelBackend",
    "LocalBackend",
    "GnuBackend",
    "MpiBackend",
    "DragonBackend",
    "get_backend",
    "BACKENDS",
]
