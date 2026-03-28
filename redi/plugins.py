"""
REDI Plugin System

Plugins extend REDI with new domains, dataloaders, or pipeline steps without
modifying the core package. They are installed as ordinary Python packages and
discovered automatically at runtime via setuptools entry points.

Plugin contract
---------------
A plugin package must:

1. Register an entry point under the ``redi.plugins`` group in its
   ``pyproject.toml``::

       [project.entry-points."redi.plugins"]
       xgc1 = "redi_xgc1"

2. Expose the following attributes in that module:

   - ``NAME: str``        — short identifier (e.g. "xgc1")
   - ``DOMAIN: str``      — domain string used with ``--domain`` (e.g. "xgc1")
   - ``DESCRIPTION: str`` — one-line description
   - ``VERSION: str``     — plugin version

3. Expose at least one of:

   - ``get_full_pipeline() -> list``      — all pipeline steps
   - ``get_ingest_steps() -> list``       — ingest steps only
   - ``get_preprocess_steps() -> list``   — preprocess steps
   - ``get_transform_steps() -> list``    — transform steps
   - ``get_structure_steps() -> list``    — structure steps
   - ``get_output_steps() -> list``       — output steps

Example plugin layout
---------------------
::

    redi-xgc1/
    ├── pyproject.toml
    └── redi_xgc1/
        ├── __init__.py   # exposes NAME, DOMAIN, DESCRIPTION, VERSION, get_full_pipeline
        └── pipeline.py   # pipeline step implementations

Installing a plugin
-------------------
::

    redi install git+https://github.com/org/redi-xgc1
    redi install ./redi-xgc1          # local directory
    redi install redi-xgc1==1.2.0     # PyPI package
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Plugin data class
# ---------------------------------------------------------------------------

@dataclass
class Plugin:
    """Represents a discovered REDI plugin."""
    name: str
    domain: str
    description: str
    version: str
    module: Any                          # the imported plugin module
    entry_point_name: str = ""           # name as registered in entry_points

    def get_full_pipeline(self) -> List:
        if hasattr(self.module, "get_full_pipeline"):
            return self.module.get_full_pipeline()
        # Assemble from stage functions if get_full_pipeline is absent
        steps = []
        for fn in ("get_ingest_steps", "get_preprocess_steps",
                   "get_transform_steps", "get_structure_steps",
                   "get_output_steps"):
            if hasattr(self.module, fn):
                steps.extend(getattr(self.module, fn)())
        return steps

    def get_stage(self, stage: str) -> List:
        fn_name = f"get_{stage}_steps"
        if hasattr(self.module, fn_name):
            return getattr(self.module, fn_name)()
        return []


# ---------------------------------------------------------------------------
# Registry — populated lazily on first access
# ---------------------------------------------------------------------------

_registry: Optional[Dict[str, Plugin]] = None


def _load_registry() -> Dict[str, Plugin]:
    """Discover all installed REDI plugins via entry points."""
    plugins: Dict[str, Plugin] = {}

    try:
        if sys.version_info >= (3, 12):
            from importlib.metadata import entry_points
            eps = entry_points(group="redi.plugins")
        else:
            from importlib.metadata import entry_points as _eps
            all_eps = _eps()
            eps = all_eps.get("redi.plugins", [])
    except Exception:
        return plugins

    for ep in eps:
        try:
            module = ep.load()
            name    = getattr(module, "NAME",        ep.name)
            domain  = getattr(module, "DOMAIN",      ep.name)
            desc    = getattr(module, "DESCRIPTION", "")
            version = getattr(module, "VERSION",     "unknown")
            plugins[domain] = Plugin(
                name=name,
                domain=domain,
                description=desc,
                version=version,
                module=module,
                entry_point_name=ep.name,
            )
        except Exception as exc:
            # Don't let a broken plugin crash REDI
            import warnings
            warnings.warn(f"Failed to load REDI plugin '{ep.name}': {exc}")

    return plugins


def get_registry() -> Dict[str, Plugin]:
    global _registry
    if _registry is None:
        _registry = _load_registry()
    return _registry


def reload_registry() -> Dict[str, Plugin]:
    """Force re-discovery of plugins (useful after install)."""
    global _registry
    _registry = _load_registry()
    return _registry


def get_plugin(domain: str) -> Optional[Plugin]:
    """Return plugin for the given domain, or None if not found."""
    return get_registry().get(domain)


def list_plugins() -> List[Plugin]:
    """Return all discovered plugins."""
    return list(get_registry().values())


# ---------------------------------------------------------------------------
# Install helper
# ---------------------------------------------------------------------------

def install(target: str, pip_args: Optional[List[str]] = None) -> int:
    """
    Install a REDI plugin via pip.

    Parameters
    ----------
    target:
        A pip-compatible install target: URL, local path, or package name.
        Examples:
          - ``git+https://github.com/org/redi-xgc1``
          - ``./redi-xgc1``
          - ``redi-xgc1==1.0.0``
    pip_args:
        Additional arguments forwarded to pip (e.g. ``["--upgrade"]``).

    Returns
    -------
    int
        pip exit code (0 = success).
    """
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", target]
    if pip_args:
        cmd.extend(pip_args)
    result = subprocess.run(cmd)
    return result.returncode
