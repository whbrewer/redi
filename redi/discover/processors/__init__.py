"""
redi/discover/processors — per-file-type analysis plug-ins.

Each processor inspects a single file and returns a FileDiscovery describing
what was found and what preprocessing steps are suggested.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingStep:
    name: str
    rationale: str
    suggested_method: str


@dataclass
class FileDiscovery:
    file_path: str
    file_type: str                          # human-readable label, e.g. "CSV"
    file_size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: list[PreprocessingStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base processor
# ---------------------------------------------------------------------------

class BaseProcessor(ABC):
    """Abstract base for all file-type processors.

    Subclasses must set ``extension`` (lower-case, with leading dot, e.g.
    ``".csv"``) and implement :meth:`analyze`.
    """

    extension: str = ""

    @abstractmethod
    def analyze(self, path: Path) -> FileDiscovery:
        """Inspect *path* and return a :class:`FileDiscovery`."""


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

class GenericProcessor(BaseProcessor):
    """Fallback for file types with no dedicated processor."""

    def analyze(self, path: Path) -> FileDiscovery:
        return FileDiscovery(
            file_path=str(path),
            file_type=path.suffix.lstrip(".").upper() or "UNKNOWN",
            file_size_bytes=path.stat().st_size,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProcessorRegistry:
    _registry: dict[str, BaseProcessor] = {}
    _generic = GenericProcessor()

    @classmethod
    def register(cls, processor: BaseProcessor) -> None:
        """Register a processor instance for its declared extension."""
        ext = processor.extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        cls._registry[ext] = processor

    @classmethod
    def get(cls, extension: str) -> BaseProcessor:
        """Return the processor for *extension*, or the generic fallback."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return cls._registry.get(ext, cls._generic)

    @classmethod
    def analyze_file(cls, path: Path) -> FileDiscovery:
        """Dispatch *path* to the appropriate processor and return results."""
        processor = cls.get(path.suffix)
        return processor.analyze(path)


# ---------------------------------------------------------------------------
# Auto-register built-in processors on first import
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from redi.discover.processors.csv_processor import CsvProcessor
    from redi.discover.processors.markdown_processor import MarkdownProcessor

    ProcessorRegistry.register(CsvProcessor())
    ProcessorRegistry.register(MarkdownProcessor())
    # TSV reuses the CSV processor
    tsv = CsvProcessor()
    tsv.extension = ".tsv"
    ProcessorRegistry.register(tsv)


_register_builtins()
