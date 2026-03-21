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
# Processor exceptions
# ---------------------------------------------------------------------------

class ProcessorError(Exception):
    """Raised by a processor when it cannot analyse a file.

    Attributes:
        path: The file that was being analysed.
        file_type: Human-readable label for the processor (e.g. ``"CSV"``).
        reason: Short description of what went wrong.
        hint: Suggestion for the user on how to resolve the issue.
        cause: The original exception, if any.
    """

    hint: str = "Check that the file is valid and not corrupted."

    def __init__(
        self,
        path: Path,
        file_type: str,
        reason: str,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.file_type = file_type
        self.reason = reason
        self.cause = cause
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.file_type}] {self.path.name}: {self.reason}"

    @property
    def user_message(self) -> str:
        """One-line message suitable for display to the user."""
        return f"{self.path.name} ({self.file_type}): {self.reason}. {self.hint}"


class FileReadError(ProcessorError):
    """Raised when a file cannot be opened or read from disk."""

    hint: str = "Check that the file exists, is readable, and is not locked by another process."


class FileParseError(ProcessorError):
    """Raised when a file can be read but its content cannot be parsed."""

    hint: str = "Check that the file content matches its extension and is not truncated or binary."


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
    def __init__(self) -> None:
        self._registry: dict[str, BaseProcessor] = {}
        self._generic = GenericProcessor()

    def register(self, processor: BaseProcessor) -> None:
        """Register a processor instance for its declared extension."""
        ext = processor.extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        self._registry[ext] = processor

    def get(self, extension: str) -> BaseProcessor:
        """Return the processor for *extension*, or the generic fallback."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return self._registry.get(ext, self._generic)

    def analyze_file(self, path: Path) -> FileDiscovery:
        """Dispatch *path* to the appropriate processor and return results.

        Raises:
            ProcessorError: If the processor raises one directly.
            ProcessorError: Wraps any unexpected exception so callers always
                receive a :class:`ProcessorError` on failure.
        """
        processor = self.get(path.suffix)
        try:
            return processor.analyze(path)
        except ProcessorError:
            raise
        except Exception as exc:
            raise ProcessorError(
                path=path,
                file_type=path.suffix.lstrip(".").upper() or "UNKNOWN",
                reason=str(exc),
                cause=exc,
            ) from exc


# ---------------------------------------------------------------------------
# Auto-register built-in processors on first import
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from redi.discover.processors.csv_processor import CsvProcessor
    from redi.discover.processors.markdown_processor import MarkdownProcessor
    from redi.discover.processors.netcdf_processor import NetCDFProcessor

    registry.register(CsvProcessor())
    registry.register(MarkdownProcessor())
    registry.register(NetCDFProcessor())
    # TSV reuses the CSV processor
    tsv = CsvProcessor()
    tsv.extension = ".tsv"
    registry.register(tsv)
    # .nc4 reuses the NetCDF processor
    nc4 = NetCDFProcessor()
    nc4.extension = ".nc4"
    registry.register(nc4)


registry = ProcessorRegistry()
_register_builtins()
