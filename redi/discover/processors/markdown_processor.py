"""
redi/discover/processors/markdown_processor.py — Markdown file analysis.

Extracts:
- Table of contents (headings / subheadings)
- File references (links, images, bare filenames)
- Groupings of references by heading, list membership, and adjacency

No preprocessing steps are suggested — markdown files are reference material.
"""

from __future__ import annotations

import re
from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery

# Regex patterns
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?$")
_SETEXT_H1 = re.compile(r"^=+\s*$")
_SETEXT_H2 = re.compile(r"^-+\s*$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_BARE_FILE = re.compile(r"(?<!\w)([\w./-]+\.(?:csv|tsv|json|nc|h5|hdf5|npy|npz|parquet|fasta|pdb|txt|yaml|yml|toml|sh|py|r|rmd|ipynb|pdf|png|jpg|jpeg|svg|gif))(?!\w)", re.IGNORECASE)
_URL_PREFIX = re.compile(r"^(?:https?|ftp|mailto)://", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*[-*+]\s+|^\s*\d+\.\s+")


class MarkdownProcessor(BaseProcessor):
    extension = ".md"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return FileDiscovery(
                file_path=str(path),
                file_type="Markdown",
                file_size_bytes=path.stat().st_size,
                metadata={"error": str(exc)},
            )

        lines = text.splitlines()
        headings = _extract_headings(lines)
        file_refs = _extract_file_references(lines)
        groups = _group_references(lines, file_refs)

        metadata = {
            "headings": headings,
            "file_references": sorted(set(file_refs)),
            "file_groups": groups,
        }

        return FileDiscovery(
            file_path=str(path),
            file_type="Markdown",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=[],  # reference material — no preprocessing
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_headings(lines: list[str]) -> list[dict]:
    """Return list of {level, text} dicts for all headings in *lines*."""
    headings = []
    for i, line in enumerate(lines):
        m = _ATX_HEADING.match(line)
        if m:
            headings.append({"level": len(m.group(1)), "text": m.group(2).strip()})
            continue
        # Setext headings require a non-empty previous line
        if i > 0 and lines[i - 1].strip():
            if _SETEXT_H1.match(line) and len(line.strip()) >= 2:
                headings.append({"level": 1, "text": lines[i - 1].strip()})
            elif _SETEXT_H2.match(line) and len(line.strip()) >= 2:
                headings.append({"level": 2, "text": lines[i - 1].strip()})
    return headings


def _extract_file_references(lines: list[str]) -> list[str]:
    """Return all non-URL file paths referenced anywhere in *lines*."""
    refs: list[str] = []
    full_text = "\n".join(lines)

    # Explicit markdown links and images
    for pattern in (_MD_LINK, _MD_IMAGE):
        for m in pattern.finditer(full_text):
            target = m.group(2).strip()
            if not _URL_PREFIX.match(target):
                refs.append(target.split("#")[0])  # strip anchors

    # Bare filenames
    for m in _BARE_FILE.finditer(full_text):
        refs.append(m.group(1))

    return refs


def _group_references(lines: list[str], all_refs: list[str]) -> dict:
    """Group file references by heading, list membership, and adjacency."""
    by_heading: dict[str, list[str]] = {}
    by_list: list[list[str]] = []
    by_adjacency: list[list[str]] = []

    current_heading = "(preamble)"
    current_list: list[str] | None = None
    adjacent_run: list[str] = []
    prev_had_ref = False

    for line in lines:
        # Track current heading
        m = _ATX_HEADING.match(line)
        if m:
            current_heading = m.group(2).strip()
            if current_list:
                by_list.append(current_list)
                current_list = None
            continue

        refs_in_line = _refs_on_line(line)
        in_list = bool(_LIST_ITEM.match(line))

        # --- by_heading ---
        for ref in refs_in_line:
            by_heading.setdefault(current_heading, []).append(ref)

        # --- by_list ---
        if in_list and refs_in_line:
            if current_list is None:
                current_list = []
            current_list.extend(refs_in_line)
        else:
            if current_list and not in_list:
                if len(current_list) > 1:
                    by_list.append(current_list)
                current_list = None

        # --- by_adjacency ---
        has_ref = bool(refs_in_line)
        if has_ref:
            adjacent_run.extend(refs_in_line)
        else:
            if len(adjacent_run) > 1:
                by_adjacency.append(adjacent_run)
            adjacent_run = []
        prev_had_ref = has_ref

    # Flush open runs
    if current_list and len(current_list) > 1:
        by_list.append(current_list)
    if len(adjacent_run) > 1:
        by_adjacency.append(adjacent_run)

    # Remove empty heading buckets
    by_heading = {h: refs for h, refs in by_heading.items() if refs}

    return {
        "by_heading": by_heading,
        "by_list": by_list,
        "by_adjacency": by_adjacency,
    }


def _refs_on_line(line: str) -> list[str]:
    """Return file references found on a single line."""
    refs: list[str] = []
    for pattern in (_MD_LINK, _MD_IMAGE):
        for m in pattern.finditer(line):
            target = m.group(2).strip()
            if not _URL_PREFIX.match(target):
                refs.append(target.split("#")[0])
    for m in _BARE_FILE.finditer(line):
        refs.append(m.group(1))
    return refs
