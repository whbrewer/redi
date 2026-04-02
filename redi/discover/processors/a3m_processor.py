"""
redi/discover/processors/a3m_processor.py — A3M multiple sequence alignment analysis.

A3M is a variant of FASTA where lowercase letters indicate insertions relative
to the query sequence.
"""

from __future__ import annotations

from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery, FileParseError, PreprocessingStep


class A3MProcessor(BaseProcessor):
    extension = ".a3m"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise FileParseError(path=path, file_type="A3M", reason=str(exc), cause=exc) from exc

        sequences = self._parse_sequences(text)
        if not sequences:
            raise FileParseError(path=path, file_type="A3M", reason="No sequences found in file")

        query_id, query_seq = sequences[0]
        # Query length = uppercase chars only (gaps/insertions excluded)
        query_len = sum(1 for c in query_seq if c.isupper() or c == "-")

        metadata = {
            "n_sequences": len(sequences),
            "query_id": query_id,
            "query_length": query_len,
            "alignment_depth": len(sequences),
        }

        steps: list[PreprocessingStep] = []

        steps.append(PreprocessingStep(
            name="Convert to gapped alignment",
            rationale="A3M format uses lowercase insertions — must be converted to a uniform gap-padded MSA for model input",
            suggested_method="remove insertion columns or expand to full multiple sequence alignment format",
        ))

        steps.append(PreprocessingStep(
            name="Filter low-quality sequences",
            rationale=f"Alignment has {len(sequences)} sequences — very deep alignments may include low-homology hits",
            suggested_method="filter by sequence identity threshold or maximum gap fraction",
        ))

        steps.append(PreprocessingStep(
            name="Encode MSA",
            rationale="Raw character sequences cannot be used directly by ML models",
            suggested_method="integer encode each residue position; OpenFold uses a 23-token vocabulary (20 AA + gap + unknown + mask)",
        ))

        return FileDiscovery(
            file_path=str(path),
            file_type="A3M",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )

    def _parse_sequences(self, text: str) -> list[tuple[str, str]]:
        sequences = []
        current_id = None
        current_seq: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif line and current_id is not None:
                current_seq.append(line)
        if current_id is not None:
            sequences.append((current_id, "".join(current_seq)))
        return sequences
