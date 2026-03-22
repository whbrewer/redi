"""
redi/discover/processors/fasta_processor.py — FASTA sequence file analysis.
"""

from __future__ import annotations

from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery, FileParseError, PreprocessingStep


class FastaProcessor(BaseProcessor):
    extension = ".fasta"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise FileParseError(path=path, file_type="FASTA", reason=str(exc), cause=exc) from exc

        sequences = self._parse_sequences(text)
        if not sequences:
            raise FileParseError(path=path, file_type="FASTA", reason="No sequences found in file")

        lengths = [len(seq) for _, seq in sequences]
        residues = set("".join(seq.upper() for _, seq in sequences))
        seq_type = self._infer_type(residues)

        metadata = {
            "n_sequences": len(sequences),
            "sequence_type": seq_type,
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_mean": round(sum(lengths) / len(lengths), 1),
            "ids": [sid for sid, _ in sequences[:5]],  # first 5 IDs as sample
        }

        steps: list[PreprocessingStep] = []

        steps.append(PreprocessingStep(
            name="Encode sequences",
            rationale=f"{seq_type} sequences cannot be used as raw strings by ML models",
            suggested_method="one-hot encoding or integer tokenization per residue",
        ))

        if min(lengths) != max(lengths):
            steps.append(PreprocessingStep(
                name="Pad or truncate to uniform length",
                rationale=f"Sequence lengths vary ({min(lengths)}–{max(lengths)} residues) — models require fixed-size input",
                suggested_method="pad with gap character or truncate to a fixed length; or use MSA-based alignment",
            ))

        return FileDiscovery(
            file_path=str(path),
            file_type="FASTA",
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

    def _infer_type(self, residues: set) -> str:
        aa_chars = set("ACDEFGHIKLMNPQRSTVWY")
        nuc_chars = set("ACGTU")
        residues_upper = {r for r in residues if r.isalpha()}
        if residues_upper <= nuc_chars:
            return "nucleotide"
        if residues_upper <= aa_chars | {"X", "B", "Z", "J", "O", "U"}:
            return "amino acid"
        return "unknown"
