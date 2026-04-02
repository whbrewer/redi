"""
redi/discover/processors/mmcif_processor.py — mmCIF protein structure file analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

from redi.discover.processors import BaseProcessor, FileDiscovery, FileParseError, PreprocessingStep


class MmCIFProcessor(BaseProcessor):
    extension = ".cif"

    def analyze(self, path: Path) -> FileDiscovery:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise FileParseError(path=path, file_type="mmCIF", reason=str(exc), cause=exc) from exc

        metadata = self._extract_metadata(text)
        steps = self._suggest_steps(metadata)

        return FileDiscovery(
            file_path=str(path),
            file_type="mmCIF",
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
            preprocessing_steps=steps,
        )

    def _extract_metadata(self, text: str) -> dict:
        meta = {}

        # PDB entry ID
        m = re.search(r"_entry\.id\s+(\S+)", text)
        if m:
            meta["pdb_id"] = m.group(1)

        # Experimental method
        m = re.search(r"_exptl\.method\s+['\"]?(.+?)['\"]?\s*\n", text, re.IGNORECASE)
        if m:
            meta["experimental_method"] = m.group(1).strip().strip("'\"")

        # Resolution
        for field in ("_refine.ls_d_res_high", "_reflns.d_resolution_high"):
            m = re.search(rf"{re.escape(field)}\s+(\d+\.?\d*)", text)
            if m:
                try:
                    meta["resolution_angstrom"] = float(m.group(1))
                except ValueError:
                    pass
                break

        # Chains
        chains = re.findall(r"_atom_site\.auth_asym_id\s+(\S+)", text)
        if not chains:
            chains = list(set(re.findall(r"\n[AH][TE][OE][MT]\s+\d+\s+\S+\s+\S+\s+(\w)\s+", text)))
        if chains:
            meta["chains"] = sorted(set(chains))
            meta["n_chains"] = len(set(chains))

        # Sequence(s) from entity_poly
        seqs = re.findall(r"_entity_poly\.pdbx_seq_one_letter_code(?:_can)?\s+[;\n]?\s*([A-Z\n]+?)\s*;", text, re.DOTALL)
        if seqs:
            clean = ["".join(s.split()) for s in seqs]
            meta["n_chains_with_sequence"] = len(clean)
            lengths = [len(s) for s in clean]
            meta["sequence_length_min"] = min(lengths)
            meta["sequence_length_max"] = max(lengths)

        return meta

    def _suggest_steps(self, meta: dict) -> list[PreprocessingStep]:
        steps: list[PreprocessingStep] = []

        steps.append(PreprocessingStep(
            name="Extract atomic coordinates",
            rationale="mmCIF is a text format — ML models require numerical coordinate tensors",
            suggested_method="parse ATOM records into (N, C, CA, O, CB) coordinate arrays per residue",
        ))

        res = meta.get("resolution_angstrom")
        if res is not None and res > 3.0:
            steps.append(PreprocessingStep(
                name="Check structure quality",
                rationale=f"Resolution is {res} Å — structures above 3.0 Å may have higher positional uncertainty",
                suggested_method="filter by resolution threshold or down-weight in training",
            ))

        n_chains = meta.get("n_chains", 1)
        if n_chains > 1:
            steps.append(PreprocessingStep(
                name="Handle multi-chain structure",
                rationale=f"Structure has {n_chains} chains — decide whether to process as a complex or split into individual chains",
                suggested_method="crop to single chain or encode chain identity as a feature",
            ))

        steps.append(PreprocessingStep(
            name="Align to sequence",
            rationale="Structure coordinates must be paired with the corresponding sequence from FASTA/MSA inputs",
            suggested_method="match residue numbering between ATOM records and sequence; handle missing residues",
        ))

        return steps
