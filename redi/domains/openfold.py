"""
REDI - OpenFold Dataset Processor

Transforms an OpenFold-style dataset (fasta/ + pdb/ + alignments/) into
per-protein NPZ feature files ready to train AlphaFold/OpenFold models.

Expected input layout:
    <root>/
        fasta/          <PDB_ID>.fasta
        pdb/            <PDB_ID>.cif
        alignments/     <PDB_ID>/
                            uniref.a3m
                            bfd.mgnify30.metaeuk30.smag30.a3m
                            pdb70.m8     (template hits — not yet used)
        protein_list.txt  (optional; one PDB ID per line)

Output (one NPZ per protein):
    aatype                   [N_res, 21]      one-hot amino acid identity
    residue_index            [N_res]          0-based position in chain
    msa                      [N_seq, N_res]   integer-encoded MSA rows
    msa_mask                 [N_seq, N_res]   1 where MSA position is valid
    deletion_matrix_int      [N_seq, N_res]   insertion counts per position
    all_atom_positions       [N_res, 37, 3]   ground-truth atom coordinates
    all_atom_mask            [N_res, 37]      1 where atom is present
    seq_length               scalar           number of residues
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — AlphaFold / OpenFold canonical orderings
# ---------------------------------------------------------------------------

# 37 heavy-atom types (same order as AlphaFold atom14 / atom37 tables)
ATOM_TYPES: List[str] = [
    "N", "CA", "C", "CB", "O",
    "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
ATOM_TYPE_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = 37

# 20 standard residue types; index 20 = unknown / non-standard
RESTYPES: List[str] = list("ARNDCQEGHILKMFPSTWYV")
RESTYPE_INDEX: Dict[str, int] = {r: i for i, r in enumerate(RESTYPES)}
NUM_RESTYPES = 21  # 20 standard + 1 unknown

# HHBLITS alphabet — used by OpenFold for MSA integer encoding (gap = 21)
# Source: openfold/np/residue_constants.py::HHBLITS_AA_TO_ID
HHBLITS_AA_TO_ID: Dict[str, int] = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
    "X": 20, "-": 21,
    # ambiguous codes map to nearest standard AA (same as OpenFold)
    "B": 2, "Z": 3, "U": 1, "O": 20, "J": 20,
}
MSA_GAP_IDX = 21  # gap token in HHBLITS encoding

# Common non-standard → standard substitutions
NON_STANDARD_MAP: Dict[str, str] = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR",
    "CSO": "CYS", "PTR": "TYR", "YCM": "CYS", "KCX": "LYS",
    "PCA": "GLU", "MLY": "LYS", "M3L": "LYS",
}

# Three-letter → one-letter map
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_openfold_root(path: Path) -> bool:
    """Return True if *path* looks like an OpenFold data root directory."""
    return (
        path.is_dir()
        and (path / "fasta").is_dir()
        and (path / "pdb").is_dir()
    )


def _parse_a3m(a3m_path: Path) -> Tuple[List[str], List[List[int]]]:
    """
    Parse an A3M file into MSA sequences and deletion matrices.

    Returns
    -------
    sequences : list of str
        MSA rows (uppercase only, insertions stripped).
    deletion_counts : list of list[int]
        Per-position insertion counts (lowercase letter runs before each column).
    """
    sequences: List[str] = []
    deletion_rows: List[List[int]] = []

    current_seq_parts: List[str] = []
    in_sequence = False

    with open(a3m_path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").replace("\x00", "")
            if line.startswith(">"):
                if in_sequence:
                    raw = "".join(current_seq_parts)
                    seq, dels = _a3m_row_to_msa(raw)
                    sequences.append(seq)
                    deletion_rows.append(dels)
                current_seq_parts = []
                in_sequence = True
            elif in_sequence:
                current_seq_parts.append(line)

    # flush last record
    if current_seq_parts:
        raw = "".join(current_seq_parts)
        seq, dels = _a3m_row_to_msa(raw)
        sequences.append(seq)
        deletion_rows.append(dels)

    return sequences, deletion_rows


def _a3m_row_to_msa(raw: str) -> Tuple[str, List[int]]:
    """
    Convert one raw A3M sequence (may contain lowercase insertions) into:
      - uppercase MSA sequence (insertions removed)
      - deletion count per MSA column (number of lowercase chars before each uppercase char)
    """
    msa_chars: List[str] = []
    del_counts: List[int] = []
    pending_dels = 0

    for ch in raw:
        if ch == "-":
            del_counts.append(pending_dels)
            msa_chars.append("-")
            pending_dels = 0
        elif ch.isupper():
            del_counts.append(pending_dels)
            msa_chars.append(ch)
            pending_dels = 0
        elif ch.islower():
            pending_dels += 1
        # other chars (spaces, digits) ignored

    return "".join(msa_chars), del_counts


def _encode_msa_row(seq: str) -> np.ndarray:
    """Map a single MSA sequence string to HHBLITS integer indices [N_res]."""
    out = np.empty(len(seq), dtype=np.int32)
    for i, ch in enumerate(seq):
        out[i] = HHBLITS_AA_TO_ID.get(ch.upper(), HHBLITS_AA_TO_ID["X"])
    return out


def _parse_mmcif(cif_path: Path) -> Tuple[str, np.ndarray, np.ndarray, float]:
    """
    Parse a mmCIF file and extract:
      sequence   : str              one-letter sequence (first chain)
      positions  : [N_res, 37, 3]  float32 atom coordinates (0 where absent)
      mask       : [N_res, 37]     float32 validity mask
      resolution : float           crystallographic resolution (0.0 if absent)
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = MMCIFParser()
        structure = parser.get_structure("x", str(cif_path))

    # Resolution from raw mmCIF dict
    mmcif_dict = MMCIF2Dict(str(cif_path))
    resolution = 0.0
    for key in ("_refine.ls_d_res_high", "_reflns.d_resolution_high"):
        val = mmcif_dict.get(key)
        if val:
            try:
                resolution = float(val[0] if isinstance(val, list) else val)
                break
            except (ValueError, TypeError):
                pass

    # Take the first model, first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())

    residues = [
        r for r in chain.get_residues()
        if r.id[0] == " "  # exclude HETATM / water
    ]

    n_res = len(residues)
    positions = np.zeros((n_res, NUM_ATOM_TYPES, 3), dtype=np.float32)
    mask = np.zeros((n_res, NUM_ATOM_TYPES), dtype=np.float32)
    seq_chars: List[str] = []

    for ri, residue in enumerate(residues):
        resname = residue.resname.strip()
        resname = NON_STANDARD_MAP.get(resname, resname)
        one_letter = THREE_TO_ONE.get(resname, "X")
        seq_chars.append(one_letter)

        for atom in residue.get_atoms():
            atom_name = atom.name.strip()
            idx = ATOM_TYPE_INDEX.get(atom_name)
            if idx is not None:
                positions[ri, idx] = atom.coord
                mask[ri, idx] = 1.0

    return "".join(seq_chars), positions, mask, resolution


def _sequence_to_aatype(seq: str) -> np.ndarray:
    """One-hot encode a sequence → int32 [N_res, 21] (OpenFold convention)."""
    n = len(seq)
    aatype = np.zeros((n, NUM_RESTYPES), dtype=np.int32)
    for i, ch in enumerate(seq):
        idx = RESTYPE_INDEX.get(ch.upper(), NUM_RESTYPES - 1)  # 20 = unknown
        aatype[i, idx] = 1
    return aatype


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_openfold_pipeline():
    """Return the full list of OpenFold pipeline steps."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.output import WriteChecksums, SaveMetadata

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------
    class LoadOpenFoldDataset(PipelineStep):
        """Detect and enumerate proteins in an OpenFold data directory."""

        name = "load_openfold_dataset"
        stage = PipelineStage.INGEST
        description = "Enumerate proteins from OpenFold dataset directory"

        def validate(self, ctx: PipelineContext) -> bool:
            return _is_openfold_root(ctx.input_path)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            root = ctx.input_path
            fasta_dir = root / "fasta"
            pdb_dir = root / "pdb"
            align_dir = root / "alignments"
            protein_list_path = root / "protein_list.txt"

            # Build protein list
            if protein_list_path.exists():
                pdb_ids = [
                    l.strip() for l in protein_list_path.read_text().splitlines()
                    if l.strip()
                ]
                logger.info(f"Using protein_list.txt: {len(pdb_ids)} entries")
            else:
                pdb_ids = sorted(
                    p.stem for p in fasta_dir.glob("*.fasta")
                )
                logger.info(f"Discovered {len(pdb_ids)} proteins from fasta/")

            proteins: List[Dict[str, Any]] = []
            missing_cif = missing_fasta = 0

            for pdb_id in pdb_ids:
                fasta_path = fasta_dir / f"{pdb_id}.fasta"
                cif_path = pdb_dir / f"{pdb_id}.cif"

                if not fasta_path.exists():
                    missing_fasta += 1
                    continue
                if not cif_path.exists():
                    missing_cif += 1
                    continue

                aln_dir = align_dir / pdb_id if align_dir.is_dir() else None
                has_msa = aln_dir is not None and aln_dir.is_dir()

                proteins.append({
                    "pdb_id": pdb_id,
                    "fasta_path": fasta_path,
                    "cif_path": cif_path,
                    "alignment_dir": aln_dir if has_msa else None,
                    "has_msa": has_msa,
                })

            logger.info(
                f"Found {len(proteins)} complete proteins "
                f"({sum(p['has_msa'] for p in proteins)} with MSA, "
                f"{missing_fasta} missing FASTA, {missing_cif} missing CIF)"
            )

            ctx.data = {"proteins": proteins}
            ctx.metadata["is_openfold"] = True
            ctx.metadata["num_proteins"] = len(proteins)
            ctx.metadata["num_with_msa"] = sum(p["has_msa"] for p in proteins)
            return ctx

    # ------------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------------
    class ParseMMCIFStructures(PipelineStep):
        """Parse mmCIF files to extract atom coordinates per protein."""

        name = "parse_mmcif_structures"
        stage = PipelineStage.PREPROCESS
        description = "Extract atom coordinates from mmCIF structure files"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("is_openfold", False)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            proteins = ctx.data["proteins"]
            failed = 0

            for prot in proteins:
                try:
                    seq, positions, mask, resolution = _parse_mmcif(prot["cif_path"])
                    prot["sequence"] = seq
                    prot["all_atom_positions"] = positions  # [N_res, 37, 3]
                    prot["all_atom_mask"] = mask            # [N_res, 37]
                    prot["resolution"] = resolution
                    prot["n_res"] = len(seq)
                except Exception as e:
                    logger.warning(f"Failed to parse {prot['pdb_id']}: {e}")
                    prot["parse_error"] = str(e)
                    failed += 1

            ok = len(proteins) - failed
            logger.info(f"Parsed {ok}/{len(proteins)} mmCIF structures")
            ctx.metadata["mmcif_parse_failures"] = failed
            return ctx

    class ParseMSAAlignments(PipelineStep):
        """Parse A3M alignment files into raw MSA sequences and deletion counts."""

        name = "parse_msa_alignments"
        stage = PipelineStage.PREPROCESS
        description = "Parse A3M MSA files into raw sequence strings and deletion counts"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("is_openfold", False)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            proteins = ctx.data["proteins"]
            total_records = 0

            for prot in proteins:
                if "parse_error" in prot:
                    continue

                seq = prot["sequence"]

                # Always include the query as the first row (zero deletions)
                raw_seqs: List[str] = [seq]
                raw_dels: List[List[int]] = [[0] * prot["n_res"]]
                seen: set = {seq}  # deduplicate across files, matching OpenFold

                if prot["has_msa"]:
                    aln_dir: Path = prot["alignment_dir"]
                    for a3m_name in [
                        "uniref.a3m",
                        "bfd.mgnify30.metaeuk30.smag30.a3m",
                    ]:
                        a3m_path = aln_dir / a3m_name
                        if not a3m_path.exists():
                            continue
                        try:
                            seqs, dels = _parse_a3m(a3m_path)
                            for s, d in zip(seqs, dels):
                                if s in seen:
                                    continue
                                seen.add(s)
                                raw_seqs.append(s)
                                raw_dels.append(d)
                        except Exception as e:
                            logger.warning(f"{prot['pdb_id']}/{a3m_name}: {e}")

                prot["raw_msa_seqs"] = raw_seqs    # list of str, length varies
                prot["raw_msa_dels"] = raw_dels    # list of list[int]
                total_records += len(raw_seqs)

            logger.info(
                f"Parsed MSAs for {len(proteins)} proteins "
                f"(avg {total_records // max(len(proteins), 1)} rows/protein)"
            )
            return ctx

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------
    class EncodeOpenFoldFeatures(PipelineStep):
        """Encode raw sequences and MSA strings into integer/one-hot tensors."""

        name = "encode_openfold_features"
        stage = PipelineStage.TRANSFORM
        description = "One-hot encode aatype; integer-encode MSA rows and deletion matrix"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("is_openfold", False)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            proteins = ctx.data["proteins"]

            for prot in proteins:
                if "parse_error" in prot or "sequence" not in prot:
                    continue

                seq = prot["sequence"]
                n_res = prot["n_res"]

                # One-hot encode the query sequence → aatype [N_res, 21]
                prot["aatype"] = _sequence_to_aatype(seq)

                # Integer-encode each MSA row; pad/truncate to query length
                raw_seqs = prot["raw_msa_seqs"]
                raw_dels = prot["raw_msa_dels"]

                encoded_rows: List[np.ndarray] = []
                encoded_dels: List[np.ndarray] = []

                for s, d in zip(raw_seqs, raw_dels):
                    enc = _encode_msa_row(s)
                    del_arr = np.array(d, dtype=np.int32)
                    # Align to query length
                    if len(enc) != n_res:
                        enc = enc[:n_res] if len(enc) > n_res else np.pad(enc, (0, n_res - len(enc)), constant_values=MSA_GAP_IDX)
                    if len(del_arr) != n_res:
                        del_arr = del_arr[:n_res] if len(del_arr) > n_res else np.pad(del_arr, (0, n_res - len(del_arr)))
                    encoded_rows.append(enc)
                    encoded_dels.append(del_arr)

                msa_arr = np.stack(encoded_rows, axis=0)                        # [N_seq, N_res]
                del_arr = np.stack(encoded_dels, axis=0)                        # [N_seq, N_res]
                prot["msa"] = msa_arr
                prot["deletion_matrix_int"] = del_arr
                # msa_mask: 1 where position is not a gap (gap=21 in HHBLITS)
                prot["msa_mask"] = (msa_arr != MSA_GAP_IDX).astype(np.float32)

            logger.info(f"Encoded features for {len(proteins)} proteins")
            return ctx

    # ------------------------------------------------------------------
    # STRUCTURE
    # ------------------------------------------------------------------
    class AssembleOpenFoldFeatures(PipelineStep):
        """Pack encoded tensors into the final per-protein feature dictionaries."""

        name = "assemble_openfold_features"
        stage = PipelineStage.STRUCTURE
        description = "Pack aatype, MSA, and atom-coordinate tensors into feature dicts"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("is_openfold", False)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            proteins = ctx.data["proteins"]
            assembled: List[Dict[str, Any]] = []

            for prot in proteins:
                if "parse_error" in prot or "aatype" not in prot:
                    continue

                n_res = prot["n_res"]
                n_seq = prot["msa"].shape[0]

                features: Dict[str, np.ndarray] = {
                    # Core sequence features
                    "aatype":                    prot["aatype"],                          # int32 [N_res, 21]
                    "between_segment_residues":  np.zeros(n_res, dtype=np.int32),         # int32 [N_res]
                    "residue_index":             np.arange(n_res, dtype=np.int32),        # int32 [N_res]
                    "seq_length":                np.full(n_res, n_res, dtype=np.int32),   # int32 [N_res]
                    # MSA features
                    "msa":                       prot["msa"],                             # int32 [N_seq, N_res]
                    "deletion_matrix_int":       prot["deletion_matrix_int"],             # int32 [N_seq, N_res]
                    "num_alignments":            np.full(n_res, n_seq, dtype=np.int32),   # int32 [N_res]
                    # Structure features (supervised training)
                    "all_atom_positions":        prot["all_atom_positions"],              # float32 [N_res, 37, 3]
                    "all_atom_mask":             prot["all_atom_mask"],                   # float32 [N_res, 37]
                    "resolution":                np.array([prot["resolution"]], dtype=np.float32),  # float32 [1]
                    "is_distillation":           np.array(0.0, dtype=np.float32),         # float32 scalar
                }
                assembled.append({"pdb_id": prot["pdb_id"], "features": features})

            logger.info(f"Assembled features for {len(assembled)} proteins")
            ctx.data = {"proteins": assembled}
            ctx.metadata["num_assembled"] = len(assembled)
            return ctx

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    class SavePerProteinNPZ(PipelineStep):
        """Save one NPZ file per protein; split proteins into train/val/test."""

        name = "save_per_protein_npz"
        stage = PipelineStage.OUTPUT
        description = "Save per-protein OpenFold feature NPZ files"

        def __init__(
            self,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            random_seed: int = 42,
        ):
            super().__init__()
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.random_seed = random_seed

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("is_openfold", False)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            proteins = ctx.data["proteins"]
            if not proteins:
                ctx.add_error("No proteins to save")
                return ctx

            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # Split at the protein level
            rng = np.random.default_rng(self.random_seed)
            n = len(proteins)
            idx = rng.permutation(n)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            split_map: Dict[int, str] = {}
            for i in idx[:train_end]:
                split_map[i] = "train"
            for i in idx[train_end:val_end]:
                split_map[i] = "val"
            for i in idx[val_end:]:
                split_map[i] = "test"

            # Create split subdirectories
            for split in ("train", "val", "test"):
                (output_dir / split).mkdir(exist_ok=True)

            saved_files: List[str] = []
            split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

            for i, prot in enumerate(proteins):
                split = split_map[i]
                pdb_id = prot["pdb_id"]
                out_path = output_dir / split / f"{pdb_id}.npz"
                np.savez_compressed(out_path, **prot["features"])
                saved_files.append(str(out_path))
                split_counts[split] += 1

            logger.info(
                f"Saved {len(saved_files)} NPZ files — "
                f"train={split_counts['train']}, "
                f"val={split_counts['val']}, "
                f"test={split_counts['test']}"
            )
            ctx.metadata["saved_files"] = saved_files
            ctx.metadata["output_format"] = "npz"
            ctx.metadata["split"] = split_counts
            return ctx

    return [
        LoadOpenFoldDataset(),
        ParseMMCIFStructures(),
        ParseMSAAlignments(),
        EncodeOpenFoldFeatures(),
        AssembleOpenFoldFeatures(),
        SavePerProteinNPZ(),
        WriteChecksums(),
        SaveMetadata(),
    ]
