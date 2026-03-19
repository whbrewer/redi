"""
DRAI Life Sciences Domain Module

Domain-specific preprocessing for biological/medical data.
Supports protein sequences, genomics, and clinical data.

Tools: biopython, presidio, scikit-learn
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ["biopython", "presidio-analyzer", "scikit-learn"]

# Standard amino acid vocabulary
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# PII patterns for anonymization
PII_PATTERNS = [
    "name", "email", "phone", "address", "ssn", "social_security",
    "patient_id", "mrn", "dob", "date_of_birth", "ip_address"
]


def get_ingest_steps():
    """Get life sciences-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error
    from ..pipeline.ingest import DetectFormat

    class LoadFASTA(PipelineStep):
        """Load FASTA sequence file"""

        name = "load_fasta"
        stage = PipelineStage.INGEST
        description = "Load FASTA sequences using BioPython"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") in ["fasta", "fastq"]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                from Bio import SeqIO

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]
                fmt = "fasta" if ctx.metadata.get("file_format") == "fasta" else "fastq"

                all_sequences = []
                for i, path in enumerate(paths):
                    if len(paths) > 1:
                        logger.debug(f"Loading file {i+1}/{len(paths)}: {path.name}")
                    seqs = list(SeqIO.parse(path, fmt))
                    all_sequences.extend(seqs)

                if len(paths) > 1:
                    logger.info(f"Loaded sequences from {len(paths)} files")

                ctx.data = {
                    "sequences": [str(seq.seq) for seq in all_sequences],
                    "ids": [seq.id for seq in all_sequences],
                    "descriptions": [seq.description for seq in all_sequences]
                }

                ctx.metadata["num_sequences"] = len(all_sequences)
                ctx.metadata["avg_length"] = sum(len(s) for s in ctx.data["sequences"]) / len(all_sequences) if all_sequences else 0
                logger.info(f"Loaded {len(all_sequences)} sequences")

            except ImportError:
                ctx.add_error(missing_dependency_error("biopython", "life_sciences"))
            except Exception as e:
                ctx.add_error(f"Error loading FASTA: {str(e)}")

            return ctx

    class LoadClinicalCSV(PipelineStep):
        """Load clinical/medical CSV data"""

        name = "load_clinical_csv"
        stage = PipelineStage.INGEST
        description = "Load clinical CSV data"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "csv" and ctx.domain == "life_sciences"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

                if len(paths) == 1:
                    df = pd.read_csv(paths[0])
                else:
                    dfs = []
                    for i, path in enumerate(paths):
                        logger.debug(f"Loading file {i+1}/{len(paths)}: {path.name}")
                        dfs.append(pd.read_csv(path))
                    df = pd.concat(dfs, axis=0, ignore_index=True)
                    logger.info(f"Concatenated {len(paths)} files")

                ctx.data = df

                # Detect PII columns
                pii_cols = []
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in PII_PATTERNS):
                        pii_cols.append(col)

                ctx.metadata["pii_columns"] = pii_cols
                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)
                logger.info(f"Loaded clinical data: {len(df)} rows, {len(pii_cols)} potential PII columns")

            except Exception as e:
                ctx.add_error(f"Error loading clinical CSV: {str(e)}")

            return ctx

    return [DetectFormat(), LoadFASTA(), LoadClinicalCSV()]


def get_preprocess_steps():
    """Get life sciences-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class AnonymizePII(PipelineStep):
        """Anonymize personally identifiable information"""

        name = "anonymize_pii"
        stage = PipelineStage.PREPROCESS
        description = "Anonymize PII columns"

        def __init__(self, method: str = "hash"):
            super().__init__()
            self.method = method  # "hash", "remove", "mask"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import hashlib

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                pii_cols = ctx.metadata.get("pii_columns", [])

                anonymized = []
                for col in pii_cols:
                    if col not in df.columns:
                        continue

                    if self.method == "hash":
                        df[col] = df[col].apply(
                            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
                            if pd.notna(x) else None
                        )
                    elif self.method == "remove":
                        df = df.drop(columns=[col])
                    elif self.method == "mask":
                        df[col] = "***ANONYMIZED***"

                    anonymized.append(col)

                ctx.data = df
                ctx.metadata["anonymized_columns"] = anonymized
                logger.info(f"Anonymized {len(anonymized)} PII columns")

            except Exception as e:
                ctx.add_error(f"Error anonymizing: {str(e)}")

            return ctx

    class FilterSequences(PipelineStep):
        """Filter sequences by length and quality"""

        name = "filter_sequences"
        stage = PipelineStage.PREPROCESS
        description = "Filter sequences by length"

        def __init__(self, min_length: int = 10, max_length: int = 5000):
            super().__init__()
            self.min_length = min_length
            self.max_length = max_length

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                if not isinstance(ctx.data, dict) or "sequences" not in ctx.data:
                    return ctx

                sequences = ctx.data["sequences"]
                ids = ctx.data.get("ids", [])
                descriptions = ctx.data.get("descriptions", [])

                # Filter by length
                filtered_seqs = []
                filtered_ids = []
                filtered_descs = []

                for i, seq in enumerate(sequences):
                    if self.min_length <= len(seq) <= self.max_length:
                        filtered_seqs.append(seq)
                        if ids:
                            filtered_ids.append(ids[i])
                        if descriptions:
                            filtered_descs.append(descriptions[i])

                ctx.data = {
                    "sequences": filtered_seqs,
                    "ids": filtered_ids,
                    "descriptions": filtered_descs
                }

                ctx.metadata["filtered_sequences"] = len(sequences) - len(filtered_seqs)
                logger.info(f"Filtered sequences: {len(sequences)} -> {len(filtered_seqs)}")

            except Exception as e:
                ctx.add_error(f"Error filtering sequences: {str(e)}")

            return ctx

    return [AnonymizePII(), FilterSequences()]


def get_transform_steps():
    """Get life sciences-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class EncodeSequences(PipelineStep):
        """Encode protein/DNA sequences as integers"""

        name = "encode_sequences"
        stage = PipelineStage.TRANSFORM
        description = "Encode sequences as integer arrays"

        def __init__(self, sequence_type: str = "protein", max_length: Optional[int] = None):
            super().__init__()
            self.sequence_type = sequence_type
            self.max_length = max_length

            if sequence_type == "protein":
                self.vocab = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding
                self.vocab["X"] = len(self.vocab) + 1  # Unknown
            else:  # DNA/RNA
                self.vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "N": 5}

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                if not isinstance(ctx.data, dict) or "sequences" not in ctx.data:
                    return ctx

                sequences = ctx.data["sequences"]

                # Determine max length
                if self.max_length:
                    max_len = self.max_length
                else:
                    max_len = max(len(s) for s in sequences)

                # Encode sequences
                encoded = np.zeros((len(sequences), max_len), dtype=np.int32)

                for i, seq in enumerate(sequences):
                    for j, char in enumerate(seq[:max_len]):
                        encoded[i, j] = self.vocab.get(char.upper(), self.vocab.get("X", 0))

                ctx.data = {
                    "X": encoded,
                    "ids": ctx.data.get("ids", []),
                    "vocab_size": len(self.vocab) + 1,  # +1 for padding
                    "max_length": max_len
                }

                ctx.metadata["encoded_shape"] = encoded.shape
                ctx.metadata["vocab_size"] = len(self.vocab) + 1
                logger.info(f"Encoded {len(sequences)} sequences to shape {encoded.shape}")

            except Exception as e:
                ctx.add_error(f"Error encoding sequences: {str(e)}")

            return ctx

    class ComputeSequenceFeatures(PipelineStep):
        """Compute sequence-based features"""

        name = "sequence_features"
        stage = PipelineStage.TRANSFORM
        description = "Compute sequence composition features"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                if not isinstance(ctx.data, dict) or "sequences" not in ctx.data:
                    return ctx

                sequences = ctx.data["sequences"]

                # Compute amino acid composition
                features = []
                for seq in sequences:
                    seq_upper = seq.upper()
                    total = len(seq_upper)
                    comp = {aa: seq_upper.count(aa) / total if total > 0 else 0 for aa in AMINO_ACIDS}
                    features.append(list(comp.values()))

                feature_names = [f"AA_{aa}" for aa in AMINO_ACIDS]
                ctx.data["composition_features"] = np.array(features, dtype=np.float32)
                ctx.data["composition_feature_names"] = feature_names

                logger.info(f"Computed {len(feature_names)} composition features")

            except Exception as e:
                ctx.add_error(f"Error computing sequence features: {str(e)}")

            return ctx

    return [EncodeSequences(), ComputeSequenceFeatures()]


def get_structure_steps():
    """Get life sciences-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class PrepareForTransformer(PipelineStep):
        """Prepare sequences for transformer models"""

        name = "prepare_transformer"
        stage = PipelineStage.STRUCTURE
        description = "Prepare data for transformer training"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                if not isinstance(ctx.data, dict) or "X" not in ctx.data:
                    return ctx

                X = ctx.data["X"]

                # Create attention masks (1 for real tokens, 0 for padding)
                attention_mask = (X != 0).astype(np.int32)

                ctx.data["attention_mask"] = attention_mask
                logger.info("Created attention masks for transformer")

            except Exception as e:
                ctx.add_error(f"Error preparing for transformer: {str(e)}")

            return ctx

    return [PrepareForTransformer()]


def get_output_steps():
    """Get life sciences-specific output steps"""
    from ..pipeline.output import get_output_steps
    return get_output_steps(output_format="hdf5")


def get_full_pipeline(input_path=None, output_format=None, max_proteins=None):
    """Get the complete life sciences preprocessing pipeline.

    If *input_path* is an OpenFold data root (has fasta/ and pdb/ subdirs),
    returns the OpenFold-specific pipeline instead of the generic one.

    If *input_path* is an OpenProteinSet root (has open_protein_set/pdb/ and
    pdb_mmcif/mmcif_files/), returns the OPS-native pipeline.

    Args:
        output_format: Passed through to the OpenFold pipeline factories.
            "npz" (default) — full IPTSO pipeline, per-protein NPZ output.
            "openfold_native" — organize raw files for OpenFoldDataModule.
        max_proteins: Cap on proteins to process (OPS pipeline only).
    """
    from pathlib import Path
    if input_path is not None:
        p = Path(input_path)
        if p.is_dir():
            from .openfold import (
                _is_openfold_root, _is_openfold_ops_root,
                get_openfold_pipeline, get_openfold_ops_pipeline,
            )
            if _is_openfold_ops_root(p):
                logger.info("Detected OpenProteinSet layout — using OPS pipeline")
                return get_openfold_ops_pipeline(
                    output_format=output_format, max_proteins=max_proteins
                )
            if _is_openfold_root(p):
                logger.info("Detected OpenFold dataset layout — using OpenFold pipeline")
                return get_openfold_pipeline(output_format=output_format)

    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
