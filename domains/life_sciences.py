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
    from pipeline import PipelineStep, PipelineStage, PipelineContext

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

                fmt = "fasta" if ctx.metadata.get("file_format") == "fasta" else "fastq"
                sequences = list(SeqIO.parse(ctx.input_path, fmt))

                ctx.data = {
                    "sequences": [str(seq.seq) for seq in sequences],
                    "ids": [seq.id for seq in sequences],
                    "descriptions": [seq.description for seq in sequences]
                }

                ctx.metadata["num_sequences"] = len(sequences)
                ctx.metadata["avg_length"] = sum(len(s) for s in ctx.data["sequences"]) / len(sequences)
                logger.info(f"Loaded {len(sequences)} sequences")

            except ImportError:
                ctx.add_error("biopython not installed")
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

                df = pd.read_csv(ctx.input_path)
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

    return [LoadFASTA(), LoadClinicalCSV()]


def get_preprocess_steps():
    """Get life sciences-specific preprocess steps"""
    from pipeline import PipelineStep, PipelineStage, PipelineContext

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
    from pipeline import PipelineStep, PipelineStage, PipelineContext

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
    from pipeline import PipelineStep, PipelineStage, PipelineContext

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
    from pipeline.output import get_output_steps
    return get_output_steps(output_format="hdf5")


def get_full_pipeline():
    """Get the complete life sciences preprocessing pipeline"""
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
