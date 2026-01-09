"""Life sciences domain FAIR validators.

Validates life sciences data against PDB and FASTA standards.
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from ..principles import DomainStandardResult


# PDB record types
PDB_REQUIRED_RECORDS = {"HEADER", "ATOM", "END"}
PDB_RECOMMENDED_RECORDS = {"TITLE", "COMPND", "SOURCE", "AUTHOR", "CRYST1"}

# Valid amino acid single letter codes
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
# Valid nucleotide codes
NUCLEOTIDES = set("ACGTU")
# Extended codes (including ambiguous)
EXTENDED_AMINO = AMINO_ACIDS | set("BXZJUO*-")
EXTENDED_NUCLEOTIDE = NUCLEOTIDES | set("RYSWKMBDHVN-")


class PDBValidator:
    """Validates PDB (Protein Data Bank) format files."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate PDB file format.

        Args:
            file_path: Path to PDB file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".pdb", ".ent"):
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            content = file_path.read_text(errors="ignore")
            lines = content.split("\n")

            # Collect record types
            record_types = set()
            atom_count = 0
            hetatm_count = 0

            for line in lines:
                if len(line) >= 6:
                    record = line[:6].strip()
                    record_types.add(record)
                    if record == "ATOM":
                        atom_count += 1
                    elif record == "HETATM":
                        hetatm_count += 1

            # Check required records
            for record in PDB_REQUIRED_RECORDS:
                if record in record_types:
                    passed.append(f"Has {record} record")
                else:
                    if record == "END":
                        warnings.append("Missing END record")
                    else:
                        failed.append(f"Missing {record} record")

            # Check recommended records
            rec_found = 0
            for record in PDB_RECOMMENDED_RECORDS:
                if record in record_types:
                    rec_found += 1

            if rec_found > 0:
                passed.append(f"{rec_found} recommended records present")
            if rec_found < len(PDB_RECOMMENDED_RECORDS) // 2:
                missing = [r for r in PDB_RECOMMENDED_RECORDS if r not in record_types]
                warnings.append(f"Consider adding: {', '.join(missing[:3])}")

            # Report atom counts
            if atom_count > 0:
                passed.append(f"{atom_count} ATOM records")
            if hetatm_count > 0:
                passed.append(f"{hetatm_count} HETATM records")

            # Check HEADER format (if present)
            header_match = re.search(r"^HEADER\s+(.{40})\s+(\d{2}-\w{3}-\d{2})", content, re.MULTILINE)
            if header_match:
                passed.append("Valid HEADER format with date")
            elif "HEADER" in record_types:
                warnings.append("HEADER present but may not follow standard format")

            # Check for CRYST1 (crystal info)
            if "CRYST1" in record_types:
                passed.append("Crystal parameters present (CRYST1)")

            # Calculate compliance
            total = len(passed) + len(failed)
            compliance = len(passed) / total if total > 0 else 0.0

            return DomainStandardResult(
                standard_name="PDB Format",
                version="3.3",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except Exception as e:
            return DomainStandardResult(
                standard_name="PDB Format",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )


class FASTAValidator:
    """Validates FASTA sequence format files."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate FASTA file format.

        Args:
            file_path: Path to FASTA file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".fasta", ".fa", ".fna", ".faa", ".fastq"):
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        try:
            content = file_path.read_text(errors="ignore")
            lines = [l.strip() for l in content.split("\n") if l.strip()]

            if not lines:
                failed.append("Empty file")
                return DomainStandardResult(
                    standard_name="FASTA",
                    compliance_percentage=0.0,
                    failed_checks=failed,
                )

            # Count sequences
            sequence_count = 0
            current_seq = []
            seq_type = None
            invalid_chars = set()

            for line in lines:
                if line.startswith(">"):
                    sequence_count += 1
                    # Validate header
                    if len(line) > 1:
                        header_content = line[1:].strip()
                        if header_content:
                            if sequence_count == 1:
                                passed.append("Valid header format")
                            # Check for structured identifier
                            if "|" in header_content or header_content.split()[0].count("_") > 0:
                                if sequence_count == 1:
                                    passed.append("Structured sequence identifier")
                    else:
                        warnings.append(f"Empty header at sequence {sequence_count}")

                    # Process previous sequence
                    if current_seq:
                        self._validate_sequence(
                            "".join(current_seq), seq_type, passed, warnings, invalid_chars
                        )
                    current_seq = []
                else:
                    current_seq.append(line.upper())

            # Process last sequence
            if current_seq:
                self._validate_sequence(
                    "".join(current_seq), seq_type, passed, warnings, invalid_chars
                )

            # Report results
            if sequence_count > 0:
                passed.append(f"{sequence_count} sequences found")
            else:
                failed.append("No sequences found (missing '>' headers)")

            if invalid_chars:
                warnings.append(f"Non-standard characters: {', '.join(sorted(invalid_chars)[:5])}")

            # Calculate compliance
            total = len(passed) + len(failed)
            compliance = len(passed) / total if total > 0 else 0.0

            return DomainStandardResult(
                standard_name="FASTA",
                compliance_percentage=compliance * 100,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
            )

        except Exception as e:
            return DomainStandardResult(
                standard_name="FASTA",
                compliance_percentage=0.0,
                failed_checks=[f"Error reading file: {str(e)}"],
            )

    def _validate_sequence(
        self,
        seq: str,
        seq_type: Optional[str],
        passed: List[str],
        warnings: List[str],
        invalid_chars: set,
    ) -> None:
        """Validate a single sequence."""
        if not seq:
            warnings.append("Empty sequence found")
            return

        seq_chars = set(seq.upper())

        # Determine sequence type
        if seq_chars <= EXTENDED_NUCLEOTIDE:
            if not any("Valid nucleotide" in p for p in passed):
                passed.append("Valid nucleotide sequence alphabet")
        elif seq_chars <= EXTENDED_AMINO:
            if not any("Valid amino acid" in p for p in passed):
                passed.append("Valid amino acid sequence alphabet")
        else:
            # Find invalid characters
            invalid = seq_chars - EXTENDED_AMINO - EXTENDED_NUCLEOTIDE
            invalid_chars.update(invalid)
