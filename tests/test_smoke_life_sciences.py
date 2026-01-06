"""
Smoke tests for Life Sciences (Bio/OpenFold) domain pipeline.

These tests verify that the life sciences pipeline handles
biological sequence data (FASTA, PDB, etc.).
"""

import pytest
from pathlib import Path


class TestLifeSciencesSmoke:
    """Smoke tests for Life Sciences domain."""

    def test_fasta_analyze(self, sample_bio_fasta):
        """Test analyzing FASTA protein data."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_bio_fasta))

        assert data_card.detected_domain == "life_sciences"

    def test_fasta_domain_detection(self, sample_bio_fasta):
        """Test that FASTA files are detected as life sciences domain."""
        from redi.cli import detect_domain

        domain = detect_domain(Path(sample_bio_fasta))
        assert domain == "life_sciences"

    def test_life_sciences_full_pipeline(self, sample_bio_fasta, temp_dir):
        """Test running full life sciences pipeline."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "bio_output"
        ctx = run_pipeline(
            input_path=str(sample_bio_fasta),
            output_dir=str(output_dir),
            domain="life_sciences"
        )

        assert ctx is not None
        assert ctx.domain == "life_sciences"

    def test_life_sciences_csv(self, temp_dir):
        """Test life sciences domain detection from CSV with bio columns."""
        from redi.analyze import analyze_dataset

        csv_path = temp_dir / "sequences.csv"
        csv_path.write_text("""protein_id,sequence,gene,chain,residue_count
P001,MKVLWAAL,BRCA1,A,8
P002,MGSSHHH,TP53,A,7
P003,MKTAYIAK,MYC,B,8
""")

        data_card = analyze_dataset(str(csv_path))
        assert data_card.detected_domain == "life_sciences"


class TestLifeSciencesDataCard:
    """Tests for Life Sciences Data Card generation."""

    def test_bio_recommendations(self, sample_bio_fasta):
        """Test life sciences pipeline recommendations."""
        from redi.analyze import get_recommendations

        recs = get_recommendations(str(sample_bio_fasta))

        assert recs["domain"] == "life_sciences"
        assert "biopython" in recs["recommended_tools"]

    def test_pii_detection_patient_data(self, temp_dir):
        """Test that patient data triggers PII detection."""
        from redi.analyze import analyze_dataset

        csv_path = temp_dir / "patient_data.csv"
        csv_path.write_text("""patient_id,patient_name,sample_id,gene_expression
1,John Doe,S001,2.5
2,Jane Smith,S002,3.1
""")

        data_card = analyze_dataset(str(csv_path))
        assert data_card.sensitive_attributes.contains_pii
        assert "patient_name" in data_card.sensitive_attributes.pii_types


class TestPDBFiles:
    """Tests for PDB file handling."""

    @pytest.fixture
    def sample_pdb(self, temp_dir):
        """Create a minimal PDB file."""
        pdb_path = temp_dir / "sample.pdb"
        pdb_path.write_text("""HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
END
""")
        return pdb_path

    def test_pdb_domain_detection(self, sample_pdb):
        """Test that PDB files are detected as life sciences domain."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_pdb))
        assert data_card.detected_domain == "life_sciences"


class TestOpenFoldIntegration:
    """Tests for OpenFold-style data handling."""

    def test_openfold_style_csv(self, temp_dir):
        """Test handling OpenFold-style alignment data."""
        from redi.analyze import analyze_dataset

        csv_path = temp_dir / "alignments.csv"
        csv_path.write_text("""sequence_id,alignment_score,chain,num_residues,msa_depth
seq_001,0.95,A,150,1024
seq_002,0.87,A,200,512
seq_003,0.92,B,175,768
""")

        data_card = analyze_dataset(str(csv_path))
        # Should detect as life sciences due to sequence/chain/residue keywords
        assert data_card.detected_domain == "life_sciences"

    def test_protein_features_csv(self, temp_dir):
        """Test protein feature data."""
        from redi.analyze import analyze_dataset

        csv_path = temp_dir / "protein_features.csv"
        csv_path.write_text("""protein_id,amino_acid_count,molecular_weight,sequence_length
P53_HUMAN,393,43653,393
BRCA1_HUMAN,1863,207721,1863
MYC_HUMAN,439,48804,439
""")

        data_card = analyze_dataset(str(csv_path))
        assert data_card.detected_domain == "life_sciences"

        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "amino_acid_count" in feature_names or "protein_id" in feature_names
