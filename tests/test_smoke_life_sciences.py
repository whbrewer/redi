"""
Smoke tests for Life Sciences (Bio/OpenFold) domain pipeline.

These tests verify that the life sciences pipeline handles
biological sequence data (FASTA, PDB, etc.).
"""

import pytest
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers for synthetic OpenProteinSet data
# ---------------------------------------------------------------------------

_RESNAME = {"A": "ALA", "G": "GLY", "M": "MET", "V": "VAL", "L": "LEU", "K": "LYS"}

def _write_minimal_mmcif(path: Path, pdb_id: str, chain: str, sequence: str, resolution: float = 2.0):
    """Write a minimal but BioPython-parseable mmCIF file."""
    lines = [
        f"data_{pdb_id}",
        f"_entry.id   {pdb_id}",
        f"_refine.ls_d_res_high  {resolution}",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for seq_id, one_letter in enumerate(sequence, start=1):
        resname = _RESNAME.get(one_letter, "ALA")
        x = float(seq_id) * 3.8
        for atom_name, symbol, dy, dz in [("N", "N", 0.0, 0.0), ("CA", "C", 1.458, 0.0), ("C", "C", 2.009, 1.420), ("O", "O", 1.246, 2.390)]:
            lines.append(
                f"ATOM   {atom_id}  {symbol}  {atom_name}  .  {resname}  {chain}  1  {seq_id}  ?  "
                f"{x:.3f}  {dy:.3f}  {dz:.3f}  1.00  10.00  {seq_id}  {resname}  {chain}  {atom_name}  1"
            )
            atom_id += 1
    path.write_text("\n".join(lines) + "\n")


def _write_minimal_a3m(path: Path, sequence: str):
    """Write a minimal A3M file with query + two homologs."""
    path.write_text(
        f">query\n{sequence}\n"
        f">homolog1\n{sequence}\n"
        f">homolog2\n{'-' * len(sequence)}\n"
    )


@pytest.fixture
def openfold_ops_root(tmp_path):
    """Minimal OpenProteinSet layout with 4 synthetic proteins."""
    proteins = [
        ("1abc", "A", "AG"),
        ("2xyz", "A", "MV"),
        ("3def", "B", "LK"),
        ("4ghi", "A", "AG"),
    ]
    ops_dir = tmp_path / "open_protein_set" / "pdb"
    mmcif_dir = tmp_path / "pdb_mmcif" / "mmcif_files"
    ops_dir.mkdir(parents=True)
    mmcif_dir.mkdir(parents=True)

    for pdb_id, chain, seq in proteins:
        chain_dir = ops_dir / f"{pdb_id}_{chain}"
        a3m_dir = chain_dir / "a3m"
        a3m_dir.mkdir(parents=True)
        _write_minimal_a3m(a3m_dir / "uniref90_hits.a3m", seq)
        _write_minimal_mmcif(mmcif_dir / f"{pdb_id}.cif", pdb_id.upper(), chain, seq)

    return tmp_path


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


class TestOpenFoldParallelPipeline:
    """Tests for the parallel OpenFold preprocessing pipeline.

    Uses synthetic mmCIF + A3M files — no dependency on the real dataset.
    """

    def test_ops_layout_detected(self, openfold_ops_root):
        """Synthetic root is recognised as OpenProteinSet layout."""
        from redi.domains.openfold import _is_openfold_ops_root
        assert _is_openfold_ops_root(openfold_ops_root)

    def test_parallel_pipeline_produces_npz(self, openfold_ops_root, tmp_path):
        """Pipeline runs end-to-end and writes one NPZ per protein."""
        from redi.domains.openfold import get_openfold_ops_pipeline
        from redi.pipeline import PipelineContext

        output_dir = tmp_path / "out"
        ctx = PipelineContext(input_path=openfold_ops_root, output_dir=output_dir)

        steps = get_openfold_ops_pipeline(output_format="npz", num_workers=2)
        for step in steps:
            ctx = step.execute(ctx)

        npz_files = list(output_dir.rglob("*.npz"))
        assert len(npz_files) == 4, f"Expected 4 NPZ files, got {len(npz_files)}"

    def test_npz_contains_expected_keys(self, openfold_ops_root, tmp_path):
        """Each NPZ file contains all required OpenFold feature tensors."""
        from redi.domains.openfold import get_openfold_ops_pipeline
        from redi.pipeline import PipelineContext

        output_dir = tmp_path / "out"
        ctx = PipelineContext(input_path=openfold_ops_root, output_dir=output_dir)

        steps = get_openfold_ops_pipeline(output_format="npz", num_workers=2)
        for step in steps:
            ctx = step.execute(ctx)

        expected_keys = {
            "aatype", "residue_index", "msa",
            "deletion_matrix_int", "all_atom_positions", "all_atom_mask", "resolution",
        }
        for npz_path in output_dir.rglob("*.npz"):
            data = np.load(npz_path)
            assert expected_keys.issubset(set(data.keys())), \
                f"{npz_path.name} missing keys: {expected_keys - set(data.keys())}"

    def test_serial_and_parallel_produce_same_output(self, openfold_ops_root, tmp_path):
        """num_workers=1 and num_workers=4 produce identical NPZ tensors."""
        from redi.domains.openfold import get_openfold_ops_pipeline
        from redi.pipeline import PipelineContext

        def run(workers, out_subdir):
            output_dir = tmp_path / out_subdir
            ctx = PipelineContext(input_path=openfold_ops_root, output_dir=output_dir)
            for step in get_openfold_ops_pipeline(output_format="npz", num_workers=workers):
                ctx = step.execute(ctx)
            return {p.name: np.load(p) for p in output_dir.rglob("*.npz")}

        serial = run(1, "serial")
        parallel = run(4, "parallel")

        assert set(serial) == set(parallel), "Different set of output files"
        for name in serial:
            for key in serial[name].keys():
                np.testing.assert_array_equal(
                    serial[name][key], parallel[name][key],
                    err_msg=f"{name}[{key}] differs between serial and parallel runs",
                )
