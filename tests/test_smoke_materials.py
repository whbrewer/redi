"""
Smoke tests for Materials Science (AFLOW) domain pipeline.

These tests verify that the materials science pipeline handles
crystal structure data and materials properties.
"""

import pytest
from pathlib import Path


class TestMaterialsSmoke:
    """Smoke tests for Materials Science domain."""

    def test_materials_analyze(self, sample_materials_csv):
        """Test analyzing materials science data."""
        from redi.analyze import analyze_dataset, generate_report

        data_card = analyze_dataset(str(sample_materials_csv))

        assert data_card.detected_domain == "materials"
        assert data_card.overview.num_records == 6
        assert data_card.readiness_level >= 1

        # Generate report should not raise
        report = generate_report(str(sample_materials_csv))
        assert "Materials" in report or "materials" in report

    def test_materials_full_pipeline(self, sample_materials_csv, temp_dir):
        """Test running full materials pipeline."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "materials_output"
        ctx = run_pipeline(
            input_path=str(sample_materials_csv),
            output_dir=str(output_dir),
            domain="materials"
        )

        assert ctx is not None
        assert ctx.domain == "materials"

    def test_materials_domain_detection(self, sample_materials_csv):
        """Test that materials domain is auto-detected."""
        from redi.cli import detect_domain

        domain = detect_domain(Path(sample_materials_csv))
        assert domain == "materials"

    def test_materials_feature_columns(self, sample_materials_csv):
        """Test that materials features are detected."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_materials_csv))

        feature_names = [f["name"] for f in data_card.characteristics.features]
        assert "formula" in feature_names
        assert "band_gap" in feature_names
        assert "formation_energy" in feature_names


class TestMaterialsDataCard:
    """Tests for Materials Data Card generation."""

    def test_materials_recommendations(self, sample_materials_csv):
        """Test materials pipeline recommendations."""
        from redi.analyze import get_recommendations

        recs = get_recommendations(str(sample_materials_csv))

        assert recs["domain"] == "materials"
        assert "pymatgen" in recs["recommended_tools"]

    def test_materials_aflow_style(self, temp_dir):
        """Test AFLOW-style materials data."""
        from redi.analyze import analyze_dataset

        # Create AFLOW-style CSV
        csv_path = temp_dir / "aflow_data.csv"
        csv_path.write_text("""auid,compound,enthalpy_formation_atom,volume_atom,density
aflow:001,NaCl,-4.26,22.5,2.16
aflow:002,MgO,-6.13,18.7,3.58
aflow:003,CaF2,-12.83,13.2,3.18
""")

        data_card = analyze_dataset(str(csv_path))
        assert data_card.detected_domain == "materials"


class TestMaterialsCIF:
    """Tests for CIF file handling (if pymatgen available)."""

    @pytest.fixture
    def sample_cif(self, temp_dir):
        """Create a sample CIF file."""
        cif_path = temp_dir / "sample.cif"
        cif_path.write_text("""data_NaCl
_symmetry_space_group_name_H-M   'F m -3 m'
_cell_length_a   5.6402
_cell_length_b   5.6402
_cell_length_c   5.6402
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
""")
        return cif_path

    def test_cif_domain_detection(self, sample_cif):
        """Test that CIF files are detected as materials domain."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_cif))
        assert data_card.detected_domain == "materials"
