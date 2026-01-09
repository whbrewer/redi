"""
Tests for FAIR assessment module.
"""

import pytest
import json
from pathlib import Path

from redi.fair import (
    FAIRAssessor,
    FAIRMetadata,
    FAIRScore,
    FAIRRecommender,
    ComplianceLevel,
    FAIRCategory,
    assess_fair,
    generate_fair_report,
    create_principle_score,
    create_all_principle_scores,
    FAIR_PRINCIPLES,
)
from redi.fair.standards import (
    get_domain_standards,
    get_standard,
    get_standards_for_extension,
    list_all_standards,
)


class TestFAIRPrinciples:
    """Tests for FAIR principle definitions."""

    def test_fair_principles_complete(self):
        """Verify all 15 FAIR principles are defined."""
        expected_ids = [
            "F1", "F2", "F3", "F4",
            "A1", "A1.1", "A1.2", "A2",
            "I1", "I2", "I3",
            "R1", "R1.1", "R1.2", "R1.3",
        ]
        for pid in expected_ids:
            assert pid in FAIR_PRINCIPLES, f"Missing principle: {pid}"

    def test_create_principle_score(self):
        """Test creating a single principle score."""
        score = create_principle_score("F1")
        assert score.principle_id == "F1"
        assert score.category == FAIRCategory.FINDABLE
        assert score.score == ComplianceLevel.NOT_ASSESSED

    def test_create_all_principle_scores(self):
        """Test creating all principle scores."""
        scores = create_all_principle_scores()
        assert len(scores) == 15
        assert "F1" in scores
        assert "R1.3" in scores

    def test_compliance_level_values(self):
        """Verify compliance level enum values."""
        assert ComplianceLevel.NOT_ASSESSED.value == 0
        assert ComplianceLevel.NOT_COMPLIANT.value == 1
        assert ComplianceLevel.PARTIALLY_COMPLIANT.value == 2
        assert ComplianceLevel.FULLY_COMPLIANT.value == 3


class TestFAIRScore:
    """Tests for FAIRScore dataclass."""

    def test_fair_score_creation(self):
        """Test creating a FAIRScore."""
        score = FAIRScore(
            dataset_name="test.csv",
            dataset_path="/path/to/test.csv",
            domain="general",
        )
        assert score.dataset_name == "test.csv"
        assert score.overall_score == 0.0

    def test_calculate_scores(self):
        """Test score calculation."""
        score = FAIRScore(
            dataset_name="test.csv",
            principles=create_all_principle_scores(),
        )

        # Set some scores
        score.principles["F1"].score = ComplianceLevel.FULLY_COMPLIANT
        score.principles["F2"].score = ComplianceLevel.PARTIALLY_COMPLIANT
        score.principles["A1"].score = ComplianceLevel.FULLY_COMPLIANT

        score.calculate_scores()

        assert score.overall_score > 0
        assert "findable" in score.category_scores
        assert "accessible" in score.category_scores

    def test_to_dict(self):
        """Test converting to dictionary."""
        score = FAIRScore(
            dataset_name="test.csv",
            domain="climate",
        )
        d = score.to_dict()

        assert d["dataset_name"] == "test.csv"
        assert d["domain"] == "climate"
        assert "overall_score" in d
        assert "category_scores" in d

    def test_to_markdown(self):
        """Test markdown generation."""
        score = FAIRScore(
            dataset_name="test.csv",
            domain="climate",
            principles=create_all_principle_scores(),
        )
        score.calculate_scores()

        md = score.to_markdown()

        assert "# FAIR Assessment Report" in md
        assert "test.csv" in md
        assert "climate" in md
        assert "Findable" in md
        assert "Accessible" in md


class TestFAIRMetadata:
    """Tests for FAIRMetadata."""

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "persistent_identifier": "10.5281/zenodo.123456",
            "license": "CC-BY-4.0",
            "is_registered": True,
        }
        metadata = FAIRMetadata.from_dict(data)

        assert metadata.persistent_identifier == "10.5281/zenodo.123456"
        assert metadata.license == "CC-BY-4.0"
        assert metadata.is_registered is True

    def test_metadata_defaults(self):
        """Test default metadata values."""
        metadata = FAIRMetadata()

        assert metadata.persistent_identifier is None
        assert metadata.license is None
        assert metadata.is_registered is False


class TestFAIRAssessor:
    """Tests for FAIRAssessor class."""

    def test_assessor_creation(self):
        """Test creating an assessor."""
        assessor = FAIRAssessor(domain="climate")
        assert assessor.domain == "climate"

    def test_assess_basic_csv(self, sample_csv):
        """Test FAIR assessment on basic CSV."""
        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(file_path=str(sample_csv))

        assert isinstance(score, FAIRScore)
        assert score.dataset_name == "sample.csv"
        assert len(score.principles) == 15

    def test_assess_with_metadata(self, sample_csv):
        """Test FAIR assessment with external metadata."""
        metadata = FAIRMetadata(
            persistent_identifier="10.5281/zenodo.123456",
            license="CC-BY-4.0",
            is_registered=True,
            registry_url="https://zenodo.org/record/123456",
        )

        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(
            file_path=str(sample_csv),
            metadata=metadata,
        )

        # With DOI, F1 should be fully compliant
        assert score.principles["F1"].score == ComplianceLevel.FULLY_COMPLIANT
        # With registration info, F4 should be fully compliant
        assert score.principles["F4"].score == ComplianceLevel.FULLY_COMPLIANT

    def test_assess_different_domains(self, sample_csv, sample_hpc_csv, sample_materials_csv):
        """Test FAIR assessment across different domains."""
        files_and_domains = [
            (sample_csv, "general"),
            (sample_hpc_csv, "hpc"),
            (sample_materials_csv, "materials"),
        ]

        for file_path, domain in files_and_domains:
            assessor = FAIRAssessor(domain=domain)
            score = assessor.assess(file_path=str(file_path))

            assert score.domain == domain
            assert score.overall_score >= 0


class TestFAIRRecommender:
    """Tests for FAIRRecommender class."""

    def test_recommender_creation(self, sample_csv):
        """Test creating a recommender."""
        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(file_path=str(sample_csv))

        recommender = FAIRRecommender(score, domain="general")
        recs = recommender.get_recommendations()

        assert "high_priority" in recs
        assert "medium_priority" in recs
        assert "domain_specific" in recs

    def test_quick_wins(self, sample_csv):
        """Test quick win recommendations."""
        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(file_path=str(sample_csv))

        recommender = FAIRRecommender(score)
        quick_wins = recommender.get_quick_wins()

        # Should suggest adding license and DOI for basic CSV
        assert len(quick_wins) > 0

    def test_to_markdown(self, sample_csv):
        """Test markdown generation."""
        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(file_path=str(sample_csv))

        recommender = FAIRRecommender(score)
        md = recommender.to_markdown()

        assert "# FAIR Improvement Recommendations" in md


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_assess_fair(self, sample_csv):
        """Test assess_fair convenience function."""
        score = assess_fair(str(sample_csv))

        assert isinstance(score, FAIRScore)
        assert score.overall_score >= 0

    def test_assess_fair_with_metadata(self, sample_csv):
        """Test assess_fair with metadata dict."""
        score = assess_fair(
            str(sample_csv),
            metadata={"license": "CC-BY-4.0"},
        )

        # R1.1 should improve with license
        assert score.principles["R1.1"].score.value > 0

    def test_generate_fair_report(self, sample_csv):
        """Test generate_fair_report convenience function."""
        report = generate_fair_report(str(sample_csv))

        assert "# FAIR Assessment Report" in report
        assert "# FAIR Improvement Recommendations" in report


class TestDomainStandards:
    """Tests for domain standards registry."""

    def test_get_domain_standards(self):
        """Test getting standards for a domain."""
        climate_standards = get_domain_standards("climate")
        assert "cf_conventions" in climate_standards
        assert "acdd" in climate_standards

        materials_standards = get_domain_standards("materials")
        assert "cif" in materials_standards

    def test_get_standard(self):
        """Test getting a specific standard."""
        cf = get_standard("climate", "cf_conventions")
        assert cf is not None
        assert cf.name == "CF Conventions"
        assert cf.version == "1.10"

    def test_get_standards_for_extension(self):
        """Test getting standards by file extension."""
        nc_standards = get_standards_for_extension(".nc")
        assert len(nc_standards) > 0

        cif_standards = get_standards_for_extension(".cif")
        assert len(cif_standards) > 0

    def test_list_all_standards(self):
        """Test listing all standards."""
        all_standards = list_all_standards()
        assert len(all_standards) > 0

        # Check structure
        for std in all_standards:
            assert "domain" in std
            assert "name" in std


class TestFAIRValidators:
    """Tests for domain-specific validators."""

    def test_climate_validator_import(self):
        """Test that climate validators can be imported."""
        from redi.fair.validators.climate import CFConventionsValidator, ACDDValidator

        cf_validator = CFConventionsValidator()
        acdd_validator = ACDDValidator()

        assert cf_validator is not None
        assert acdd_validator is not None

    def test_materials_validator_import(self):
        """Test that materials validators can be imported."""
        from redi.fair.validators.materials import CIFValidator, OPTIMADEValidator

        cif_validator = CIFValidator()
        optimade_validator = OPTIMADEValidator()

        assert cif_validator is not None
        assert optimade_validator is not None

    def test_life_sciences_validator_import(self):
        """Test that life sciences validators can be imported."""
        from redi.fair.validators.life_sciences import PDBValidator, FASTAValidator

        pdb_validator = PDBValidator()
        fasta_validator = FASTAValidator()

        assert pdb_validator is not None
        assert fasta_validator is not None

    def test_fasta_validation(self, sample_bio_fasta):
        """Test FASTA file validation."""
        from redi.fair.validators.life_sciences import FASTAValidator

        validator = FASTAValidator()
        result = validator.validate(sample_bio_fasta)

        assert result is not None
        assert result.standard_name == "FASTA"
        assert result.compliance_percentage > 0

    def test_hpc_validator_import(self):
        """Test that HPC validators can be imported."""
        from redi.fair.validators.hpc import SlurmValidator, PBSValidator

        slurm_validator = SlurmValidator()
        pbs_validator = PBSValidator()

        assert slurm_validator is not None
        assert pbs_validator is not None


class TestIntegration:
    """Integration tests for FAIR assessment."""

    def test_full_assessment_workflow(self, sample_csv, temp_dir):
        """Test complete assessment workflow."""
        # 1. Create metadata
        metadata = FAIRMetadata(
            persistent_identifier="10.5281/zenodo.123456",
            license="CC-BY-4.0",
            is_registered=True,
            access_url="https://example.com/data.csv",
        )

        # 2. Run assessment
        assessor = FAIRAssessor(domain="general")
        score = assessor.assess(
            file_path=str(sample_csv),
            metadata=metadata,
        )

        # 3. Get recommendations
        recommender = FAIRRecommender(score)
        recs = recommender.get_recommendations()

        # 4. Generate report
        report = score.to_markdown()

        # 5. Save report
        report_path = temp_dir / "FAIR_ASSESSMENT.md"
        report_path.write_text(report)

        assert report_path.exists()
        assert score.overall_score > 0
        assert len(report) > 100

    def test_assessment_json_output(self, sample_csv):
        """Test JSON output from assessment."""
        score = assess_fair(str(sample_csv))
        score_dict = score.to_dict()

        # Verify JSON serializable
        json_str = json.dumps(score_dict)
        parsed = json.loads(json_str)

        assert parsed["dataset_name"] == "sample.csv"
        assert "overall_score" in parsed
        assert "principles" in parsed
