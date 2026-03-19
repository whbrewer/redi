"""
Smoke tests for the combustion domain pipeline (AMReX block-structured AMR).

Tests cover:
- AMReX plotfile format detection
- Header and FAB binary parsing
- Full pipeline (ingest → preprocess → transform → structure → output)
- Config-driven options: structure_mode, target_level, normalize, derive
- Data Card / domain registration
"""

import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Reader unit tests
# ---------------------------------------------------------------------------

class TestAMReXReader:
    """Unit tests for the native AMReX plotfile reader."""

    def test_header_parsing(self, sample_amrex_plotfile):
        from redi.domains.combustion import _read_plotfile_header

        hdr = _read_plotfile_header(sample_amrex_plotfile)

        assert hdr["version"] == "HyperCLaw-V1.1"
        assert hdr["ncomp"] == 5
        assert hdr["var_names"] == ["x_velocity", "y_velocity", "temp", "Y(O2)", "Y(N2)"]
        assert hdr["spacedim"] == 2
        assert hdr["time"] == 0.0
        assert hdr["finest_level"] == 1
        assert hdr["num_levels"] == 2
        assert hdr["ref_ratios"] == [2]
        assert len(hdr["domain_boxes"]) == 2
        assert hdr["dx_per_level"][0] == pytest.approx([0.001, 0.001])
        assert hdr["dx_per_level"][1] == pytest.approx([0.0005, 0.0005])

    def test_level_header_parsing(self, sample_amrex_plotfile):
        from redi.domains.combustion import _read_level_header

        entries = _read_level_header(sample_amrex_plotfile / "Level_0")

        assert len(entries) == 1
        fname, offset = entries[0]
        assert fname == "Cell_D_00000"
        assert offset == 0

    def test_fab_reading_level0(self, sample_amrex_plotfile):
        from redi.domains.combustion import _read_fab

        level_dir = sample_amrex_plotfile / "Level_0"
        data, lo, hi = _read_fab(level_dir, "Cell_D_00000", 0)

        # Shape: (ncomp, nx, ny) = (5, 8, 16)
        assert data.shape == (5, 8, 16)
        assert lo == [0, 0]
        assert hi == [7, 15]

        # Check values we wrote in the fixture
        assert data[0].mean() == pytest.approx(0.5)    # x_velocity
        assert data[2].mean() == pytest.approx(300.0)  # temp at level 0
        assert data[3].mean() == pytest.approx(0.23)   # Y(O2)

    def test_fab_reading_level1(self, sample_amrex_plotfile):
        from redi.domains.combustion import _read_fab

        level_dir = sample_amrex_plotfile / "Level_1"
        data, lo, hi = _read_fab(level_dir, "Cell_D_00000", 0)

        assert data.shape == (5, 16, 32)
        assert data[2].mean() == pytest.approx(350.0)  # temp at level 1 (300 + 50)

    def test_full_plotfile_load(self, sample_amrex_plotfile):
        from redi.domains.combustion import _load_amrex_plotfile

        amr = _load_amrex_plotfile(sample_amrex_plotfile)

        assert "header" in amr
        assert "levels" in amr
        assert len(amr["levels"]) == 2

        lev0 = amr["levels"][0]
        assert lev0["level"] == 0
        assert len(lev0["boxes"]) == 1
        assert "x_velocity" in lev0["data"]
        assert "temp" in lev0["data"]
        assert len(lev0["data"]["temp"]) == 1      # 1 box
        assert lev0["data"]["temp"][0].shape == (8, 16)

    def test_composite_to_uniform(self, sample_amrex_plotfile):
        from redi.domains.combustion import _load_amrex_plotfile, _composite_to_uniform

        amr = _load_amrex_plotfile(sample_amrex_plotfile)
        composite = _composite_to_uniform(amr)

        # Finest level is 1 → domain box (0,0)-(15,31) → shape (16, 32)
        assert composite["temp"].shape == (16, 32)
        assert composite["x_velocity"].shape == (16, 32)

        # Temperature should be ~350 K (finer level data overwrites coarser)
        assert composite["temp"].mean() == pytest.approx(350.0)

    def test_composite_to_coarser_level(self, sample_amrex_plotfile):
        from redi.domains.combustion import _load_amrex_plotfile, _composite_to_uniform

        amr = _load_amrex_plotfile(sample_amrex_plotfile)
        composite = _composite_to_uniform(amr, target_level=0)

        # Level 0 domain: (0,0)-(7,15) → shape (8, 16)
        assert composite["temp"].shape == (8, 16)
        assert composite["temp"].mean() == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class TestAMReXDetection:
    """Tests for AMReX plotfile auto-detection."""

    def test_detect_format_via_detectformat_step(self, sample_amrex_plotfile, temp_dir):
        from redi.pipeline.ingest import DetectFormat
        from redi.pipeline import PipelineContext

        ctx = PipelineContext(
            input_path=sample_amrex_plotfile,
            output_dir=temp_dir,
            domain="combustion",
        )
        step = DetectFormat()
        ctx = step.execute(ctx)

        assert ctx.metadata.get("file_format") == "amrex-plotfile"

    def test_auto_domain_detection(self, sample_amrex_plotfile):
        from redi.cli import detect_domain

        domain = detect_domain(sample_amrex_plotfile)
        assert domain == "combustion"

    def test_non_amrex_dir_not_detected(self, temp_dir):
        from redi.cli import detect_domain

        # Directory without a Header file — should not be combustion
        domain = detect_domain(temp_dir)
        assert domain != "combustion"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class TestCombustionPipelineStages:
    """Tests for individual pipeline stages."""

    def test_ingest_stage(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "ingest_out"),
            domain="combustion",
            stages=["ingest"],
        )

        assert ctx is not None
        assert isinstance(ctx.data, dict)
        assert "levels" in ctx.data
        assert ctx.metadata["num_levels"] == 2
        assert ctx.metadata["ncomp"] == 5
        assert ctx.metadata["species"] == ["Y(O2)", "Y(N2)"]

    def test_preprocess_stage(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "preproc_out"),
            domain="combustion",
            stages=["ingest", "preprocess"],
        )

        assert ctx is not None
        # Y(O2) should still be in [0, 1] after clipping
        for lev in ctx.data["levels"]:
            for arr in lev["data"]["Y(O2)"]:
                assert arr.min() >= 0.0
                assert arr.max() <= 1.0

    def test_transform_stage_derives_fields(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "transform_out"),
            domain="combustion",
            stages=["ingest", "preprocess", "transform"],
        )

        assert ctx is not None
        derived = ctx.metadata.get("derived_fields", [])
        assert "Y(fuel_proxy)" in derived
        assert "vorticity" in derived
        # Derived fields registered in header var_names
        assert "Y(fuel_proxy)" in ctx.data["header"]["var_names"]
        assert "vorticity" in ctx.data["header"]["var_names"]

    def test_transform_stage_normalizes(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "norm_out"),
            domain="combustion",
            stages=["ingest", "preprocess", "transform"],
        )

        norm_meta = ctx.metadata.get("normalization", {})
        assert norm_meta.get("method") == "reference_scale"
        # temp normalized by 2000 K; fixture sets 300 K at level 0, 350 K at level 1
        raw_temps = {0: 300.0, 1: 350.0}
        for lev_data in ctx.data["levels"]:
            lev = lev_data["level"]
            expected = raw_temps[lev] / 2000.0
            for arr in lev_data["data"]["temp"]:
                assert arr.mean() == pytest.approx(expected, rel=0.01)

    def test_structure_stage_composite(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "struct_out"),
            domain="combustion",
            stages=["ingest", "preprocess", "transform", "structure"],
        )

        assert isinstance(ctx.data, dict)
        assert "X" in ctx.data
        X = ctx.data["X"]
        # Shape: (nvar, nx_finest, ny_finest) = (6, 16, 32)
        # 4 original + 2 derived (fuel_proxy, vorticity)
        assert X.ndim == 3
        assert X.shape[1] == 16   # finest level nx
        assert X.shape[2] == 32   # finest level ny
        assert X.shape[0] == len(ctx.data["var_names"])
        assert "x_coords" in ctx.data
        assert "y_coords" in ctx.data


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestCombustionFullPipeline:
    """End-to-end pipeline tests."""

    def test_full_pipeline_produces_snapshot_npz(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        output_dir = temp_dir / "full_out"
        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(output_dir),
            domain="combustion",
        )

        assert ctx is not None
        assert ctx.domain == "combustion"
        assert len(ctx.errors) == 0

        npz_path = output_dir / "snapshot.npz"
        assert npz_path.exists()

        data = np.load(str(npz_path), allow_pickle=True)
        assert "X" in data
        assert "var_names" in data
        assert "x_coords" in data
        assert "y_coords" in data
        assert "time" in data
        assert data["X"].ndim == 3
        assert not np.any(np.isnan(data["X"]))

    def test_full_pipeline_no_errors(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "clean_out"),
            domain="combustion",
        )
        assert ctx.errors == []

    def test_real_plotfile_if_available(self, temp_dir):
        """Run against the real HotBubble2D dataset if present on this machine."""
        real_path = Path("/opt/drai/combustion/plt00000")
        if not real_path.exists():
            pytest.skip("Real AMReX dataset not available at /opt/drai/combustion/plt00000")

        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(real_path),
            output_dir=str(temp_dir / "real_out"),
            domain="combustion",
        )

        assert ctx.errors == []
        assert ctx.metadata["num_levels"] == 4
        assert ctx.metadata["finest_level"] == 3
        assert "Y(O2)" in ctx.metadata["var_names"]
        assert "Y(N2)" in ctx.metadata["var_names"]

        npz = np.load(str(temp_dir / "real_out" / "snapshot.npz"), allow_pickle=True)
        assert npz["X"].shape[1] == 512
        assert npz["X"].shape[2] == 1024


# ---------------------------------------------------------------------------
# Config-driven options
# ---------------------------------------------------------------------------

class TestCombustionConfig:
    """Tests for YAML config options."""

    def test_config_target_level(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "lvl0_out"),
            domain="combustion",
            config={"combustion": {"target_level": 0}},
        )

        X = ctx.data["X"]
        assert X.shape[1] == 8   # Level_0: 8×16
        assert X.shape[2] == 16

    def test_config_normalize_disabled(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "nonorm_out"),
            domain="combustion",
            config={"combustion": {"normalize": {"enabled": False}}},
        )

        # temp should be raw Kelvin, not divided by 2000
        for lev in ctx.data["levels"] if "levels" in ctx.data else []:
            for arr in lev["data"].get("temp", []):
                assert arr.mean() > 1.0   # raw K, not normalized

    def test_config_normalize_custom_scale(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        custom_scale = 500.0
        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "customscale_out"),
            domain="combustion",
            stages=["ingest", "preprocess", "transform"],
            config={"combustion": {"normalize": {"enabled": True, "scales": {"temp": custom_scale}}}},
        )

        # temp / 500.0 = 300/500 = 0.6 at level 0
        for lev in ctx.data["levels"]:
            for arr in lev["data"]["temp"]:
                expected = pytest.approx(arr.mean() * custom_scale / custom_scale, rel=0.01)
                assert arr.min() > 0  # still positive

    def test_config_derive_disabled(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "noderive_out"),
            domain="combustion",
            stages=["ingest", "preprocess", "transform"],
            config={"combustion": {"derive": {"vorticity": False, "fuel_proxy": False}}},
        )

        derived = ctx.metadata.get("derived_fields", [])
        assert "vorticity" not in derived
        assert "Y(fuel_proxy)" not in derived

    def test_config_per_level_structure(self, sample_amrex_plotfile, temp_dir):
        from redi.cli import run_pipeline

        ctx = run_pipeline(
            input_path=str(sample_amrex_plotfile),
            output_dir=str(temp_dir / "perlevel_out"),
            domain="combustion",
            config={"combustion": {"structure_mode": "per_level"}},
        )

        assert isinstance(ctx.data, dict)
        assert "levels" in ctx.data
        assert "level_0" in ctx.data["levels"]
        assert "level_1" in ctx.data["levels"]
        assert ctx.metadata.get("structure_mode") == "per_level"


# ---------------------------------------------------------------------------
# Domain registration
# ---------------------------------------------------------------------------

class TestCombustionDomainRegistration:
    """Tests for domain enum and pipeline registry."""

    def test_domain_enum_exists(self):
        from redi.domains import Domain
        assert Domain.COMBUSTION.value == "combustion"

    def test_get_domain_pipeline_returns_module(self):
        from redi.domains import Domain, get_domain_pipeline
        import redi.domains.combustion as combustion_mod

        mod = get_domain_pipeline(Domain.COMBUSTION)
        assert mod is combustion_mod

    def test_pipeline_exports_all_stage_functions(self):
        from redi.domains import combustion

        for fn in ("get_ingest_steps", "get_preprocess_steps", "get_transform_steps",
                   "get_structure_steps", "get_output_steps", "get_full_pipeline"):
            assert callable(getattr(combustion, fn)), f"Missing: {fn}"

    def test_full_pipeline_has_expected_step_names(self):
        from redi.domains.combustion import get_full_pipeline

        steps = get_full_pipeline()
        names = [s.name for s in steps]

        assert "detect_format" in names
        assert "load_amrex_plotfile" in names
        assert "clip_species_fractions" in names
        assert "validate_physics" in names
        assert "derive_combustion_features" in names
        assert "normalize_combustion_fields" in names
        assert "composite_amr_levels" in names
        assert "save_combustion_npz" in names

    def test_scientific_domain_enum_includes_combustion(self):
        from redi.datacard import ScientificDomain
        assert ScientificDomain.COMBUSTION.value == "combustion"

    def test_domain_pipeline_recommendations(self):
        from redi.assess import PipelineRecommender, DOMAIN_PIPELINES
        from redi.datacard import ScientificDomain

        assert ScientificDomain.COMBUSTION in DOMAIN_PIPELINES
        pipeline = DOMAIN_PIPELINES[ScientificDomain.COMBUSTION]
        assert "steps" in pipeline
        assert "tools" in pipeline
        step_names = [s[0] for s in pipeline["steps"]]
        assert "ingest" in step_names
        assert "composite" in step_names
        assert "shard" in step_names
