"""
Unit tests for the OC22/OC20 LMDB path of the atomistic domain.

All tests use synthetic in-memory data — no real LMDB files required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    input_path: str = "/tmp/fake.lmdb",
    output_dir: str = "/tmp/redi_atomistic_test",
    config: dict = None,
    metadata: dict = None,
):
    """Create a minimal pipeline context object."""
    ctx              = SimpleNamespace()
    ctx.input_path   = input_path
    ctx.output_dir   = output_dir
    ctx.config       = config or {}
    ctx.metadata     = metadata or {}
    ctx._errors      = []
    ctx.domain       = "hydragnn"

    def add_error(msg):
        ctx._errors.append(msg)

    ctx.add_error = add_error
    return ctx


def _make_raw_record(
    natoms: int = 5,
    energy: float = -3.14,
    has_tags: bool = True,
    has_cell: bool = True,
    bad_energy: bool = False,
    bad_forces: bool = False,
    zero_atoms: bool = False,
) -> SimpleNamespace:
    """Create a synthetic raw record (mimics PyG Data object)."""
    n = 0 if zero_atoms else natoms
    obj              = SimpleNamespace()
    obj.natoms       = n
    obj.y            = float("nan") if bad_energy else energy
    obj.atomic_numbers = np.array([6, 8, 1, 1, 14][:n], dtype=np.int32)
    obj.pos          = np.random.default_rng(42).random((n, 3)).astype(np.float32)
    obj.force        = (
        np.full((n, 3), float("nan"), dtype=np.float32)
        if bad_forces
        else np.random.default_rng(7).random((n, 3)).astype(np.float32)
    )
    obj.tags         = np.zeros(n, dtype=np.int32) if has_tags else None
    obj.cell         = np.eye(3, dtype=np.float32) * 5.0 if has_cell else None
    obj.fixed        = np.zeros(n, dtype=np.int32)
    obj.sid          = "test-sid-42"
    return obj


def _make_normalized_record(natoms: int = 5, rng_seed: int = 0) -> Dict[str, Any]:
    """Return a synthetic normalized record dict (output of BuildGraphs)."""
    rng = np.random.default_rng(rng_seed)
    return {
        "atomic_numbers":  np.array([6, 8, 1, 1, 14][:natoms], dtype=np.int32),
        "pos":             rng.random((natoms, 3)).astype(np.float32),
        "forces":          rng.random((natoms, 3)).astype(np.float32),
        "energy_per_atom": np.float32(-0.62),
        "natoms":          natoms,
        "tags":            np.zeros(natoms, dtype=np.int32),
        "sid":             f"sid-{rng_seed}",
        "edge_index":      np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_attr":       np.ones((2, 1), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class TestDomainRegistration:
    def test_imports(self):
        from redi.domains import atomistic as redi_atomistics
        assert redi_atomistics.NAME        == "atomistic"
        assert redi_atomistics.DOMAIN      == "atomistic"
        assert redi_atomistics.VERSION     == "0.4.0"
        assert "HydraGNN" in redi_atomistics.DESCRIPTION

    def test_get_full_pipeline_returns_list(self):
        from redi.domains import atomistic as redi_atomistics
        pipeline = redi_atomistics.get_full_pipeline()
        assert isinstance(pipeline, list)
        assert len(pipeline) == 9

    def test_get_full_pipeline_step_names(self):
        from redi.domains import atomistic as redi_atomistics
        pipeline = redi_atomistics.get_full_pipeline()
        names = [s.name for s in pipeline]
        assert names == [
            "load_ocp_lmdb",
            "load_atom_table",
            "load_atom_h5",
            "filter_records",
            "flag_force_outliers",
            "reference_energies",
            "build_graphs",
            "assemble_pyg_data",
            "save_output",
        ]

    def test_get_full_pipeline_oc20(self):
        from redi.domains import atomistic as redi_atomistics
        pipeline = redi_atomistics.get_full_pipeline(dataset="oc20")
        # First step should be LoadOCPLMDB with dataset=oc20
        step = pipeline[0]
        assert step.dataset == "oc20"

    def test_output_format_propagates(self):
        from redi.domains import atomistic as redi_atomistics
        pipeline = redi_atomistics.get_full_pipeline(output_format="pt")
        save_step = pipeline[-1]
        assert save_step.output_format == "pt"


# ---------------------------------------------------------------------------
# FilterRecords
# ---------------------------------------------------------------------------

class TestFilterRecords:
    def test_keeps_valid_records(self):
        from redi.domains.atomistic import FilterRecords

        records = [_make_raw_record(natoms=5) for _ in range(10)]
        ctx     = _make_ctx(metadata={"records": records})

        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_valid"]   == 10
        assert ctx.metadata["n_dropped"] == 0
        assert len(ctx.metadata["records"]) == 10
        assert not ctx._errors

    def test_drops_nan_energy(self):
        from redi.domains.atomistic import FilterRecords

        records = [_make_raw_record(natoms=5)] * 3 + [_make_raw_record(bad_energy=True)] * 2
        ctx     = _make_ctx(metadata={"records": records})

        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_valid"]   == 3
        assert ctx.metadata["n_dropped"] == 2

    def test_drops_nan_forces(self):
        from redi.domains.atomistic import FilterRecords

        records = [_make_raw_record(natoms=4)] * 5 + [_make_raw_record(bad_forces=True)] * 1
        ctx     = _make_ctx(metadata={"records": records})

        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_dropped"] == 1

    def test_drops_zero_atoms(self):
        from redi.domains.atomistic import FilterRecords

        records = [_make_raw_record()] * 2 + [_make_raw_record(zero_atoms=True)] * 3
        ctx     = _make_ctx(metadata={"records": records})

        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_valid"]   == 2
        assert ctx.metadata["n_dropped"] == 3

    def test_empty_input(self):
        from redi.domains.atomistic import FilterRecords

        ctx  = _make_ctx(metadata={"records": []})
        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_valid"]   == 0
        assert ctx.metadata["n_dropped"] == 0

    def test_dict_records_supported(self):
        """FilterRecords must also work with plain dict records."""
        from redi.domains.atomistic import FilterRecords

        rec = {
            "natoms": 3,
            "y":      -1.0,
            "force":  np.ones((3, 3), dtype=np.float32),
            "atomic_numbers": np.array([1, 1, 8], dtype=np.int32),
            "pos":    np.zeros((3, 3), dtype=np.float32),
            "tags":   np.zeros(3, dtype=np.int32),
            "sid":    "dict-sid",
        }
        ctx  = _make_ctx(metadata={"records": [rec]})
        step = FilterRecords()
        ctx  = step.execute(ctx)

        assert ctx.metadata["n_valid"] == 1


# ---------------------------------------------------------------------------
# BuildGraphs (LMDB path: centre positions, z-score forces, build graph)
# ---------------------------------------------------------------------------

class TestBuildGraphs:
    def test_basic_normalization(self):
        from redi.domains.atomistic import BuildGraphs

        records = [_make_raw_record(natoms=5) for _ in range(4)]
        ctx     = _make_ctx(metadata={"records": records, "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        assert "normalized" in ctx.metadata
        assert len(ctx.metadata["normalized"]) == 4
        assert not ctx._errors

    def test_positions_centered(self):
        from redi.domains.atomistic import BuildGraphs

        records = [_make_raw_record(natoms=6)]
        ctx     = _make_ctx(metadata={"records": records, "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        pos = ctx.metadata["normalized"][0]["pos"]
        # Mean of centered positions should be ~0
        assert np.allclose(pos.mean(axis=0), 0.0, atol=1e-5)

    def test_energy_per_atom(self):
        from redi.domains.atomistic import BuildGraphs

        natoms = 8
        energy = -4.0
        rec    = _make_raw_record(natoms=natoms, energy=energy)
        ctx    = _make_ctx(metadata={"records": [rec], "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        e_per = ctx.metadata["normalized"][0]["energy_per_atom"]
        assert float(e_per) == pytest.approx(energy / natoms)

    def test_forces_z_scored(self):
        from redi.domains.atomistic import BuildGraphs

        records = [_make_raw_record(natoms=5)]
        ctx     = _make_ctx(metadata={"records": records, "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        forces = ctx.metadata["normalized"][0]["forces"]
        # After z-scoring, abs mean should be small
        assert np.abs(forces.mean()) < 0.1

    def test_oc22_fixed_field_extracted(self):
        from redi.domains.atomistic import BuildGraphs

        rec = _make_raw_record(natoms=5)
        rec.fixed = np.array([0, 1, 0, 0, 1], dtype=np.int32)
        ctx = _make_ctx(metadata={"records": [rec], "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        assert "fixed" in ctx.metadata["normalized"][0]

    def test_oc20_no_fixed_field(self):
        from redi.domains.atomistic import BuildGraphs

        rec = _make_raw_record(natoms=5)
        rec.fixed = None            # the raw record carries no fixed-atom mask
        ctx = _make_ctx(metadata={"records": [rec], "dataset": "oc20"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        assert "fixed" not in ctx.metadata["normalized"][0]

    def test_output_keys(self):
        from redi.domains.atomistic import BuildGraphs

        records = [_make_raw_record(natoms=4)]
        ctx     = _make_ctx(metadata={"records": records, "dataset": "oc22"})

        step = BuildGraphs()
        ctx  = step.execute(ctx)

        keys = set(ctx.metadata["normalized"][0].keys())
        expected = {"atomic_numbers", "pos", "forces", "energy_per_atom",
                    "natoms", "tags", "sid", "edge_index", "edge_attr"}
        assert expected.issubset(keys)


# ---------------------------------------------------------------------------
# SaveOutput — NPZ
# ---------------------------------------------------------------------------

class TestSaveOutputNPZ:
    def _make_records(self, n: int = 6) -> List[dict]:
        return [_make_normalized_record(natoms=4, rng_seed=i) for i in range(n)]

    def test_npz_written_all_split(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        records  = self._make_records(6)
        ctx      = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      None,        # trigger 80/10/10
            },
        )

        step = SaveOutput(output_format="npz")
        ctx  = step.execute(ctx)

        assert not ctx._errors
        # At least train.npz must exist for 6 records
        assert (tmp_path / "train.npz").exists()
        # index.json always written
        assert (tmp_path / "index.json").exists()

    def test_npz_arrays_correct(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(4)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "train",
            },
        )

        step = SaveOutput(output_format="npz")
        ctx  = step.execute(ctx)

        assert not ctx._errors
        data = np.load(tmp_path / "train.npz")
        assert "atomic_numbers" in data
        assert "positions"      in data
        assert "forces"         in data
        assert "energy_per_atom" in data
        assert "natoms"         in data
        assert "split_index"    in data
        assert "tags"           in data
        # 4 structures × 4 atoms = 16 atoms total
        assert data["atomic_numbers"].shape[0] == 16
        assert data["natoms"].shape[0]          == 4

    def test_npz_split_index_correct(self, tmp_path):
        """split_index[i] should equal sum(natoms[:i])."""
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(3)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "val",
            },
        )

        step = SaveOutput(output_format="npz")
        ctx  = step.execute(ctx)

        data   = np.load(tmp_path / "val.npz")
        natoms = data["natoms"]
        sidx   = data["split_index"]
        expected = np.cumsum([0] + list(natoms[:-1]), dtype=np.int32)
        np.testing.assert_array_equal(sidx, expected)

    def test_index_json_content(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(5)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "test",
            },
        )

        step = SaveOutput(output_format="npz")
        ctx  = step.execute(ctx)

        with open(tmp_path / "index.json") as f:
            index = json.load(f)

        assert index["dataset"]       == "oc22"
        assert index["n_structures"]  == 5
        assert "splits"               in index
        assert "features"             in index
        assert "positions"            in index["features"]

    def test_json_format_sets_hydragnn_flag(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(3)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "train",
            },
        )

        step = SaveOutput(output_format="json")
        ctx  = step.execute(ctx)

        with open(tmp_path / "index.json") as f:
            index = json.load(f)

        assert index.get("hydragnn_native") is True
        assert index["output_format"] == "json"

    def test_80_10_10_split(self, tmp_path):
        """With split=None and 100 records, should produce ~80/10/10."""
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(100)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      None,
            },
        )

        step = SaveOutput(output_format="npz")
        ctx  = step.execute(ctx)

        with open(tmp_path / "index.json") as f:
            index = json.load(f)

        splits = index["splits"]
        total  = sum(splits.values())
        assert total == 100
        assert splits.get("train", 0) == 80
        assert splits.get("val",   0) == 10
        assert splits.get("test",  0) == 10


# ---------------------------------------------------------------------------
# SaveOutput — PT (skipped if torch unavailable)
# ---------------------------------------------------------------------------

_requires_pyg = pytest.mark.skipif(
    not (__import__("importlib").util.find_spec("torch")
         and __import__("importlib").util.find_spec("torch_geometric")),
    reason="torch / torch_geometric not installed",
)


class TestSaveOutputPT:
    def _assemble(self, ctx):
        from redi.domains.atomistic import AssemblePyGData
        ctx = AssemblePyGData().execute(ctx)
        assert not ctx._errors
        return ctx

    def _make_records(self, n: int = 4) -> List[dict]:
        return [_make_normalized_record(natoms=3, rng_seed=i) for i in range(n)]

    @_requires_pyg
    def test_pt_files_written(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(4)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "train",
            },
        )

        ctx  = self._assemble(ctx)
        step = SaveOutput(output_format="pt")
        ctx  = step.execute(ctx)

        assert not ctx._errors
        pt_files = list((tmp_path / "train").glob("*.pt"))
        assert len(pt_files) == 4
        assert (tmp_path / "index.json").exists()

    @_requires_pyg
    def test_pt_data_object_fields(self, tmp_path):
        import torch
        from redi.domains.atomistic import SaveOutput

        records = self._make_records(2)
        ctx     = _make_ctx(
            output_dir=str(tmp_path),
            metadata={
                "normalized": records,
                "dataset":    "oc22",
                "split":      "val",
            },
        )

        ctx  = self._assemble(ctx)
        step = SaveOutput(output_format="pt")
        ctx  = step.execute(ctx)

        pt_file = sorted((tmp_path / "val").glob("*.pt"))[0]
        data    = torch.load(str(pt_file), weights_only=False)

        assert hasattr(data, "x")
        assert hasattr(data, "pos")
        assert hasattr(data, "y")
        assert hasattr(data, "force")
        assert data.x.shape == (3, 2)          # node features: [Z/100, tag]
        assert data.edge_index.shape[0] == 2

    def test_pt_without_pyg_data_gives_clear_error(self, tmp_path):
        from redi.domains.atomistic import SaveOutput

        ctx = _make_ctx(
            output_dir=str(tmp_path),
            metadata={"normalized": self._make_records(2), "dataset": "oc22", "split": "train"},
        )
        ctx = SaveOutput(output_format="pt").execute(ctx)

        assert any("AssemblePyGData" in e for e in ctx._errors)
        assert not (tmp_path / "index.json").exists()
