"""
Unit tests for per-atom table input (LoadAtomTable and the steps it enables).

All tests use small synthetic Parquet/CSV tables written to tmp_path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

L = 10.0   # cubic cell edge (Angstrom)


def _make_ctx(input_path, output_dir="/tmp/redi_atom_table_test", config=None):
    ctx            = SimpleNamespace()
    ctx.input_path = str(input_path)
    ctx.output_dir = str(output_dir)
    ctx.config     = config or {}
    ctx.metadata   = {}
    ctx._errors    = []
    ctx.add_error  = ctx._errors.append
    return ctx


def _make_table(groups=("mol-boot", "mol-0000", "mol-0001", "mol-0002"),
                frames_per_group=3, natoms=6, seed=0):
    """Per-atom rows: Be2F4 frames in a cubic cell, positions partly outside the cell."""
    rng  = np.random.default_rng(seed)
    rows = []
    fid  = 0
    for g in groups:
        for step in range(frames_per_group):
            energy = -100.0 + rng.normal(scale=0.1)
            for a in range(natoms):
                x, y, z = rng.uniform(-0.5 * L, 1.5 * L, size=3)
                fx, fy, fz = rng.normal(size=3)
                rows.append(dict(
                    source_file="data-0000.h5", frame_id=fid, atom_idx=a,
                    frame_label=f"{g}-{step:010d}", species="Be" if a % 3 == 0 else "F",
                    x=x, y=y, z=z, fx=fx, fy=fy, fz=fz, energy=energy,
                    **{f"cell_{i}{j}": (L if i == j else 0.0) for i in range(3) for j in range(3)},
                ))
            fid += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # unsorted rows


def _write(df, tmp_path, name="frames.parquet"):
    path = tmp_path / name
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        pytest.importorskip("pyarrow")
        df.to_parquet(path)
    return path


def _load(tmp_path, df=None, config=None, name="frames.parquet"):
    from redi.domains.atomistic import LoadAtomTable
    path = _write(_make_table() if df is None else df, tmp_path, name)
    ctx  = _make_ctx(path, tmp_path / "out", config)
    return LoadAtomTable().execute(ctx)


# ---------------------------------------------------------------------------
# Registration / dispatch
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_domain_constants(self):
        from redi.domains import atomistic
        assert atomistic.DOMAIN == "atomistic"

    def test_cli_accepts_atomistic_and_hydragnn_without_plugin(self):
        from redi import cli
        parser = cli.create_parser()
        for dom in ("atomistic", "hydragnn"):
            args = parser.parse_args(["run", "x.parquet", "--domain", dom])
            assert args.domain == dom

    def test_loader_dispatch_by_input_type(self, tmp_path):
        from redi.domains.atomistic import LoadAtomTable, LoadOCPLMDB
        table = _make_ctx(tmp_path / "frames.parquet")
        lmdb  = _make_ctx(tmp_path / "shard.lmdb")
        assert LoadAtomTable().validate(table) and not LoadOCPLMDB().validate(table)
        assert LoadOCPLMDB().validate(lmdb) and not LoadAtomTable().validate(lmdb)

    def test_table_steps_skip_for_lmdb_records(self):
        from redi.domains.atomistic import FlagForceOutliers, ReferenceEnergies
        ctx = _make_ctx("/tmp/shard.lmdb")
        ctx.metadata["records"] = []
        assert not FlagForceOutliers().validate(ctx)
        assert not ReferenceEnergies().validate(ctx)

    def test_element_table(self):
        from redi.domains.atomistic import ATOMIC_NUMBERS
        assert len(ATOMIC_NUMBERS) == 118
        assert ATOMIC_NUMBERS["Be"] == 4 and ATOMIC_NUMBERS["F"] == 9 and ATOMIC_NUMBERS["Og"] == 118


# ---------------------------------------------------------------------------
# LoadAtomTable
# ---------------------------------------------------------------------------

class TestLoadAtomTable:
    def test_groups_rows_into_structures(self, tmp_path):
        ctx = _load(tmp_path)
        assert not ctx._errors
        records = ctx.metadata["records"]
        assert len(records) == 12
        assert all(r.natoms == 6 for r in records)
        assert [r.frame_id for r in records] == list(range(12))

    def test_atom_order_and_species(self, tmp_path):
        df  = _make_table()
        ctx = _load(tmp_path, df)
        r0  = ctx.metadata["records"][0]
        expected = df[df.frame_id == 0].sort_values("atom_idx")
        assert np.allclose(r0.pos, expected[["x", "y", "z"]].to_numpy())
        assert r0.atomic_numbers.tolist() == [4, 9, 9, 4, 9, 9]

    def test_energy_cell_and_group(self, tmp_path):
        ctx = _load(tmp_path)
        r   = ctx.metadata["records"][4]
        assert r.y == r.energy
        assert np.allclose(r.cell, np.eye(3) * L)
        assert r.group == "mol-0000"
        assert r.sid == "mol-0000-0000000001"

    def test_metadata_and_summary(self, tmp_path):
        ctx = _load(tmp_path)
        md  = ctx.metadata
        assert md["loader"] == "atom_table"
        assert md["split_by"] == "group"
        assert md["structure_options"]["pbc"] is True
        assert md["structure_options"]["normalize_forces"] is False
        s = md["atomistic_summary"]
        assert s["n_structures"] == 12 and s["n_atoms"] == 72
        assert s["species_counts"] == {"Be": 24, "F": 48}
        assert s["n_groups"] == 4
        assert s["n_atoms_outside_cell"] > 0
        assert s["units"]["energy"] == "eV"

    def test_drops_inconsistent_energy(self, tmp_path):
        df = _make_table()
        df.loc[(df.frame_id == 2) & (df.atom_idx == 3), "energy"] += 1.0
        ctx = _load(tmp_path, df)
        assert len(ctx.metadata["records"]) == 11
        assert 2 not in [r.frame_id for r in ctx.metadata["records"]]
        assert ctx.metadata["atomistic_summary"]["n_inconsistent_dropped"] == 1

    def test_csv_and_numeric_species(self, tmp_path):
        df = _make_table()
        df["species"] = df["species"].map({"Be": 4, "F": 9})
        ctx = _load(tmp_path, df, name="frames.csv")
        assert not ctx._errors
        assert ctx.metadata["records"][0].atomic_numbers.tolist() == [4, 9, 9, 4, 9, 9]

    def test_no_cell_disables_pbc(self, tmp_path):
        df  = _make_table().drop(columns=[f"cell_{i}{j}" for i in range(3) for j in range(3)])
        ctx = _load(tmp_path, df)
        assert ctx.metadata["structure_options"]["pbc"] is False
        assert ctx.metadata["records"][0].cell is None

    def test_column_mapping_and_group_column(self, tmp_path):
        df  = _make_table().rename(columns={"frame_id": "structure", "energy": "E"})
        ctx = _load(tmp_path, df, config={
            "columns": {"frame": "structure", "energy": "E"}, "group_column": "source_file",
        })
        assert not ctx._errors
        assert {r.group for r in ctx.metadata["records"]} == {"data-0000.h5"}

    def test_missing_columns_error(self, tmp_path):
        ctx = _load(tmp_path, _make_table().drop(columns=["fx"]))
        assert any("missing columns" in e for e in ctx._errors)

    def test_unknown_element_error(self, tmp_path):
        df = _make_table()
        df.loc[0, "species"] = "Xx"
        ctx = _load(tmp_path, df)
        assert any("unknown element" in e for e in ctx._errors)


# ---------------------------------------------------------------------------
# FlagForceOutliers / ReferenceEnergies
# ---------------------------------------------------------------------------

class TestFlagForceOutliers:
    def test_reports_without_dropping(self, tmp_path):
        from redi.domains.atomistic import FlagForceOutliers
        ctx = FlagForceOutliers().execute(_load(tmp_path))
        assert len(ctx.metadata["records"]) == 12
        out = ctx.metadata["atomistic_summary"]["force_outliers"]
        assert out["n_dropped"] == 0
        assert len(out["top"]) == 10
        assert all(hasattr(r, "max_force") for r in ctx.metadata["records"])

    def test_drops_above_threshold(self, tmp_path):
        from redi.domains.atomistic import FlagForceOutliers
        df = _make_table()
        df.loc[(df.frame_id == 5) & (df.atom_idx == 0), "fx"] = 50.0
        ctx = FlagForceOutliers().execute(_load(tmp_path, df, config={"max_force": 20.0}))
        assert 5 not in [r.frame_id for r in ctx.metadata["records"]]
        assert ctx.metadata["atomistic_summary"]["force_outliers"]["n_dropped"] == 1


class TestReferenceEnergies:
    def _records(self, compositions, mu, offset=0.0):
        recs = []
        for i, counts in enumerate(compositions):
            z = np.concatenate([np.full(n, el) for el, n in counts.items()])
            e = sum(n * mu[el] for el, n in counts.items()) + offset
            recs.append(SimpleNamespace(atomic_numbers=z, y=e, natoms=len(z), sid=str(i)))
        return recs

    def test_recovers_references_when_identifiable(self):
        from redi.domains.atomistic import ReferenceEnergies
        mu   = {4: -380.0, 9: -650.0}
        recs = self._records([{4: 2, 9: 4}, {4: 3, 9: 4}, {4: 1, 9: 5}], mu)
        ctx  = _make_ctx("x.parquet")
        ctx.metadata.update(records=recs, loader="atom_table")
        ctx  = ReferenceEnergies().execute(ctx)
        ref  = ctx.metadata["atomistic_summary"]["energy_reference"]
        assert ref["identifiable"] is True
        assert ref["per_element"]["Be"] == pytest.approx(-380.0)
        assert ref["per_element"]["F"] == pytest.approx(-650.0)
        assert all(abs(r.energy_referenced) < 1e-6 for r in recs)

    def test_single_composition_is_constant_shift(self):
        from redi.domains.atomistic import ReferenceEnergies
        recs = self._records([{4: 2, 9: 4}] * 3, {4: -380.0, 9: -650.0})
        for r, d in zip(recs, (-0.1, 0.0, 0.1)):
            r.y += d
        ctx = _make_ctx("x.parquet")
        ctx.metadata.update(records=recs, loader="atom_table")
        ctx = ReferenceEnergies().execute(ctx)
        assert ctx.metadata["atomistic_summary"]["energy_reference"]["identifiable"] is False
        assert [r.energy_referenced for r in recs] == pytest.approx([-0.1, 0.0, 0.1])

    def test_disabled_by_config(self):
        from redi.domains.atomistic import ReferenceEnergies
        ctx = _make_ctx("x.parquet", config={"energy_reference": "none"})
        ctx.metadata.update(records=[], loader="atom_table")
        assert not ReferenceEnergies().validate(ctx)


# ---------------------------------------------------------------------------
# Periodic graph construction
# ---------------------------------------------------------------------------

class TestPeriodicGraph:
    def test_wrap_positions(self):
        from redi.domains.atomistic import _wrap_positions
        pos  = np.array([[-1.0, 11.0, 25.0], [3.0, 4.0, 5.0]])
        wrap = _wrap_positions(pos, np.eye(3) * L)
        assert np.allclose(wrap, [[9.0, 1.0, 5.0], [3.0, 4.0, 5.0]])

    def test_wrap_triclinic(self):
        from redi.domains.atomistic import _wrap_positions
        cell = np.array([[10.0, 0, 0], [5.0, 8.0, 0], [0, 0, 9.0]])
        pos  = np.array([[0.2, 0.3, 0.4]]) @ cell + cell[0] - 2 * cell[1]
        assert np.allclose(_wrap_positions(pos, cell), np.array([[0.2, 0.3, 0.4]]) @ cell)

    def test_edge_across_boundary(self):
        from redi.domains.atomistic import _radius_graph_pbc
        pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        ei, ea, shifts = _radius_graph_pbc(pos, np.eye(3) * L, r=2.0, max_neighbours=100)
        assert ei.T.tolist() == [[0, 1], [1, 0]]
        assert np.allclose(ea[:, 0], 1.0)
        # edge vector = pos[dst] + shift @ cell - pos[src]
        vec = pos[ei[1]] + shifts @ (np.eye(3) * L) - pos[ei[0]]
        assert np.allclose(np.linalg.norm(vec, axis=1), ea[:, 0])

    def test_no_self_edges_and_symmetric(self):
        from redi.domains.atomistic import _radius_graph_pbc
        pos = np.random.default_rng(1).uniform(0, L, size=(20, 3))
        ei, ea, shifts = _radius_graph_pbc(pos, np.eye(3) * L, r=4.0, max_neighbours=100)
        assert not np.any((ei[0] == ei[1]) & (np.abs(shifts).sum(axis=1) == 0))
        fwd = {(s, d, tuple(sh)) for s, d, sh in zip(ei[0], ei[1], shifts.astype(int))}
        assert all((d, s, tuple(-np.array(sh))) in fwd for s, d, sh in fwd)

    def test_matches_brute_force_minimum_image(self):
        from redi.domains.atomistic import _radius_graph_pbc
        pos = np.random.default_rng(2).uniform(0, L, size=(15, 3))
        ei, _, _ = _radius_graph_pbc(pos, np.eye(3) * L, r=4.5, max_neighbours=100)
        d = pos[None] - pos[:, None]
        d -= L * np.round(d / L)
        dist = np.linalg.norm(d, axis=-1)
        expected = {(i, j) for i in range(15) for j in range(15) if i != j and dist[i, j] <= 4.5}
        assert set(map(tuple, ei.T.tolist())) == expected

    def test_process_structure_periodic_keeps_raw_forces(self, tmp_path):
        from redi.domains.atomistic import _process_one_structure
        ctx = _load(tmp_path)
        rec = ctx.metadata["records"][0]
        out = _process_one_structure(rec, ctx.metadata["structure_options"])
        assert np.all((out["pos"] >= 0) & (out["pos"] <= L))
        assert np.allclose(out["forces"], rec.force, atol=1e-6)
        assert "edge_shifts" in out
        assert out["energy"] == rec.energy and out["group"] == rec.group

    def test_default_options_keep_oc22_behaviour(self, tmp_path):
        from redi.domains.atomistic import _process_one_structure
        rec = _load(tmp_path).metadata["records"][0]
        out = _process_one_structure(rec)
        assert np.allclose(out["pos"].mean(axis=0), 0.0, atol=1e-5)
        assert out["forces"].std() == pytest.approx(1.0, abs=1e-3)
        assert "edge_shifts" not in out and "group" not in out


# ---------------------------------------------------------------------------
# Splits and end-to-end output
# ---------------------------------------------------------------------------

class TestGroupSplits:
    def test_groups_never_straddle_splits(self):
        from redi.domains.atomistic import SaveOutput
        groups = [f"g{i // 5}" for i in range(100)]   # 20 groups of 5
        splits = SaveOutput()._make_group_splits(groups)
        seen = {}
        for name, idxs in splits.items():
            for i in idxs:
                assert seen.setdefault(groups[i], name) == name
        assert sorted(i for idxs in splits.values() for i in idxs) == list(range(100))
        assert all(splits[name] for name in ("train", "val", "test"))

    def test_val_test_not_empty_with_few_groups(self):
        from redi.domains.atomistic import SaveOutput
        groups = ["big"] * 90 + ["a"] * 5 + ["b"] * 5
        splits = SaveOutput()._make_group_splits(groups)
        assert all(splits[name] for name in ("train", "val", "test"))


class TestEndToEnd:
    def _run(self, tmp_path, output_format="npz", config=None):
        from redi.domains import atomistic as redi_atomistics
        path = _write(_make_table(), tmp_path)
        ctx  = _make_ctx(path, tmp_path / "out", config)
        for step in redi_atomistics.get_full_pipeline(output_format=output_format):
            if step.validate(ctx):
                ctx = step.execute(ctx)
        return ctx

    def test_npz_output(self, tmp_path):
        ctx = self._run(tmp_path)
        errors = [e for e in ctx._errors if "torch" not in e]
        assert not errors
        out   = tmp_path / "out"
        index = json.loads((out / "index.json").read_text())
        assert index["split_method"] == "group"
        assert sum(index["splits"].values()) == 12
        groups = [g for gs in index["split_groups"].values() for g in gs]
        assert len(groups) == len(set(groups)) == 4
        assert index["atomistic"]["energy_reference"]["identifiable"] is False
        assert "force_outliers" in index["atomistic"]
        assert not {"records", "normalized", "pyg_data"} & set(ctx.metadata)
        assert ctx.metadata["output_records"] == {
            f"{split}.npz": n for split, n in index["splits"].items() if n
        }

        train = np.load(out / "train.npz")
        for key in ("cell", "energy", "energy_referenced", "group", "frame_id", "max_force"):
            assert key in train.files
        n = len(train["natoms"])
        assert train["cell"].shape == (n, 3, 3)
        assert train["positions"].shape == (train["natoms"].sum(), 3)
        assert np.all((train["positions"] >= 0) & (train["positions"] <= L))

    def test_parquet_matches_npz(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
        self._run(tmp_path / "a")
        ctx = self._run(tmp_path / "b", output_format="parquet")
        assert not [e for e in ctx._errors if "torch" not in e]
        out   = tmp_path / "b" / "out"
        index = json.loads((out / "index.json").read_text())
        assert index["output_format"] == "parquet"
        assert ctx.metadata["output_records"] == {
            f"{split}.parquet": n for split, n in index["splits"].items() if n
        }
        assert not list(out.glob("*.npz"))

        npz = np.load(tmp_path / "a" / "out" / "train.npz")
        tbl = pq.read_table(out / "train.parquet")
        assert tbl.num_rows == len(npz["natoms"])
        np.testing.assert_array_equal(tbl["natoms"].to_numpy(), npz["natoms"])
        rows = tbl.to_pylist()
        np.testing.assert_allclose(np.concatenate([r["positions"] for r in rows]), npz["positions"])
        np.testing.assert_allclose(np.concatenate([r["forces"] for r in rows]), npz["forces"])
        np.testing.assert_array_equal(np.concatenate([r["atomic_numbers"] for r in rows]),
                                      npz["atomic_numbers"])
        np.testing.assert_allclose(np.array([r["cell"] for r in rows]), npz["cell"])
        assert [r["group"] for r in rows] == npz["group"].tolist()

    def test_unknown_format_errors(self, tmp_path):
        ctx = self._run(tmp_path, output_format="zarr")
        assert any("not supported" in e for e in ctx._errors)
        assert not (tmp_path / "out" / "index.json").exists()

    def test_pt_output_has_pbc_fields(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch
        ctx = self._run(tmp_path, output_format="pt")
        assert not ctx._errors
        files = sorted((tmp_path / "out").rglob("*.pt"))
        assert len(files) == 12
        data = torch.load(files[0], weights_only=False)
        assert data.cell.shape == (3, 3)
        assert data.edge_shifts.shape == (data.edge_index.shape[1], 3)
        assert hasattr(data, "energy_referenced")


# ---------------------------------------------------------------------------
# Step-level change reports (feed REDI's metadata.json pipeline_steps)
# ---------------------------------------------------------------------------

class TestStepChanges:
    def _run(self, tmp_path, df=None, config=None):
        from redi.domains import atomistic as redi_atomistics
        path = _write(_make_table() if df is None else df, tmp_path)
        ctx  = _make_ctx(path, tmp_path / "out", config)
        for step in redi_atomistics.get_full_pipeline():
            if step.validate(ctx):
                ctx = step.execute(ctx)
        return ctx.metadata["step_changes"]

    def test_every_executed_step_reports(self, tmp_path):
        changes = self._run(tmp_path)
        assert "load_ocp_lmdb" not in changes          # skipped: input is a table
        for name in ("load_atom_table", "filter_records", "flag_force_outliers",
                     "reference_energies", "build_graphs", "save_output"):
            assert name in changes, name

    def test_load_report(self, tmp_path):
        c = self._run(tmp_path)["load_atom_table"]
        assert c["rows_read"] == 72 and c["structures_built"] == 12
        assert c["structures_dropped_inconsistent"] == 0
        assert c["atoms_outside_cell"] > 0
        assert c["species_to_atomic_number"] == {"Be": 4, "F": 9}

    def test_drops_are_reported_through_the_chain(self, tmp_path):
        df = _make_table()
        df.loc[(df.frame_id == 2) & (df.atom_idx == 3), "energy"] += 1.0       # inconsistent
        df.loc[(df.frame_id == 5) & (df.atom_idx == 0), "fx"] = 50.0            # force outlier
        changes = self._run(tmp_path, df, config={"max_force": 20.0})
        assert changes["load_atom_table"]["structures_dropped_inconsistent"] == 1
        assert changes["load_atom_table"]["structures_built"] == 11
        assert changes["flag_force_outliers"]["records_in"] == 11
        assert changes["flag_force_outliers"]["dropped"] == 1
        assert changes["build_graphs"]["records_in"] == 10
        assert changes["save_output"]["structures"] == 10

    def test_graph_report_describes_transforms(self, tmp_path):
        c = self._run(tmp_path)["build_graphs"]
        assert c["periodic"] is True
        assert c["positions"] == "wrapped into the cell"
        assert c["atoms_wrapped"] > 0 and c["edges"] > 0
        assert c["forces"] == "unchanged"
        assert c["records_in"] == c["records_out"] == 12 and c["dropped"] == 0

    def test_graph_report_without_cell(self, tmp_path):
        df = _make_table().drop(columns=[f"cell_{i}{j}" for i in range(3) for j in range(3)])
        c = self._run(tmp_path, df)["build_graphs"]
        assert c["periodic"] is False
        assert c["positions"] == "centred on the structure mean"
        assert c["atoms_wrapped"] == 0

    def test_save_report(self, tmp_path):
        c = self._run(tmp_path)["save_output"]
        assert c["split_method"] == "by group"
        assert sum(c["splits"].values()) == c["structures"] == 12
