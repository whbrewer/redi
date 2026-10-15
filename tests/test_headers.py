"""Tests for `redi inspect --headers-only` (redi.inspect.headers)."""

import numpy as np
import pytest

from redi.inspect.headers import collect_headers, expand_inputs

h5py = pytest.importorskip("h5py")


def _write_h5(path, n):
    with h5py.File(path, "w") as f:
        f.attrs["source"] = "unit-test"
        g = f.create_group("grp")
        g.create_dataset("energy", data=np.zeros(n))
        g.create_dataset("forces", data=np.zeros((n, 4, 3), dtype="float32"))


def test_hdf5_directory_groups_by_schema(tmp_path):
    _write_h5(tmp_path / "a.h5", 5)
    _write_h5(tmp_path / "b.h5", 7)
    res = collect_headers(str(tmp_path))
    assert res["n_files"] == 2 and res["n_errors"] == 0
    assert len(res["schemas"]) == 1
    ds = res["schemas"][0]["datasets"]["grp/forces"]
    assert ds["trailing_shape"] == [4, 3]
    assert (ds["leading_dim_min"], ds["leading_dim_max"], ds["leading_dim_total"]) == (5, 7, 12)
    assert res["files"][0]["attrs"]["source"] == "unit-test"


def test_glob_and_mixed_formats(tmp_path):
    np.save(tmp_path / "x.npy", np.zeros((3, 2)))
    np.savez(tmp_path / "y.npz", a=np.zeros(4), b=np.zeros((4, 2)))
    (tmp_path / "z.csv").write_text("c1,c2,c3\n1,2,3\n")
    (tmp_path / "ignored.txt").write_text("nope")
    res = collect_headers(str(tmp_path / "*"))
    assert res["n_files"] == 3
    by_name = {f["path"].rsplit("/", 1)[-1]: f for f in res["files"]}
    assert by_name["x.npy"]["datasets"]["array"]["shape"] == [3, 2]
    assert set(by_name["y.npz"]["datasets"]) == {"a", "b"}
    assert list(by_name["z.csv"]["datasets"]) == ["c1", "c2", "c3"]


def test_corrupt_file_reported_not_fatal(tmp_path):
    _write_h5(tmp_path / "good.h5", 2)
    (tmp_path / "bad.h5").write_bytes(b"not hdf5")
    res = collect_headers(str(tmp_path))
    assert res["n_files"] == 2 and res["n_errors"] == 1


def test_no_array_data_read(tmp_path, monkeypatch):
    _write_h5(tmp_path / "a.h5", 5)

    def boom(*a, **k):
        raise AssertionError("dataset data was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", boom)
    assert collect_headers(str(tmp_path / "a.h5"))["n_errors"] == 0


def test_expand_inputs_empty(tmp_path):
    assert expand_inputs(str(tmp_path / "nothing*.h5")) == []
