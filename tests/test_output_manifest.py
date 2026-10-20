"""
Tests for the metadata.json output manifest (redi.output_manifest).
"""

import enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class TestJsonSafe:
    def test_plain_values_pass_through(self):
        from redi.output_manifest import json_safe
        value = {"a": 1, "b": [1, 2.5, "x", None, True], "c": {"d": "e"}}
        assert json_safe(value) == value

    def test_numpy_path_enum(self):
        from redi.output_manifest import json_safe

        class Color(enum.Enum):
            RED = "red"

        assert json_safe(np.float32(1.5)) == 1.5
        assert json_safe(Path("/a/b")) == "/a/b"
        assert json_safe(Color.RED) == "red"
        assert json_safe(np.arange(3)) == [0, 1, 2]

    def test_large_array_summarized(self):
        from redi.output_manifest import json_safe
        out = json_safe(np.zeros((2000, 3), dtype=np.float32))
        assert out == {"__summary__": "ndarray", "shape": [2000, 3], "dtype": "float32"}

    def test_long_object_list_summarized(self):
        from redi.output_manifest import json_safe
        out = json_safe([SimpleNamespace(x=i) for i in range(100)])
        assert out["__summary__"] == "list"
        assert out["length"] == 100
        assert out["item_types"] == ["SimpleNamespace"]

    def test_short_object_list_kept_per_item(self):
        from redi.output_manifest import json_safe
        out = json_safe([SimpleNamespace(x=1)])
        assert out == ["namespace(x=1)"]

    def test_list_of_dicts_not_capped(self):
        from redi.output_manifest import json_safe
        rows = [{"step": i} for i in range(500)]
        assert json_safe(rows) == rows

    def test_long_primitive_list_summarized(self):
        from redi.output_manifest import json_safe, MAX_PRIMITIVE_LIST
        assert json_safe(list(range(MAX_PRIMITIVE_LIST + 1))) == {
            "__summary__": "list", "length": MAX_PRIMITIVE_LIST + 1,
        }


class TestSafeMetadata:
    def test_omits_oversized_values(self):
        from redi.output_manifest import safe_metadata
        md = {"small": "ok", "big": [{"blob": "x" * 2000} for _ in range(1000)]}
        out = safe_metadata(md)
        assert out["small"] == "ok"
        assert out["big"]["__summary__"] == "omitted"
        assert out["big"]["length"] == 1000

    def test_keeps_top_level_type_filter(self):
        from redi.output_manifest import safe_metadata
        out = safe_metadata({"arr": np.zeros(3), "obj": object(), "n": 3})
        assert out == {"n": 3}

    def test_records_from_plugin_are_summarized(self):
        from redi.output_manifest import safe_metadata
        records = [SimpleNamespace(pos=np.zeros((201, 3))) for _ in range(1998)]
        out = safe_metadata({"records": records})
        assert out["records"]["__summary__"] == "list"
        assert len(json.dumps(out)) < 200


class TestDescribeOutputs:
    def test_npz_arrays_and_records(self, tmp_path):
        from redi.output_manifest import describe_outputs
        np.savez_compressed(tmp_path / "train.npz", X=np.zeros((100, 4), np.float32), y=np.zeros(100))
        out = describe_outputs(tmp_path)
        (entry,) = out["files"]
        assert entry["path"] == "train.npz" and entry["format"] == "npz"
        assert entry["n_records"] == 100
        assert entry["arrays"] == {
            "X": {"shape": [100, 4], "dtype": "float32"},
            "y": {"shape": [100], "dtype": "float64"},
        }
        assert entry["sha256"] == hashlib.sha256((tmp_path / "train.npz").read_bytes()).hexdigest()
        assert out["n_files"] == 1 and out["total_size_bytes"] == entry["size_bytes"]

    def test_npz_mixed_leading_dims_needs_record_count(self, tmp_path):
        from redi.output_manifest import describe_outputs
        np.savez(tmp_path / "train.npz", positions=np.zeros((60, 3)), natoms=np.full(10, 6))
        (entry,) = describe_outputs(tmp_path)["files"]
        assert "n_records" not in entry
        (entry,) = describe_outputs(tmp_path, record_counts={"train.npz": 10})["files"]
        assert entry["n_records"] == 10

    def test_npy_csv_and_text(self, tmp_path):
        from redi.output_manifest import describe_outputs
        np.save(tmp_path / "X.npy", np.zeros((7, 2), np.int32))
        (tmp_path / "data.csv").write_text("a,b\n1,2\n3,4\n5,6")
        (tmp_path / "README.md").write_text("hi")
        files = {e["path"]: e for e in describe_outputs(tmp_path)["files"]}
        assert files["X.npy"]["arrays"]["X"] == {"shape": [7, 2], "dtype": "int32"}
        assert files["X.npy"]["n_records"] == 7
        assert files["data.csv"]["n_records"] == 3
        assert files["data.csv"]["columns"] == ["a", "b"]
        assert files["README.md"]["format"] == "markdown"
        assert "n_records" not in files["README.md"]

    def test_parquet(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        from redi.output_manifest import describe_outputs
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_parquet(tmp_path / "t.parquet")
        (entry,) = describe_outputs(tmp_path)["files"]
        assert entry["n_records"] == 3
        assert list(entry["columns"]) == ["a", "b"]
        assert entry["columns"]["a"] == "int64"

    def test_hdf5(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from redi.output_manifest import describe_outputs
        with h5py.File(tmp_path / "d.h5", "w") as f:
            f["grp/X"] = np.zeros((5, 3))
            f["y"] = np.zeros(5)
        (entry,) = describe_outputs(tmp_path)["files"]
        assert entry["arrays"]["grp/X"] == {"shape": [5, 3], "dtype": "float64"}
        assert entry["n_records"] == 5

    def test_many_files_collapse_into_group(self, tmp_path):
        from redi.output_manifest import describe_outputs, MAX_FILES_PER_GROUP
        split = tmp_path / "train"
        split.mkdir()
        n = MAX_FILES_PER_GROUP + 10
        for i in range(n):
            (split / f"graphdata_{i:05d}.pt").write_bytes(b"x" * 10)
        out = describe_outputs(tmp_path, record_counts={"train": n})
        (entry,) = out["files"]
        assert entry["path"] == "train/*.pt"
        assert entry["n_files"] == n and entry["n_records"] == n
        assert entry["size_bytes"] == 10 * n
        assert entry["first"] == "graphdata_00000.pt"
        assert out["n_files"] == n

    def test_zarr_store_is_one_entry(self, tmp_path):
        from redi.output_manifest import describe_outputs
        arr = tmp_path / "data.zarr" / "tas"
        arr.mkdir(parents=True)
        (arr / ".zarray").write_text(json.dumps({"shape": [10, 4], "dtype": "<f4"}))
        (arr / "0.0").write_bytes(b"chunk")
        (entry,) = describe_outputs(tmp_path)["files"]
        assert entry["path"] == "data.zarr/" and entry["format"] == "zarr"
        assert entry["n_files"] == 2
        assert entry["arrays"] == {"tas": {"shape": [10, 4], "dtype": "<f4"}}

    def test_excludes_metadata_json_and_checksum_optional(self, tmp_path):
        from redi.output_manifest import describe_outputs
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "index.json").write_text("{}")
        out = describe_outputs(tmp_path, checksum=False)
        assert [e["path"] for e in out["files"]] == ["index.json"]
        assert "sha256" not in out["files"][0]

    def test_unreadable_file_reports_error(self, tmp_path):
        from redi.output_manifest import describe_outputs
        (tmp_path / "bad.npz").write_bytes(b"not a zip")
        (entry,) = describe_outputs(tmp_path)["files"]
        assert "records_error" in entry

    def test_missing_directory(self, tmp_path):
        from redi.output_manifest import describe_outputs
        assert describe_outputs(tmp_path / "nope")["files"] == []


class TestBuildMetadata:
    def _ctx(self, tmp_path, **metadata):
        return SimpleNamespace(
            output_dir=tmp_path, config={}, completed_stages=[], errors=[], metadata=metadata,
        )

    def test_document(self, tmp_path):
        from redi.output_manifest import build_metadata
        np.savez(tmp_path / "train.npz", X=np.zeros((4, 2)))
        doc = build_metadata(self._ctx(tmp_path, output_format="npz"), "in.csv", "general", {"v": 1})
        assert doc["input_file"] == "in.csv" and doc["domain"] == "general"
        assert doc["output_format"] == "npz" and doc["provenance"] == {"v": 1}
        assert doc["outputs"]["files"][0]["n_records"] == 4
        json.dumps(doc)

    def test_output_records_hint(self, tmp_path):
        from redi.output_manifest import build_metadata
        np.savez(tmp_path / "train.npz", positions=np.zeros((12, 3)), natoms=np.full(2, 6))
        doc = build_metadata(self._ctx(tmp_path, output_records={"train.npz": 2}), "x", "atomistic")
        assert doc["outputs"]["files"][0]["n_records"] == 2


class TestPipelineWritesManifest:
    def test_run_pipeline_metadata_has_outputs(self, sample_csv, temp_dir):
        from redi.cli import run_pipeline
        out = temp_dir / "out"
        run_pipeline(input_path=str(sample_csv), output_dir=str(out), domain="general", style="simple")
        meta = json.loads((out / "metadata.json").read_text())
        files = {e["path"]: e for e in meta["outputs"]["files"]}
        assert "metadata.json" not in files
        assert "READINESS_REPORT.md" in files
        npz = [e for e in files.values() if e["format"] == "npz"]
        assert npz and all("arrays" in e and "sha256" in e for e in npz)
