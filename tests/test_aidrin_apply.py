"""REDI acting on an AIDRIN report: stratified split, protected target, constant columns.

Two things must hold together: with the report, the generic pipeline really does these
things, and without it nothing about the pipeline changes.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from redi.aidrin_manifest import derive_settings, load_report, record_outcomes
from redi.pipeline import PipelineContext
from redi.pipeline.output import TrainValTestSplit, get_output_steps
from redi.pipeline.transform import (
    EncodeCategorical,
    EncodeTarget,
    NormalizeNumeric,
    SelectFeatures,
    get_transform_steps,
)

REPO = Path(__file__).resolve().parent.parent


def ctx_for(data, tmp_path):
    return PipelineContext(input_path=tmp_path / "in.csv", output_dir=tmp_path / "out", data=data)


# ---------------------------------------------------------------------------
# defaults are untouched
# ---------------------------------------------------------------------------

class TestDefaultsUnchanged:
    def test_transform_steps_are_the_same_without_options(self):
        assert [s.name for s in get_transform_steps()] == ["normalize_numeric", "encode_categorical"]

    def test_output_steps_do_not_stratify_by_default(self):
        split = get_output_steps()[0]
        assert isinstance(split, TrainValTestSplit) and split.stratify is False

    def test_default_split_matches_the_original_algorithm(self, tmp_path):
        n = 100
        X, y = np.arange(n * 2, dtype=float).reshape(n, 2), np.arange(n)
        ctx = TrainValTestSplit().execute(ctx_for({"X": X, "y": y}, tmp_path))
        rng_idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(rng_idx)
        assert list(ctx.data["train"]["y"]) == list(y[rng_idx[:80]])
        assert list(ctx.data["val"]["y"]) == list(y[rng_idx[80:90]])
        assert "stratified" not in ctx.metadata["split"]

    def test_normalize_and_encode_touch_everything_without_exclude(self, tmp_path):
        df = pd.DataFrame({"n": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
        ctx = ctx_for(df.copy(), tmp_path)
        ctx = NormalizeNumeric().execute(ctx)
        ctx = EncodeCategorical().execute(ctx)
        assert abs(ctx.data["n"].mean()) < 1e-9 and "c" not in ctx.data.columns


# ---------------------------------------------------------------------------
# stratified split
# ---------------------------------------------------------------------------

def imbalanced(n=1000, minority=100):
    y = np.array([0] * (n - minority) + [1] * minority)
    return {"X": np.arange(n * 2, dtype=float).reshape(n, 2), "y": y}


class TestStratifiedSplit:
    def test_each_split_keeps_the_class_share(self, tmp_path):
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(imbalanced(), tmp_path))
        for name in ("train", "val", "test"):
            share = ctx.data[name]["y"].mean()
            assert abs(share - 0.10) < 0.02, (name, share)
        assert ctx.metadata["split"]["stratified"] is True

    def test_sizes_add_up_and_nothing_is_lost_or_duplicated(self, tmp_path):
        data = imbalanced()
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        sizes = [len(ctx.data[s]["y"]) for s in ("train", "val", "test")]
        assert sum(sizes) == 1000
        seen = np.concatenate([ctx.data[s]["X"][:, 0] for s in ("train", "val", "test")])
        assert sorted(seen) == sorted(data["X"][:, 0])

    def test_x_and_y_stay_aligned(self, tmp_path):
        data = imbalanced()
        data["X"][:, 1] = data["y"]  # second feature mirrors the label
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        for s in ("train", "val", "test"):
            assert (ctx.data[s]["X"][:, 1] == ctx.data[s]["y"]).all()

    def test_reproducible_for_a_seed(self, tmp_path):
        a = TrainValTestSplit(stratify=True).execute(ctx_for(imbalanced(), tmp_path)).data["train"]["X"]
        b = TrainValTestSplit(stratify=True).execute(ctx_for(imbalanced(), tmp_path)).data["train"]["X"]
        assert (a == b).all()

    def test_a_single_sample_class_is_never_test_only(self, tmp_path):
        data = imbalanced(n=200, minority=1)
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        assert 1 in set(ctx.data["train"]["y"])

    def test_string_free_integer_classes_with_many_labels(self, tmp_path):
        y = np.repeat(np.arange(5), 40)
        data = {"X": np.arange(200, dtype=float).reshape(200, 1), "y": y}
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        for s in ("train", "val", "test"):
            assert set(ctx.data[s]["y"]) == set(range(5))

    def test_continuous_target_falls_back_and_says_so(self, tmp_path):
        data = {"X": np.zeros((50, 1)), "y": np.linspace(0, 1, 50)}
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        assert ctx.metadata["split"]["stratified"] is False
        assert "continuous" in ctx.metadata["split"]["stratify_note"]
        assert len(ctx.data["train"]["y"]) == 40

    def test_too_many_distinct_values_falls_back(self, tmp_path):
        data = {"X": np.zeros((200, 1)), "y": np.arange(200)}
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for(data, tmp_path))
        assert ctx.metadata["split"]["stratified"] is False
        assert "too many" in ctx.metadata["split"]["stratify_note"]

    def test_no_target_falls_back(self, tmp_path):
        ctx = TrainValTestSplit(stratify=True).execute(ctx_for({"X": np.zeros((50, 2))}, tmp_path))
        assert ctx.metadata["split"]["stratified"] is False
        assert "no target" in ctx.metadata["split"]["stratify_note"]


# ---------------------------------------------------------------------------
# target protection and constant columns
# ---------------------------------------------------------------------------

class TestTargetProtection:
    def test_text_target_is_label_encoded_with_a_recorded_mapping(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "t": ["b", "a", "b", "c"]})
        ctx = EncodeTarget("t").execute(ctx_for(df, tmp_path))
        assert list(ctx.data["t"]) == [1, 0, 1, 2]
        assert ctx.metadata["target_encoding"]["mapping"] == {"0": "a", "1": "b", "2": "c"}

    def test_numeric_target_is_left_alone(self, tmp_path):
        df = pd.DataFrame({"t": [0, 1, 1, 0]})
        ctx = EncodeTarget("t").execute(ctx_for(df, tmp_path))
        assert list(ctx.data["t"]) == [0, 1, 1, 0]
        assert ctx.metadata["target_encoding"]["method"] == "none"

    def test_missing_target_is_an_error_not_silence(self, tmp_path):
        ctx = EncodeTarget("nope").execute(ctx_for(pd.DataFrame({"f": [1, 2]}), tmp_path))
        assert any("Target column 'nope' not found" in e for e in ctx.errors)

    def test_excluded_target_is_not_normalized_or_one_hot_encoded(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "t": [0, 1, 0, 1], "s": ["x", "y", "x", "y"]})
        ctx = ctx_for(df, tmp_path)
        for step in get_transform_steps(target_column="t"):
            ctx = step.execute(ctx)
        assert list(ctx.data["t"]) == [0, 1, 0, 1]          # labels not z-scored
        assert abs(ctx.data["f"].mean()) < 1e-9             # features still are
        assert "s_x" in ctx.data.columns                    # other categoricals still encoded

    def test_text_target_survives_the_whole_transform_stage(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "t": ["no", "yes", "no", "no"]})
        ctx = ctx_for(df, tmp_path)
        for step in get_transform_steps(target_column="t"):
            ctx = step.execute(ctx)
        assert "t" in ctx.data.columns and ctx.data["t"].dtype.kind == "i"
        assert not [c for c in ctx.data.columns if c.startswith("t_")]


class TestConstantColumns:
    def test_numeric_constants_dropped_and_listed(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "k": [7.0, 7.0, 7.0], "i": [5, 5, 5]})
        ctx = SelectFeatures(method="variance", threshold=1e-12).execute(ctx_for(df, tmp_path))
        assert list(ctx.data.columns) == ["f"]
        assert sorted(ctx.metadata["feature_selection"]["removed"]) == ["i", "k"]

    def test_text_constants_are_not_removed(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0], "s": ["same", "same"]})
        ctx = SelectFeatures(method="variance", threshold=1e-12).execute(ctx_for(df, tmp_path))
        assert "s" in ctx.data.columns

    def test_the_target_is_never_dropped_even_if_constant(self, tmp_path):
        df = pd.DataFrame({"f": [1.0, 2.0], "t": [1, 1]})
        ctx = SelectFeatures(method="variance", threshold=1e-12, exclude=["t"]).execute(ctx_for(df, tmp_path))
        assert "t" in ctx.data.columns

    def test_step_order_with_both_options(self):
        names = [s.name for s in get_transform_steps(target_column="t", drop_constant=True)]
        assert names == ["encode_target", "select_features", "normalize_numeric", "encode_categorical"]


# ---------------------------------------------------------------------------
# settings and outcomes
# ---------------------------------------------------------------------------

def manifest_dict(data_path, roles=None, findings=None):
    return {
        "schema_version": "aidrin.assessment/v1",
        "tool": {"name": "aidrin", "version": "t"},
        "created_at": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"path": str(data_path), "format": "csv", "rows": 1, "columns": 1, "schema": []},
        "redaction": "redacted",
        "roles": {"target": None, "sensitive": [], "quasi_identifiers": [], "id": None, "batch": None,
                  "confirmed_by_user": False, **(roles or {})},
        "metrics": [],
        "findings": [{"metric": "m", "scope": "dataset", "name": n, "value": v, "direction": "informational"}
                     for n, v in (findings or {}).items()],
    }


def load(tmp_path, **kw):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest_dict(tmp_path / "d.csv", **kw)))
    return load_report(p)


class TestDeriveSettings:
    def test_confirmed_target_and_imbalance(self, tmp_path):
        s = derive_settings(load(tmp_path, roles={"target": "y", "confirmed_by_user": True},
                                 findings={"aidrin.fairness.class_imbalance": 0.4}))
        assert (s.target_column, s.stratify, s.drop_constant) == ("y", True, False)

    def test_unconfirmed_target_is_not_used(self, tmp_path):
        s = derive_settings(load(tmp_path, roles={"target": "y"},
                                 findings={"aidrin.fairness.class_imbalance": 0.4}))
        assert (s.target_column, s.stratify) == (None, False)

    def test_balanced_target_does_not_stratify(self, tmp_path):
        s = derive_settings(load(tmp_path, roles={"target": "y", "confirmed_by_user": True},
                                 findings={"aidrin.fairness.class_imbalance": 0.0}))
        assert (s.target_column, s.stratify) == ("y", False)

    def test_constants_need_no_roles(self, tmp_path):
        s = derive_settings(load(tmp_path, findings={"aidrin.structure.constant_feature_count": 3}))
        assert s.drop_constant is True and s.target_column is None

    def test_empty_report_changes_nothing(self, tmp_path):
        s = derive_settings(load(tmp_path))
        assert (s.target_column, s.stratify, s.drop_constant) == (None, False, False)


class TestRecordOutcomes:
    def block(self, *topics):
        return {"aidrin_assessment": {"applied": [{"topic": t, "outcome": None, "detail": None} for t in topics]}}

    def test_successes(self):
        md = self.block("target-role", "stratified-split", "constant-features")
        md.update(numpy_structure={"target_column": "t"},
                  target_encoding={"column": "t", "method": "label", "mapping": {"0": "a"}},
                  split={"stratified": True}, feature_selection={"removed": ["k"]})
        assert record_outcomes(md) == []
        got = {a["topic"]: a for a in md["aidrin_assessment"]["applied"]}
        assert all(a["outcome"] == "applied" for a in got.values())
        assert "label-encoded" in got["target-role"]["detail"] and "k" in got["constant-features"]["detail"]

    def test_wired_but_not_done_is_reported_as_not_applied(self):
        md = self.block("target-role", "stratified-split", "constant-features")
        md.update(split={"stratified": False, "stratify_note": "target is continuous, not class labels"})
        failed = record_outcomes(md)
        assert {f["topic"] for f in failed} == {"target-role", "stratified-split", "constant-features"}
        details = {f["topic"]: f["detail"] for f in failed}
        assert "continuous" in details["stratified-split"]

    def test_requested_target_that_was_not_found_is_not_applied(self):
        # numpy_structure echoes the requested name even when the column is absent
        md = self.block("target-role")
        md.update(numpy_structure={"target_column": "ghost"})
        failed = record_outcomes(md)
        assert [f["topic"] for f in failed] == ["target-role"]

    def test_no_report_no_work(self):
        assert record_outcomes({}) == []


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def run_redi(*argv):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONWARNINGS="ignore")
    return subprocess.run([sys.executable, "-m", "redi.cli", *argv], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=300)


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "f1": rng.rand(n), "f2": rng.rand(n) * 10,
        "k": 3.0,                                    # numeric constant
        "label": np.where(np.arange(n) % 10 == 0, "rare", "common"),   # 10% minority, text
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def write_manifest(tmp_path, data, roles=None, findings=None):
    m = manifest_dict(data, roles=roles, findings=findings)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p


CONFIRMED = {"target": "label", "confirmed_by_user": True}
FINDINGS = {"aidrin.fairness.class_imbalance": 0.8, "aidrin.structure.constant_feature_count": 1}


def split_files(out):
    return {s: np.load(out / f"{s}.npz", allow_pickle=False) for s in ("train", "val", "test")}


class TestEndToEnd:
    def test_report_drives_the_pipeline(self, tmp_path, dataset):
        manifest = write_manifest(tmp_path, dataset, CONFIRMED, FINDINGS)
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.count("[applied]") == 3
        assert "was not applied" not in proc.stdout

        splits = split_files(out)  # allow_pickle=False: y must be a real numeric array
        for s in splits.values():
            assert s["y"].dtype.kind == "i"
        shares = [splits[n]["y"].mean() for n in ("train", "val", "test")]
        assert all(abs(x - 0.10) < 0.03 for x in shares), shares

        meta = json.loads((out / "metadata.json").read_text())
        assert meta["numpy_structure"]["target_column"] == "label"
        assert "k" not in meta["numpy_structure"]["feature_columns"]
        assert meta["target_encoding"]["mapping"] == {"0": "common", "1": "rare"}
        assert meta["split"]["stratified"] is True
        applied = {a["topic"]: a for a in meta["aidrin_assessment"]["applied"]}
        assert set(applied) == {"target-role", "stratified-split", "constant-features"}
        assert all(a["outcome"] == "applied" for a in applied.values())
        assert "k" in applied["constant-features"]["detail"]

    def test_target_is_not_normalized(self, tmp_path, dataset):
        manifest = write_manifest(tmp_path, dataset, CONFIRMED, {})
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        ys = np.concatenate([s["y"] for s in split_files(out).values()])
        assert set(np.unique(ys)) == {0, 1}

    def test_without_the_flag_nothing_changes(self, tmp_path, dataset):
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        meta = json.loads((out / "metadata.json").read_text())
        assert "aidrin_assessment" not in meta
        assert "stratified" not in meta.get("split", {})
        assert "target_encoding" not in meta
        assert "k" in meta["numpy_structure"]["feature_columns"]  # constant column kept without the report
        assert "y" not in split_files(out)["train"].files

    def test_no_apply_flag_is_advisory_only(self, tmp_path, dataset):
        manifest = write_manifest(tmp_path, dataset, CONFIRMED, FINDINGS)
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--aidrin-report", str(manifest),
                        "--aidrin-no-apply", "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[applied]" not in proc.stdout
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["aidrin_assessment"]["mode"] == "advisory"
        assert meta["aidrin_assessment"]["applied"] == []
        assert "y" not in split_files(out)["train"].files

    def test_unconfirmed_target_is_not_used(self, tmp_path, dataset):
        manifest = write_manifest(tmp_path, dataset, {"target": "label"}, FINDINGS)
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        meta = json.loads((out / "metadata.json").read_text())
        assert {a["topic"] for a in meta["aidrin_assessment"]["applied"]} == {"constant-features"}
        assert "y" not in split_files(out)["train"].files

    def test_missing_target_column_is_reported_not_hidden(self, tmp_path, dataset):
        manifest = write_manifest(tmp_path, dataset, {"target": "no_such_column", "confirmed_by_user": True}, FINDINGS)
        out = tmp_path / "out"
        proc = run_redi("run", str(dataset), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "AIDRIN action 'target-role' was not applied" in proc.stdout
        meta = json.loads((out / "metadata.json").read_text())
        outcomes = {a["topic"]: a["outcome"] for a in meta["aidrin_assessment"]["applied"]}
        assert outcomes["target-role"] == "not applied"
        assert outcomes["stratified-split"] == "not applied"
