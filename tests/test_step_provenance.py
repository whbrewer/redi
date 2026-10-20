"""
Tests for per-step provenance in metadata.json (pipeline_steps / pipeline_timing).
"""

import json
import types

import pytest


def _make_plugin(monkeypatch, *, fail_validate=(), raise_in=None, changes=True):
    """Register a fake plugin whose steps use string stage names (as plugin steps do)."""
    import redi.plugins

    def make_step(stage_name):
        class _Step:
            name = f"{stage_name}_step"
            stage = stage_name
            description = ""

            def validate(self, ctx):
                return self.name not in fail_validate

            def execute(self, ctx):
                if self.name == raise_in:
                    raise RuntimeError("boom")
                if stage_name == "preprocess":
                    ctx.metadata["kept"] = 8
                    if changes:
                        ctx.metadata.setdefault("step_changes", {})[self.name] = {
                            "records_in": 10, "records_out": 8, "dropped": 2,
                        }
                return ctx

        return _Step()

    steps = lambda: [make_step(s) for s in ("ingest", "preprocess", "output")]
    plugin = types.SimpleNamespace(
        module=types.SimpleNamespace(get_full_pipeline=steps), get_full_pipeline=steps,
    )
    monkeypatch.setattr(redi.plugins, "get_plugin", lambda domain: plugin)


def _run(sample_csv, temp_dir):
    from redi.cli import run_pipeline
    out = temp_dir / "out"
    run_pipeline(input_path=str(sample_csv), output_dir=str(out), domain="fake_plugin", style="simple")
    return json.loads((out / "metadata.json").read_text())


class TestPipelineSteps:
    def test_every_step_recorded_in_order_with_timing(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch)
        meta = _run(sample_csv, temp_dir)
        steps = meta["pipeline_steps"]
        assert [s["name"] for s in steps] == ["ingest_step", "preprocess_step", "output_step"]
        assert [s["order"] for s in steps] == [1, 2, 3]
        assert [s["stage"] for s in steps] == ["ingest", "preprocess", "output"]
        for s in steps:
            assert s["status"] == "success"
            assert isinstance(s["seconds"], float) and s["seconds"] >= 0
            assert s["started_at"] <= s["ended_at"]

    def test_pipeline_timing(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch)
        timing = _run(sample_csv, temp_dir)["pipeline_timing"]
        assert timing["started_at"] <= timing["ended_at"]
        assert timing["total_seconds"] >= 0

    def test_metadata_keys_added_are_listed(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch)
        steps = {s["name"]: s for s in _run(sample_csv, temp_dir)["pipeline_steps"]}
        assert "kept" in steps["preprocess_step"]["metadata_added"]
        assert "step_changes" not in steps["preprocess_step"]["metadata_added"]

    def test_step_reported_changes_are_attached_and_not_duplicated(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch)
        meta = _run(sample_csv, temp_dir)
        steps = {s["name"]: s for s in meta["pipeline_steps"]}
        assert steps["preprocess_step"]["changes"] == {"records_in": 10, "records_out": 8, "dropped": 2}
        assert "changes" not in steps["ingest_step"]
        assert "step_changes" not in meta

    def test_skipped_step_recorded(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch, fail_validate=("preprocess_step",))
        steps = {s["name"]: s for s in _run(sample_csv, temp_dir)["pipeline_steps"]}
        assert steps["preprocess_step"]["status"] == "skipped"
        assert steps["preprocess_step"]["reason"]
        assert steps["ingest_step"]["status"] == "success"

    def test_failed_step_recorded_with_message(self, sample_csv, temp_dir, monkeypatch):
        _make_plugin(monkeypatch, raise_in="preprocess_step")
        steps = {s["name"]: s for s in _run(sample_csv, temp_dir)["pipeline_steps"]}
        assert steps["preprocess_step"]["status"] == "error"
        assert "boom" in steps["preprocess_step"]["message"]

    def test_core_domain_records_data_shape(self, sample_csv, temp_dir):
        from redi.cli import run_pipeline
        out = temp_dir / "out"
        run_pipeline(input_path=str(sample_csv), output_dir=str(out), domain="general", style="simple")
        steps = json.loads((out / "metadata.json").read_text())["pipeline_steps"]
        shapes = [s["data_shape"] for s in steps if "data_shape" in s]
        assert shapes, "no step recorded a data shape"
        assert any(sh["before"] != sh["after"] for sh in shapes)
