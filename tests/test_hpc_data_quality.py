"""Data-quality guards in the HPC job mapper: sentinels, epoch-sized limits, the sanity report."""
import json

import numpy as np
import pandas as pd
import pytest

from redi.domains.hpc.job_mapper import (
    ApplyColumnScalars, AutoMapColumns, BuildJobDicts, DeriveJobFields, FieldDeriver, SaveJobDicts,
    ColumnMapping, field_sanity_report,
)
from redi.pipeline import create_context

UINT32_MAX = 4294967295


def _frame(time_limit, n=12):
    return pd.DataFrame({
        "job_id": np.arange(1, n + 1), "nodes": [2] * n,
        "submit": pd.date_range("2024-01-01 08:00", periods=n, freq="min"),
        "start": pd.date_range("2024-01-01 08:30", periods=n, freq="min"),
        "end": pd.date_range("2024-01-01 09:30", periods=n, freq="min"),
        "user": [f"u{i % 3}" for i in range(n)], "timelimit": time_limit,
    })


OVERRIDES = {"job_id": "id", "nodes": "nodes_required", "submit": "submit_time", "start": "start_time",
             "end": "end_time", "user": "user", "timelimit": "time_limit"}


def _run(df, scalars=None, tmp=None):
    ctx = create_context(input_path="x", output_dir=str(tmp or "."), domain="hpc")
    ctx.data = df
    for step in (AutoMapColumns(overrides=OVERRIDES), ApplyColumnScalars(scalars), DeriveJobFields(),
                 BuildJobDicts(system_name="t")):
        ctx = step.execute(ctx)
    assert not ctx.errors, ctx.errors
    return ctx


def test_scalar_step_removes_unlimited_sentinel_before_scaling():
    df = _frame([60, UINT32_MAX] + [120] * 10)
    ctx = create_context(input_path="x", output_dir=".", domain="hpc"); ctx.data = df
    out = ApplyColumnScalars({"timelimit": 60}).execute(ctx).data["timelimit"]
    assert out.iloc[0] == 3600 and out.iloc[2] == 7200
    assert np.isnan(out.iloc[1])                       # NOT 4294967295 * 60 = 2.6e11
    note = ctx.metadata["field_sanity"][0]
    assert note["sentinel_rows"] == 1 and note["total_rows"] == 12


def test_unlimited_and_scaled_limits_end_up_as_unknown_and_seconds():
    ctx = _run(_frame([60, UINT32_MAX] + [120] * 10), scalars={"timelimit": 60})
    tl = [j["time_limit"] for j in ctx.data]
    assert tl[0] == 3600 and tl[1] == 0 and tl[2] == 7200
    assert max(tl) < 1e8


def test_epoch_sized_time_limit_is_not_passed_through():
    epoch = 1725866960                                   # a submit time in seconds, not a duration
    ctx = _run(_frame([epoch] * 12))
    assert all(j["time_limit"] == 0 for j in ctx.data)
    note = next(n for n in ctx.metadata["field_sanity"] if n.get("epoch_sized_rows"))
    assert note["field"] == "time_limit" and note["epoch_sized_rows"] == 12


def test_real_limits_are_left_alone():
    ctx = _run(_frame([3600] * 12))
    assert all(j["time_limit"] == 3600 for j in ctx.data)
    assert not any(n["field"] == "time_limit" and "values are not durations" in n["issue"]
                   for n in ctx.metadata.get("field_sanity", []))


def test_sanity_report_flags_constant_zero_negative_and_dead_traces():
    jobs = [{"nodes_required": 4, "cpu_cores_required": 0, "gpu_units_required": -1, "time_limit": 0,
             "expected_run_time": 100.0 + i, "priority": 0, "cpu_trace": [0.0, 0.0], "gpu_trace": 0}
            for i in range(20)]
    issues = {(n["field"], n["issue"].split(" (")[0]) for n in field_sanity_report(jobs)}
    assert ("nodes_required", "constant") in issues
    assert ("gpu_units_required", "negative values") in issues
    assert ("time_limit", "all zero") in issues
    assert ("cpu_trace", "all zero in every job") in issues and ("gpu_trace", "all zero in every job") in issues
    assert not any(f == "expected_run_time" for f, _ in issues)       # varies: fine


def test_manifest_records_the_sanity_report(tmp_path):
    ctx = _run(_frame([UINT32_MAX] * 12), scalars={"timelimit": 60}, tmp=tmp_path)
    ctx = SaveJobDicts(output_format="json", system_name="t").execute(ctx)
    meta = json.load(open(tmp_path / "raps_mapping_metadata.json"))
    assert any(n["field"] == "timelimit" and n.get("sentinel_rows") == 12 for n in meta["field_sanity"])


def test_derived_names_are_deterministic():
    df = _frame([3600] * 12)
    mapping = ColumnMapping(mapped={"id": "job_id", "nodes_required": "nodes", "submit_time": "submit",
                                    "start_time": "start", "end_time": "end"}, confidence={}, unmapped_columns=[],
                            missing_required=[])
    a, _ = FieldDeriver().derive(df.copy(), mapping)
    b, _ = FieldDeriver().derive(df.copy(), mapping)
    assert list(a["_derived_name"]) == list(b["_derived_name"])
    assert len(set(a["_derived_name"])) == 12 and all(len(x) == 6 for x in a["_derived_name"])


def test_power_trace_without_power_config_is_reported_not_silently_zeroed(tmp_path):
    from redi.domains.hpc.job_mapper import TraceNormalizer
    n = TraceNormalizer(power_config=None)
    assert n.normalize_trace([900.0, 1100.0], "cpu_trace", "cpu_power_consumption", nodes_required=1) == [0.0, 0.0]
    assert len(n.sanity) == 1 and "no power_config" in n.sanity[0]["issue"]
    n.normalize_trace([1000.0], "cpu_trace", "cpu_power_consumption", nodes_required=1)
    assert len(n.sanity) == 1                                  # reported once per column

    df = _frame([3600] * 12).assign(cpu_power=[900.0] * 12)
    ctx = create_context(input_path="x", output_dir=str(tmp_path), domain="hpc"); ctx.data = df
    for step in (AutoMapColumns(overrides={**OVERRIDES, "cpu_power": "cpu_trace"}), DeriveJobFields(), BuildJobDicts(system_name="t")):
        ctx = step.execute(ctx)
    assert any("no power_config" in n["issue"] for n in ctx.metadata["field_sanity"])
    assert any(n["field"] == "cpu_trace" and "all zero" in n["issue"] for n in ctx.metadata["field_sanity"])
