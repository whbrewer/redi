"""Tests for the OLCF Frontier job-centric loader, the signed-utilization option,
and the user/account split in the HPC job mapper."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from redi.domains.hpc import get_raps_pipeline, get_ingest_steps
from redi.domains.hpc.job_mapper import (
    ColumnMapper, ColumnMapping, JobDictBuilder, TraceNormalizer,
)
from redi.pipeline import create_context
from redi.pipeline.ingest import DetectFormat

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "hpc" / "frontier_jobcentric.yaml"
QUANTA = 15


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def dataset(tmp_path):
    """A tiny job-centric dataset: 3 jobs ended 2024-01-17; job 3 has no telemetry."""
    root = tmp_path / "telemetry"
    rows = [
        # job_idx, nodes, start, end, wall_time(min), state
        ("000001", 2, "2024-01-17 00:00:00", "2024-01-17 00:01:00", 5, "COMPLETED"),
        ("000002", 4, "2024-01-17 00:10:00", "2024-01-17 00:11:10", 1, "TIMEOUT"),
        ("000003", 1, "2024-01-17 00:20:00", "2024-01-17 00:21:00", 5, "CANCELLED"),
    ]
    info = pd.DataFrame({
        "job_idx": [r[0] for r in rows], "date_dir": "2024-01-17",
        "node_count": [r[1] for r in rows],
        "submit_time": [_ts(r[2]) - pd.Timedelta(seconds=30) for r in rows],
        "start_time": [_ts(r[2]) for r in rows], "end_time": [_ts(r[3]) for r in rows],
        "wall_time": [r[4] for r in rows], "completion_state": [r[5] for r in rows],
        "queue": "batch-normal",
        "user_id": ["user0001", "user0002", "user0003"],
        "project_id": ["AAA001", "AAA001", "BBB002"],
        "node_list": [["x2103c7s7b0n0", "x2103c7s7b1n0"], ["x2104c5s7b0n0"] * 4, ["x2105c1s1b0n0"]],
    })
    root.mkdir()
    info.to_parquet(root / "frontier-completed-job-info.parquet")
    # known 1 s power: CPU 80 W per node (below Frontier's 90 W idle), GPU = ramp so bin means are easy to check
    for jid, nodes, secs in (("000001", 2, 60), ("000002", 4, 70)):
        d = root / "job-telemetry" / "2024-01-17" / jid
        d.mkdir(parents=True)
        pd.DataFrame({
            "cpu_agg_total_power": np.full(secs, 80.0 * nodes),
            "gpu_agg_total_power": np.arange(secs, dtype=float) + 1000.0 * nodes,
        }).to_parquet(d / f"{jid}-cleaned-power.parquet")
    return root


def _load(dataset, **config):
    cfg = {"date_dir": "2024-01-17", "raps_jobs": {"trace_quanta": QUANTA}}
    cfg.update(config)
    ctx = create_context(input_path=str(dataset), output_dir=str(dataset.parent / "out"), domain="hpc", **cfg)
    for step in get_ingest_steps():
        if step.validate(ctx):
            ctx = step.execute(ctx)
    return ctx


def test_detect_format(dataset):
    ctx = create_context(input_path=str(dataset), output_dir=str(dataset.parent / "o"), domain="hpc")
    ctx = DetectFormat().execute(ctx)
    assert ctx.metadata["file_format"] == "frontier-jobcentric"


def test_loader_skips_jobs_without_telemetry_and_bins_to_quanta(dataset):
    ctx = _load(dataset)
    assert not ctx.errors, ctx.errors
    df = ctx.data
    assert list(df["job_id"]) == [1, 2]                      # job 3 has no telemetry file
    assert ctx.metadata["frontier_jobcentric"]["jobs_without_telemetry"] == 1
    # job 1: 60 s -> 4 bins of 15 s; GPU ramp 2000..2059, mean of each bin
    gpu = df.loc[0, "gpu_agg_total_power"]
    assert len(gpu) == 4
    assert gpu[0] == pytest.approx(np.mean(np.arange(15) + 2000.0))
    # job 2: 70 s -> 5 bins, last is a partial interval averaged over its 10 real samples
    gpu2 = df.loc[1, "gpu_agg_total_power"]
    assert len(gpu2) == 5
    assert gpu2[4] == pytest.approx(np.mean(np.arange(60, 70) + 4000.0))
    # time_limit (s): max(wall_time*60, ceil(run/quanta)*quanta)
    assert df.loc[0, "time_limit"] == 300                    # 5 min limit > 60 s run
    assert df.loc[1, "time_limit"] == 75                     # 1 min limit < 70 s run -> 5 quanta
    assert df.loc[0, "name"] == "AAA001/000001"


def test_loader_requires_a_partition(dataset):
    ctx = _load(dataset, date_dir=None)
    assert any("date_dir" in e for e in ctx.errors)


def _run_pipeline(dataset, clamp_negative, tmp_path):
    cfg = yaml.safe_load(EXAMPLE.read_text())
    r = cfg["raps_jobs"]
    steps = get_raps_pipeline(
        column_overrides=r["column_overrides"], power_config=r["power_config"], trace_quanta=r["trace_quanta"],
        system_name="frontier", topology=r["topology"], priority_policy=r["priority_policy"],
        output_format="json", clamp_negative=clamp_negative, account_source=r["account_source"])
    ctx = create_context(input_path=str(dataset), output_dir=str(tmp_path / "o"), domain="hpc",
                         date_dir="2024-01-17", raps_jobs={"trace_quanta": QUANTA})
    for step in steps:
        if step.name.startswith(("save_", "write_")):
            continue
        if step.validate(ctx):
            ctx = step.execute(ctx)
    assert not ctx.errors, ctx.errors
    return {j["id"]: j for j in ctx.data}


def test_end_to_end_user_account_and_traces(dataset, tmp_path):
    jobs = _run_pipeline(dataset, clamp_negative=False, tmp_path=tmp_path)
    j = jobs["1"]
    assert j["user"] == "user0001" and j["project"] == "AAA001"
    assert j["account"] == "user0001"                        # RAPS slot carries the user
    assert j["nodes_required"] == 2 and j["end_state"] == "COMPLETED"
    assert j["time_limit"] == 300 and j["expected_run_time"] == 60
    assert len(j["scheduled_nodes"]) == 2 and all(isinstance(n, int) for n in j["scheduled_nodes"])
    # util = (watts/idle - per_node)/(max/idle - 1), watts = agg / nodes (RAPS formula)
    cpu0 = (80.0 * 2 / 2 / 90 - 1) / (280 / 90 - 1)
    assert cpu0 < 0
    assert j["cpu_trace"][0] == pytest.approx(cpu0)          # below idle: kept signed
    gpu0 = (np.mean(np.arange(15) + 2000.0) / 2 / 88 - 4) / (560 / 88 - 1)
    assert j["gpu_trace"][0] == pytest.approx(gpu0)


def test_clamp_option_controls_negative_utilization(dataset, tmp_path):
    signed = _run_pipeline(dataset, clamp_negative=False, tmp_path=tmp_path)["1"]["cpu_trace"][0]
    clamped = _run_pipeline(dataset, clamp_negative=True, tmp_path=tmp_path)["1"]["cpu_trace"][0]
    assert signed < 0                                        # 80 W/node is below the 90 W idle
    assert clamped == 0.0


def test_trace_normalizer_default_still_clamps():
    n = TraceNormalizer({"cpu": {"idle": 90, "max": 280, "per_node": 1}})
    assert n.normalize_trace([50.0], "cpu_trace", "cpu_power", nodes_required=1) == 0.0
    s = TraceNormalizer({"cpu": {"idle": 90, "max": 280, "per_node": 1}}, clamp_negative=False)
    assert s.normalize_trace([50.0], "cpu_trace", "cpu_power", nodes_required=1) < 0.0


# ---------------------------------------------------------------- user / account split

def _frame():
    return pd.DataFrame({
        "job_id": [1], "n": [2], "submit": ["2024-01-01 08:00:00"], "start": ["2024-01-01 08:30:00"],
        "end": ["2024-01-01 09:30:00"], "who": ["alice"], "proj": ["P42"],
    })


def _mapping(**extra):
    m = {"id": "job_id", "nodes_required": "n", "submit_time": "submit", "start_time": "start", "end_time": "end"}
    m.update(extra)
    return ColumnMapping(mapped=m, confidence={k: 1.0 for k in m}, unmapped_columns=[], missing_required=[])


def test_user_goes_to_raps_account_slot_and_project_is_kept():
    jobs, _ = JobDictBuilder().build_jobs(_frame(), _mapping(user="who", account="proj"))
    assert jobs[0]["user"] == "alice" and jobs[0]["account"] == "alice" and jobs[0]["project"] == "P42"


def test_account_source_account_keeps_the_project_in_the_slot():
    jobs, _ = JobDictBuilder(account_source="account").build_jobs(_frame(), _mapping(user="who", account="proj"))
    assert jobs[0]["user"] == "alice" and jobs[0]["account"] == "P42"


def test_legacy_mapping_without_user_is_unchanged():
    jobs, _ = JobDictBuilder().build_jobs(_frame(), _mapping(account="who"))
    assert jobs[0]["account"] == "alice" and not jobs[0].get("user") and "project" not in jobs[0]


def test_aliases_separate_user_from_project_and_ignore_group_id():
    res = ColumnMapper().map_columns(["job_id", "user_id", "group_id", "project_name", "num_nodes"])
    assert res.mapped["user"] == "user_id"
    assert res.mapped["account"] == "project_name"
    assert "group_id" not in res.mapped.values()             # no accidental token match on "id"
