"""
Unit tests for the HPC Job Mapper module.

Tests ColumnMapper, FieldDeriver, TraceNormalizer, and JobDictBuilder
with column schemas from multiple real HPC systems.
"""

import json
import pytest
import numpy as np
import pandas as pd

from redi.domains.hpc_job_mapper import (
    FIELD_REGISTRY,
    ColumnMapper,
    ColumnMapping,
    FieldDeriver,
    TraceNormalizer,
    JobDictBuilder,
    _parse_duration,
    _normalize_state,
    _parse_node_list,
    _parse_trace_string,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return ColumnMapper()


@pytest.fixture
def deriver():
    return FieldDeriver()


@pytest.fixture
def trace_normalizer():
    return TraceNormalizer()


@pytest.fixture
def trace_normalizer_with_power():
    return TraceNormalizer(power_config={
        "cpu": {"idle": 100, "max": 500},
        "gpu": {"idle": 50, "max": 400},
    })


@pytest.fixture
def builder():
    return JobDictBuilder()


# ---------------------------------------------------------------------------
# System column schemas from RAPS dataloaders
# ---------------------------------------------------------------------------

SYSTEM_SCHEMAS = {
    "frontier": [
        "primary_job_id", "user_name", "account", "num_nodes",
        "time_submit", "time_start", "time_end", "timelimit",
        "runtime", "job_state", "partition", "exit_code",
    ],
    "fugaku": [
        "jid", "usr", "nnumr", "sdt", "edt", "elpl",
        "status", "pri",
    ],
    "mit_supercloud": [
        "job_id", "user", "num_nodes_alloc", "submit_time",
        "start_time", "end_time", "timelimit", "run_time",
        "cpu_util", "gpu_util", "job_state", "partition",
    ],
    "marconi100": [
        "id_job", "id_user", "nodes_alloc", "time_submit",
        "time_start", "time_end", "timelimit", "cpu_power_consumption",
        "gpu_power_consumption", "exit_code", "partition",
    ],
    "lassen": [
        "allocation_id", "hashed_user_id", "node_count",
        "queued_timestamp", "start_timestamp", "end_timestamp",
        "runtime_seconds", "exit_status",
    ],
    "kestrel": [
        "job_id", "user", "nodes_required", "submit_time",
        "start_time", "end_time", "time_limit", "run_time",
        "job_state", "partition", "cpus", "gpus",
    ],
    "blue_waters": [
        "jobid", "user", "nodect", "qtime", "start",
        "end", "walltime", "status", "queue",
    ],
    "philly": [
        "jobid", "user", "num_gpus", "submitted_time",
        "begin_time", "end_time", "status",
    ],
    "google_cloud_v2": [
        "job_id", "user", "scheduling_class", "cpu_usage",
        "start_time", "end_time",
    ],
    "adastra_mi250": [
        "job_ID", "name", "account", "num_nodes",
        "time_submit", "time_start", "time_end",
        "sum_cpu0_power", "sum_gpu_power", "partition",
    ],
}


# ---------------------------------------------------------------------------
# TestColumnMapper
# ---------------------------------------------------------------------------

class TestColumnMapper:
    """Tests for the multi-phase ColumnMapper."""

    @pytest.mark.parametrize("system,columns", list(SYSTEM_SCHEMAS.items()))
    def test_maps_system_columns(self, mapper, system, columns):
        """Each real HPC system's columns should produce at least a few mappings."""
        result = mapper.map_columns(columns)

        assert isinstance(result, ColumnMapping)
        assert len(result.mapped) > 0, f"No mappings for {system}: {columns}"

        # Every system should map 'id' (they all have some job id column)
        assert "id" in result.mapped, (
            f"{system}: 'id' not mapped. mapped={result.mapped}"
        )

    def test_frontier_full_mapping(self, mapper):
        result = mapper.map_columns(SYSTEM_SCHEMAS["frontier"])
        mapped = result.mapped

        assert mapped["id"] == "primary_job_id"
        assert mapped["account"] in ("user_name", "account")
        assert mapped["nodes_required"] == "num_nodes"
        assert mapped["submit_time"] == "time_submit"
        assert mapped["start_time"] == "time_start"
        assert mapped["end_time"] == "time_end"
        assert mapped["expected_run_time"] == "runtime"
        assert mapped["end_state"] in ("job_state", "exit_code")

    def test_fugaku_mapping(self, mapper):
        result = mapper.map_columns(SYSTEM_SCHEMAS["fugaku"])
        mapped = result.mapped

        assert mapped["id"] == "jid"
        assert mapped["account"] == "usr"
        assert mapped["nodes_required"] == "nnumr"

    def test_manual_overrides(self):
        mapper = ColumnMapper(overrides={"weird_col": "id"})
        result = mapper.map_columns(["weird_col", "other"])
        assert result.mapped["id"] == "weird_col"
        assert result.confidence["id"] == 1.0

    def test_case_insensitive(self, mapper):
        result = mapper.map_columns(["JOB_ID", "User", "NUM_NODES"])
        assert "id" in result.mapped
        assert "account" in result.mapped
        assert "nodes_required" in result.mapped

    def test_unmapped_columns_tracked(self, mapper):
        result = mapper.map_columns(["job_id", "xyzzy_foo", "blarg_baz"])
        assert "xyzzy_foo" in result.unmapped_columns
        assert "blarg_baz" in result.unmapped_columns

    def test_missing_required_tracked(self, mapper):
        # Only map a few fields, missing key required fields
        result = mapper.map_columns(["status", "queue"])
        # id, nodes_required, start_time, end_time should be missing
        assert "id" in result.missing_required
        assert "nodes_required" in result.missing_required

    def test_no_duplicate_mappings(self, mapper):
        """Each canonical field should map to at most one column."""
        result = mapper.map_columns(SYSTEM_SCHEMAS["kestrel"])
        # Check no duplicates in values
        cols_used = list(result.mapped.values())
        assert len(cols_used) == len(set(cols_used))


# ---------------------------------------------------------------------------
# TestFieldDeriver
# ---------------------------------------------------------------------------

class TestFieldDeriver:
    def test_derive_run_time_from_timestamps(self, deriver):
        df = pd.DataFrame({
            "start": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00"]),
            "end": pd.to_datetime(["2024-01-01 11:00:00", "2024-01-01 12:30:00"]),
            "jid": ["1", "2"],
        })
        mapping = ColumnMapping(
            mapped={"id": "jid", "start_time": "start", "end_time": "end"},
            confidence={"id": 0.95, "start_time": 0.95, "end_time": 0.95},
            unmapped_columns=[],
            missing_required=["nodes_required", "expected_run_time"],
        )
        df, new_mapping = deriver.derive(df, mapping)

        assert "expected_run_time" in new_mapping.mapped
        assert df[new_mapping.mapped["expected_run_time"]].iloc[0] == 3600.0
        assert df[new_mapping.mapped["expected_run_time"]].iloc[1] == 5400.0

    def test_derive_name_from_id(self, deriver):
        df = pd.DataFrame({"jid": [101, 102]})
        mapping = ColumnMapping(
            mapped={"id": "jid"},
            confidence={"id": 0.95},
            unmapped_columns=[],
            missing_required=["nodes_required", "start_time", "end_time"],
        )
        df, new_mapping = deriver.derive(df, mapping)

        assert "name" in new_mapping.mapped
        assert df[new_mapping.mapped["name"]].iloc[0] == "job_101"

    def test_derive_nodes_from_node_list(self, deriver):
        df = pd.DataFrame({
            "jid": [1],
            "nodelist": ["node001,node002,node003"],
        })
        mapping = ColumnMapping(
            mapped={"id": "jid", "scheduled_nodes": "nodelist"},
            confidence={"id": 0.95, "scheduled_nodes": 0.95},
            unmapped_columns=[],
            missing_required=["nodes_required", "start_time", "end_time"],
        )
        df, new_mapping = deriver.derive(df, mapping)

        assert "nodes_required" in new_mapping.mapped
        assert df[new_mapping.mapped["nodes_required"]].iloc[0] == 3

    def test_derive_submit_time_fallback(self, deriver):
        df = pd.DataFrame({"start": ["2024-01-01"], "jid": [1]})
        mapping = ColumnMapping(
            mapped={"id": "jid", "start_time": "start"},
            confidence={"id": 0.95, "start_time": 0.95},
            unmapped_columns=[],
            missing_required=["nodes_required", "end_time"],
        )
        df, new_mapping = deriver.derive(df, mapping)

        assert "submit_time" in new_mapping.mapped
        assert new_mapping.mapped["submit_time"] == "start"


# ---------------------------------------------------------------------------
# TestTraceNormalizer
# ---------------------------------------------------------------------------

class TestTraceNormalizer:
    def test_utilization_passthrough(self, trace_normalizer):
        val = trace_normalizer.normalize_trace(0.75, "cpu_trace", "cpu_util")
        assert val == pytest.approx(0.75)

    def test_percentage_to_utilization(self, trace_normalizer):
        val = trace_normalizer.normalize_trace(85.0, "cpu_trace", "cpu_util")
        assert val == pytest.approx(0.85)

    def test_power_with_config(self, trace_normalizer_with_power):
        # 300W with idle=100, max=500 → (300-100)/(500-100) = 0.5
        val = trace_normalizer_with_power.normalize_trace(300.0, "cpu_trace", "cpu_power")
        assert val == pytest.approx(0.5)

    def test_power_without_config(self, trace_normalizer):
        # Power detected by value > 100, but no config → 0
        val = trace_normalizer.normalize_trace(300.0, "cpu_trace", "cpu_power")
        assert val == pytest.approx(0.0)

    def test_list_trace_percentage(self, trace_normalizer):
        val = trace_normalizer.normalize_trace([50.0, 80.0, 100.0], "cpu_trace", "cpu_util")
        assert isinstance(val, list)
        assert len(val) == 3
        assert val[0] == pytest.approx(0.5)
        assert val[2] == pytest.approx(1.0)

    def test_string_trace_list(self, trace_normalizer):
        val = trace_normalizer.normalize_trace("[50, 80]", "cpu_trace", "cpu_util")
        assert isinstance(val, list)
        assert len(val) == 2
        assert val[0] == pytest.approx(0.5)

    def test_none_returns_zero(self, trace_normalizer):
        assert trace_normalizer.normalize_trace(None, "cpu_trace", "cpu") == 0

    def test_nan_returns_zero(self, trace_normalizer):
        assert trace_normalizer.normalize_trace(float("nan"), "cpu_trace", "cpu") == 0

    def test_clamp_percentage_over_100(self, trace_normalizer):
        # 99% should normalize to 0.99 (detected as percentage since max <= 100)
        val = trace_normalizer.normalize_trace(99.0, "cpu_trace", "cpu_util")
        assert val == pytest.approx(0.99)

    def test_utilization_clamp(self, trace_normalizer):
        # Values > 1 but <= 100 are treated as percentage; verify single value
        val = trace_normalizer.normalize_trace(50.0, "cpu_trace", "cpu_util")
        assert val == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestJobDictBuilder
# ---------------------------------------------------------------------------

class TestJobDictBuilder:
    def test_build_basic_jobs(self, builder):
        df = pd.DataFrame({
            "job_id": [1001, 1002],
            "num_nodes": [4, 8],
            "submit_time": ["2024-01-01 08:00:00", "2024-01-01 09:00:00"],
            "start_time": ["2024-01-01 08:30:00", "2024-01-01 09:30:00"],
            "end_time": ["2024-01-01 09:30:00", "2024-01-01 10:30:00"],
            "exit_code": [0, 1],
            "user": ["alice", "bob"],
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "submit_time": "submit_time",
                "start_time": "start_time",
                "end_time": "end_time",
                "end_state": "exit_code",
                "account": "user",
            },
            confidence={k: 0.95 for k in [
                "id", "nodes_required", "submit_time", "start_time",
                "end_time", "end_state", "account",
            ]},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)

        assert len(jobs) == 2
        assert jobs[0]["id"] == "1001"
        assert jobs[0]["nodes_required"] == 4
        assert jobs[0]["account"] == "alice"
        assert jobs[0]["end_state"] == "COMPLETED"
        assert jobs[1]["end_state"] == "FAILED"

        # Times should be relative (submit of job 1 is ref = 0)
        assert jobs[0]["submit_time"] == pytest.approx(0.0)
        assert jobs[0]["start_time"] == pytest.approx(1800.0)  # 30 min later
        assert jobs[0]["end_time"] == pytest.approx(5400.0)     # 90 min later

    def test_skip_zero_nodes(self, builder):
        df = pd.DataFrame({
            "job_id": [1],
            "num_nodes": [0],
            "start_time": ["2024-01-01 08:00:00"],
            "end_time": ["2024-01-01 09:00:00"],
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "start_time": "start_time",
                "end_time": "end_time",
            },
            confidence={},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)
        assert len(jobs) == 0

    def test_skip_start_after_end(self, builder):
        df = pd.DataFrame({
            "job_id": [1],
            "num_nodes": [4],
            "start_time": ["2024-01-01 10:00:00"],
            "end_time": ["2024-01-01 08:00:00"],  # end before start
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "start_time": "start_time",
                "end_time": "end_time",
            },
            confidence={},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)
        assert len(jobs) == 0

    def test_all_canonical_fields_present(self, builder):
        df = pd.DataFrame({
            "job_id": [1],
            "num_nodes": [2],
            "start_time": ["2024-01-01 08:00:00"],
            "end_time": ["2024-01-01 09:00:00"],
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "start_time": "start_time",
                "end_time": "end_time",
            },
            confidence={},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)
        assert len(jobs) == 1
        job = jobs[0]

        # Check all canonical fields exist with defaults
        canonical_names = {fs.canonical_name for fs in FIELD_REGISTRY}
        for name in canonical_names:
            assert name in job, f"Missing canonical field: {name}"

        # Check RAPS trace-metadata fields are present
        raps_fields = [
            "current_state", "current_run_time", "trace_time",
            "trace_start_time", "trace_end_time", "trace_quanta",
            "trace_missing_values", "allocated_cpu_cores",
            "allocated_gpu_units", "downscale", "dilated",
        ]
        for name in raps_fields:
            assert name in job, f"Missing RAPS metadata field: {name}"

        # partition should be int, not str
        assert isinstance(job["partition"], int)

    def test_duration_string_parsing(self, builder):
        df = pd.DataFrame({
            "job_id": [1],
            "num_nodes": [2],
            "start_time": ["2024-01-01 08:00:00"],
            "end_time": ["2024-01-01 09:00:00"],
            "walltime": ["1:30:00"],
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "start_time": "start_time",
                "end_time": "end_time",
                "time_limit": "walltime",
            },
            confidence={},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)
        assert jobs[0]["time_limit"] == pytest.approx(5400.0)

    def test_node_list_parsing(self, builder):
        df = pd.DataFrame({
            "job_id": [1],
            "num_nodes": [3],
            "start_time": ["2024-01-01 08:00:00"],
            "end_time": ["2024-01-01 09:00:00"],
            "nodelist": ["nid001,nid002,nid003"],
        })
        mapping = ColumnMapping(
            mapped={
                "id": "job_id",
                "nodes_required": "num_nodes",
                "start_time": "start_time",
                "end_time": "end_time",
                "scheduled_nodes": "nodelist",
            },
            confidence={},
            unmapped_columns=[],
            missing_required=[],
        )
        jobs = builder.build_jobs(df, mapping)
        assert len(jobs[0]["scheduled_nodes"]) == 3
        assert all(isinstance(v, int) for v in jobs[0]["scheduled_nodes"])


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestParseDuration:
    def test_seconds(self):
        assert _parse_duration(3600) == 3600.0

    def test_hhmmss(self):
        assert _parse_duration("1:30:00") == 5400.0

    def test_dhhmmss(self):
        assert _parse_duration("1-02:00:00") == 93600.0

    def test_mmss(self):
        assert _parse_duration("30:00") == 1800.0

    def test_string_number(self):
        assert _parse_duration("7200") == 7200.0

    def test_invalid(self):
        assert _parse_duration("not_a_duration") == 0.0


class TestNormalizeState:
    def test_completed(self):
        assert _normalize_state("completed") == "COMPLETED"

    def test_failed(self):
        assert _normalize_state("FAILED") == "FAILED"

    def test_exit_code_zero(self):
        assert _normalize_state(0) == "COMPLETED"

    def test_exit_code_nonzero(self):
        assert _normalize_state(1) == "FAILED"

    def test_timeout(self):
        assert _normalize_state("TIMEOUT") == "TIMEOUT"

    def test_cancelled(self):
        assert _normalize_state("cancelled") == "CANCELLED"
        assert _normalize_state("canceled") == "CANCELLED"

    def test_none(self):
        assert _normalize_state(None) == "COMPLETED"


class TestParseNodeList:
    def test_comma_separated(self):
        result = _parse_node_list("n1,n2,n3")
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_space_separated(self):
        result = _parse_node_list("n1 n2 n3")
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_single(self):
        result = _parse_node_list("nid00042")
        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_list_input(self):
        result = _parse_node_list(["a", "b"])
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_numeric_nodes(self):
        assert _parse_node_list("1,2,3") == [1, 2, 3]
        assert _parse_node_list([1, 2]) == [1, 2]

    def test_none(self):
        assert _parse_node_list(None) == []

    def test_nan(self):
        assert _parse_node_list(float("nan")) == []


class TestParseTraceString:
    def test_json_list(self):
        assert _parse_trace_string("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]

    def test_comma_separated(self):
        assert _parse_trace_string("1.0, 2.0") == [1.0, 2.0]

    def test_scalar(self):
        assert _parse_trace_string("42.5") == 42.5

    def test_empty(self):
        assert _parse_trace_string("") == 0
