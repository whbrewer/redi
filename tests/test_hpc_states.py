"""Job-state normalization: SLURM numeric state codes, Philly vocabulary, exit-code legacy, duplicate overrides."""
import logging

import numpy as np
import pandas as pd
import pytest

from redi.domains.hpc.job_mapper import (
    ApplyColumnScalars, AutoMapColumns, BuildJobDicts, ColumnMapper, DeriveJobFields, _normalize_state,
)
from redi.pipeline import create_context


@pytest.mark.parametrize("code,expected", [(3, "COMPLETED"), (4, "CANCELLED"), (5, "FAILED"), (6, "TIMEOUT"),
                                           (7, "NODE_FAIL"), (11, "OUT_OF_MEMORY"), (1024, "FAILED")])
def test_slurm_state_codes(code, expected):
    assert _normalize_state(code, "slurm_code") == expected
    assert _normalize_state(float(code), "slurm_code") == expected       # pandas often gives floats


def test_numbers_are_still_exit_codes_by_default():
    assert _normalize_state(0) == "COMPLETED" and _normalize_state(256) == "FAILED" and _normalize_state(3) == "FAILED"


@pytest.mark.parametrize("raw,expected", [("Pass", "COMPLETED"), ("Failed", "FAILED"), ("Killed", "CANCELLED"),
                                          ("COMPLETED", "COMPLETED"), ("completed", "COMPLETED"), ("TIMEOUT", "TIMEOUT")])
def test_philly_and_slurm_words(raw, expected):
    assert _normalize_state(raw) == expected


def test_state_column_wins_over_exit_code_end_to_end():
    # SLURM records CANCELLED (4) and TIMEOUT (6) jobs with exit_code 0
    n = 6
    df = pd.DataFrame({"job_id": range(1, n + 1), "nodes": [1] * n,
                       "submit": pd.date_range("2024-01-01 08:00", periods=n, freq="min"),
                       "start": pd.date_range("2024-01-01 08:10", periods=n, freq="min"),
                       "end": pd.date_range("2024-01-01 09:00", periods=n, freq="min"),
                       "state": [3, 4, 5, 6, 3, 11], "exit_code": [0, 0, 256, 0, 0, 0]})
    ov = {"job_id": "id", "nodes": "nodes_required", "submit": "submit_time", "start": "start_time",
          "end": "end_time", "state": "end_state"}
    ctx = create_context(input_path="x", output_dir=".", domain="hpc"); ctx.data = df
    for step in (AutoMapColumns(overrides=ov), DeriveJobFields(), BuildJobDicts(system_name="t", state_encoding="slurm_code")):
        ctx = step.execute(ctx)
    assert [j["end_state"] for j in ctx.data] == ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "COMPLETED", "OUT_OF_MEMORY"]


def test_two_overrides_to_one_field_warn(caplog):
    with caplog.at_level(logging.WARNING):
        res = ColumnMapper(overrides={"state": "end_state", "exit_code": "end_state"}).map_columns(["job_id", "state", "exit_code"])
    assert res.mapped["end_state"] == "exit_code"                       # later wins (unchanged behaviour) ...
    assert any("Two overrides map to 'end_state'" in r.message for r in caplog.records)   # ... but not silently
