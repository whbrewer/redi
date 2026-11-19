"""ML-path preprocess/transform steps: array cells, pandas string dtype, label codes, negative counts."""
import numpy as np
import pandas as pd

from redi.domains.hpc import get_preprocess_steps, get_transform_steps
from redi.domains.hpc.job_mapper import AutoMapColumns, BuildJobDicts, DeriveJobFields
from redi.pipeline import create_context


def _ctx(df):
    ctx = create_context(input_path="x", output_dir=".", domain="hpc")
    ctx.data = df
    return ctx


def _step(steps, cls_name):
    return next(s for s in steps if type(s).__name__ == cls_name)


def test_clean_step_survives_array_cells_and_drops_duplicates():
    df = pd.DataFrame({"job_id": [1, 1, 2, 3], "run": [10, 10, 20, 30],
                       "nodes": [np.array([1, 2]), np.array([1, 2]), np.array([3]), np.array([4])]})
    ctx = _step(get_preprocess_steps(), "CleanHPCData").execute(_ctx(df))
    assert not ctx.errors, ctx.errors          # used to be: unhashable type: numpy.ndarray (swallowed)
    assert len(ctx.data) == 3                   # the exact duplicate row went


def test_encoder_handles_string_dtype_and_skips_array_columns():
    df = pd.DataFrame({"user": pd.array(["a", "b", "a", "c"], dtype="string"),
                       "nodes": [np.array([1]), np.array([2]), np.array([3]), np.array([4])], "x": [1.0, 2.0, 3.0, 4.0]})
    ctx = _step(get_transform_steps(), "EncodeHPCCategoricals").execute(_ctx(df))
    assert not ctx.errors, ctx.errors
    assert list(ctx.data["user"]) == [0, 1, 0, 2]                  # encoded although dtype is not 'object'
    assert isinstance(ctx.data["nodes"].iloc[0], np.ndarray)        # array column left alone
    assert ctx.metadata["encoded_column_names"] == ["user"]


def test_normalizer_leaves_label_codes_alone():
    df = pd.DataFrame({"user": ["a", "b", "a", "c", "b", "a"], "nodes": [1, 2, 4, 8, 16, 32]})
    ctx = _ctx(df)
    for name in ("EncodeHPCCategoricals", "NormalizeHPCFeatures"):
        ctx = _step(get_transform_steps(), name).execute(ctx)
    assert not ctx.errors
    assert list(ctx.data["user"]) == [0, 1, 0, 2, 1, 0]             # codes, not z-scores
    assert abs(ctx.data["nodes"].mean()) < 1e-9 and abs(ctx.data["nodes"].std() - 1) < 1e-9
    assert "user" not in ctx.metadata["normalization_stats"] and "nodes" in ctx.metadata["normalization_stats"]


def test_negative_gpu_count_sentinel_becomes_zero_and_is_reported():
    n = 12
    df = pd.DataFrame({"job": range(n), "nodes": [1] * n, "gpus": [-1] * n,
                       "submit": pd.date_range("2024-01-01", periods=n, freq="min"),
                       "start": pd.date_range("2024-01-01 00:10", periods=n, freq="min"),
                       "end": pd.date_range("2024-01-01 01:00", periods=n, freq="min")})
    ov = {"job": "id", "nodes": "nodes_required", "gpus": "gpu_units_required", "submit": "submit_time",
          "start": "start_time", "end": "end_time"}
    ctx = _ctx(df)
    for step in (AutoMapColumns(overrides=ov), DeriveJobFields(), BuildJobDicts(system_name="t")):
        ctx = step.execute(ctx)
    assert all(j["gpu_units_required"] == 0 for j in ctx.data)
    assert any(n_["field"] == "gpu_units_required" and n_.get("rows") == 12 and "negative count" in n_["issue"]
               for n_ in ctx.metadata["field_sanity"])
