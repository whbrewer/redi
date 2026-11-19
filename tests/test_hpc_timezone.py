"""Timestamps and date windows must not depend on the machine's timezone when `timezone` is configured."""
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from redi.domains.hpc import _get_date_filter_step, _parse_date_range, get_ingest_steps
from redi.pipeline import create_context

LA = ZoneInfo("America/Los_Angeles")
ROOT = str(Path(__file__).resolve().parent.parent)


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc).timestamp()


def test_date_strings_are_read_in_the_configured_zone():
    t0, t1 = _parse_date_range("2017-10-03", "2017-10-04T12:00", LA)
    assert t0 == _utc(2017, 10, 3, 7)            # midnight PDT = 07:00 UTC
    assert t1 == _utc(2017, 10, 4, 19)
    assert _parse_date_range("1507014000", None, LA)[0] == 1507014000.0     # bare epoch seconds untouched


def test_epochs_do_not_depend_on_the_host_timezone():
    code = ("from zoneinfo import ZoneInfo; from redi.domains.hpc import _parse_date_range;"
            "print(_parse_date_range('2017-10-03', '2017-10-04', ZoneInfo('America/Los_Angeles')))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "TZ": tz, "PYTHONPATH": ROOT}).stdout
            for tz in ("America/New_York", "Asia/Tokyo", "UTC")}
    assert len(outs) == 1 and outs.pop().strip()
    legacy = "from redi.domains.hpc import _parse_date_range; print(_parse_date_range('2017-10-03', None))"
    legacy_outs = {subprocess.run([sys.executable, "-c", legacy], capture_output=True, text=True,
                                  env={**os.environ, "TZ": tz, "PYTHONPATH": ROOT}).stdout
                   for tz in ("America/New_York", "Asia/Tokyo")}
    assert len(legacy_outs) == 2                  # documents why the key exists: unset = machine-dependent


def test_philly_parser_uses_the_configured_zone():
    step = next(s for s in get_ingest_steps() if s.name == "load_philly")
    step._tz = LA
    assert step._parse_ts("2017-10-03 22:39:55") == 1507095595.0     # 05:39:55 UTC on Oct 4
    step._tz = ZoneInfo("UTC")
    assert step._parse_ts("2017-10-03 22:39:55") == _utc(2017, 10, 3, 22, 39, 55)


def test_host_local_fallback_warns_once(caplog):
    import redi.domains.hpc as hpc
    hpc._TZ_WARNED.discard("host_tz")
    with caplog.at_level(logging.WARNING):
        _parse_date_range("2017-10-03", "2017-10-04")
        _parse_date_range("2017-10-03", "2017-10-04")
    assert sum("HOST's local timezone" in r.message for r in caplog.records) == 1


def _filter(df, **config):
    ctx = create_context(input_path="x", output_dir=".", domain="hpc", **config)
    ctx.data = df
    ctx.metadata["raps_column_mapping"] = {}
    return _get_date_filter_step().execute(ctx)


def test_date_filter_handles_timezone_aware_columns():
    # tz-aware UTC timestamps (Marconi100's layout) used to raise "Cannot subtract tz-naive and tz-aware"
    t = pd.to_datetime(["2020-10-09 23:30", "2020-10-10 00:10", "2020-10-10 23:50", "2020-10-11 00:05"], utc=True)
    df = pd.DataFrame({"job": range(4), "submit_time": t, "start_time": t})
    ctx = _filter(df, start="2020-10-10", end="2020-10-11", timezone="UTC", date_filter_field="start_time")
    assert not ctx.errors, ctx.errors
    assert list(ctx.data["job"]) == [1, 2]


def test_date_filter_reads_naive_columns_in_the_configured_zone():
    # wall-clock times in Los Angeles; the window is 2017-10-03 LA time
    t = pd.to_datetime(["2017-10-02 23:30", "2017-10-03 00:10", "2017-10-03 23:50", "2017-10-04 00:05"])
    df = pd.DataFrame({"job": range(4), "submit_time": t, "start_time": t})
    ctx = _filter(df, start="2017-10-03", end="2017-10-04", timezone="America/Los_Angeles")
    assert list(ctx.data["job"]) == [1, 2]
