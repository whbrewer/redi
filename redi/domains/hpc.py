"""
DRAI HPC/Operational Domain Module

Domain-specific preprocessing for HPC job accounting,
system logs, and operational data.

Tools: pandas, scikit-learn, polars
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ["pandas", "scikit-learn", "polars"]

# Common HPC job fields
HPC_FIELDS = {
    "identifiers": ["jobid", "job_id", "job", "user", "project", "account", "allocation"],
    "resources": ["nodes", "cores", "cpus", "gpus", "memory", "mem"],
    "timing": ["walltime", "runtime", "elapsed", "wait_time", "queue_time", "submit", "start", "end"],
    "status": ["exit_code", "exitcode", "status", "state"],
    "queue": ["queue", "partition", "qos"],
    "system": ["system", "machine", "cluster", "host"],
}

# Categorical fields that need encoding
CATEGORICAL_FIELDS = ["user", "project", "queue", "partition", "status", "state", "system", "machine"]


def _parse_date_range(start_str, end_str):
    """Parse start/end strings to Unix epoch seconds (floats).

    Accepts ISO format (2017-10-03T00:00, 2017-10-03, 20170328) or bare floats.
    Returns (ts_start, ts_end) where either may be None if the input was None.
    """
    from datetime import datetime

    def _parse(s):
        if s is None:
            return None
        s = str(s).strip().strip('"').strip("'")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).timestamp()
            except ValueError:
                pass
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Cannot parse date: {s!r}. Use ISO format e.g. 2017-10-03T00:00")

    return _parse(start_str), _parse(end_str)


def get_ingest_steps():
    """Get HPC-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.ingest import DetectFormat

    class LoadHPCData(PipelineStep):
        """Load HPC job accounting data"""

        name = "load_hpc"
        stage = PipelineStage.INGEST
        description = "Load HPC job accounting CSV"

        def validate(self, ctx: PipelineContext) -> bool:
            # Domain check is optional since it may not be set during ingest
            fmt_ok = ctx.metadata.get("file_format") in ["csv", "parquet"]
            domain_ok = ctx.domain in ["hpc", "general"] or not ctx.domain
            return fmt_ok and domain_ok

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]
                file_format = ctx.metadata.get("file_format")

                if len(paths) == 1:
                    # Single file
                    if file_format == "parquet":
                        df = pd.read_parquet(paths[0])
                    else:
                        df = pd.read_csv(paths[0], low_memory=False)
                else:
                    # Multiple files - concatenate
                    dfs = []
                    for i, path in enumerate(paths):
                        logger.debug(f"Loading file {i+1}/{len(paths)}: {path.name}")
                        if file_format == "parquet":
                            dfs.append(pd.read_parquet(path))
                        else:
                            dfs.append(pd.read_csv(path, low_memory=False))
                    df = pd.concat(dfs, axis=0, ignore_index=True)
                    logger.info(f"Concatenated {len(paths)} files")

                ctx.data = df

                # Identify field types
                cols_lower = {c.lower(): c for c in df.columns}
                identified = {}

                for field_type, keywords in HPC_FIELDS.items():
                    for keyword in keywords:
                        for col_lower, col in cols_lower.items():
                            if keyword in col_lower:
                                identified.setdefault(field_type, []).append(col)

                ctx.metadata["identified_fields"] = identified
                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)

                # Detect potential target columns (runtime prediction)
                runtime_cols = identified.get("timing", [])
                if runtime_cols:
                    ctx.metadata["target_candidates"] = runtime_cols

                logger.info(f"Loaded HPC data: {len(df)} jobs, {len(identified)} field types identified")

            except Exception as e:
                ctx.add_error(f"Error loading HPC data: {str(e)}")

            return ctx

    class LoadTorqueData(PipelineStep):
        """Load PBS/Torque accounting logs from a directory layout."""

        name = "load_torque"
        stage = PipelineStage.INGEST
        description = "Load PBS/Torque accounting logs"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "torque-pbs"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                root = ctx.input_path

                # Locate log files: prefer torque_logs/ subdir, fall back to root
                torque_dir = root / "torque_logs"
                log_dir = torque_dir if torque_dir.is_dir() else root
                log_files = sorted(
                    f for f in log_dir.iterdir()
                    if f.is_file() and f.name.isdigit() and len(f.name) == 8
                )

                if not log_files:
                    ctx.add_error(f"No PBS accounting log files found in {log_dir}")
                    return ctx

                logger.debug(f"     Found {len(log_files)} PBS log files in {log_dir}")

                records = []
                skipped = 0
                for log_file in log_files:
                    with open(log_file, errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(";", 3)
                            if len(parts) < 4 or parts[1] != "E":
                                skipped += 1
                                continue
                            _, _, job_id, attrs_str = parts
                            record = {"jobid": job_id}
                            for token in attrs_str.split():
                                if "=" in token:
                                    key, _, val = token.partition("=")
                                    record[key.lower()] = val
                            records.append(record)

                if not records:
                    ctx.add_error("No completed job records (event type 'E') found in PBS logs")
                    return ctx

                df = pd.DataFrame(records)
                ctx.data = df

                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)
                ctx.metadata["columns"] = list(df.columns)
                ctx.metadata["log_files_parsed"] = len(log_files)
                ctx.metadata["records_skipped"] = skipped

                logger.info(
                    f"Loaded PBS/Torque logs: {len(df):,} completed jobs from "
                    f"{len(log_files)} files ({skipped:,} non-E records skipped)"
                )

            except Exception as e:
                ctx.add_error(f"Error loading PBS/Torque logs: {str(e)}")

            return ctx

    class LoadGCloudV2Data(PipelineStep):
        """Load Google Cluster Trace v2 from its multi-table directory layout.

        Joins job_events, task_events, machine_events, and task_usage to produce
        a flat DataFrame with nodes_required and per-job CPU traces, matching
        the logic from raps/dataloaders/gcloudv2.py.
        """

        name = "load_gcloud_v2"
        stage = PipelineStage.INGEST
        description = "Load Google Cluster Trace v2"

        # Column schemas for each sub-table (headerless CSVs)
        _COLUMNS = {
            "job_events": [
                "timestamp", "missing_info", "job_ID", "event_type",
                "user_name", "scheduling_class", "job_name", "logical_job_name",
            ],
            "machine_events": [
                "timestamp", "machine_ID", "event_type", "platform_ID",
                "CPU_capacity", "memory_capacity",
            ],
            "task_events": [
                "timestamp", "missing_info", "job_ID", "task_index",
                "machine_ID", "event_type", "user_name", "scheduling_class",
                "priority", "CPU_request", "memory_request",
                "disk_space_request", "different_machine_constraint",
            ],
            "task_usage": [
                "start_time", "end_time", "job_ID", "task_index", "machine_ID",
                "CPU_usage_rate", "memory_usage_avg", "memory_usage_max",
                "assigned_memory", "unmapped_page_cache_memory",
                "page_cache_memory", "maximum_memory_usage",
                "disk_IO_time_avg", "disk_IO_time_max", "local_disk_space_used",
                "cycles_per_instruction", "memory_accesses_per_instruction",
                "sampling_rate", "aggregation_type", "missing_col_19",
            ],
        }

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "gcloud-v2"

        def _load_table(self, base: Path, table: str) -> "pd.DataFrame":
            import pandas as pd
            table_dir = base / table
            files = sorted(table_dir.glob("part-*.csv.gz"))
            if not files:
                raise FileNotFoundError(f"No part files in {table_dir}")
            names = self._COLUMNS[table]
            dfs = [pd.read_csv(f, compression="gzip", header=None,
                               names=names, dtype={names[0]: float})
                   for f in files]
            return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np
                import pandas as pd

                base = ctx.input_path
                logger.debug(f"     Loading Google Cluster v2 from {base}")

                # 1. Machine capacity (mode of CPU/memory across all machines)
                machine_df = self._load_table(base, "machine_events")
                cpu_cap = machine_df["CPU_capacity"].mode()[0]
                mem_cap = machine_df["memory_capacity"].mode()[0]
                logger.debug(f"     Machine capacity: cpu={cpu_cap}, mem={mem_cap}")

                # 2. nodes_required per job from task_events
                task_df = self._load_table(base, "task_events")
                task_df = task_df[task_df["event_type"] == 0]
                job_res = task_df.groupby("job_ID").agg(
                    CPU_request=("CPU_request", "sum"),
                    memory_request=("memory_request", "sum"),
                ).reset_index()
                job_res["nodes_required"] = np.maximum(
                    np.ceil(job_res["CPU_request"] / cpu_cap),
                    np.ceil(job_res["memory_request"] / mem_cap),
                ).astype(int)
                job_res = job_res[job_res["nodes_required"] > 0]
                nodes_map = job_res.set_index("job_ID")["nodes_required"].to_dict()
                logger.debug(f"     {len(nodes_map):,} jobs with resource requests")

                # 3. Job submit events (event_type=0), timestamps μs → s
                job_df = self._load_table(base, "job_events")
                job_df = job_df[job_df["event_type"] == 0].copy()
                job_df["timestamp"] = job_df["timestamp"].astype(float) / 1e6
                t0 = job_df["timestamp"].min()

                # 4. Task usage → per-job start/end times and CPU traces
                usage_df = self._load_table(base, "task_usage")
                usage_df["start_time"] = usage_df["start_time"].astype(float) / 1e6
                usage_df["end_time"] = usage_df["end_time"].astype(float) / 1e6
                usage_df["job_ID"] = usage_df["job_ID"].astype(int)

                start_map = usage_df.groupby("job_ID")["start_time"].min().to_dict()
                end_map   = usage_df.groupby("job_ID")["end_time"].max().to_dict()
                trace_map = (usage_df.groupby("job_ID")["CPU_usage_rate"]
                             .apply(lambda s: s.to_numpy()).to_dict())

                # 5. Filter jobs that have both usage data and resource requests
                valid_ids = set(trace_map) & set(nodes_map)
                job_df = job_df[job_df["job_ID"].isin(valid_ids)].copy()
                logger.debug(f"     {len(job_df):,} jobs after filtering to valid set")

                # 6. Build flat DataFrame
                job_df["submit_time"] = job_df["timestamp"] - t0
                job_df["start_time"]  = job_df["job_ID"].map(start_map) - t0
                job_df["end_time"]    = job_df["job_ID"].map(end_map) - t0
                job_df["nodes_required"] = job_df["job_ID"].map(nodes_map)
                job_df["cpu_trace"]   = job_df["job_ID"].map(trace_map)

                cols = [
                    "job_ID", "user_name", "scheduling_class", "job_name",
                    "submit_time", "start_time", "end_time",
                    "nodes_required", "cpu_trace",
                ]
                ctx.data = job_df[cols].reset_index(drop=True)

                ctx.metadata["num_rows"] = len(ctx.data)
                ctx.metadata["num_columns"] = len(ctx.data.columns)
                ctx.metadata["columns"] = list(ctx.data.columns)

                logger.info(
                    f"Loaded Google Cluster v2: {len(ctx.data):,} jobs "
                    f"({len(valid_ids):,} with usage+resources)"
                )

            except Exception as e:
                ctx.add_error(f"Error loading Google Cluster v2 data: {str(e)}")

            return ctx

    class LoadPhillyData(PipelineStep):
        """Load Microsoft Philly GPU cluster traces.

        Reads cluster_job_log (JSON), flattens nested attempts to extract
        start/end times and nodes_required, then joins per-machine CPU/GPU
        utilization traces from cpu_by_day/ and gpu_by_day/ if present.
        """

        name = "load_philly"
        stage = PipelineStage.INGEST
        description = "Load Philly GPU cluster traces"

        _DATE_FMT = "%Y-%m-%d %H:%M:%S"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "philly"

        def _parse_ts(self, val) -> float:
            """Convert Philly timestamp (str or epoch) to epoch seconds."""
            if val is None or val == "None":
                return None
            if isinstance(val, (int, float)):
                return float(val)
            val = val.replace(" PST", "").replace(" PDT", "")
            try:
                from datetime import datetime
                return datetime.strptime(val, self._DATE_FMT).timestamp()
            except ValueError:
                return None

        def _load_daily_traces(self, trace_dir: Path, colname: str,
                               min_date=None, max_date=None) -> dict:
            """Load daily CSVs → dict of date-string → DataFrame.

            Args:
                min_date: datetime.date — skip files strictly before this date
                max_date: datetime.date — skip files strictly after this date
            """
            import pandas as pd
            from datetime import datetime as _dt
            traces = {}
            for f in sorted(trace_dir.glob("*.csv")):
                if min_date is not None or max_date is not None:
                    try:
                        file_date = _dt.strptime(f.stem, "%Y-%m-%d").date()
                        if min_date is not None and file_date < min_date:
                            continue
                        if max_date is not None and file_date > max_date:
                            continue
                    except ValueError:
                        pass  # unparseable filename — load anyway
                try:
                    df = pd.read_csv(f, dtype={"machine_id": str, colname: float})
                    df["time"] = pd.to_datetime(df["time"], errors="coerce",
                                                format=self._DATE_FMT)
                    traces[f.stem] = df  # stem = "2017-10-03"
                except Exception:
                    pass
            return traces

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np
                import pandas as pd
                from datetime import datetime, timedelta

                trace_dir = ctx.input_path
                job_log_path = trace_dir / "cluster_job_log"

                logger.debug(f"     Loading Philly job log from {job_log_path}")
                import json
                with open(job_log_path, encoding="utf-8") as f:
                    raw_jobs = json.load(f)

                logger.debug(f"     {len(raw_jobs):,} raw jobs in log")

                # Flatten job records
                records = []
                for raw in raw_jobs:
                    attempts = raw.get("attempts", [])
                    if not attempts:
                        continue

                    # Use first attempt's start, last attempt's end
                    start_ts = self._parse_ts(attempts[0].get("start_time"))
                    end_ts   = self._parse_ts(attempts[-1].get("end_time"))
                    submitted_ts = self._parse_ts(raw.get("submitted_time"))

                    if not start_ts or not end_ts:
                        continue

                    # Collect machines and GPU count from last attempt
                    detail = attempts[-1].get("detail", [])
                    machine_ids = [d["ip"] for d in detail if "ip" in d]
                    num_gpus = sum(len(d.get("gpus", [])) for d in detail)

                    if not machine_ids:
                        continue

                    records.append({
                        "jobid":          raw.get("jobid"),
                        "user":           raw.get("user"),
                        "status":         raw.get("status"),
                        "vc":             raw.get("vc"),
                        "submitted_time": submitted_ts,
                        "start_time":     start_ts,
                        "end_time":       end_ts,
                        "nodes_required": len(machine_ids),
                        "num_gpus":       num_gpus,
                        "_machine_ids":   machine_ids,
                    })

                if not records:
                    ctx.add_error("No valid Philly job records found")
                    return ctx

                df = pd.DataFrame(records)
                logger.debug(f"     {len(df):,} jobs with valid attempts")

                # Apply date range filter before the expensive per-job trace loop
                _ds = ctx.config.get('start')
                _de = ctx.config.get('end')
                ts0, ts1 = None, None
                if _ds or _de:
                    try:
                        ts0, ts1 = _parse_date_range(_ds, _de)
                        filter_col = "submitted_time" if "submitted_time" in df.columns else "start_time"
                        pre = len(df)
                        if ts0 is not None:
                            df = df[df[filter_col] >= ts0]
                        if ts1 is not None:
                            df = df[df[filter_col] < ts1]
                        df = df.reset_index(drop=True)
                        logger.info(f"Date filter [{_ds} – {_de}]: {pre:,} → {len(df):,} jobs")
                    except Exception as _e:
                        logger.warning(f"Date filter failed: {_e}")

                # Load CPU/GPU traces if directories exist
                cpu_dir = trace_dir / "cpu_by_day"
                gpu_dir = trace_dir / "gpu_by_day"
                has_traces = cpu_dir.is_dir() and gpu_dir.is_dir()

                if has_traces and not df.empty:
                    # Compute the date window for trace loading.
                    # If the user specified a date range, clamp to that window
                    # (jobs may run much longer than the requested window; loading
                    # all their trace files would be extremely slow).
                    from datetime import date as _date, timedelta as _td
                    _ts0 = ts0 if (_ds or _de) else None
                    _ts1 = ts1 if (_ds or _de) else None
                    if _ts0 is not None:
                        _min_date = datetime.fromtimestamp(_ts0).date() - _td(days=1)
                    else:
                        _min_date = (datetime.fromtimestamp(df["start_time"].min()).date()
                                     - _td(days=1))
                    if _ts1 is not None:
                        _max_date = datetime.fromtimestamp(_ts1).date() + _td(days=1)
                    else:
                        _max_date = (datetime.fromtimestamp(df["end_time"].max()).date()
                                     + _td(days=1))
                    logger.debug(
                        f"     Loading CPU/GPU daily traces "
                        f"[{_min_date} – {_max_date}]..."
                    )
                    all_cpu = self._load_daily_traces(cpu_dir, "cpu_util",
                                                      min_date=_min_date, max_date=_max_date)
                    all_gpu = self._load_daily_traces(gpu_dir, "gpu_util",
                                                      min_date=_min_date, max_date=_max_date)
                    logger.debug(f"     Loaded {len(all_cpu)} CPU days, {len(all_gpu)} GPU days")

                    # Pre-concat all daily frames then build a machine_id → DataFrame
                    # index so per-job lookup only touches that machine's rows.
                    def _build_machine_index(daily_dict):
                        if not daily_dict:
                            return {}
                        combined = pd.concat(list(daily_dict.values()), ignore_index=True)
                        return {mid: grp.reset_index(drop=True)
                                for mid, grp in combined.groupby("machine_id")}

                    cpu_by_machine = _build_machine_index(all_cpu)
                    gpu_by_machine = _build_machine_index(all_gpu)

                    def extract_trace(by_machine, colname, machines, start_dt, end_dt):
                        sub_frames = [by_machine[m] for m in machines if m in by_machine]
                        if not sub_frames:
                            return np.array([], dtype=float)
                        sub = (pd.concat(sub_frames, ignore_index=True)
                               if len(sub_frames) > 1 else sub_frames[0])
                        mask = (sub["time"] >= start_dt) & (sub["time"] <= end_dt)
                        filtered = sub.loc[mask]
                        if filtered.empty:
                            return np.array([], dtype=float)
                        if len(machines) > 1:
                            filtered = (filtered.groupby("time")[colname]
                                        .mean().reset_index())
                        return filtered[colname].to_numpy() * 0.01

                    cpu_traces, gpu_traces = [], []
                    for _, row in df.iterrows():
                        machines = row["_machine_ids"]
                        start_dt = datetime.fromtimestamp(row["start_time"])
                        end_dt   = datetime.fromtimestamp(row["end_time"])
                        cpu_traces.append(extract_trace(cpu_by_machine, "cpu_util",
                                                        machines, start_dt, end_dt))
                        gpu_traces.append(extract_trace(gpu_by_machine, "gpu_util",
                                                        machines, start_dt, end_dt))

                    df["cpu_trace"] = cpu_traces
                    df["gpu_trace"] = gpu_traces
                    logger.debug(f"     Traces attached to {len(df):,} jobs")

                # Drop internal helper column
                df = df.drop(columns=["_machine_ids"])
                ctx.data = df

                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)
                ctx.metadata["columns"] = list(df.columns)
                ctx.metadata["has_traces"] = has_traces

                logger.info(
                    f"Loaded Philly: {len(df):,} jobs"
                    + (" with CPU/GPU traces" if has_traces else " (no traces)")
                )

            except Exception as e:
                ctx.add_error(f"Error loading Philly data: {str(e)}")

            return ctx

    return [DetectFormat(), LoadHPCData(), LoadTorqueData(), LoadGCloudV2Data(), LoadPhillyData()]


def get_preprocess_steps():
    """Get HPC-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CleanHPCData(PipelineStep):
        """Clean HPC job data"""

        name = "clean_hpc"
        stage = PipelineStage.PREPROCESS
        description = "Clean HPC job records"

        def __init__(self, remove_failed: bool = False):
            super().__init__()
            self.remove_failed = remove_failed

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                initial_count = len(df)

                # Remove jobs with zero/negative numeric timing values
                # (skip datetime columns — they are timestamps, not durations)
                timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                for col in timing_cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        df = df[df[col] > 0]

                # Optionally remove failed jobs
                if self.remove_failed:
                    status_cols = ctx.metadata.get("identified_fields", {}).get("status", [])
                    for col in status_cols:
                        if col in df.columns:
                            # Keep exit_code == 0
                            if "exit" in col.lower():
                                df = df[df[col] == 0]

                # Remove duplicates
                df = df.drop_duplicates()

                ctx.data = df
                ctx.metadata["rows_removed"] = initial_count - len(df)
                logger.info(f"Cleaned HPC data: {initial_count} -> {len(df)} jobs")

            except Exception as e:
                ctx.add_error(f"Error cleaning HPC data: {str(e)}")

            return ctx

    class ParseTimestamps(PipelineStep):
        """Parse timestamp columns"""

        name = "parse_timestamps"
        stage = PipelineStage.PREPROCESS
        description = "Parse and standardize timestamps"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                timestamp_keywords = ["submit", "start", "end", "time", "date"]

                parsed = []
                for col in df.columns:
                    if any(kw in col.lower() for kw in timestamp_keywords):
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                parsed.append(col)
                            except Exception:
                                pass

                ctx.data = df
                ctx.metadata["parsed_timestamps"] = parsed
                logger.info(f"Parsed {len(parsed)} timestamp columns")

            except Exception as e:
                ctx.add_error(f"Error parsing timestamps: {str(e)}")

            return ctx

    return [CleanHPCData(), ParseTimestamps()]


def get_transform_steps():
    """Get HPC-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class EngineerHPCFeatures(PipelineStep):
        """Engineer HPC-specific features"""

        name = "engineer_hpc_features"
        stage = PipelineStage.TRANSFORM
        description = "Create derived HPC features"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                created = []
                cols_lower = {c.lower(): c for c in df.columns}

                # Compute wait time if submit and start exist
                if "submit" in cols_lower and "start" in cols_lower:
                    submit_col = cols_lower["submit"]
                    start_col = cols_lower["start"]
                    if pd.api.types.is_datetime64_any_dtype(df[submit_col]):
                        df["wait_time_seconds"] = (df[start_col] - df[submit_col]).dt.total_seconds()
                        created.append("wait_time_seconds")

                # Compute core-hours
                timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                resource_cols = ctx.metadata.get("identified_fields", {}).get("resources", [])

                for time_col in timing_cols:
                    if "runtime" in time_col.lower() or "elapsed" in time_col.lower():
                        for res_col in resource_cols:
                            if "core" in res_col.lower() or "cpu" in res_col.lower():
                                # Runtime in seconds, cores -> core-hours
                                df["core_hours"] = df[time_col] * df[res_col] / 3600
                                created.append("core_hours")
                                break
                        break

                # Node utilization (if nodes requested vs allocated)
                if "nodes" in cols_lower:
                    nodes_col = cols_lower["nodes"]
                    if "cores" in cols_lower:
                        cores_col = cols_lower["cores"]
                        # Approximate cores per node
                        df["cores_per_node"] = df[cores_col] / df[nodes_col].replace(0, 1)
                        created.append("cores_per_node")

                # Time features from submit time
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[f"{col}_hour"] = df[col].dt.hour
                        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                        df[f"{col}_month"] = df[col].dt.month
                        created.extend([f"{col}_hour", f"{col}_dayofweek", f"{col}_month"])

                ctx.data = df
                ctx.metadata["engineered_features"] = created
                logger.info(f"Engineered {len(created)} HPC features")

            except Exception as e:
                ctx.add_error(f"Error engineering features: {str(e)}")

            return ctx

    class EncodeHPCCategoricals(PipelineStep):
        """Encode categorical HPC fields"""

        name = "encode_hpc_categoricals"
        stage = PipelineStage.TRANSFORM
        description = "Encode categorical fields (user, queue, etc.)"

        def __init__(self, method: str = "label", max_categories: int = 100):
            super().__init__()
            self.method = method
            self.max_categories = max_categories

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                encoded_cols = []

                for col in df.columns:
                    # Check if it's a categorical column
                    if df[col].dtype == 'object':
                        n_unique = df[col].nunique()

                        if n_unique > self.max_categories:
                            # High cardinality - use frequency encoding
                            freq = df[col].value_counts(normalize=True)
                            df[f"{col}_freq"] = df[col].map(freq)
                            df = df.drop(columns=[col])
                            encoded_cols.append(f"{col}_freq (frequency)")
                        elif self.method == "label":
                            df[col], _ = pd.factorize(df[col])
                            encoded_cols.append(f"{col} (label)")
                        elif self.method == "onehot" and n_unique <= 20:
                            dummies = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                            encoded_cols.append(f"{col} (onehot)")

                ctx.data = df
                ctx.metadata["encoded_categoricals"] = encoded_cols
                logger.info(f"Encoded {len(encoded_cols)} categorical columns")

            except Exception as e:
                ctx.add_error(f"Error encoding categoricals: {str(e)}")

            return ctx

    class NormalizeHPCFeatures(PipelineStep):
        """Normalize numerical features"""

        name = "normalize_hpc"
        stage = PipelineStage.TRANSFORM
        description = "Normalize HPC numerical features"

        def __init__(self, method: str = "zscore"):
            super().__init__()
            self.method = method

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Exclude datetime-derived features from normalization
                exclude_patterns = ["_hour", "_dayofweek", "_month", "exit"]
                feature_cols = [c for c in numeric_cols
                               if not any(p in c.lower() for p in exclude_patterns)]

                stats = {}
                for col in feature_cols:
                    if self.method == "zscore":
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[col] = (df[col] - mean) / std
                        stats[col] = {"mean": float(mean), "std": float(std)}
                    elif self.method == "log":
                        # Log transform for skewed distributions
                        if df[col].min() >= 0:
                            df[col] = np.log1p(df[col])
                            stats[col] = {"transform": "log1p"}

                ctx.data = df
                ctx.metadata["normalization_stats"] = stats
                logger.info(f"Normalized {len(stats)} features")

            except Exception as e:
                ctx.add_error(f"Error normalizing: {str(e)}")

            return ctx

    return [EngineerHPCFeatures(), EncodeHPCCategoricals(), NormalizeHPCFeatures()]


def get_structure_steps():
    """Get HPC-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class PrepareForML(PipelineStep):
        """Prepare data for ML model training"""

        name = "prepare_hpc_ml"
        stage = PipelineStage.STRUCTURE
        description = "Prepare HPC data for ML training"

        def __init__(self, target_column: Optional[str] = None):
            super().__init__()
            self.target_column = target_column

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data

                # Determine target column
                target = self.target_column
                if not target:
                    # Default to first timing column (runtime prediction)
                    timing_cols = ctx.metadata.get("identified_fields", {}).get("timing", [])
                    for col in timing_cols:
                        if "runtime" in col.lower() or "elapsed" in col.lower():
                            target = col
                            break

                # Get numeric feature columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Exclude non-feature columns
                exclude = ["jobid", "job_id", target] if target else ["jobid", "job_id"]
                feature_cols = [c for c in numeric_cols if c.lower() not in [e.lower() for e in exclude]]

                # Remove any columns with all NaN
                feature_cols = [c for c in feature_cols if df[c].notna().any()]

                # Fill remaining NaN with median
                for col in feature_cols:
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].median())

                X = df[feature_cols].values.astype(np.float32)

                result = {"X": X, "feature_names": feature_cols}

                if target and target in df.columns:
                    y = df[target].values.astype(np.float32)
                    result["y"] = y
                    ctx.metadata["target_column"] = target

                ctx.data = result
                ctx.metadata["array_shape"] = X.shape
                ctx.metadata["feature_columns"] = feature_cols
                logger.info(f"Prepared ML data: X shape {X.shape}, target={target}")

            except Exception as e:
                ctx.add_error(f"Error preparing for ML: {str(e)}")

            return ctx

    return [PrepareForML()]


def get_output_steps():
    """Get HPC-specific output steps"""
    from ..pipeline.output import get_output_steps
    return get_output_steps(output_format="npz")


def _get_raps_preprocess_steps():
    """Preprocess steps for RAPS job extraction.

    Mirrors the cleaning that RAPS dataloaders apply:
      1. Drop rows with missing start/end timestamps
      2. Deduplicate by job ID (keep last)
      3. Drop rows with null node counts

    Heavier validation (start <= end, wall_time > 0, nodes > 0) is handled
    downstream by the JobDictBuilder which has access to the mapped fields.
    """
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CleanHPCForRAPS(PipelineStep):
        """Clean HPC data for RAPS job extraction."""

        name = "clean_hpc_raps"
        stage = PipelineStage.PREPROCESS
        description = "Clean HPC data for RAPS job extraction"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                initial_count = len(df)

                # Drop rows with missing start timestamps (RAPS convention:
                # only require start_time, not end_time — NaN end_time jobs
                # are filtered later by JobDictBuilder)
                start_keywords = ["start"]
                start_cols = [
                    c for c in df.columns
                    if any(kw in c.lower() for kw in start_keywords)
                    and ("time" in c.lower() or "date" in c.lower()
                         or "stamp" in c.lower() or c.lower() in ("start", "sdt"))
                    and "batch" not in c.lower()  # exclude batch_time_start
                ]
                if start_cols:
                    df = df.dropna(subset=start_cols)

                # Deduplicate by job ID column (keep last, matching RAPS convention)
                id_keywords = ["job_id", "jobid", "jid", "id_job", "job_ID",
                               "primary_job_id", "allocation_id"]
                id_col = None
                for kw in id_keywords:
                    if kw in df.columns:
                        id_col = kw
                        break
                if id_col:
                    df = df.drop_duplicates(subset=id_col, keep="last")
                else:
                    df = df.drop_duplicates()

                ctx.data = df.reset_index(drop=True)
                removed = initial_count - len(df)
                if removed:
                    logger.info(f"RAPS clean: {initial_count} → {len(df)} jobs "
                                f"({removed} removed)")
                else:
                    logger.info(f"RAPS clean: {len(df)} jobs (no rows removed)")

            except Exception as e:
                ctx.add_error(f"Error cleaning HPC data for RAPS: {str(e)}")

            return ctx

    return [CleanHPCForRAPS()]


def _get_load_jobprofile_step(jobprofile_path=None):
    """Return a LoadJobProfile step (or empty list if no path given)."""
    if not jobprofile_path:
        return []

    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class LoadJobProfile(PipelineStep):
        """Load jobprofile telemetry data for per-timestep trace arrays."""

        name = "load_jobprofile"
        stage = PipelineStage.INGEST
        description = "Load jobprofile power telemetry"

        def __init__(self, path: str):
            super().__init__()
            self.path = path

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            import glob as _glob
            import pandas as pd

            pattern = self.path
            # Expand globs
            if '*' in pattern or '?' in pattern:
                files = sorted(_glob.glob(pattern, recursive=True))
            elif Path(pattern).is_dir():
                files = sorted(str(p) for p in Path(pattern).glob("*.parquet"))
            else:
                files = [pattern]

            if not files:
                logger.warning(f"No jobprofile files found at {pattern}")
                return ctx

            dfs = []
            for f in files:
                try:
                    dfs.append(pd.read_parquet(f))
                except Exception as e:
                    logger.warning(f"Failed to read jobprofile file {f}: {e}")
            if not dfs:
                logger.warning("No jobprofile data loaded")
                return ctx

            jp_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

            # Convert timestamp and sort
            if "timestamp" in jp_df.columns:
                jp_df["timestamp"] = pd.to_datetime(jp_df["timestamp"], errors="coerce")
                jp_df = jp_df.sort_values("timestamp").reset_index(drop=True)

            ctx.metadata["_raps_jobprofile_df"] = jp_df
            logger.info(f"Loaded jobprofile: {len(jp_df)} rows, "
                        f"{jp_df['allocation_id'].nunique() if 'allocation_id' in jp_df.columns else '?'} allocations")
            return ctx

    return [LoadJobProfile(jobprofile_path)]


def _get_date_filter_step():
    """Return a FilterByDateRange step that reads start/end from ctx.config.

    Runs after AutoMapColumns so column names are canonical (submit_time / start_time).
    Is a no-op when neither start nor end is set in the config.
    """
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class FilterByDateRange(PipelineStep):
        """Filter jobs to the configured date window."""

        name = "filter_date_range"
        stage = PipelineStage.PREPROCESS
        description = "Filter jobs by date range"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            import pandas as pd

            _ds = ctx.config.get('start')
            _de = ctx.config.get('end')
            if not _ds and not _de:
                return ctx
            if not isinstance(ctx.data, pd.DataFrame):
                return ctx

            try:
                ts0, ts1 = _parse_date_range(_ds, _de)
            except Exception as e:
                ctx.add_error(f"Date filter: {e}")
                return ctx

            df = ctx.data
            # Prefer canonical submit_time; fall back to start_time
            col = next((c for c in ("submit_time", "start_time") if c in df.columns), None)
            if col is None:
                logger.warning("Date filter: no submit_time/start_time column found")
                return ctx

            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                series = series.astype("int64") / 1e9  # ns → s
            else:
                series = pd.to_numeric(series, errors="coerce")

            pre = len(df)
            mask = pd.Series(True, index=df.index)
            if ts0 is not None:
                mask &= series >= ts0
            if ts1 is not None:
                mask &= series < ts1

            ctx.data = df[mask].reset_index(drop=True)
            logger.info(
                f"Date filter [{_ds} – {_de}] on {col}: {pre:,} → {len(ctx.data):,} jobs"
            )
            return ctx

    return FilterByDateRange()


def get_raps_pipeline(column_overrides=None, power_config=None, trace_quanta=None,
                      output_format="npz", system_name="unknown",
                      jobprofile_path=None, topology=None, priority_policy=None):
    """Get the RAPS Job extraction pipeline.

    Args:
        column_overrides: Dict mapping original_column_name → canonical_field_name
        power_config: Dict like {"cpu": {"idle": 100, "max": 500}} for power→utilization
        trace_quanta: Trace time-step size from RAPS scheduler config (seconds)
        output_format: Output format — npz (default), json, parquet, csv
        system_name: RAPS system name stored in the NPZ snapshot (e.g. "adastraMI250")
        jobprofile_path: Path/glob to jobprofile parquet files for per-timestep traces
        topology: System topology dict (sc_shape, blades_per_chassis, nodes_per_blade)
        priority_policy: Named priority policy to apply when priority is absent from
            the source data. Supported: "frontier" (aging boost by node count).
    """
    from .hpc_job_mapper import AutoMapColumns, DeriveJobFields, BuildJobDicts, SaveJobDicts

    ingest = get_ingest_steps()                          # DetectFormat + LoadHPCData
    jobprofile = _get_load_jobprofile_step(jobprofile_path)  # optional jobprofile load
    preprocess = _get_raps_preprocess_steps()             # RAPS-specific cleaning
    raps_steps = [
        AutoMapColumns(overrides=column_overrides),      # map columns → canonical fields
        _get_date_filter_step(),                         # filter by start/end (no-op if unset)
        DeriveJobFields(),                               # derive missing fields
        BuildJobDicts(power_config=power_config, trace_quanta=trace_quanta,
                      topology=topology, system_name=system_name,
                      priority_policy=priority_policy),
        SaveJobDicts(output_format=output_format, system_name=system_name),
    ]
    return ingest + jobprofile + preprocess + raps_steps


def get_full_pipeline(raps_jobs=False, **kwargs):
    """Get the complete HPC preprocessing pipeline.

    Args:
        raps_jobs: If True, return the RAPS Job extraction pipeline instead of
                   the default ML-prep pipeline.
        **kwargs: Passed to get_raps_pipeline() when raps_jobs=True
                  (column_overrides, power_config, output_format).
    """
    if raps_jobs:
        return get_raps_pipeline(**kwargs)
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
