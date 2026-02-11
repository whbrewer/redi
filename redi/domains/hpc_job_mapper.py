"""
RAPS Job Mapper - Auto-extract RAPS Job objects from HPC telemetry data.

Takes arbitrary HPC telemetry DataFrames and automatically maps columns to
RAPS Job fields, derives missing fields, normalizes traces, and outputs
RAPS-compatible Job dictionaries — without needing per-system custom dataloaders.

Pipeline steps (activated by --raps-jobs flag):
  AutoMapColumns  → maps arbitrary columns to canonical Job fields
  DeriveJobFields → computes missing fields from available data
  BuildJobDicts   → assembles RAPS job_dict per row, normalizes traces
  SaveJobDicts    → serializes as JSON + optional pickle
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..pipeline import PipelineStep, PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Field Registry
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """Specification for a single RAPS Job field."""
    canonical_name: str
    aliases: List[str]
    field_type: str  # "int", "str", "timestamp", "trace", "state", "list", "float"
    required: bool = False
    default: Any = None


FIELD_REGISTRY: List[FieldSpec] = [
    FieldSpec(
        canonical_name="id",
        aliases=["job_id", "id_job", "jobid", "jid", "primary_job_id", "job_id_field",
                 "job_ID", "allocation_id"],
        field_type="str",
        required=True,
    ),
    FieldSpec(
        canonical_name="name",
        aliases=["name", "job_name", "jobname", "jnam", "name_job"],
        field_type="str",
        required=False,
        default="",
    ),
    FieldSpec(
        canonical_name="account",
        aliases=["account", "user", "user_id", "id_user", "usr",
                 "hashed_user_id", "user_name"],
        field_type="str",
        required=False,
        default="unknown",
    ),
    FieldSpec(
        canonical_name="nodes_required",
        aliases=["nodes_required", "node_count", "num_nodes", "num_nodes_alloc",
                 "nodes_alloc", "nnumr", "nodect"],
        field_type="int",
        required=True,
    ),
    FieldSpec(
        canonical_name="submit_time",
        aliases=["submit_time", "time_submit", "time_submission", "adt",
                 "submitted_time", "qtime", "job_submit_time", "queued_timestamp"],
        field_type="timestamp",
        required=False,
    ),
    FieldSpec(
        canonical_name="start_time",
        aliases=["start_time", "time_start", "sdt", "begin_time",
                 "start_timestamp", "start"],
        field_type="timestamp",
        required=True,
    ),
    FieldSpec(
        canonical_name="end_time",
        aliases=["end_time", "time_end", "edt", "end_timestamp", "end"],
        field_type="timestamp",
        required=True,
    ),
    FieldSpec(
        canonical_name="time_limit",
        aliases=["time_limit", "timelimit", "walltime", "elpl"],
        field_type="float",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="expected_run_time",
        aliases=["run_time", "runtime", "runtime_seconds", "wall_time",
                 "duration", "elapsed"],
        field_type="float",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="cpu_trace",
        aliases=["cpu_trace", "cpu_utilisation", "cpu_util", "cpu_usage",
                 "CPU_usage_rate", "cpu_power_consumption", "sum_cpu0_power",
                 "perf1", "avgpcon"],
        field_type="trace",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="gpu_trace",
        aliases=["gpu_trace", "gpu_util", "gpu_usage", "gpu_power",
                 "sum_gpu_power", "gpu_power_consumption", "gpu_energy"],
        field_type="trace",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="ntx_trace",
        aliases=["ntx_trace", "ib_tx", "net_tx", "ipogif0_tx_bytes"],
        field_type="trace",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="nrx_trace",
        aliases=["nrx_trace", "ib_rx", "net_rx", "ipogif0_rx_bytes"],
        field_type="trace",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="end_state",
        aliases=["end_state", "job_state", "state_current", "state_end",
                 "exit_status", "status", "exit_code"],
        field_type="state",
        required=False,
        default="COMPLETED",
    ),
    FieldSpec(
        canonical_name="priority",
        aliases=["priority", "pri", "scheduling_class"],
        field_type="int",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="partition",
        aliases=["partition", "queue", "qos"],
        field_type="str",
        required=False,
        default="",
    ),
    FieldSpec(
        canonical_name="scheduled_nodes",
        aliases=["scheduled_nodes", "nodes", "xnames", "exec_host",
                 "nodelist", "machineId"],
        field_type="list",
        required=False,
        default=[],
    ),
    FieldSpec(
        canonical_name="cpu_cores_required",
        aliases=["cpu_cores_required", "cpus", "num_cpus", "cores", "ppn"],
        field_type="int",
        required=False,
        default=0,
    ),
    FieldSpec(
        canonical_name="gpu_units_required",
        aliases=["gpu_units_required", "gpus", "num_gpus", "gres_gpu"],
        field_type="int",
        required=False,
        default=0,
    ),
]

# Pre-build lookup: alias (lowercase) → FieldSpec
_ALIAS_TO_FIELD: Dict[str, FieldSpec] = {}
for _fs in FIELD_REGISTRY:
    for _alias in _fs.aliases:
        _ALIAS_TO_FIELD[_alias.lower()] = _fs
    # Also add canonical name itself
    _ALIAS_TO_FIELD[_fs.canonical_name.lower()] = _fs


# ---------------------------------------------------------------------------
# 2. ColumnMapper
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """Result of column-to-canonical mapping."""
    mapped: Dict[str, str]            # canonical_name → original column name
    confidence: Dict[str, float]      # canonical_name → confidence score
    unmapped_columns: List[str]       # columns that didn't map to anything
    missing_required: List[str]       # required canonical fields with no mapping


class ColumnMapper:
    """Multi-phase column matching algorithm.

    Phases:
      1. Manual overrides (from config) — confidence 1.0
      2. Exact match (case-insensitive against alias list) — confidence 0.95
      3. Token match (split on separators, compute overlap) — confidence 0.5–0.8
      4. Fuzzy match (SequenceMatcher ratio) — confidence 0.6+
    """

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        """
        Args:
            overrides: Dict mapping original_column_name → canonical_field_name.
        """
        self.overrides = overrides or {}

    def map_columns(self, columns: List[str]) -> ColumnMapping:
        """Map DataFrame columns to canonical RAPS Job field names."""
        mapped: Dict[str, str] = {}          # canonical → original col
        confidence: Dict[str, float] = {}
        used_columns: set = set()

        # Phase 1: manual overrides
        for col, canonical in self.overrides.items():
            if col in columns:
                mapped[canonical] = col
                confidence[canonical] = 1.0
                used_columns.add(col)

        # Phase 2: exact match (case-insensitive)
        for col in columns:
            if col in used_columns:
                continue
            col_lower = col.lower()
            if col_lower in _ALIAS_TO_FIELD:
                fs = _ALIAS_TO_FIELD[col_lower]
                if fs.canonical_name not in mapped:
                    mapped[fs.canonical_name] = col
                    confidence[fs.canonical_name] = 0.95
                    used_columns.add(col)

        # Phase 3: token match
        for col in columns:
            if col in used_columns:
                continue
            col_tokens = set(_tokenize(col.lower()))
            if not col_tokens:
                continue

            best_score = 0.0
            best_field: Optional[FieldSpec] = None
            for fs in FIELD_REGISTRY:
                if fs.canonical_name in mapped:
                    continue
                for alias in fs.aliases:
                    alias_tokens = set(_tokenize(alias.lower()))
                    if not alias_tokens:
                        continue
                    overlap = col_tokens & alias_tokens
                    if overlap:
                        score = len(overlap) / max(len(col_tokens), len(alias_tokens))
                        # Scale to 0.5–0.8 range
                        score = 0.5 + score * 0.3
                        if score > best_score:
                            best_score = score
                            best_field = fs

            if best_field and best_score >= 0.55:
                mapped[best_field.canonical_name] = col
                confidence[best_field.canonical_name] = best_score
                used_columns.add(col)

        # Phase 4: fuzzy match (remaining columns only)
        for col in columns:
            if col in used_columns:
                continue
            col_lower = col.lower()

            best_ratio = 0.0
            best_field_f: Optional[FieldSpec] = None
            for fs in FIELD_REGISTRY:
                if fs.canonical_name in mapped:
                    continue
                for alias in fs.aliases:
                    ratio = difflib.SequenceMatcher(None, col_lower, alias.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_field_f = fs

            if best_field_f and best_ratio >= 0.6:
                mapped[best_field_f.canonical_name] = col
                confidence[best_field_f.canonical_name] = best_ratio
                used_columns.add(col)

        # Compute unmapped and missing
        unmapped_columns = [c for c in columns if c not in used_columns]
        missing_required = [
            fs.canonical_name
            for fs in FIELD_REGISTRY
            if fs.required and fs.canonical_name not in mapped
        ]

        return ColumnMapping(
            mapped=mapped,
            confidence=confidence,
            unmapped_columns=unmapped_columns,
            missing_required=missing_required,
        )


def _tokenize(s: str) -> List[str]:
    """Split a string into tokens on common separators."""
    return [t for t in re.split(r'[_\-.\s]+', s) if t]


# ---------------------------------------------------------------------------
# 3. FieldDeriver
# ---------------------------------------------------------------------------

class FieldDeriver:
    """Derives missing canonical fields from available data."""

    def derive(self, df: pd.DataFrame, mapping: ColumnMapping) -> Tuple[pd.DataFrame, ColumnMapping]:
        """Add derived columns and update the mapping.

        Returns:
            Updated (df, mapping) with derived columns added.
        """
        mapped = dict(mapping.mapped)
        conf = dict(mapping.confidence)

        # expected_run_time ← end_time - start_time
        if "expected_run_time" not in mapped and "start_time" in mapped and "end_time" in mapped:
            start_col = mapped["start_time"]
            end_col = mapped["end_time"]
            derived_col = "_derived_expected_run_time"
            start = pd.to_datetime(df[start_col], errors="coerce")
            end = pd.to_datetime(df[end_col], errors="coerce")
            df[derived_col] = (end - start).dt.total_seconds()
            mapped["expected_run_time"] = derived_col
            conf["expected_run_time"] = 0.9
            logger.info("Derived expected_run_time from end_time - start_time")

        # name ← f"job_{id}"
        if "name" not in mapped and "id" in mapped:
            id_col = mapped["id"]
            derived_col = "_derived_name"
            df[derived_col] = "job_" + df[id_col].astype(str)
            mapped["name"] = derived_col
            conf["name"] = 0.8
            logger.info("Derived name from id")

        # nodes_required ← len(scheduled_nodes)
        if "nodes_required" not in mapped and "scheduled_nodes" in mapped:
            sn_col = mapped["scheduled_nodes"]
            derived_col = "_derived_nodes_required"
            df[derived_col] = df[sn_col].apply(_count_nodes)
            mapped["nodes_required"] = derived_col
            conf["nodes_required"] = 0.85
            logger.info("Derived nodes_required from scheduled_nodes")

        # submit_time ← start_time (fallback)
        if "submit_time" not in mapped and "start_time" in mapped:
            mapped["submit_time"] = mapped["start_time"]
            conf["submit_time"] = 0.5
            logger.info("Derived submit_time as fallback from start_time")

        # Update missing_required
        missing_required = [
            fs.canonical_name
            for fs in FIELD_REGISTRY
            if fs.required and fs.canonical_name not in mapped
        ]

        new_mapping = ColumnMapping(
            mapped=mapped,
            confidence=conf,
            unmapped_columns=mapping.unmapped_columns,
            missing_required=missing_required,
        )
        return df, new_mapping


def _count_nodes(val) -> int:
    """Count nodes from a node-list string or list."""
    if pd.isna(val) or val is None:
        return 0
    if isinstance(val, (list, tuple)):
        return len(val)
    s = str(val).strip()
    if not s:
        return 0
    # Comma-separated: "node001,node002,node003"
    if "," in s:
        return len([n for n in s.split(",") if n.strip()])
    # Space-separated
    parts = s.split()
    if len(parts) > 1:
        return len(parts)
    return 1


# ---------------------------------------------------------------------------
# 4. TraceNormalizer
# ---------------------------------------------------------------------------

class TraceNormalizer:
    """Detects trace data format and converts to RAPS-compatible utilization (0–1)."""

    def __init__(self, power_config: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            power_config: Optional dict like {"cpu": {"idle": 100, "max": 500}, ...}
        """
        self.power_config = power_config or {}

    def normalize_trace(self, value: Any, canonical_name: str, col_name: str) -> Any:
        """Normalize a single trace value to 0-1 scale.

        Args:
            value: The raw trace value (scalar, list, or string)
            canonical_name: e.g. "cpu_trace", "gpu_trace"
            col_name: Original column name (for power detection)

        Returns:
            Normalized value (scalar or list of floats in 0-1 range)
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0

        # Parse string representations of lists
        if isinstance(value, str):
            value = _parse_trace_string(value)

        # Single scalar
        if isinstance(value, (int, float)):
            return self._normalize_scalar(float(value), canonical_name, col_name)

        # Array/list
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float)
            if len(arr) == 0:
                return 0
            return self._normalize_array(arr, canonical_name, col_name).tolist()

        return 0

    def _detect_format(self, values: np.ndarray, canonical_name: str, col_name: str) -> str:
        """Detect trace format: 'power', 'percentage', 'utilization', or 'unknown'."""
        col_lower = col_name.lower()

        # Check column name for power indicators
        power_keywords = ["power", "watt", "energy", "pcon"]
        if any(kw in col_lower for kw in power_keywords):
            return "power"

        # Check value range
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return "unknown"

        max_val = float(np.max(valid))

        if max_val > 100:
            return "power"
        elif max_val > 1.0:
            return "percentage"
        elif max_val <= 1.0:
            return "utilization"

        return "unknown"

    def _normalize_scalar(self, value: float, canonical_name: str, col_name: str) -> float:
        """Normalize a single scalar trace value."""
        fmt = self._detect_format(np.array([value]), canonical_name, col_name)
        return self._apply_normalization(value, fmt, canonical_name, col_name)

    def _normalize_array(self, arr: np.ndarray, canonical_name: str, col_name: str) -> np.ndarray:
        """Normalize an array of trace values."""
        fmt = self._detect_format(arr, canonical_name, col_name)
        return np.array([self._apply_normalization(v, fmt, canonical_name, col_name) for v in arr])

    def _apply_normalization(self, value: float, fmt: str, canonical_name: str, col_name: str) -> float:
        """Apply normalization based on detected format."""
        if np.isnan(value):
            return 0.0

        # Determine trace type key for power_config
        trace_type = canonical_name.replace("_trace", "")  # "cpu", "gpu", etc.

        if fmt == "power":
            cfg = self.power_config.get(trace_type, {})
            idle = cfg.get("idle", 0)
            max_p = cfg.get("max", 0)
            if max_p > idle:
                normalized = (value - idle) / (max_p - idle)
            else:
                # No config — can't accurately normalize power, return 0
                normalized = 0.0
            return float(np.clip(normalized, 0.0, 1.0))

        elif fmt == "percentage":
            return float(np.clip(value / 100.0, 0.0, 1.0))

        elif fmt == "utilization":
            return float(np.clip(value, 0.0, 1.0))

        return 0.0


def _parse_trace_string(s: str) -> Any:
    """Parse a string that might represent a trace (list or scalar)."""
    s = s.strip()
    if not s:
        return 0

    # Try JSON parse: "[1, 2, 3]"
    if s.startswith("["):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try comma-separated numbers: "1.0, 2.0, 3.0"
    if "," in s:
        try:
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        except ValueError:
            pass

    # Try single number
    try:
        return float(s)
    except ValueError:
        pass

    return 0


# ---------------------------------------------------------------------------
# 5. JobDictBuilder
# ---------------------------------------------------------------------------

# RAPS end-state mapping
_STATE_MAP = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "timeout": "TIMEOUT",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
    "node_fail": "NODE_FAIL",
    "oom": "OUT_OF_MEMORY",
    "out_of_memory": "OUT_OF_MEMORY",
    "running": "RUNNING",
    "pending": "PENDING",
}


class JobDictBuilder:
    """Builds RAPS-compatible job_dict dictionaries from mapped DataFrame rows."""

    def __init__(self, power_config: Optional[Dict[str, Dict[str, float]]] = None):
        self.trace_normalizer = TraceNormalizer(power_config=power_config)

    def build_jobs(self, df: pd.DataFrame, mapping: ColumnMapping) -> List[Dict[str, Any]]:
        """Build list of RAPS job dicts from a DataFrame.

        Args:
            df: DataFrame with columns that have been mapped via ColumnMapper
            mapping: The column mapping result

        Returns:
            List of RAPS-compatible job dictionaries
        """
        jobs: List[Dict[str, Any]] = []
        mapped = mapping.mapped

        # Find reference time (earliest submit or start time)
        ref_time = self._find_reference_time(df, mapped)

        skipped = 0
        for idx, row in df.iterrows():
            try:
                job = self._build_one(row, mapped, ref_time)
                if job is not None:
                    jobs.append(job)
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                skipped += 1

        if skipped:
            logger.info(f"Skipped {skipped} invalid rows")
        logger.info(f"Built {len(jobs)} RAPS job dicts")

        return jobs

    def _find_reference_time(self, df: pd.DataFrame, mapped: Dict[str, str]) -> float:
        """Find the earliest timestamp to use as reference (epoch seconds)."""
        ref = None
        for time_field in ["submit_time", "start_time"]:
            if time_field in mapped:
                col = mapped[time_field]
                series = pd.to_datetime(df[col], errors="coerce")
                valid = series.dropna()
                if not valid.empty:
                    epoch = valid.min().timestamp()
                    if ref is None or epoch < ref:
                        ref = epoch
        return ref if ref is not None else 0.0

    def _build_one(self, row: pd.Series, mapped: Dict[str, str],
                   ref_time: float) -> Optional[Dict[str, Any]]:
        """Build a single job dict from a DataFrame row."""
        job: Dict[str, Any] = {}

        for fs in FIELD_REGISTRY:
            canon = fs.canonical_name
            if canon in mapped:
                raw = row.get(mapped[canon])
                value = self._convert_value(raw, fs, mapped.get(canon, ""), ref_time)
            else:
                value = fs.default
                # Deep copy mutable defaults
                if isinstance(value, list):
                    value = list(value)

            job[canon] = value

        # Validation: skip jobs with nodes_required <= 0
        if job.get("nodes_required") is not None and job["nodes_required"] <= 0:
            logger.debug(f"Skipping job {job.get('id', '?')}: nodes_required={job['nodes_required']}")
            return None

        # Validation: skip if start > end
        if (job.get("start_time") is not None and job.get("end_time") is not None
                and isinstance(job["start_time"], (int, float))
                and isinstance(job["end_time"], (int, float))
                and job["start_time"] > job["end_time"]):
            logger.debug(f"Skipping job {job.get('id', '?')}: start_time > end_time")
            return None

        # Add RAPS trace-metadata fields required by raps.job.Job / job_dict
        start = job.get("start_time", 0) or 0
        end = job.get("end_time", 0) or 0
        submit = job.get("submit_time", 0) or 0

        job["current_state"] = "COMPLETED"  # historical data → all completed
        job["current_run_time"] = 0
        job["trace_time"] = submit  # same convention as RAPS dataloaders
        job["trace_start_time"] = start
        job["trace_end_time"] = end
        job["trace_quanta"] = 1
        job["trace_missing_values"] = False
        job["allocated_cpu_cores"] = job.get("cpu_cores_required", 0)
        job["allocated_gpu_units"] = job.get("gpu_units_required", 0)
        job["downscale"] = 1
        job["dilated"] = False

        # Ensure partition is int (RAPS expects int | None, not str)
        part = job.get("partition", 0)
        if isinstance(part, str):
            try:
                job["partition"] = int(part) if part else 0
            except ValueError:
                job["partition"] = 0

        return job

    def _convert_value(self, raw: Any, fs: FieldSpec, col_name: str,
                       ref_time: float) -> Any:
        """Convert a raw value to the expected type for a canonical field."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            default = fs.default
            if isinstance(default, list):
                return list(default)
            return default

        if fs.field_type == "int":
            try:
                return int(float(raw))
            except (ValueError, TypeError):
                return fs.default

        if fs.field_type == "float":
            return _parse_duration(raw) if fs.canonical_name in ("time_limit", "expected_run_time") else _to_float(raw, fs.default)

        if fs.field_type == "str":
            return str(raw)

        if fs.field_type == "timestamp":
            return self._convert_timestamp(raw, ref_time)

        if fs.field_type == "trace":
            return self.trace_normalizer.normalize_trace(raw, fs.canonical_name, col_name)

        if fs.field_type == "state":
            return _normalize_state(raw)

        if fs.field_type == "list":
            return _parse_node_list(raw)

        return raw

    def _convert_timestamp(self, raw: Any, ref_time: float) -> float:
        """Convert a timestamp value to relative seconds from ref_time."""
        # Already numeric (epoch seconds)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val = float(raw)
            # If value is very large, assume it's epoch seconds
            if val > 1e9:
                return val - ref_time
            # Otherwise assume it's already relative
            return val

        # Pandas Timestamp
        if isinstance(raw, pd.Timestamp):
            if pd.isna(raw):
                return 0.0
            return raw.timestamp() - ref_time

        # String → parse as datetime
        if isinstance(raw, str):
            try:
                ts = pd.to_datetime(raw)
                return ts.timestamp() - ref_time
            except (ValueError, TypeError):
                pass
            # Try as duration string
            dur = _parse_duration(raw)
            if dur > 0:
                return dur

        return 0.0


def _to_float(val: Any, default: Any = 0) -> float:
    """Safely convert to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default) if default is not None else 0.0


def _parse_duration(val: Any) -> float:
    """Parse duration string (HH:MM:SS, D-HH:MM:SS, or seconds) to seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0

    s = val.strip()

    # Pure numeric
    try:
        return float(s)
    except ValueError:
        pass

    # D-HH:MM:SS
    m = re.match(r'^(\d+)-(\d+):(\d+):(\d+)$', s)
    if m:
        days, hours, mins, secs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return days * 86400 + hours * 3600 + mins * 60 + secs

    # HH:MM:SS
    m = re.match(r'^(\d+):(\d+):(\d+)$', s)
    if m:
        hours, mins, secs = int(m[1]), int(m[2]), int(m[3])
        return hours * 3600 + mins * 60 + secs

    # MM:SS
    m = re.match(r'^(\d+):(\d+)$', s)
    if m:
        mins, secs = int(m[1]), int(m[2])
        return mins * 60 + secs

    return 0.0


def _normalize_state(val: Any) -> str:
    """Normalize job state to RAPS format."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "COMPLETED"

    s = str(val).strip().lower()

    # Direct lookup
    if s in _STATE_MAP:
        return _STATE_MAP[s]

    # Numeric exit codes: 0 → COMPLETED, else FAILED
    try:
        code = int(float(s))
        return "COMPLETED" if code == 0 else "FAILED"
    except (ValueError, TypeError):
        pass

    # Partial match
    for key, raps_state in _STATE_MAP.items():
        if key in s:
            return raps_state

    return s.upper() if s else "COMPLETED"


def _parse_node_list(val: Any) -> List[int]:
    """Parse a node list from various formats into integer node indices.

    RAPS requires ``scheduled_nodes`` to be ``list[int]`` (node indices)
    or an empty list.  When the raw data contains hostname strings we
    hash them to stable integer IDs so RAPS can consume them.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        items = list(val)
    else:
        s = str(val).strip()
        if not s:
            return []
        # Comma-separated: "node001,node002,node003"
        if "," in s:
            items = [n.strip() for n in s.split(",") if n.strip()]
        # Space-separated
        elif len(s.split()) > 1:
            items = s.split()
        else:
            items = [s]

    # Convert each item to int — if it's already numeric use it,
    # otherwise hash the hostname string to a stable integer index.
    result: List[int] = []
    for item in items:
        try:
            result.append(int(float(item)))
        except (ValueError, TypeError):
            # Hash hostname to a stable positive int
            result.append(abs(hash(str(item))) % (10**9))
    return result


# ---------------------------------------------------------------------------
# 6. RAPS Compatibility
# ---------------------------------------------------------------------------

def get_job_class():
    """Try to import RAPS Job class; fall back to local dict builder."""
    try:
        from raps.job import Job, job_dict
        return Job, job_dict
    except ImportError:
        return None, _local_job_dict


def _local_job_dict(**kwargs) -> Dict[str, Any]:
    """Local fallback that produces the same dict structure as raps.job.job_dict."""
    return {
        "id": kwargs.get("id", ""),
        "name": kwargs.get("name", ""),
        "account": kwargs.get("account", "unknown"),
        "nodes_required": kwargs.get("nodes_required", 1),
        # Allocation
        "current_state": kwargs.get("current_state", "COMPLETED"),
        "end_state": kwargs.get("end_state", "COMPLETED"),
        "scheduled_nodes": kwargs.get("scheduled_nodes", []),
        "priority": kwargs.get("priority", 0),
        "partition": kwargs.get("partition", 0),
        # Resource requests and allocations
        "cpu_cores_required": kwargs.get("cpu_cores_required", 0),
        "gpu_units_required": kwargs.get("gpu_units_required", 0),
        "allocated_cpu_cores": kwargs.get("allocated_cpu_cores", 0),
        "allocated_gpu_units": kwargs.get("allocated_gpu_units", 0),
        # Traces
        "cpu_trace": kwargs.get("cpu_trace", 0),
        "gpu_trace": kwargs.get("gpu_trace", 0),
        "ntx_trace": kwargs.get("ntx_trace", []),
        "nrx_trace": kwargs.get("nrx_trace", []),
        # Times
        "submit_time": kwargs.get("submit_time", 0),
        "time_limit": kwargs.get("time_limit", 0),
        "start_time": kwargs.get("start_time", 0),
        "end_time": kwargs.get("end_time", 0),
        "expected_run_time": kwargs.get("expected_run_time", 0),
        "current_run_time": kwargs.get("current_run_time", 0),
        "trace_time": kwargs.get("trace_time", 0),
        "trace_start_time": kwargs.get("trace_start_time", 0),
        "trace_end_time": kwargs.get("trace_end_time", 0),
        "trace_quanta": kwargs.get("trace_quanta", 1),
        "trace_missing_values": kwargs.get("trace_missing_values", False),
        "dilated": kwargs.get("dilated", False),
        "downscale": kwargs.get("downscale", 1),
    }


# ---------------------------------------------------------------------------
# 7. Pipeline Steps
# ---------------------------------------------------------------------------

class AutoMapColumns(PipelineStep):
    """Map arbitrary HPC telemetry columns to canonical RAPS Job field names."""

    name = "auto_map_columns"
    stage = PipelineStage.INGEST
    description = "Map columns to RAPS Job fields"

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        super().__init__()
        self.mapper = ColumnMapper(overrides=overrides)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not isinstance(ctx.data, pd.DataFrame):
            ctx.add_error("AutoMapColumns requires a DataFrame")
            return ctx

        mapping = self.mapper.map_columns(list(ctx.data.columns))
        ctx.metadata["raps_column_mapping"] = mapping.mapped
        ctx.metadata["raps_mapping_confidence"] = mapping.confidence
        ctx.metadata["raps_unmapped_columns"] = mapping.unmapped_columns
        ctx.metadata["raps_missing_required"] = mapping.missing_required
        ctx.metadata["_raps_mapping_obj"] = mapping  # internal use

        logger.info(
            f"Mapped {len(mapping.mapped)}/{len(FIELD_REGISTRY)} fields "
            f"(unmapped cols: {len(mapping.unmapped_columns)}, "
            f"missing required: {len(mapping.missing_required)})"
        )

        if mapping.missing_required:
            logger.warning(f"Missing required fields (will attempt derivation): {mapping.missing_required}")

        return ctx


class DeriveJobFields(PipelineStep):
    """Compute missing canonical fields from available data."""

    name = "derive_job_fields"
    stage = PipelineStage.TRANSFORM
    description = "Derive missing RAPS Job fields"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not isinstance(ctx.data, pd.DataFrame):
            ctx.add_error("DeriveJobFields requires a DataFrame")
            return ctx

        mapping: Optional[ColumnMapping] = ctx.metadata.get("_raps_mapping_obj")
        if mapping is None:
            ctx.add_error("DeriveJobFields: no column mapping found (run AutoMapColumns first)")
            return ctx

        deriver = FieldDeriver()
        ctx.data, new_mapping = deriver.derive(ctx.data, mapping)

        ctx.metadata["raps_column_mapping"] = new_mapping.mapped
        ctx.metadata["raps_mapping_confidence"] = new_mapping.confidence
        ctx.metadata["raps_missing_required"] = new_mapping.missing_required
        ctx.metadata["_raps_mapping_obj"] = new_mapping

        if new_mapping.missing_required:
            logger.warning(f"Still missing required fields after derivation: {new_mapping.missing_required}")

        return ctx


class BuildJobDicts(PipelineStep):
    """Assemble RAPS job_dict per row, normalize traces."""

    name = "build_job_dicts"
    stage = PipelineStage.STRUCTURE
    description = "Build RAPS Job dictionaries"

    def __init__(self, power_config: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__()
        self.power_config = power_config

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not isinstance(ctx.data, pd.DataFrame):
            ctx.add_error("BuildJobDicts requires a DataFrame")
            return ctx

        mapping: Optional[ColumnMapping] = ctx.metadata.get("_raps_mapping_obj")
        if mapping is None:
            ctx.add_error("BuildJobDicts: no column mapping found")
            return ctx

        builder = JobDictBuilder(power_config=self.power_config)
        jobs = builder.build_jobs(ctx.data, mapping)

        ctx.data = jobs
        ctx.metadata["raps_job_count"] = len(jobs)

        return ctx


class SaveJobDicts(PipelineStep):
    """Save RAPS job dicts in the requested format.

    Supported formats:
    - **npz** (default): RAPS-compatible snapshot that ``raps run -f <file>``
      can load directly via ``Telemetry.load_snapshot()``.
    - **json**: Human-readable JSON array of job dicts.
    - **parquet**: Columnar format via pandas.
    - **csv**: Flat CSV via pandas.

    A JSON sidecar and mapping-metadata file are always written alongside
    the primary output.
    """

    name = "save_job_dicts"
    stage = PipelineStage.OUTPUT
    description = "Save RAPS Job dictionaries"

    def __init__(self, output_format: str = "npz", system_name: str = "unknown"):
        super().__init__()
        self.output_format = output_format.lower() if output_format else "npz"
        self.system_name = system_name

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not isinstance(ctx.data, list):
            ctx.add_error("SaveJobDicts: expected list of job dicts")
            return ctx

        output_dir = Path(ctx.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        jobs = ctx.data
        saved_files: List[str] = []

        # --- Compute telemetry envelope ---
        telemetry_start = 0
        telemetry_end = 0
        for j in jobs:
            end = j.get("end_time", 0) or 0
            if isinstance(end, (int, float)) and end > telemetry_end:
                telemetry_end = int(end)

        fmt = self.output_format

        # --- Primary output ---
        if fmt == "npz":
            self._save_npz(jobs, output_dir, telemetry_start, telemetry_end, saved_files)
        elif fmt == "json":
            self._save_json(jobs, output_dir, saved_files)
        elif fmt == "parquet":
            self._save_parquet(jobs, output_dir, saved_files)
        elif fmt == "csv":
            self._save_csv(jobs, output_dir, saved_files)
        else:
            logger.warning(f"Unknown output format '{fmt}', falling back to npz")
            self._save_npz(jobs, output_dir, telemetry_start, telemetry_end, saved_files)

        # --- Save mapping metadata (always) ---
        meta_path = output_dir / "raps_mapping_metadata.json"
        meta = {
            "column_mapping": ctx.metadata.get("raps_column_mapping", {}),
            "mapping_confidence": ctx.metadata.get("raps_mapping_confidence", {}),
            "unmapped_columns": ctx.metadata.get("raps_unmapped_columns", []),
            "missing_required": ctx.metadata.get("raps_missing_required", []),
            "job_count": ctx.metadata.get("raps_job_count", 0),
            "telemetry_start": telemetry_start,
            "telemetry_end": telemetry_end,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        saved_files.append(str(meta_path))

        ctx.metadata["saved_files"] = saved_files
        primary = saved_files[0] if saved_files else output_dir
        logger.info(f"Saved {len(jobs)} RAPS jobs to {primary}")

        return ctx

    # ---- format-specific writers ----

    def _save_npz(self, jobs, output_dir, telemetry_start, telemetry_end, saved_files):
        """RAPS-compatible NPZ snapshot."""
        from datetime import datetime, timezone
        from types import SimpleNamespace

        start_date = datetime.now(timezone.utc)

        Job_cls, _ = get_job_class()
        if Job_cls is not None:
            job_objs = []
            for jd in jobs:
                try:
                    job_objs.append(Job_cls(jd))
                except Exception as e:
                    logger.debug(f"Skipping job {jd.get('id', '?')} in RAPS Job creation: {e}")
            snapshot_jobs = np.array([vars(j) for j in job_objs], dtype=object)
        else:
            snapshot_jobs = np.array(jobs, dtype=object)

        # RAPS load_from_files() accesses args_from_file.system_name
        args = SimpleNamespace(system=self.system_name, system_name=self.system_name)

        npz_path = output_dir / "raps_jobs.npz"
        np.savez_compressed(
            str(npz_path),
            jobs=snapshot_jobs,
            telemetry_start=telemetry_start,
            telemetry_end=telemetry_end,
            start_date=start_date,
            args=args,
        )
        saved_files.append(str(npz_path))

    def _save_json(self, jobs, output_dir, saved_files):
        """JSON array of job dicts."""
        json_path = output_dir / "raps_jobs.json"
        with open(json_path, "w") as f:
            json.dump(jobs, f, indent=2, default=str)
        saved_files.append(str(json_path))

    def _save_parquet(self, jobs, output_dir, saved_files):
        """Parquet via pandas — flattens scalar fields, drops trace arrays."""
        df = self._jobs_to_dataframe(jobs)
        parquet_path = output_dir / "raps_jobs.parquet"
        df.to_parquet(str(parquet_path), index=False)
        saved_files.append(str(parquet_path))

    def _save_csv(self, jobs, output_dir, saved_files):
        """CSV via pandas — flattens scalar fields, drops trace arrays."""
        df = self._jobs_to_dataframe(jobs)
        csv_path = output_dir / "raps_jobs.csv"
        df.to_csv(str(csv_path), index=False)
        saved_files.append(str(csv_path))

    @staticmethod
    def _jobs_to_dataframe(jobs):
        """Convert job dicts to a DataFrame, keeping only scalar fields."""
        scalar_rows = []
        for j in jobs:
            row = {}
            for k, v in j.items():
                if isinstance(v, (list, np.ndarray)):
                    continue  # skip trace arrays for tabular formats
                row[k] = v
            scalar_rows.append(row)
        return pd.DataFrame(scalar_rows)
