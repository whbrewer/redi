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
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..pipeline import PipelineStep, PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


# Timestamp unit thresholds (absolute epoch magnitudes).
# Values above these ranges are normalized to epoch seconds.
_EPOCH_NS_THRESHOLD = 1e17
_EPOCH_US_THRESHOLD = 1e14
_EPOCH_MS_THRESHOLD = 1e11
_EPOCH_SECONDS_THRESHOLD = 1e9


def _normalize_epoch_seconds(value: float) -> float:
    """Normalize epoch value to seconds when input is ns/us/ms."""
    abs_value = abs(value)
    if abs_value >= _EPOCH_NS_THRESHOLD:
        return value / 1e9
    if abs_value >= _EPOCH_US_THRESHOLD:
        return value / 1e6
    if abs_value >= _EPOCH_MS_THRESHOLD:
        return value / 1e3
    return value


def _normalize_epoch_array(values: np.ndarray) -> np.ndarray:
    """Vectorized epoch unit normalization to seconds."""
    arr = values.astype(float, copy=True)
    abs_arr = np.abs(arr)

    ns_mask = abs_arr >= _EPOCH_NS_THRESHOLD
    us_mask = (abs_arr >= _EPOCH_US_THRESHOLD) & (abs_arr < _EPOCH_NS_THRESHOLD)
    ms_mask = (abs_arr >= _EPOCH_MS_THRESHOLD) & (abs_arr < _EPOCH_US_THRESHOLD)

    arr[ns_mask] = arr[ns_mask] / 1e9
    arr[us_mask] = arr[us_mask] / 1e6
    arr[ms_mask] = arr[ms_mask] / 1e3
    return arr


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
        default=None,
    ),
    FieldSpec(
        canonical_name="nrx_trace",
        aliases=["nrx_trace", "ib_rx", "net_rx", "ipogif0_rx_bytes"],
        field_type="trace",
        required=False,
        default=None,
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
            start_num = pd.to_numeric(df[start_col], errors="coerce")
            end_num = pd.to_numeric(df[end_col], errors="coerce")
            if start_num.notna().any() and end_num.notna().any():
                # Epoch-like numeric timestamps (common in scheduler logs).
                df[derived_col] = end_num - start_num
            else:
                start = pd.to_datetime(df[start_col], errors="coerce")
                end = pd.to_datetime(df[end_col], errors="coerce")
                df[derived_col] = (end - start).dt.total_seconds()
            mapped["expected_run_time"] = derived_col
            conf["expected_run_time"] = 0.9
            logger.info("Derived expected_run_time from end_time - start_time")

        # name ← uuid4[:6] (matches RAPS convention)
        if "name" not in mapped:
            derived_col = "_derived_name"
            df[derived_col] = [uuid.uuid4().hex[:6] for _ in range(len(df))]
            mapped["name"] = derived_col
            conf["name"] = 0.8
            logger.info("Derived name as uuid4[:6]")

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
    """Detects trace data format and converts to RAPS-compatible utilization.

    When ``power_config`` contains per-node counts (``per_node``) the RAPS
    power formula is used::

        watts = sum_power / (wall_time * nodes_required)
        util  = (watts / idle - per_node) / (max / idle - 1.0)

    Otherwise a simple linear normalization is applied.  Scalar inputs are
    returned as scalar floats (not wrapped in a list) and values are **not**
    clamped to ``[0, 1]`` — RAPS allows >1 values for GPU utilization.
    """

    def __init__(self, power_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            power_config: Dict like::

                {"cpu": {"idle": 90, "max": 280, "per_node": 1},
                 "gpu": {"idle": 44, "max": 238, "per_node": 8},
                 "nics_per_node": 4, "power_nic": 20}
        """
        self.power_config = power_config or {}

    def normalize_trace(self, value: Any, canonical_name: str, col_name: str,
                        wall_time: float = 0, nodes_required: int = 1) -> Any:
        """Normalize a single trace value.

        Args:
            value: The raw trace value (scalar, list, or string)
            canonical_name: e.g. "cpu_trace", "gpu_trace"
            col_name: Original column name (for power detection)
            wall_time: Job wall-clock time in seconds (for power → utilization)
            nodes_required: Number of nodes (for power → utilization)

        Returns:
            Normalized value — scalar float for scalar input, list for array input.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0

        # Parse string representations of lists
        if isinstance(value, str):
            value = _parse_trace_string(value)

        # Single scalar
        if isinstance(value, (int, float)):
            return self._normalize_scalar(
                float(value), canonical_name, col_name, wall_time, nodes_required)

        # Array/list
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float)
            if len(arr) == 0:
                return []
            normalized = self._normalize_array(
                arr, canonical_name, col_name, wall_time, nodes_required)
            # Return scalar for single-element arrays
            if len(normalized) == 1:
                return float(normalized[0])
            return normalized.tolist()

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

    def _normalize_scalar(self, value: float, canonical_name: str, col_name: str,
                          wall_time: float = 0, nodes_required: int = 1) -> float:
        """Normalize a single scalar trace value."""
        fmt = self._detect_format(np.array([value]), canonical_name, col_name)
        return self._apply_normalization(
            value, fmt, canonical_name, col_name, wall_time, nodes_required)

    def _normalize_array(self, arr: np.ndarray, canonical_name: str, col_name: str,
                         wall_time: float = 0, nodes_required: int = 1) -> np.ndarray:
        """Normalize an array of trace values."""
        fmt = self._detect_format(arr, canonical_name, col_name)
        return np.array([
            self._apply_normalization(v, fmt, canonical_name, col_name, wall_time, nodes_required)
            for v in arr
        ])

    def _apply_normalization(self, value: float, fmt: str, canonical_name: str,
                             col_name: str, wall_time: float = 0,
                             nodes_required: int = 1) -> float:
        """Apply normalization based on detected format.

        For power data with full RAPS config (per_node present), uses::

            watts = sum_power / (wall_time * nodes_required)
            util  = (watts / idle - per_node) / (max / idle - 1.0)

        Otherwise falls back to simple linear normalization.
        Values are NOT clamped — RAPS allows >1 for GPU utilization.
        """
        if np.isnan(value):
            return 0.0

        # Determine trace type key for power_config
        trace_type = canonical_name.replace("_trace", "")  # "cpu", "gpu", etc.

        if fmt == "power":
            cfg = self.power_config.get(trace_type, {})
            idle = cfg.get("idle", 0)
            max_p = cfg.get("max", 0)
            per_node = cfg.get("per_node", 0)

            if idle > 0 and max_p > 0 and per_node > 0 and wall_time > 0 and nodes_required > 0:
                # Full RAPS formula
                watts = value / (wall_time * nodes_required)
                util = (watts / idle - per_node) / (max_p / idle - 1.0)
                return max(0.0, float(util))
            elif max_p > idle > 0:
                # Simple linear normalization (no per_node info)
                return max(0.0, float((value - idle) / (max_p - idle)))
            else:
                return 0.0

        elif fmt == "percentage":
            return float(value / 100.0)

        elif fmt == "utilization":
            return float(value)

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


# ---------------------------------------------------------------------------
# Priority policies
# ---------------------------------------------------------------------------

def _priority_frontier(nodes_required: int) -> int:
    """Frontier aging boost policy.

    Source: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#job-priority-by-node-count
    Reproduced from raps/dataloaders/frontier.py::aging_boost().
    """
    if nodes_required >= 5645:
        return 8 * 24 * 3600   # 691200 s
    elif nodes_required >= 1882:
        return 4 * 24 * 3600   # 345600 s
    return 0


_PRIORITY_POLICIES: Dict[str, Any] = {
    "frontier": _priority_frontier,
}


class JobDictBuilder:
    """Builds RAPS-compatible job_dict dictionaries from mapped DataFrame rows."""

    def __init__(self, power_config: Optional[Dict[str, Dict[str, float]]] = None,
                 trace_quanta: Optional[int] = None,
                 jobprofile_df: Optional[pd.DataFrame] = None,
                 topology: Optional[Dict[str, Any]] = None,
                 system_name: str = "unknown",
                 priority_policy: Optional[str] = None):
        self.trace_normalizer = TraceNormalizer(power_config=power_config)
        self.power_config = power_config or {}
        self.trace_quanta = trace_quanta
        self.jobprofile_df = jobprofile_df
        self.topology = topology
        self.system_name = system_name
        self._jp_groups: Optional[Dict[str, pd.DataFrame]] = None
        self._priority_fn = _PRIORITY_POLICIES.get(priority_policy) if priority_policy else None

    def build_jobs(self, df: pd.DataFrame, mapping: ColumnMapping) -> Tuple[List[Dict[str, Any]], float]:
        """Build list of RAPS job dicts from a DataFrame.

        Args:
            df: DataFrame with columns that have been mapped via ColumnMapper
            mapping: The column mapping result

        Returns:
            Tuple of (list of RAPS-compatible job dictionaries, ref_time in epoch seconds)
        """
        jobs: List[Dict[str, Any]] = []
        mapped = mapping.mapped

        # Pre-group jobprofile by allocation_id for O(1) lookups
        if self.jobprofile_df is not None and "allocation_id" in self.jobprofile_df.columns:
            self._jp_groups = {
                str(k): v for k, v in self.jobprofile_df.groupby("allocation_id")
            }
            logger.info(f"Jobprofile: {len(self._jp_groups)} unique allocations")

        # Find reference time
        ref_time = self._find_reference_time(df, mapped)
        timestamp_cache = self._build_timestamp_cache(df, mapped, ref_time)

        skipped = 0
        total_rows = len(df)
        started_at = time.perf_counter()
        progress_every = 50000

        for pos, (idx, row) in enumerate(df.iterrows(), start=1):
            try:
                job = self._build_one(row, mapped, ref_time, timestamp_cache=timestamp_cache, row_pos=pos - 1)
                if job is not None:
                    jobs.append(job)
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                skipped += 1

            if pos % progress_every == 0 or pos == total_rows:
                elapsed = time.perf_counter() - started_at
                rate = pos / elapsed if elapsed > 0 else 0.0
                if total_rows >= progress_every:
                    print(
                        f"\r  ⏳ build_job_dicts ... {pos:,}/{total_rows:,} rows ({rate:.0f} rows/s)",
                        end="",
                        flush=True,
                        file=sys.stderr,
                    )
                logger.info(
                    f"build_job_dicts progress: {pos}/{total_rows} rows "
                    f"({rate:.0f} rows/s, skipped={skipped})"
                )

        if skipped:
            logger.info(f"Skipped {skipped} invalid rows")
        logger.info(f"Built {len(jobs)} RAPS job dicts")

        return jobs, ref_time

    def _build_timestamp_cache(
        self, df: pd.DataFrame, mapped: Dict[str, str], ref_time: float
    ) -> Dict[str, np.ndarray]:
        """Precompute timestamp fields once to avoid per-row datetime parsing."""
        cache: Dict[str, np.ndarray] = {}

        for fs in FIELD_REGISTRY:
            canon = fs.canonical_name
            if fs.field_type != "timestamp" or canon not in mapped:
                continue

            col = mapped[canon]
            if col not in df.columns:
                continue

            series = df[col]

            # Numeric timestamps: treat >1e9 as epoch seconds, else relative seconds.
            if pd.api.types.is_numeric_dtype(series):
                raw = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                raw_sec = _normalize_epoch_array(raw)
                rel = np.where(
                    np.isnan(raw_sec),
                    0.0,
                    np.where(raw_sec > _EPOCH_SECONDS_THRESHOLD, raw_sec - ref_time, raw_sec),
                )
                cache[canon] = rel
                continue

            # Object columns that are mostly numeric epoch timestamps.
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_valid_ratio = float(numeric.notna().mean()) if len(series) else 0.0
            if numeric_valid_ratio >= 0.9:
                raw = numeric.to_numpy(dtype=float)
                raw_sec = _normalize_epoch_array(raw)
                rel = np.where(
                    np.isnan(raw_sec),
                    0.0,
                    np.where(raw_sec > _EPOCH_SECONDS_THRESHOLD, raw_sec - ref_time, raw_sec),
                )
                cache[canon] = rel
                continue

            # Datetime-like strings/objects: parse once, then fallback to duration parsing.
            parsed = pd.to_datetime(series, errors="coerce")
            ns = parsed.to_numpy(dtype="datetime64[ns]")
            rel = np.zeros(len(series), dtype=float)
            valid = ~pd.isna(ns)
            if valid.any():
                rel[valid] = ns[valid].astype("int64") / 1e9 - ref_time

            invalid = ~valid
            if invalid.any():
                fallback = series[invalid].map(_parse_duration).to_numpy(dtype=float)
                rel[invalid] = np.where(fallback > 0, fallback, 0.0)

            cache[canon] = rel

        return cache

    def _find_reference_time(self, df: pd.DataFrame, mapped: Dict[str, str]) -> float:
        """Find the earliest timestamp to use as reference (epoch seconds).

        When jobprofile is available, uses ``jobprofile_df['timestamp'].min()``
        (matching RAPS convention).  Otherwise falls back to ``start_time.min()``.
        """
        # Primary: use jobprofile timestamp.min() when available
        if self.jobprofile_df is not None and "timestamp" in self.jobprofile_df.columns:
            ts = self.jobprofile_df["timestamp"]
            valid = ts.dropna()
            if not valid.empty:
                return valid.min().timestamp()

        # Fallback: use start_time.min()
        if "start_time" in mapped:
            col = mapped["start_time"]
            raw = df[col]
            numeric = pd.to_numeric(raw, errors="coerce").dropna()
            numeric_sec = numeric.map(_normalize_epoch_seconds)
            numeric_valid = numeric_sec[numeric_sec > _EPOCH_SECONDS_THRESHOLD]
            if not numeric_valid.empty:
                return float(numeric_valid.min())
            series = pd.to_datetime(raw, errors="coerce")
            valid = series.dropna()
            if not valid.empty:
                return valid.min().timestamp()
        # Last resort: submit_time
        if "submit_time" in mapped:
            col = mapped["submit_time"]
            raw = df[col]
            numeric = pd.to_numeric(raw, errors="coerce").dropna()
            numeric_sec = numeric.map(_normalize_epoch_seconds)
            numeric_valid = numeric_sec[numeric_sec > _EPOCH_SECONDS_THRESHOLD]
            if not numeric_valid.empty:
                return float(numeric_valid.min())
            series = pd.to_datetime(raw, errors="coerce")
            valid = series.dropna()
            if not valid.empty:
                return valid.min().timestamp()
        return 0.0

    def _build_one(
        self,
        row: pd.Series,
        mapped: Dict[str, str],
        ref_time: float,
        timestamp_cache: Optional[Dict[str, np.ndarray]] = None,
        row_pos: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a single job dict from a DataFrame row."""
        job: Dict[str, Any] = {}
        trace_fields: List[Tuple[FieldSpec, Any, str]] = []  # (spec, raw, col_name)

        for fs in FIELD_REGISTRY:
            canon = fs.canonical_name
            if canon in mapped:
                raw = row.get(mapped[canon])
                if fs.field_type == "trace":
                    # Defer trace conversion until we have wall_time/nodes
                    trace_fields.append((fs, raw, mapped.get(canon, "")))
                    job[canon] = fs.default
                    if isinstance(job[canon], list):
                        job[canon] = list(job[canon])
                    continue
                if (
                    fs.field_type == "timestamp"
                    and timestamp_cache is not None
                    and row_pos is not None
                    and canon in timestamp_cache
                ):
                    value = float(timestamp_cache[canon][row_pos])
                else:
                    value = self._convert_value(raw, fs, mapped.get(canon, ""), ref_time)
            else:
                value = fs.default
                # Deep copy mutable defaults
                if isinstance(value, list):
                    value = list(value)

            job[canon] = value

        # Apply priority policy when the source data has no priority column
        if self._priority_fn is not None and not job.get("priority"):
            job["priority"] = self._priority_fn(int(job.get("nodes_required") or 0))

        # Validation: skip if raw end_time is NaN/NaT (still-running jobs)
        if "end_time" in mapped:
            raw_end = row.get(mapped["end_time"])
            if raw_end is None or pd.isna(raw_end):
                logger.debug(f"Skipping job {job.get('id', '?')}: NaN/NaT end_time")
                return None

        # Now convert traces with wall_time / nodes context
        wall_time = job.get("expected_run_time", 0) or 0
        nodes_req = job.get("nodes_required", 1) or 1

        # Jobprofile join: build per-timestep traces from power telemetry
        if self._jp_groups is not None:
            # Get allocation_id from the raw row (joblive column)
            alloc_id = self._get_allocation_id(row, mapped)
            if alloc_id is not None:
                got_profile = self._build_traces_from_jobprofile(job, alloc_id, nodes_req)
                if not got_profile:
                    # No matching jobprofile → skip job (matches RAPS behavior)
                    logger.debug(f"Skipping job {job.get('id', '?')}: no jobprofile for allocation {alloc_id}")
                    return None
            else:
                logger.debug(f"Skipping job {job.get('id', '?')}: no allocation_id")
                return None
        else:
            # No jobprofile — use scalar trace normalization
            for fs, raw, col_name in trace_fields:
                job[fs.canonical_name] = self._convert_trace(
                    raw, fs, col_name, wall_time, nodes_req)

        # Convert scheduled_nodes with system-specific xname converter
        if self.topology and isinstance(job.get("scheduled_nodes"), list):
            raw_nodes = job["scheduled_nodes"]
            if raw_nodes:
                # raw_nodes are already parsed int hashes — re-parse from original
                raw_node_val = row.get(mapped.get("scheduled_nodes", ""))
                if raw_node_val is not None and not (isinstance(raw_node_val, float) and np.isnan(raw_node_val)):
                    parsed = _parse_node_list_raw(raw_node_val)
                    job["scheduled_nodes"] = convert_node_names(
                        parsed, self.system_name, self.topology)

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
        wall_time = job.get("expected_run_time", 0) or 0

        # Determine whether we have real per-timestep trace arrays
        cpu_trace = job.get("cpu_trace")
        has_timeseries = isinstance(cpu_trace, (list, np.ndarray)) and len(cpu_trace) > 1

        job["current_state"] = "PENDING"      # scheduler moves to RUNNING
        job["current_run_time"] = 0
        job["allocated_cpu_cores"] = 0        # simulator fills these
        job["allocated_gpu_units"] = 0        # simulator fills these
        job["downscale"] = 1
        job["dilated"] = False

        if has_timeseries:
            # Real per-timestep trace data
            quanta = self.trace_quanta or 1
            job["trace_quanta"] = quanta
            job["trace_time"] = len(cpu_trace) * quanta
            job["trace_start_time"] = end     # RAPS convention
            job["trace_end_time"] = start     # RAPS convention
            job["trace_missing_values"] = False
        else:
            # Scalar or no trace data — RAPS convention for scalar traces
            job["trace_quanta"] = self.trace_quanta
            job["trace_time"] = wall_time
            job["trace_start_time"] = end     # RAPS convention
            job["trace_end_time"] = start     # RAPS convention
            job["trace_missing_values"] = True

        # Ensure partition is int (RAPS expects int | None, not str)
        part = job.get("partition", 0)
        if isinstance(part, str):
            try:
                job["partition"] = int(part) if part else 0
            except ValueError:
                job["partition"] = 0

        return job

    def _get_allocation_id(self, row: pd.Series, mapped: Dict[str, str]) -> Optional[str]:
        """Extract allocation_id from a joblive row.

        Checks for an explicit 'allocation_id' column first, then falls
        back to the mapped 'id' field.
        """
        # Direct column access (allocation_id is often present but not mapped)
        if "allocation_id" in row.index:
            val = row["allocation_id"]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return str(val)
        # Fall back to whatever is mapped as 'id'
        id_col = mapped.get("id")
        if id_col and id_col in row.index:
            val = row[id_col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return str(val)
        return None

    def _build_traces_from_jobprofile(self, job: Dict[str, Any],
                                       allocation_id: str,
                                       nodes_required: int) -> bool:
        """Build cpu_trace and gpu_trace arrays from jobprofile power data.

        Applies the RAPS power formula:
            util = (power - nodes * idle * per_node) / (nodes * max * per_node - nodes * idle * per_node)
            trace = util * per_node

        Returns True if traces were built, False if no matching profile found.
        """
        profile = self._jp_groups.get(allocation_id)
        if profile is None or len(profile) == 0:
            return False

        nodes = max(nodes_required, 1)
        cpu_cfg = self.power_config.get("cpu", {})
        gpu_cfg = self.power_config.get("gpu", {})

        # CPU trace
        if "sum_cpu0_power" in profile.columns:
            cpu_power = profile["sum_cpu0_power"].values.astype(float)
            cpu_idle = cpu_cfg.get("idle", 0)
            cpu_max = cpu_cfg.get("max", 0)
            cpu_per_node = cpu_cfg.get("per_node", 1)
            cpu_pmin = nodes * cpu_idle * cpu_per_node
            cpu_pmax = nodes * cpu_max * cpu_per_node
            if cpu_pmax > cpu_pmin:
                cpu_util = (cpu_power - cpu_pmin) / (cpu_pmax - cpu_pmin)
                cpu_trace = cpu_util * cpu_per_node
            else:
                cpu_trace = np.zeros_like(cpu_power)
            job["cpu_trace"] = np.nan_to_num(cpu_trace).tolist()
        else:
            job["cpu_trace"] = 0

        # GPU trace
        if "sum_gpu_power" in profile.columns:
            gpu_power = profile["sum_gpu_power"].values.astype(float)
            gpu_idle = gpu_cfg.get("idle", 0)
            gpu_max = gpu_cfg.get("max", 0)
            gpu_per_node = gpu_cfg.get("per_node", 1)
            gpu_pmin = nodes * gpu_idle * gpu_per_node
            gpu_pmax = nodes * gpu_max * gpu_per_node
            if gpu_pmax > gpu_pmin:
                gpu_util = (gpu_power - gpu_pmin) / (gpu_pmax - gpu_pmin)
                gpu_trace = gpu_util * gpu_per_node
            else:
                gpu_trace = np.zeros_like(gpu_power)
            job["gpu_trace"] = np.nan_to_num(gpu_trace).tolist()
        else:
            job["gpu_trace"] = 0

        return True

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
            return self.trace_normalizer.normalize_trace(
                raw, fs.canonical_name, col_name)

        if fs.field_type == "state":
            return _normalize_state(raw)

        if fs.field_type == "list":
            return _parse_node_list(raw)

        return raw

    def _convert_trace(self, raw: Any, fs: FieldSpec, col_name: str,
                       wall_time: float, nodes_required: int) -> Any:
        """Convert a trace value with wall_time/nodes context for RAPS formula."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            default = fs.default
            if isinstance(default, list):
                return list(default)
            return default
        return self.trace_normalizer.normalize_trace(
            raw, fs.canonical_name, col_name, wall_time, nodes_required)

    def _convert_timestamp(self, raw: Any, ref_time: float) -> float:
        """Convert a timestamp value to relative seconds from ref_time."""
        # Already numeric (epoch seconds)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val = _normalize_epoch_seconds(float(raw))
            # If value is very large, assume it's epoch seconds
            if val > _EPOCH_SECONDS_THRESHOLD:
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


# ---------------------------------------------------------------------------
# Node name converters (system-specific xname → index)
# ---------------------------------------------------------------------------

def _xname_to_index(xname: str, topology: Dict[str, Any]) -> int:
    """Convert a Frontier xname string to a linear node index.

    Ported from raps.dataloaders.frontier.xname_to_index.
    Xname format: x{row}{col:02}c{chassis}s{slot}b{node}
    Example: x2301c4s2b1
    """
    row, col = int(xname[2]), int(xname[3:5])
    chassis, slot, node = int(xname[6]), int(xname[8]), int(xname[10])
    if row == 6:
        col -= 9
    rack_index = row * 12 + col
    nodes_per_rack = topology["sc_shape"][2]
    node_index = (chassis * topology["blades_per_chassis"] * topology["nodes_per_blade"]
                  + slot * topology["nodes_per_blade"] + node)
    return rack_index * nodes_per_rack + node_index


_NODE_CONVERTERS = {"frontier": _xname_to_index}


def convert_node_names(names: List[Any], system_name: str,
                       topology: Optional[Dict[str, Any]] = None) -> List[int]:
    """Convert node name strings to integer indices.

    Uses a system-specific converter when available (e.g. Frontier xname),
    otherwise falls back to stable hashing.
    """
    fn = _NODE_CONVERTERS.get(system_name)
    result: List[int] = []
    for n in names:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        s = str(n).strip()
        if not s:
            continue
        if fn and topology:
            try:
                result.append(fn(s, topology))
            except (IndexError, ValueError):
                result.append(abs(hash(s)) % (10**9))
        else:
            try:
                result.append(int(float(s)))
            except (ValueError, TypeError):
                result.append(abs(hash(s)) % (10**9))
    return result


def _parse_node_list_raw(val: Any) -> List[str]:
    """Parse a node list into raw string items (before int conversion).

    Used by convert_node_names to get the original hostnames for
    system-specific conversion (e.g. xname → index).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(x) for x in val]
    s = str(val).strip()
    if not s:
        return []
    if "," in s:
        return [n.strip() for n in s.split(",") if n.strip()]
    parts = s.split()
    if len(parts) > 1:
        return parts
    return [s]


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


def load_raps_system_params(system_name: str) -> Optional[Dict[str, Any]]:
    """Load power, system, and scheduler params from an installed RAPS config.

    Reads the RAPS YAML for *system_name* and returns a dict with:
      - ``power_config``: for :class:`TraceNormalizer`
      - ``trace_quanta``: from the scheduler section

    Returns ``None`` when RAPS is not installed or the system is not found.
    """
    try:
        from raps.system_config import get_system_config
    except ImportError:
        logger.debug("RAPS not installed — cannot load system config")
        return None

    try:
        cfg = get_system_config(system_name)
    except FileNotFoundError:
        logger.warning(f"RAPS system config '{system_name}' not found")
        return None

    pwr = cfg.power
    sys = cfg.system
    return {
        "power_config": {
            "cpu": {
                "idle": pwr.power_cpu_idle,
                "max": pwr.power_cpu_max,
                "per_node": sys.cpus_per_node,
            },
            "gpu": {
                "idle": pwr.power_gpu_idle,
                "max": pwr.power_gpu_max,
                "per_node": sys.gpus_per_node,
            },
            "nics_per_node": sys.nics_per_node,
            "power_nic": pwr.power_nic or 0,
            "power_mem": pwr.power_mem,
        },
        "trace_quanta": cfg.scheduler.trace_quanta,
        "topology": {
            "sc_shape": sys.sc_shape,
            "blades_per_chassis": sys.blades_per_chassis,
            "nodes_per_blade": sys.nodes_per_blade,
        },
    }


# Backwards-compat alias
def load_raps_power_config(system_name: str) -> Optional[Dict[str, Any]]:
    """Load power config from RAPS. Returns just the power_config dict."""
    params = load_raps_system_params(system_name)
    if params is None:
        return None
    return params["power_config"]


def _local_job_dict(**kwargs) -> Dict[str, Any]:
    """Local fallback that produces the same dict structure as raps.job.job_dict."""
    return {
        "id": kwargs.get("id", ""),
        "name": kwargs.get("name", ""),
        "account": kwargs.get("account", "unknown"),
        "nodes_required": kwargs.get("nodes_required", 1),
        # Allocation
        "current_state": kwargs.get("current_state", "PENDING"),
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
        "ntx_trace": kwargs.get("ntx_trace", None),
        "nrx_trace": kwargs.get("nrx_trace", None),
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
        "trace_quanta": kwargs.get("trace_quanta", None),
        "trace_missing_values": kwargs.get("trace_missing_values", True),
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

    def __init__(self, power_config: Optional[Dict[str, Dict[str, float]]] = None,
                 trace_quanta: Optional[int] = None,
                 topology: Optional[Dict[str, Any]] = None,
                 system_name: str = "unknown",
                 priority_policy: Optional[str] = None):
        super().__init__()
        self.power_config = power_config
        self.trace_quanta = trace_quanta
        self.topology = topology
        self.system_name = system_name
        self.priority_policy = priority_policy

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not isinstance(ctx.data, pd.DataFrame):
            ctx.add_error("BuildJobDicts requires a DataFrame")
            return ctx

        mapping: Optional[ColumnMapping] = ctx.metadata.get("_raps_mapping_obj")
        if mapping is None:
            ctx.add_error("BuildJobDicts: no column mapping found")
            return ctx

        # Get jobprofile_df from context (set by LoadJobProfile step)
        jobprofile_df = ctx.metadata.get("_raps_jobprofile_df")

        builder = JobDictBuilder(
            power_config=self.power_config,
            trace_quanta=self.trace_quanta,
            jobprofile_df=jobprofile_df,
            topology=self.topology,
            system_name=self.system_name,
            priority_policy=self.priority_policy,
        )
        jobs, ref_time = builder.build_jobs(ctx.data, mapping)

        ctx.data = jobs
        ctx.metadata["raps_job_count"] = len(jobs)
        ctx.metadata["_raps_ref_time"] = ref_time

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
        ref_time = ctx.metadata.get("_raps_ref_time", 0.0)
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
            self._save_npz(jobs, output_dir, telemetry_start, telemetry_end, saved_files, ref_time)
        elif fmt == "json":
            self._save_json(jobs, output_dir, saved_files)
        elif fmt == "parquet":
            self._save_parquet(jobs, output_dir, saved_files)
        elif fmt == "csv":
            self._save_csv(jobs, output_dir, saved_files)
        else:
            logger.warning(f"Unknown output format '{fmt}', falling back to npz")
            self._save_npz(jobs, output_dir, telemetry_start, telemetry_end, saved_files, ref_time)

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

    def _save_npz(self, jobs, output_dir, telemetry_start, telemetry_end, saved_files, ref_time=0.0):
        """RAPS-compatible NPZ snapshot."""
        from datetime import datetime, timezone
        from types import SimpleNamespace

        # Use actual telemetry start datetime (from ref_time) instead of now()
        ref_time_sec = _normalize_epoch_seconds(float(ref_time)) if ref_time else 0.0
        if ref_time_sec > 0:
            try:
                start_date = datetime.fromtimestamp(ref_time_sec, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                logger.warning(
                    "Invalid ref_time for datetime conversion (%s); falling back to now().",
                    ref_time,
                )
                start_date = datetime.now(timezone.utc)
        else:
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
