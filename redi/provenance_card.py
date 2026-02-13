"""Generate a human-readable provenance card from Flowcept JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

SENSITIVE_KEY_PATTERNS = ("api_key", "token", "secret", "password")
SENSITIVE_VALUE_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]+")

try:
    from flowcept import __version__ as FLOWCEPT_VERSION
except Exception:
    FLOWCEPT_VERSION = "unknown"

try:
    from flowcept.configs import DUMP_BUFFER_PATH
except Exception:
    DUMP_BUFFER_PATH = "/tmp/flowcept_buffer.jsonl"


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value if value > 0 else None


def _to_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value)


def _fmt_ts(value: Any) -> str:
    ts = _as_float(value)
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _elapsed(started_at: Any, ended_at: Any) -> Optional[float]:
    start = _as_float(started_at)
    end = _as_float(ended_at)
    if start is None or end is None or end < start:
        return None
    return end - start


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"


def _fmt_percent(value: Optional[float]) -> str:
    v = _positive(_as_float(value))
    if v is None:
        return "-"
    return f"{v:.1f}%"


def _fmt_count(value: Optional[float]) -> str:
    v = _positive(_as_float(value))
    if v is None:
        return "-"
    return f"{int(v):,}"


def _fmt_bytes(value: Optional[float]) -> str:
    v = _positive(_as_float(value))
    if v is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while v >= 1024 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(v)} {units[idx]}"
    return f"{v:.2f} {units[idx]}"


def _redact_key_value(key: str, value: Any) -> Any:
    key_l = key.lower()
    if any(pat in key_l for pat in SENSITIVE_KEY_PATTERNS):
        return "REDACTED"
    if isinstance(value, str) and SENSITIVE_VALUE_PATTERN.search(value):
        return "REDACTED"
    return value


def sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = sanitize_json_like(_redact_key_value(str(k), v))
        return out
    if isinstance(value, list):
        return [sanitize_json_like(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json_like(v) for v in value]
    if isinstance(value, str) and SENSITIVE_VALUE_PATTERN.search(value):
        return "REDACTED"
    return value


def _safe_json_text(value: Any) -> str:
    safe = sanitize_json_like(value)
    try:
        return json.dumps(safe, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(safe), ensure_ascii=False)


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    return records, skipped


def resolve_flowcept_buffer_path(flowcept_ref: Any = None) -> Path:
    """Resolve the concrete Flowcept buffer file path from runtime/config."""
    candidates: List[Path] = []

    if flowcept_ref is not None and hasattr(flowcept_ref, "get_buffer"):
        try:
            buffer_records = flowcept_ref.get_buffer()
            if isinstance(buffer_records, list):
                workflow_record = next(
                    (
                        rec
                        for rec in buffer_records
                        if isinstance(rec, dict) and rec.get("type") == "workflow"
                    ),
                    None,
                )
                if isinstance(workflow_record, dict):
                    dump_cfg = (
                        workflow_record.get("flowcept_settings", {})
                        .get("project", {})
                        .get("dump_buffer", {})
                    )
                    configured_path = dump_cfg.get("path")
                    if configured_path:
                        configured_base = Path(str(configured_path)).expanduser()
                        candidates.append(configured_base)

                        if dump_cfg.get("append_workflow_id_to_path") or dump_cfg.get("append_id_to_path"):
                            workflow_id = workflow_record.get("workflow_id")
                            pattern = (
                                f"{configured_base.stem}_{workflow_id}*{configured_base.suffix}"
                                if workflow_id
                                else f"{configured_base.stem}*{configured_base.suffix}"
                            )
                            if configured_base.parent.exists():
                                candidates.extend(
                                    sorted(
                                        configured_base.parent.glob(pattern),
                                        key=lambda p: p.stat().st_mtime,
                                        reverse=True,
                                    )
                                )
        except Exception:
            pass

    if DUMP_BUFFER_PATH:
        candidates.append(Path(str(DUMP_BUFFER_PATH)).expanduser())
    candidates.append(Path("/tmp/flowcept_buffer.jsonl"))

    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        c = str(candidate)
        if c not in seen:
            seen.add(c)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0]


def generate_provenance_artifacts(
    output_dir: Path,
    flowcept_ref: Any = None,
) -> Dict[str, Any]:
    """
    Move Flowcept buffer to output_dir and generate PROVENANCE_CARD.md.

    Returns a dict with generated artifact paths and generation stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_dst = output_dir / "flowcept_buffer.jsonl"
    buffer_src = None

    # Do not force an extra dump here; Flowcept lifecycle already flushes buffer on stop.
    # We only consume/move the persisted buffer file.

    # Fallback: resolve existing buffer path and move it.
    if not buffer_dst.exists():
        buffer_src = resolve_flowcept_buffer_path(flowcept_ref=flowcept_ref)
        if buffer_src.exists():
            if buffer_dst.exists():
                buffer_dst.unlink()
            shutil.move(str(buffer_src), str(buffer_dst))

    card_path = output_dir / "PROVENANCE_CARD.md"
    if buffer_dst.exists():
        input_jsonl = buffer_dst
    elif buffer_src is not None:
        input_jsonl = buffer_src
    else:
        input_jsonl = resolve_flowcept_buffer_path(flowcept_ref=flowcept_ref)
    stats = generate_provenance_card(
        input_jsonl=str(input_jsonl),
        output_markdown=str(card_path),
    )
    return {
        "buffer_file": str(buffer_dst) if buffer_dst.exists() else None,
        "card_file": str(card_path),
        "stats": stats,
    }


def link_datacard_to_provenance(datacard_path: Path, provenance_filename: str = "PROVENANCE_CARD.md") -> bool:
    """Append a provenance link block to DATA_CARD.md if missing."""
    if not datacard_path.exists():
        return False
    existing = datacard_path.read_text(encoding="utf-8")
    if "Data Transformation Pipeline Provenance Card" in existing:
        return False
    link_block = (
        "\n## Provenance\n"
        f"- [Data Transformation Pipeline Provenance Card]({provenance_filename})\n"
    )
    datacard_path.write_text(existing.rstrip() + "\n" + link_block, encoding="utf-8")
    return True


def _choose_provenance_output_dir(default_output_dir: Path, saved_files: Any) -> Path:
    if isinstance(saved_files, list):
        for sf in saved_files:
            sf_path = Path(sf)
            if sf_path.suffix:
                return sf_path.parent
    return default_output_dir


def _ensure_saved_files_list(ctx: Any) -> List[str]:
    if "saved_files" not in ctx.metadata or not isinstance(ctx.metadata.get("saved_files"), list):
        ctx.metadata["saved_files"] = []
    return ctx.metadata["saved_files"]


def finalize_provenance_for_context(
    ctx: Any,
    flowcept_ref: Any = None,
    verbose: bool = False,
    logger: Any = None,
) -> Dict[str, Any]:
    """Generate provenance artifacts and register them in context metadata."""
    if not getattr(ctx, "with_flowcept", False):
        return {"enabled": False, "generated": False, "error": None}

    try:
        provenance_output_dir = _choose_provenance_output_dir(
            default_output_dir=Path(ctx.output_dir),
            saved_files=ctx.metadata.get("saved_files", []),
        )
        result = generate_provenance_artifacts(
            output_dir=provenance_output_dir,
            flowcept_ref=flowcept_ref,
        )
        ctx.output_dir = provenance_output_dir

        saved_files = _ensure_saved_files_list(ctx)
        buffer_file = result.get("buffer_file")
        card_file = result.get("card_file")
        if buffer_file and str(buffer_file) not in saved_files:
            saved_files.append(str(buffer_file))
        if card_file and str(card_file) not in saved_files:
            saved_files.append(str(card_file))

        if verbose and logger is not None:
            stats = result.get("stats", {})
            logger.info(
                "Provenance card generated: parsed=%s tasks=%s skipped=%s output=%s",
                stats.get("parsed_lines"),
                stats.get("tasks"),
                stats.get("skipped_lines"),
                stats.get("output"),
            )

        return {
            "enabled": True,
            "generated": True,
            "error": None,
            "card_path": Path(card_file) if card_file else None,
            "buffer_path": Path(buffer_file) if buffer_file else None,
        }
    except Exception as e:
        if logger is not None:
            logger.warning("Could not generate provenance card: %s", e)
        return {"enabled": True, "generated": False, "error": str(e)}


def resolve_provenance_paths(ctx: Any) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve provenance card and buffer paths from context metadata/output_dir."""
    provenance_path: Optional[Path] = None
    flowcept_buffer_path: Optional[Path] = None

    saved_files = ctx.metadata.get("saved_files", [])
    if isinstance(saved_files, list):
        for sf in saved_files:
            sf_path = Path(sf)
            if sf_path.name in {"PROVENANCE_CARD.md", "provenance_card.md"} and sf_path.exists():
                provenance_path = sf_path
            if sf_path.name == "flowcept_buffer.jsonl" and sf_path.exists():
                flowcept_buffer_path = sf_path

    fallback_card_upper = Path(ctx.output_dir) / "PROVENANCE_CARD.md"
    if provenance_path is None and fallback_card_upper.exists():
        provenance_path = fallback_card_upper
    fallback_card_lower = Path(ctx.output_dir) / "provenance_card.md"
    if provenance_path is None and fallback_card_lower.exists():
        provenance_path = fallback_card_lower

    fallback_buffer = Path(ctx.output_dir) / "flowcept_buffer.jsonl"
    if flowcept_buffer_path is None and fallback_buffer.exists():
        flowcept_buffer_path = fallback_buffer

    return provenance_path, flowcept_buffer_path


def render_provenance_summary(
    ctx: Any,
    style: str,
    datacard_path: Optional[Path] = None,
    print_section: Optional[Callable[[str], None]] = None,
    printer: Callable[[str], None] = print,
    colors: Any = None,
) -> bool:
    """Render provenance saved information for CLI outputs."""
    provenance_path, flowcept_buffer_path = resolve_provenance_paths(ctx)
    if provenance_path is None:
        return False

    if datacard_path is not None and datacard_path.exists():
        try:
            link_datacard_to_provenance(datacard_path, provenance_filename="PROVENANCE_CARD.md")
        except Exception:
            pass

    if style == "simple":
        printer("Provenance Card:")
        printer("  Saved: PROVENANCE_CARD.md")
        if flowcept_buffer_path is not None:
            printer("  Saved: flowcept_buffer.jsonl")
        return True

    if style != "rich":
        return True

    bold = getattr(colors, "BOLD", "") if colors else ""
    dim = getattr(colors, "DIM", "") if colors else ""
    end = getattr(colors, "END", "") if colors else ""

    if print_section is not None:
        print_section("Provenance Card")
    else:
        printer("\n▸ Provenance Card")
    printer(f"    📋 {bold}PROVENANCE_CARD.md{end} {dim}(saved){end}")
    if flowcept_buffer_path is not None:
        printer(f"    📋 {bold}flowcept_buffer.jsonl{end} {dim}(saved){end}")
    return True


def _resolve_records(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    workflows = [r for r in records if r.get("type") == "workflow"]
    workflow = workflows[-1] if workflows else {}
    workflow_id = workflow.get("workflow_id")

    tasks = [r for r in records if r.get("type") == "task"]
    if workflow_id:
        tasks = [t for t in tasks if t.get("workflow_id") == workflow_id]
    return workflow, tasks


def _task_io(task: Dict[str, Any]) -> Tuple[str, str]:
    used = task.get("used", {})
    if not isinstance(used, dict):
        return "unknown", "unknown"
    return _to_str(used.get("input_path")), _to_str(used.get("output_dir"))


def _render_table(headers: List[str], rows: List[List[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    if not rows:
        body = ["| " + " | ".join(["-"] * len(headers)) + " |"]
    return "\n".join([head, sep] + body)


def _iter_task_metadata(task: Dict[str, Any]) -> Dict[str, Any]:
    cmeta = task.get("custom_metadata", {})
    if not isinstance(cmeta, dict):
        return {}
    meta = cmeta.get("metadata", {})
    if not isinstance(meta, dict):
        return {}
    return meta


def _collect_saved_files(tasks: Iterable[Dict[str, Any]]) -> List[str]:
    files: List[str] = []
    for task in tasks:
        meta = _iter_task_metadata(task)
        saved = meta.get("saved_files")
        if isinstance(saved, list):
            files.extend(str(v) for v in saved)
    return sorted(set(files))


def _deep_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _sub_or_none(a: Any, b: Any) -> Optional[float]:
    af = _as_float(a)
    bf = _as_float(b)
    if af is None or bf is None:
        return None
    return bf - af


def compute_telemetry_delta(start: Dict[str, Any], end: Dict[str, Any]) -> Dict[str, Any]:
    cpu_start = _deep_get(start, ["cpu"]) or {}
    cpu_end = _deep_get(end, ["cpu"]) or {}
    cpu_times_start = _deep_get(cpu_start, ["times_avg"]) or {}
    cpu_times_end = _deep_get(cpu_end, ["times_avg"]) or {}

    disk_start = _deep_get(start, ["disk"]) or {}
    disk_end = _deep_get(end, ["disk"]) or {}
    disk_usage_start = _deep_get(disk_start, ["disk_usage"]) or {}
    disk_usage_end = _deep_get(disk_end, ["disk_usage"]) or {}
    io_start = _deep_get(disk_start, ["io_sum"]) or {}
    io_end = _deep_get(disk_end, ["io_sum"]) or {}

    mem_start = _deep_get(start, ["memory", "virtual"]) or {}
    mem_end = _deep_get(end, ["memory", "virtual"]) or {}

    return {
        "cpu": {
            "user_seconds": _positive(_sub_or_none(cpu_times_start.get("user"), cpu_times_end.get("user"))),
            "system_seconds": _positive(_sub_or_none(cpu_times_start.get("system"), cpu_times_end.get("system"))),
            "idle_seconds": _positive(_sub_or_none(cpu_times_start.get("idle"), cpu_times_end.get("idle"))),
            "percent_points": _positive(_sub_or_none(cpu_start.get("percent_all"), cpu_end.get("percent_all"))),
        },
        "memory": {
            "used_bytes": _positive(_sub_or_none(mem_start.get("used"), mem_end.get("used"))),
            "percent_points": _positive(_sub_or_none(mem_start.get("percent"), mem_end.get("percent"))),
        },
        "disk": {
            "used_bytes": _positive(_sub_or_none(disk_usage_start.get("used"), disk_usage_end.get("used"))),
            "percent_points": _positive(_sub_or_none(disk_usage_start.get("percent"), disk_usage_end.get("percent"))),
            "io_sum": {
                "read_count": _positive(_sub_or_none(io_start.get("read_count"), io_end.get("read_count"))),
                "write_count": _positive(_sub_or_none(io_start.get("write_count"), io_end.get("write_count"))),
                "read_bytes": _positive(_sub_or_none(io_start.get("read_bytes"), io_end.get("read_bytes"))),
                "write_bytes": _positive(_sub_or_none(io_start.get("write_bytes"), io_end.get("write_bytes"))),
                "read_time": _positive(_sub_or_none(io_start.get("read_time"), io_end.get("read_time"))),
                "write_time": _positive(_sub_or_none(io_start.get("write_time"), io_end.get("write_time"))),
            },
        },
    }


def _sum_numeric(values: Iterable[Optional[float]]) -> float:
    total = 0.0
    for v in values:
        if isinstance(v, (int, float)):
            total += float(v)
    return total


def _compact_columns(columns: Any, limit: int = 8) -> str:
    if not isinstance(columns, list) or not columns:
        return "-"
    if len(columns) <= limit:
        return ", ".join(str(c) for c in columns)
    head = ", ".join(str(c) for c in columns[:limit])
    return f"{head}, ... (+{len(columns) - limit} more)"


def _activity_metadata_lines(task: Dict[str, Any]) -> List[str]:
    cmeta = task.get("custom_metadata", {}) if isinstance(task.get("custom_metadata"), dict) else {}
    meta = _iter_task_metadata(task)
    activity = _to_str(task.get("activity_id"))
    stage = _to_str(cmeta.get("current_stage"))
    status = _to_str(task.get("status"))
    elapsed = _elapsed(task.get("started_at"), task.get("ended_at"))

    lines = [f"- **{activity}** ({stage}, {status}, {_fmt_seconds(elapsed)}s)"]

    file_name = meta.get("file_name")
    file_format = meta.get("file_format")
    file_size = _fmt_bytes(_as_float(meta.get("file_size_bytes")))
    if file_name or file_format or file_size != "-":
        lines.append(
            f"  - Input File: `{_to_str(file_name, '-')}` | Format: `{_to_str(file_format, '-')}` | Size: `{file_size}`"
        )

    num_rows = meta.get("num_rows")
    num_cols = meta.get("num_columns")
    if num_rows is not None or num_cols is not None:
        lines.append(f"  - Table Shape: `{_to_str(num_rows, '-')}` rows x `{_to_str(num_cols, '-')}` columns")

    columns = meta.get("columns")
    if columns:
        lines.append(f"  - Columns: {_compact_columns(columns)}")

    mv = meta.get("missing_values_handled")
    if isinstance(mv, dict):
        lines.append(
            f"  - Missing Values: strategy=`{_to_str(mv.get('strategy'), '-')}`, before=`{_to_str(mv.get('before'), '-')}`, after=`{_to_str(mv.get('after'), '-')}`"
        )

    dup = meta.get("duplicates_removed")
    if dup is not None:
        lines.append(f"  - Duplicates Removed: `{_to_str(dup)}`")

    dtypes = meta.get("dtypes_after")
    if isinstance(dtypes, dict) and dtypes:
        dtype_preview = ", ".join(f"{k}:{v}" for k, v in list(dtypes.items())[:8])
        if len(dtypes) > 8:
            dtype_preview += f", ... (+{len(dtypes) - 8} more)"
        lines.append(f"  - Final Dtypes: {dtype_preview}")

    outliers = meta.get("outliers")
    if isinstance(outliers, dict):
        found = outliers.get("found")
        found_summary = "-"
        if isinstance(found, dict):
            non_zero = {k: v for k, v in found.items() if _as_float(v) and _as_float(v) > 0}
            if non_zero:
                top = sorted(non_zero.items(), key=lambda kv: kv[1], reverse=True)[:5]
                found_summary = ", ".join(f"{k}:{v}" for k, v in top)
        lines.append(
            "  - Outliers: "
            f"method=`{_to_str(outliers.get('method'), '-')}`, "
            f"threshold=`{_to_str(outliers.get('threshold'), '-')}`, "
            f"action=`{_to_str(outliers.get('action'), '-')}`, affected=`{found_summary}`"
        )

    norm = meta.get("normalization")
    if isinstance(norm, dict):
        method = _to_str(norm.get("method"), "-")
        cols = norm.get("columns")
        stats = norm.get("stats")
        norm_line = f"  - Normalization: method=`{method}`"
        if isinstance(cols, list):
            norm_line += f", columns=`{len(cols)}`"
        lines.append(norm_line)
        if isinstance(stats, dict) and stats:
            preview = []
            for col, stat in list(stats.items())[:3]:
                if isinstance(stat, dict):
                    mean = _as_float(stat.get("mean"))
                    std = _as_float(stat.get("std"))
                    preview.append(f"{col}(mean={_fmt_seconds(mean) if mean is not None else '-'}, std={_fmt_seconds(std) if std is not None else '-'})")
            if preview:
                lines.append(f"  - Normalization Stats Sample: {', '.join(preview)}")

    nps = meta.get("numpy_structure")
    if isinstance(nps, dict):
        lines.append(
            f"  - Numpy Structure: X Shape=`{_to_str(nps.get('X_shape'), '-')}`, Target=`{_to_str(nps.get('target_column'), '-')}`"
        )

    output_format = meta.get("output_format")
    if output_format is not None:
        lines.append(f"  - Output Format: `{_to_str(output_format)}`")

    saved_files = meta.get("saved_files")
    if isinstance(saved_files, list) and saved_files:
        lines.append(f"  - Saved Files: {', '.join(str(p) for p in saved_files)}")

    return lines


def _filter_empty_columns(headers: List[str], rows: List[List[Any]], keep_indices: List[int]) -> Tuple[List[str], List[List[Any]]]:
    keep = set(keep_indices)
    for col_idx in range(len(headers)):
        if col_idx in keep:
            continue
        has_value = any(row[col_idx] != "-" for row in rows)
        if has_value:
            keep.add(col_idx)
    kept_indices = [i for i in range(len(headers)) if i in keep]
    return [headers[i] for i in kept_indices], [[row[i] for i in kept_indices] for row in rows]


def _resource_rows(tasks_sorted: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]], List[Dict[str, Any]]]:
    headers = [
        "Activity",
        "Elapsed (s)",
        "CPU User (s)",
        "CPU System (s)",
        "CPU (%)",
        "Memory Used",
        "Memory (%)",
        "Disk Used",
        "Disk (%)",
        "Read",
        "Write",
        "Read Ops",
        "Write Ops",
    ]
    rows: List[List[Any]] = []
    deltas: List[Dict[str, Any]] = []

    for task in tasks_sorted:
        delta = compute_telemetry_delta(
            task.get("telemetry_at_start", {}) if isinstance(task.get("telemetry_at_start"), dict) else {},
            task.get("telemetry_at_end", {}) if isinstance(task.get("telemetry_at_end"), dict) else {},
        )
        deltas.append(delta)

        elapsed = _elapsed(task.get("started_at"), task.get("ended_at"))
        rows.append(
            [
                _to_str(task.get("activity_id")),
                _fmt_seconds(elapsed),
                _fmt_seconds(delta["cpu"].get("user_seconds")),
                _fmt_seconds(delta["cpu"].get("system_seconds")),
                _fmt_percent(delta["cpu"].get("percent_points")),
                _fmt_bytes(delta["memory"].get("used_bytes")),
                _fmt_percent(delta["memory"].get("percent_points")),
                _fmt_bytes(delta["disk"].get("used_bytes")),
                _fmt_percent(delta["disk"].get("percent_points")),
                _fmt_bytes(delta["disk"]["io_sum"].get("read_bytes")),
                _fmt_bytes(delta["disk"]["io_sum"].get("write_bytes")),
                _fmt_count(delta["disk"]["io_sum"].get("read_count")),
                _fmt_count(delta["disk"]["io_sum"].get("write_count")),
            ]
        )

    headers, rows = _filter_empty_columns(headers, rows, keep_indices=[0, 1])
    return headers, rows, deltas


def _top_slowest(tasks_sorted: List[Dict[str, Any]], n: int = 5) -> List[Tuple[str, float]]:
    elapsed_pairs: List[Tuple[str, float]] = []
    for t in tasks_sorted:
        elapsed = _elapsed(t.get("started_at"), t.get("ended_at"))
        if elapsed is not None:
            elapsed_pairs.append((_to_str(t.get("activity_id")), elapsed))
    elapsed_pairs.sort(key=lambda x: x[1], reverse=True)
    return elapsed_pairs[:n]


def _extract_git_info(workflow: Dict[str, Any]) -> Dict[str, Any]:
    repo = workflow.get("code_repository", {})
    if not isinstance(repo, dict):
        return {}
    return sanitize_json_like(
        {
            "remote": _to_str(repo.get("remote"), ""),
            "branch": _to_str(repo.get("branch")),
            "short_sha": _to_str(repo.get("short_sha")),
            "dirty": _to_str(repo.get("dirty")),
        }
    )


def render_pipeline_section(task_records: List[Dict[str, Any]]) -> str:
    tasks_sorted = sorted(task_records, key=lambda t: _as_float(t.get("started_at")) or float("inf"))

    input_counter: Counter[str] = Counter()
    for task in tasks_sorted:
        used = task.get("used", {})
        if isinstance(used, dict):
            ip = used.get("input_path")
            if isinstance(ip, str) and ip.strip():
                input_counter[ip.strip()] += 1
    input_path = input_counter.most_common(1)[0][0] if input_counter else "unknown"

    def _task_time(task: Dict[str, Any]) -> float:
        return _as_float(task.get("ended_at")) or _as_float(task.get("started_at")) or float("-inf")

    output_path = "unknown"
    saved_candidates: List[Tuple[float, str]] = []
    for task in tasks_sorted:
        cmeta = task.get("custom_metadata", {})
        if not isinstance(cmeta, dict):
            continue
        meta = cmeta.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        saved = meta.get("saved_files")
        if isinstance(saved, list) and saved:
            first = saved[0]
            if isinstance(first, str) and first.strip():
                saved_candidates.append((_task_time(task), first.strip()))
    if saved_candidates:
        saved_candidates.sort(key=lambda x: x[0])
        output_path = saved_candidates[-1][1]
    else:
        out_candidates: List[Tuple[float, str]] = []
        for task in tasks_sorted:
            used = task.get("used", {})
            if not isinstance(used, dict):
                continue
            out_dir = used.get("output_dir")
            if isinstance(out_dir, str) and out_dir.strip():
                out_candidates.append((_task_time(task), out_dir.strip()))
        if out_candidates:
            out_candidates.sort(key=lambda x: x[0])
            output_path = out_candidates[-1][1]

    rail = "        │"
    down = "        ▼"

    ordered: List[Tuple[str, str]] = []
    for task in tasks_sorted:
        cmeta = task.get("custom_metadata", {})
        stage = "unknown"
        if isinstance(cmeta, dict):
            stage = _to_str(cmeta.get("current_stage"), "unknown")
        ordered.append((_to_str(task.get("activity_id")), stage))

    diagram: List[str] = [input_path, rail, down]
    if not ordered:
        diagram.extend([down, output_path])
    else:
        for idx, (activity, stage) in enumerate(ordered):
            diagram.append(f" {activity}")
            if idx == len(ordered) - 1:
                continue
            next_stage = ordered[idx + 1][1]
            if next_stage != stage:
                diagram.append(f"{rail}──────── {stage}")
            else:
                diagram.append(rail)
        diagram.append(down)
        diagram.append(output_path)

    return "## Pipeline Structure\n\n```text\n" + "\n".join(diagram) + "\n```"


def generate_provenance_card(
    input_jsonl: str = "/tmp/flowcept_buffer.jsonl",
    output_markdown: str = "PROVENANCE_CARD.md",
) -> Dict[str, Any]:
    """Generate PROVENANCE_CARD.md from Flowcept JSONL."""
    input_path = Path(input_jsonl)
    output_path = Path(output_markdown)
    if not input_path.exists():
        raise FileNotFoundError(f"Flowcept JSONL not found: {input_path}")

    records, skipped = _read_jsonl(input_path)
    workflow, tasks = _resolve_records(records)
    workflow = sanitize_json_like(workflow)
    tasks = [sanitize_json_like(t) for t in tasks]

    tasks_sorted = sorted(tasks, key=lambda t: _as_float(t.get("started_at")) or float("inf"))

    starts = [_as_float(t.get("started_at")) for t in tasks_sorted if _as_float(t.get("started_at")) is not None]
    ends = [_as_float(t.get("ended_at")) for t in tasks_sorted if _as_float(t.get("ended_at")) is not None]
    min_start = min(starts) if starts else None
    max_end = max(ends) if ends else None
    total_elapsed = (max_end - min_start) if (min_start is not None and max_end is not None) else None

    unique_inputs = sorted({_task_io(t)[0] for t in tasks_sorted})
    unique_outputs = sorted({_task_io(t)[1] for t in tasks_sorted})
    saved_files = _collect_saved_files(tasks_sorted)

    status_counts = Counter(_to_str(t.get("status")) for t in tasks_sorted)
    stage_counts = Counter(
        _to_str((t.get("custom_metadata", {}) if isinstance(t.get("custom_metadata"), dict) else {}).get("current_stage"))
        for t in tasks_sorted
    )
    timing_rows: List[List[Any]] = []
    for t in tasks_sorted:
        cmeta = t.get("custom_metadata", {}) if isinstance(t.get("custom_metadata"), dict) else {}
        elapsed = _elapsed(t.get("started_at"), t.get("ended_at"))
        timing_rows.append(
            [
                _to_str(t.get("activity_id")),
                _to_str(cmeta.get("current_stage")),
                _to_str(t.get("status")),
                _fmt_ts(t.get("started_at")),
                _fmt_ts(t.get("ended_at")),
                _fmt_seconds(elapsed),
            ]
        )

    resource_headers, resource_rows, telemetry_deltas = _resource_rows(tasks_sorted)

    total_read_bytes = _sum_numeric(d["disk"]["io_sum"].get("read_bytes") for d in telemetry_deltas)
    total_write_bytes = _sum_numeric(d["disk"]["io_sum"].get("write_bytes") for d in telemetry_deltas)
    total_read_count = _sum_numeric(d["disk"]["io_sum"].get("read_count") for d in telemetry_deltas)
    total_write_count = _sum_numeric(d["disk"]["io_sum"].get("write_count") for d in telemetry_deltas)
    total_mem_used = _sum_numeric(d["memory"].get("used_bytes") for d in telemetry_deltas)
    total_disk_used = _sum_numeric(d["disk"].get("used_bytes") for d in telemetry_deltas)

    cpu_percent_values = [d["cpu"].get("percent_points") for d in telemetry_deltas if d["cpu"].get("percent_points") is not None]
    avg_cpu_percent = (sum(cpu_percent_values) / len(cpu_percent_values)) if cpu_percent_values else None

    slowest = _top_slowest(tasks_sorted, n=5)
    io_heavy = sorted(
        (
            (
                _to_str(t.get("activity_id")),
                (_as_float(d["disk"]["io_sum"].get("read_bytes")) or 0.0),
                (_as_float(d["disk"]["io_sum"].get("write_bytes")) or 0.0),
            )
            for t, d in zip(tasks_sorted, telemetry_deltas)
        ),
        key=lambda x: x[1] + x[2],
        reverse=True,
    )[:5]
    cpu_active = sorted(
        (
            (_to_str(t.get("activity_id")), (_as_float(d["cpu"].get("percent_points")) or 0.0))
            for t, d in zip(tasks_sorted, telemetry_deltas)
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    git_info = _extract_git_info(workflow)
    workflow_name = _to_str(workflow.get("name"), "unknown")
    title_name = workflow_name if workflow_name and workflow_name != "unknown" else _to_str(workflow.get("workflow_id"))

    lines: List[str] = []
    lines.append(f"# Data Transformation Pipeline Provenance Card: {title_name}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- **Workflow Name:** `{workflow_name}`")
    lines.append(f"- **Workflow ID:** `{_to_str(workflow.get('workflow_id'))}`")
    lines.append(f"- **Campaign ID:** `{_to_str(workflow.get('campaign_id'))}`")
    lines.append(f"- **Execution Start (UTC):** `{_fmt_ts(min_start)}`")
    lines.append(f"- **Execution End (UTC):** `{_fmt_ts(max_end)}`")
    lines.append(f"- **Total Elapsed (s):** `{_fmt_seconds(total_elapsed)}`")
    lines.append(f"- **System Name:** `{_to_str(workflow.get('sys_name'))}`")
    lines.append(f"- **Environment ID:** `{_to_str(workflow.get('environment_id'))}`")
    lines.append(
        f"- **Code Repository:** `branch={_to_str(git_info.get('branch'))}, short_sha={_to_str(git_info.get('short_sha'))}, dirty={_to_str(git_info.get('dirty'))}`"
    )
    lines.append(f"- **Git Remote:** `{_to_str(git_info.get('remote'))}`")
    lines.append(f"- **Input Paths:** `{unique_inputs}`")
    lines.append(f"- **Output Directories:** `{unique_outputs}`")
    if saved_files:
        lines.append(f"- **Saved Files:** `{saved_files}`")
    lines.append("")

    lines.append("## Workflow-level Summary")
    lines.append(f"- **Total Data Transformations:** `{len(tasks_sorted)}`")
    lines.append(f"- **Status Counts:** `{dict(status_counts)}`")
    lines.append(f"- **Stage Counts:** `{dict(stage_counts)}`")
    lines.append(f"- **Total Elapsed Workflow Time (s):** `{_fmt_seconds(total_elapsed)}`")
    lines.append("- **Top 5 Slowest Data Transformations:**")
    for name, sec in slowest:
        lines.append(f"  - `{name}`: `{_fmt_seconds(sec)} s`")
    lines.append("- **Resource Totals:**")
    lines.append(f"  - `Memory Used`: `{_fmt_bytes(total_mem_used)}`")
    disk_used = _fmt_bytes(total_disk_used)
    if disk_used != "-":
        lines.append(f"  - `Disk Used`: `{disk_used}`")
    lines.append(f"  - `Average CPU (%)`: `{_fmt_percent(avg_cpu_percent)}`")
    lines.append("  - **IO:**")
    lines.append(f"    - `Read`: `{_fmt_bytes(total_read_bytes)}`")
    lines.append(f"    - `Write`: `{_fmt_bytes(total_write_bytes)}`")
    lines.append(f"    - `Read Ops`: `{_fmt_count(total_read_count)}`")
    lines.append(f"    - `Write Ops`: `{_fmt_count(total_write_count)}`")
    lines.append("- **Key Observations:**")
    if slowest:
        lines.append(f"  - Slowest data transformation: `{slowest[0][0]}` at `{_fmt_seconds(slowest[0][1])} s`")
    if io_heavy:
        top_io = io_heavy[0]
        lines.append(
            f"  - Largest IO data transformation: `{top_io[0]}` with Read `{_fmt_bytes(top_io[1])}` and Write `{_fmt_bytes(top_io[2])}`"
        )

    missing_ended = [_to_str(t.get("activity_id")) for t in tasks_sorted if t.get("ended_at") is None]
    if missing_ended:
        lines.append(f"  - Missing end timestamp for: `{missing_ended}`")

    norm_zero_std = []
    for t in tasks_sorted:
        meta = _iter_task_metadata(t)
        norm = meta.get("normalization")
        if isinstance(norm, dict):
            stats = norm.get("stats")
            if isinstance(stats, dict):
                for col, stat in stats.items():
                    if isinstance(stat, dict) and stat.get("std") == 0:
                        norm_zero_std.append(f"{_to_str(t.get('activity_id'))}:{col}")
    if norm_zero_std:
        lines.append(f"  - Normalization std==0 observed for: `{norm_zero_std}`")
    lines.append("")

    lines.append(render_pipeline_section(tasks_sorted))
    lines.append("")

    lines.append("## Timing Report")
    lines.append("Rows are sorted by **Started At** (ascending).")
    lines.append("")
    lines.append(
        _render_table(
            [
                "Data Transformation",
                "Stage",
                "Status",
                "Started At",
                "Ended At",
                "Elapsed (s)",
            ],
            timing_rows,
        )
    )
    lines.append("")

    lines.append("## Resource Usage")
    lines.append(
        _render_table(resource_headers, resource_rows)
    )
    lines.append("")
    lines.append("- **Interpretation:**")
    lines.append("  - Most IO-heavy data transformations (Read + Write):")
    for name, read_b, write_b in io_heavy:
        read_s = _fmt_bytes(read_b)
        write_s = _fmt_bytes(write_b)
        if read_s != "-" or write_s != "-":
            lines.append(f"    - `{name}`: Read={read_s}, Write={write_s}")
    lines.append("  - Most CPU-active data transformations (CPU %):")
    for name, cpu_pct in cpu_active:
        pct = _fmt_percent(cpu_pct)
        if pct != "-":
            lines.append(f"    - `{name}`: CPU={pct}")
    lines.append("")

    lines.append("## Data Transformation Details")
    for task in tasks_sorted:
        lines.extend(_activity_metadata_lines(task))
    lines.append("")
    lines.append("---")
    generated_at = datetime.now().astimezone().strftime("%b %d, %Y at %I:%M %p %Z")
    if FLOWCEPT_VERSION != "unknown":
        lines.append(
            "Provenance card generated by [Flowcept](https://flowcept.org/) | "
            "[GitHub](https://github.com/ORNL/flowcept) | "
            f"[Version: {FLOWCEPT_VERSION}](https://github.com/ORNL/flowcept/releases/tag/v{FLOWCEPT_VERSION}) "
            f"on {generated_at}"
        )
    else:
        lines.append(
            "Provenance card generated by [Flowcept](https://flowcept.org/) | "
            "[GitHub](https://github.com/ORNL/flowcept) | "
            f"Version: unknown on {generated_at}"
        )
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "parsed_lines": len(records),
        "tasks": len(tasks_sorted),
        "skipped_lines": skipped,
        "output": str(output_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate provenance card from Flowcept JSONL")
    parser.add_argument("--input", default="/tmp/flowcept_buffer.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="PROVENANCE_CARD.md", help="Output markdown path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    stats = generate_provenance_card(input_jsonl=args.input, output_markdown=args.output)
    print(
        f"parsed_lines={stats['parsed_lines']} tasks={stats['tasks']} skipped_lines={stats['skipped_lines']} output={stats['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
