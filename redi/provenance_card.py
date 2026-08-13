"""Lightweight provenance helpers backed by Flowcept reports."""

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flowcept.configs import DUMP_BUFFER_PATH
from flowcept.report.service import generate_report


def generate_provenance_artifacts(output_dir: Path, flowcept_ref: Any = None) -> Dict[str, Any]:
    """Copy/move Flowcept buffer into output_dir and generate PROVENANCE_CARD.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_dst = output_dir / "flowcept_buffer.jsonl"
    buffer_src = Path(str(DUMP_BUFFER_PATH)).expanduser()

    if buffer_src.exists() and buffer_src.resolve() != buffer_dst.resolve():
        if buffer_dst.exists():
            buffer_dst.unlink()
        shutil.move(str(buffer_src), str(buffer_dst))
    if not buffer_dst.exists():
        raise FileNotFoundError(f"Flowcept JSONL not found after move: {buffer_dst}")

    card_path = output_dir / "PROVENANCE_CARD.md"
    pdf_path = output_dir / "PROVENANCE_REPORT.pdf"
    report = generate_report(
        report_type="provenance_card",
        format="markdown",
        output_path=str(card_path),
        input_jsonl_path=str(buffer_dst),
    )
    pdf_file = None
    pdf_error = None
    try:
        generate_report(
            report_type="provenance_report",
            format="pdf",
            output_path=str(pdf_path),
            input_jsonl_path=str(buffer_dst),
        )
        pdf_file = str(pdf_path)
    except ModuleNotFoundError as exc:
        pdf_error = str(exc)
    stats = {
        "parsed_lines": report.get("total_records"),
        "tasks": report.get("total_tasks"),
        "skipped_lines": report.get("skipped_lines", 0),
        "output": report.get("output", str(card_path)),
        "flowcept_report": report,
        "pdf_error": pdf_error,
    }

    return {
        "buffer_file": str(buffer_dst) if buffer_dst.exists() else None,
        "card_file": str(card_path),
        "pdf_file": pdf_file,
        "stats": stats,
    }


def link_datacard_to_provenance(datacard_path: Path, provenance_filename: str = "PROVENANCE_CARD.md") -> bool:
    """Insert a provenance link under the '## 2. Provenance' section of the given report, if missing."""
    if not datacard_path.exists():
        return False

    content = datacard_path.read_text(encoding="utf-8")
    if "Data Transformation Pipeline Provenance Card" in content:
        return False

    link_line = f"- [Data Transformation Pipeline Provenance Card]({provenance_filename})"
    section_header = "## 2. Provenance"
    if section_header in content:
        lines = content.splitlines()
        updated_lines = []
        inserted = False
        for i, line in enumerate(lines):
            updated_lines.append(line)
            if line.strip() == section_header and not inserted:
                # Insert after the section header and optional blank line.
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    updated_lines.append(lines[j])
                    j += 1
                updated_lines.append(link_line)
                inserted = True
        if inserted:
            datacard_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")
            return True

    # Fallback: append without creating an extra trailing provenance section.
    datacard_path.write_text(content.rstrip() + "\n" + link_line + "\n", encoding="utf-8")
    return True


def finalize_provenance_for_context(
    ctx: Any,
    flowcept_ref: Any = None,
    verbose: bool = False,
    logger: Any = None,
) -> Dict[str, Any]:
    """Generate provenance artifacts and register them into context metadata."""
    if not getattr(ctx, "with_flowcept", False):
        return {"enabled": False, "generated": False, "error": None}

    try:
        result = generate_provenance_artifacts(output_dir=Path(ctx.output_dir), flowcept_ref=flowcept_ref)

        saved_files = ctx.metadata.setdefault("saved_files", [])
        if not isinstance(saved_files, list):
            saved_files = []
            ctx.metadata["saved_files"] = saved_files
        for key in ("buffer_file", "card_file", "pdf_file"):
            value = result.get(key)
            if value and value not in saved_files:
                saved_files.append(value)

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
            "card_path": Path(result["card_file"]) if result.get("card_file") else None,
            "buffer_path": Path(result["buffer_file"]) if result.get("buffer_file") else None,
        }
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not generate provenance card: %s", exc)
        return {"enabled": True, "generated": False, "error": str(exc)}


def render_provenance_summary(
    ctx: Any,
    style: str,
    datacard_path: Optional[Path] = None,
    print_section: Optional[Callable[[str], None]] = None,
    printer: Callable[[str], None] = print,
    colors: Any = None,
) -> bool:
    """Render provenance artifact summary in CLI output."""
    saved_files = ctx.metadata.get("saved_files", [])
    if not isinstance(saved_files, list):
        return False

    file_names = {Path(str(sf)).name for sf in saved_files}
    has_provenance = "PROVENANCE_CARD.md" in file_names
    has_buffer = "flowcept_buffer.jsonl" in file_names
    has_pdf = "PROVENANCE_REPORT.pdf" in file_names

    if not has_provenance:
        return False

    if datacard_path is not None and datacard_path.exists():
        try:
            link_datacard_to_provenance(datacard_path, provenance_filename="PROVENANCE_CARD.md")
        except Exception:
            pass

    if style == "simple":
        printer("Provenance Card:")
        printer("  Saved: PROVENANCE_CARD.md")
        if has_pdf:
            printer("  Saved: PROVENANCE_REPORT.pdf")
        if has_buffer:
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
    if has_pdf:
        printer(f"    📋 {bold}PROVENANCE_REPORT.pdf{end} {dim}(saved){end}")
    if has_buffer:
        printer(f"    📋 {bold}flowcept_buffer.jsonl{end} {dim}(saved){end}")
    return True


def merge_flowcept_buffers(buffer_paths: list, output_path: Path) -> int:
    """
    Concatenate multiple flowcept_buffer.jsonl files into one.

    Returns the number of lines written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w") as out:
        for p in buffer_paths:
            p = Path(p)
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        n += 1
    return n


def aggregate_parallel_timings(buffer_path: Path) -> dict:
    """
    Compute timing summary from a merged flowcept_buffer.jsonl.

    Returns a dict with:
      - wall_time_s: max(ended_at) - min(started_at)
      - total_compute_s: sum of per-task durations
      - n_tasks: number of completed tasks
      - n_ranks: number of distinct ranks (inferred from unique worker metadata)
      - per_stage: {stage: {count, min_s, mean_s, max_s, total_s}}
    """
    import json as _json
    from collections import defaultdict

    starts, ends = [], []
    stage_durations: dict = defaultdict(list)
    n_tasks = 0

    with open(buffer_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue

            t_start = rec.get("started_at")
            t_end = rec.get("ended_at")
            if t_start is None or t_end is None:
                continue

            duration = float(t_end) - float(t_start)
            if duration < 0:
                continue

            starts.append(float(t_start))
            ends.append(float(t_end))
            n_tasks += 1

            # Stage from tags (first tag) or activity_id
            tags = rec.get("tags") or []
            stage = tags[0] if tags else rec.get("activity_id", "unknown")
            stage_durations[stage].append(duration)

    if not starts:
        return {"wall_time_s": 0.0, "total_compute_s": 0.0, "n_tasks": 0,
                "per_stage": {}}

    wall_time = max(ends) - min(starts)
    total_compute = sum(d for durations in stage_durations.values() for d in durations)

    per_stage = {}
    for stage, durations in sorted(stage_durations.items()):
        per_stage[stage] = {
            "count":   len(durations),
            "min_s":   min(durations),
            "mean_s":  sum(durations) / len(durations),
            "max_s":   max(durations),
            "total_s": sum(durations),
        }

    return {
        "wall_time_s":     wall_time,
        "total_compute_s": total_compute,
        "n_tasks":         n_tasks,
        "per_stage":       per_stage,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate provenance card from Flowcept JSONL")
    parser.add_argument("--output", default="PROVENANCE_CARD.md", help="Output markdown path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = generate_provenance_artifacts(output_dir=Path(args.output).parent)
    stats = result["stats"]
    print(
        f"parsed_lines={stats['parsed_lines']} tasks={stats['tasks']} "
        f"skipped_lines={stats['skipped_lines']} output={stats['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
