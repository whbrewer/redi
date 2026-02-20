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
    report = generate_report(
        report_type="provenance_card",
        format="markdown",
        output_path=str(card_path),
        input_jsonl_path=str(buffer_dst),
    )
    stats = {
        "parsed_lines": report.get("total_records"),
        "tasks": report.get("total_tasks"),
        "skipped_lines": report.get("skipped_lines", 0),
        "output": report.get("output", str(card_path)),
        "flowcept_report": report,
    }

    return {
        "buffer_file": str(buffer_dst) if buffer_dst.exists() else None,
        "card_file": str(card_path),
        "stats": stats,
    }


def link_datacard_to_provenance(datacard_path: Path, provenance_filename: str = "PROVENANCE_CARD.md") -> bool:
    """Append provenance link to DATA_CARD.md if missing."""
    if not datacard_path.exists():
        return False

    content = datacard_path.read_text(encoding="utf-8")
    if "Data Transformation Pipeline Provenance Card" in content:
        return False

    link_block = (
        "\n## Provenance\n"
        f"- [Data Transformation Pipeline Provenance Card]({provenance_filename})\n"
    )
    datacard_path.write_text(content.rstrip() + "\n" + link_block, encoding="utf-8")
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
        for key in ("buffer_file", "card_file"):
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
    if has_buffer:
        printer(f"    📋 {bold}flowcept_buffer.jsonl{end} {dim}(saved){end}")
    return True


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
