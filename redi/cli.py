#!/usr/bin/env python3
"""
REDI CLI - Data Readiness for AI Command Line Interface

Unified entry point for running REDI pipelines.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

# Setup logging - quieter by default
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger("redi")


def print_banner():
    """Print the REDI banner"""
    print(f"""{Colors.BOLD}{Colors.CYAN}
 ██████╗ ███████╗██████╗ ██╗
 ██╔══██╗██╔════╝██╔══██╗██║
 ██████╔╝█████╗  ██║  ██║██║
 ██╔══██╗██╔══╝  ██║  ██║██║
 ██║  ██║███████╗██████╔╝██║
 ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝
{Colors.END}{Colors.DIM} Readiness Engine for Data Integration{Colors.END}
""")


def print_step(icon: str, name: str, status: str, detail: str = "", timing: str = ""):
    """Print a pipeline step with status"""
    status_colors = {
        "running": Colors.YELLOW,
        "success": Colors.GREEN,
        "skipped": Colors.DIM,
        "error": Colors.RED,
    }
    color = status_colors.get(status, Colors.END)

    timing_str = f" {Colors.DIM}({timing}){Colors.END}" if timing else ""
    detail_str = f" {Colors.DIM}→ {detail}{Colors.END}" if detail else ""

    print(f"  {icon} {color}{name}{Colors.END}{detail_str}{timing_str}")


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ {title}{Colors.END}")


def print_metric(label: str, value: str, indent: int = 4):
    """Print a metric"""
    print(f"{' ' * indent}{Colors.DIM}{label}:{Colors.END} {Colors.BOLD}{value}{Colors.END}")


def detect_domain(input_path: Path, columns: Optional[List[str]] = None) -> str:
    """Auto-detect domain from file and content"""
    from .domains import detect_domain as _detect_domain, DOMAIN_EXTENSIONS

    ext = input_path.suffix.lower()
    for domain, extensions in DOMAIN_EXTENSIONS.items():
        if ext in extensions:
            return domain.value

    if ext in [".csv", ".tsv", ".parquet", ".json"]:
        if columns is None:
            try:
                import pandas as pd
                if ext == ".parquet":
                    df = pd.read_parquet(input_path, columns=None)
                    columns = list(df.columns)
                else:
                    df = pd.read_csv(input_path, nrows=0)
                    columns = list(df.columns)
            except Exception:
                columns = []

        detected = _detect_domain(columns=columns, file_extension=ext)
        return detected.value

    return "general"


def run_pipeline(
    input_path: str,
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None,
    verbose: bool = False,
    style: str = "rich"
):
    """Run the REDI pipeline with engaging visual output"""
    from .pipeline import Pipeline, PipelineStage, create_context

    input_path = Path(input_path)
    if not input_path.exists():
        if style == "rich":
            print(f"{Colors.RED}✗ Input file not found: {input_path}{Colors.END}")
        else:
            print(f"Error: Input file not found: {input_path}")
        return None

    # Print banner (rich mode only)
    if style == "rich":
        print_banner()

    # Show input info
    file_size = input_path.stat().st_size
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"

    if style == "rich":
        print_section("Input")
        print_metric("File", str(input_path))
        print_metric("Size", size_str)
        print_metric("Format", input_path.suffix.upper().replace(".", ""))
    elif style == "simple":
        print(f"Input: {input_path} ({size_str})")

    # Auto-detect domain
    if domain is None:
        domain = detect_domain(input_path)

    domain_icons = {
        "climate": "🌍",
        "materials": "⚛️ ",
        "life_sciences": "🧬",
        "fusion": "☀️ ",
        "hpc": "🖥️ ",
        "general": "📊",
    }
    icon = domain_icons.get(domain, "📊")

    if style == "rich":
        print_section("Domain Detection")
        print(f"    {icon} Detected: {Colors.BOLD}{Colors.CYAN}{domain.upper()}{Colors.END}")
    elif style == "simple":
        print(f"Domain: {domain}")

    # Create context
    ctx = create_context(str(input_path), output_dir, domain, **(config or {}))

    # Load domain-specific pipeline
    if style == "rich":
        print_section("Loading Pipeline")

    try:
        if domain == "climate":
            from .domains.climate import get_full_pipeline
        elif domain == "materials":
            from .domains.materials import get_full_pipeline
        elif domain == "life_sciences":
            from .domains.life_sciences import get_full_pipeline
        elif domain == "fusion":
            from .domains.fusion import get_full_pipeline
        elif domain == "hpc":
            from .domains.hpc import get_full_pipeline
        else:
            get_full_pipeline = None

        if get_full_pipeline:
            pipeline = Pipeline(domain)
            steps = get_full_pipeline()
            for step in steps:
                if stages is None or step.stage.value in (stages if isinstance(stages, list) else [stages]):
                    pipeline.add_step(step)
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} steps for {domain} domain")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps")
        else:
            # Fall back to generic pipeline
            from .pipeline.ingest import get_ingest_steps
            from .pipeline.preprocess import get_preprocess_steps
            from .pipeline.transform import get_transform_steps
            from .pipeline.structure import get_structure_steps
            from .pipeline.output import get_output_steps

            pipeline = Pipeline(domain)
            all_stages = ["ingest", "preprocess", "transform", "structure", "output"]
            if stages is None:
                stages = all_stages

            if "ingest" in stages:
                for step in get_ingest_steps():
                    pipeline.add_step(step)
            if "preprocess" in stages:
                for step in get_preprocess_steps(domain):
                    pipeline.add_step(step)
            if "transform" in stages:
                for step in get_transform_steps(domain):
                    pipeline.add_step(step)
            if "structure" in stages:
                for step in get_structure_steps(domain):
                    pipeline.add_step(step)
            if "output" in stages:
                for step in get_output_steps():
                    pipeline.add_step(step)
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} generic steps")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps")

    except ImportError as e:
        if style == "rich":
            print(f"    {Colors.YELLOW}⚠ Could not load domain module: {e}{Colors.END}")
        else:
            print(f"Error: Could not load domain module: {e}")
        return None

    # Execute pipeline with visual feedback
    if style == "rich":
        print_section("Processing Pipeline")
    elif style == "simple":
        print("Processing...")

    total_start = time.time()
    step_results = []
    prev_error_count = 0

    for i, step in enumerate(pipeline.steps, 1):
        step_start = time.time()

        # Show running status (rich mode)
        if style == "rich":
            print(f"  ⏳ {Colors.YELLOW}{step.name}{Colors.END} ...", end="", flush=True)

        # Validate
        if not step.validate(ctx):
            step_time = time.time() - step_start
            if style == "rich":
                print(f"\r  ⏭️  {Colors.DIM}{step.name}{Colors.END} → skipped (validation) {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
            elif style == "simple":
                print(f"  [SKIP] {step.name}")
            step_results.append(("skipped", step.name, "", step_time))
            continue

        # Execute
        try:
            old_metadata = dict(ctx.metadata)
            ctx = step.execute(ctx)
            step_time = time.time() - step_start

            # Determine what changed
            detail = ""
            if ctx.metadata.get("records_loaded") and "records_loaded" not in old_metadata:
                detail = f"{ctx.metadata['records_loaded']:,} records"
            elif ctx.metadata.get("array_shape") and "array_shape" not in old_metadata:
                detail = f"shape {ctx.metadata['array_shape']}"
            elif ctx.metadata.get("features_added"):
                detail = f"+{len(ctx.metadata['features_added'])} features"
            elif ctx.metadata.get("engineered_features") and "engineered_features" not in old_metadata:
                detail = f"{len(ctx.metadata['engineered_features'])} features"
            elif ctx.metadata.get("saved_files") and "saved_files" not in old_metadata:
                detail = f"{len(ctx.metadata['saved_files'])} files"
            elif ctx.metadata.get("split") and "split" not in old_metadata:
                split = ctx.metadata["split"]
                detail = f"train={split.get('train_size', '?')}, val={split.get('val_size', '?')}, test={split.get('test_size', '?')}"

            # Check for errors added during this step
            new_errors = len(ctx.errors) > prev_error_count
            prev_error_count = len(ctx.errors)

            if new_errors:
                if style == "rich":
                    print(f"\r  ⚠️  {Colors.YELLOW}{step.name}{Colors.END} → {ctx.errors[-1][:40]}... {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                elif style == "simple":
                    print(f"  [WARN] {step.name}: {ctx.errors[-1][:60]}")
                step_results.append(("warning", step.name, ctx.errors[-1], step_time))
            else:
                if style == "rich":
                    detail_str = f" → {detail}" if detail else ""
                    print(f"\r  ✅ {Colors.GREEN}{step.name}{Colors.END}{detail_str} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                elif style == "simple":
                    detail_str = f": {detail}" if detail else ""
                    print(f"  [OK] {step.name}{detail_str}")
                step_results.append(("success", step.name, detail, step_time))

        except Exception as e:
            step_time = time.time() - step_start
            ctx.add_error(f"Error in step {step.name}: {str(e)}")
            prev_error_count = len(ctx.errors)
            if style == "rich":
                print(f"\r  ❌ {Colors.RED}{step.name}{Colors.END} → {str(e)[:50]} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
            elif style == "simple":
                print(f"  [FAIL] {step.name}: {str(e)[:60]}")
            step_results.append(("error", step.name, str(e), step_time))

    total_time = time.time() - total_start

    # Summary
    success_count = sum(1 for r in step_results if r[0] == "success")
    warning_count = sum(1 for r in step_results if r[0] == "warning")
    error_count = sum(1 for r in step_results if r[0] == "error")
    skipped_count = sum(1 for r in step_results if r[0] == "skipped")

    if style == "rich":
        print_section("Results")

        status_parts = []
        if success_count:
            status_parts.append(f"{Colors.GREEN}✓ {success_count} succeeded{Colors.END}")
        if warning_count:
            status_parts.append(f"{Colors.YELLOW}⚠ {warning_count} warnings{Colors.END}")
        if error_count:
            status_parts.append(f"{Colors.RED}✗ {error_count} failed{Colors.END}")
        if skipped_count:
            status_parts.append(f"{Colors.DIM}⏭ {skipped_count} skipped{Colors.END}")

        print(f"    {' │ '.join(status_parts)}")
        print(f"    {Colors.DIM}Total time: {total_time:.2f}s{Colors.END}")

        # Output files
        if ctx.metadata.get("saved_files"):
            print_section("Output Files")
            for f in ctx.metadata["saved_files"]:
                fpath = Path(f)
                if fpath.exists():
                    fsize = fpath.stat().st_size
                    size_str = f"{fsize / 1024:.1f} KB" if fsize < 1024*1024 else f"{fsize / (1024*1024):.2f} MB"
                    print(f"    📄 {Colors.BOLD}{fpath.name}{Colors.END} {Colors.DIM}({size_str}){Colors.END}")
                else:
                    print(f"    📄 {fpath.name}")

        # Data summary
        if ctx.metadata.get("array_shape"):
            print_section("Data Summary")
            print_metric("Shape", str(ctx.metadata["array_shape"]))
            if ctx.metadata.get("split"):
                split = ctx.metadata["split"]
                print_metric("Train", str(split.get("train_size", "N/A")))
                print_metric("Val", str(split.get("val_size", "N/A")))
                print_metric("Test", str(split.get("test_size", "N/A")))

        # Generate and save Data Card
        print_section("Data Card")
        try:
            from .analyze import generate_report
            report = generate_report(str(input_path))
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))
            print(f"    📋 {Colors.BOLD}DATA_CARD.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
        except Exception as e:
            print(f"    {Colors.YELLOW}⚠ Could not generate Data Card: {e}{Colors.END}")

        # Final status
        print()
        if error_count == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Pipeline completed successfully!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Pipeline completed with {error_count} error(s){Colors.END}")
        print()

    elif style == "simple":
        print(f"Done: {success_count} ok, {warning_count} warnings, {error_count} errors ({total_time:.2f}s)")
        if ctx.metadata.get("saved_files"):
            for f in ctx.metadata["saved_files"]:
                print(f"  Output: {f}")

    elif style == "json":
        import json
        result = {
            "input": str(input_path),
            "domain": domain,
            "steps": [{"status": r[0], "name": r[1], "detail": r[2], "time_ms": r[3]*1000} for r in step_results],
            "summary": {"success": success_count, "warnings": warning_count, "errors": error_count, "skipped": skipped_count},
            "total_time_s": total_time,
            "output_files": ctx.metadata.get("saved_files", []),
            "errors": ctx.errors
        }
        print(json.dumps(result, indent=2))

    return ctx


def analyze_dataset(input_path: str) -> str:
    """Run data card analysis only"""
    print_banner()
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from .analyze import generate_report
        return generate_report(input_path)
    except ImportError as e:
        return f"{Colors.RED}Error: Could not import analyze module: {e}{Colors.END}"


def main():
    parser = argparse.ArgumentParser(
        description="REDI - Data Readiness for AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                      # Auto-detect and run full pipeline
  %(prog)s data.csv --domain hpc         # Specify domain
  %(prog)s data.csv -o ./processed       # Custom output directory
  %(prog)s data.csv --stages ingest,preprocess  # Run specific stages
  %(prog)s --analyze data.csv            # Generate data card only
        """
    )

    parser.add_argument("input", help="Input file path")
    parser.add_argument("-d", "--domain", choices=["climate", "materials", "life_sciences", "fusion", "hpc", "general"],
                       help="Scientific domain (auto-detected if not specified)")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-s", "--stages", help="Comma-separated list of stages to run")
    parser.add_argument("-f", "--format", choices=["npz", "parquet", "csv", "hdf5", "zarr"],
                       default="npz", help="Output format: npz (default), parquet, csv, hdf5, zarr")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-a", "--analyze", action="store_true", help="Only analyze and generate data card")
    parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                       help="Output style: rich (default), simple, or json")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle analyze mode
    if args.analyze:
        result = analyze_dataset(args.input)
        print(result)
        return

    # Run pipeline
    stages = args.stages.split(",") if args.stages else None
    ctx = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        domain=args.domain,
        stages=stages,
        config={"output_format": args.format},
        verbose=args.verbose,
        style=args.style
    )


if __name__ == "__main__":
    main()
