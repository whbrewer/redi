#!/usr/bin/env python3
"""
REDI CLI - Data Readiness for AI Command Line Interface

Unified entry point for running REDI pipelines.
"""

import argparse
import glob
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


def expand_input_paths(input_args: List[str]) -> List[Path]:
    """Expand input arguments to a list of file paths.

    Supports:
    - Single file paths
    - Multiple file paths
    - Glob patterns (*.nc, data_*.csv, etc.)

    Args:
        input_args: List of input paths or patterns

    Returns:
        Sorted list of Path objects
    """
    expanded_paths = []

    for arg in input_args:
        # Check if it's a glob pattern
        if '*' in arg or '?' in arg or '[' in arg:
            # Expand glob pattern
            matches = glob.glob(arg, recursive=True)
            if matches:
                expanded_paths.extend(Path(m) for m in matches)
            else:
                # No matches - keep original (will fail validation later)
                expanded_paths.append(Path(arg))
        else:
            expanded_paths.append(Path(arg))

    # Sort by filename for consistent ordering (important for time-series)
    expanded_paths = sorted(set(expanded_paths), key=lambda p: p.name)

    return expanded_paths


def run_pipeline(
    input_path: Optional[str] = None,
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None,
    verbose: bool = False,
    style: str = "rich",
    input_paths: Optional[List[str]] = None
):
    """Run the REDI pipeline with engaging visual output

    Args:
        input_path: Single input file path (for backward compatibility)
        output_dir: Output directory
        domain: Scientific domain (auto-detected if not specified)
        stages: List of stages to run
        config: Additional configuration
        verbose: Enable verbose output
        style: Output style (rich, simple, json)
        input_paths: List of input file paths for multi-file processing
    """
    from .pipeline import Pipeline, PipelineStage, create_context

    # Handle input paths
    if input_paths:
        # Multi-file mode - paths already expanded
        paths = [Path(p) for p in input_paths]
    elif input_path:
        # Single file mode (backward compatible)
        paths = [Path(input_path)]
    else:
        if style == "rich":
            print(f"{Colors.RED}✗ No input files specified{Colors.END}")
        else:
            print("Error: No input files specified")
        return None

    # Validate all files exist
    missing_files = [p for p in paths if not p.exists()]
    if missing_files:
        if style == "rich":
            print(f"{Colors.RED}✗ Input file(s) not found:{Colors.END}")
            for f in missing_files[:5]:
                print(f"    {f}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
        else:
            print(f"Error: Input file(s) not found: {missing_files[0]}")
        return None

    # Use first file as primary for backward compatibility
    primary_path = paths[0]

    # Print banner (rich mode only)
    if style == "rich":
        print_banner()

    # Show input info
    if len(paths) == 1:
        # Single file
        file_size = primary_path.stat().st_size
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"

        if style == "rich":
            print_section("Input")
            print_metric("File", str(primary_path))
            print_metric("Size", size_str)
            print_metric("Format", primary_path.suffix.upper().replace(".", ""))
        elif style == "simple":
            print(f"Input: {primary_path} ({size_str})")
    else:
        # Multiple files
        total_size = sum(p.stat().st_size for p in paths)
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024*1024 else f"{total_size / (1024*1024):.1f} MB"
        if total_size >= 1024*1024*1024:
            size_str = f"{total_size / (1024*1024*1024):.2f} GB"

        if style == "rich":
            print_section("Input")
            print_metric("Files", f"{len(paths)} files")
            print_metric("Total Size", size_str)
            print_metric("Format", primary_path.suffix.upper().replace(".", ""))
            print_metric("First", paths[0].name)
            print_metric("Last", paths[-1].name)
        elif style == "simple":
            print(f"Input: {len(paths)} files ({size_str})")

    # Auto-detect domain
    domain_was_auto = domain is None
    if domain is None:
        domain = detect_domain(primary_path)

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
        if domain == "general" and domain_was_auto:
            print(f"    {Colors.YELLOW}⚠ Could not auto-detect domain{Colors.END}")
            print(f"    {Colors.DIM}Tip: Use -d/--domain to specify: climate, materials, life_sciences, fusion, hpc{Colors.END}")
        else:
            print(f"    {icon} Detected: {Colors.BOLD}{Colors.CYAN}{domain.upper()}{Colors.END}")
    elif style == "simple":
        if domain == "general" and domain_was_auto:
            print(f"Warning: Could not auto-detect domain. Use -d to specify.")
        print(f"Domain: {domain}")

    # Create context
    safe_config = dict(config or {})
    for reserved_key in ("input_path", "input", "output_dir", "domain", "input_paths"):
        safe_config.pop(reserved_key, None)

    # Pass all paths to context
    ctx = create_context(
        input_path=str(primary_path),
        output_dir=output_dir,
        domain=domain,
        input_paths=[str(p) for p in paths],
        **safe_config
    )

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
            if verbose:
                # In verbose mode, print on own line so logs don't get overwritten
                print(f"  ⏳ {Colors.YELLOW}{step.name}{Colors.END}: {step.description}")
            else:
                print(f"  ⏳ {Colors.YELLOW}{step.name}{Colors.END} ...", end="", flush=True)

        # Validate
        if not step.validate(ctx):
            step_time = time.time() - step_start
            if style == "rich":
                if verbose:
                    print(f"     {Colors.DIM}↳ skipped (validation failed){Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                else:
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
                error_msg = ctx.errors[-1]
                # Check for critical errors that should stop the pipeline
                is_critical = any(phrase in error_msg.lower() for phrase in [
                    "unknown file format",
                    "file not found",
                    "no data loaded",
                    "all files failed",
                    "missing dependency",
                    "install with:"
                ])

                if style == "rich":
                    if is_critical:
                        if verbose:
                            print(f"     {Colors.RED}↳ error: {error_msg}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                        else:
                            print(f"\r  ❌ {Colors.RED}{step.name}{Colors.END} → {error_msg[:40]}... {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                    else:
                        if verbose:
                            print(f"     {Colors.YELLOW}↳ warning: {error_msg}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                        else:
                            print(f"\r  ⚠️  {Colors.YELLOW}{step.name}{Colors.END} → {error_msg[:40]}... {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                elif style == "simple":
                    print(f"  [{'FAIL' if is_critical else 'WARN'}] {step.name}: {error_msg[:60]}")

                if is_critical:
                    step_results.append(("error", step.name, error_msg, step_time))
                    if style == "rich":
                        print(f"\n    {Colors.RED}Pipeline stopped due to critical error.{Colors.END}")
                    break
                else:
                    step_results.append(("warning", step.name, error_msg, step_time))
            else:
                if style == "rich":
                    detail_str = f" → {detail}" if detail else ""
                    if verbose:
                        print(f"     {Colors.GREEN}↳ done{Colors.END}{detail_str} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                    else:
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
                if verbose:
                    print(f"     {Colors.RED}↳ error: {str(e)}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                else:
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
            from .analyze import generate_report, analyze_dataset, PipelineRecommender

            # Generate full report for file (use primary path for multi-file)
            report = generate_report(str(primary_path))
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))
            print(f"    📋 {Colors.BOLD}DATA_CARD.md{Colors.END} {Colors.DIM}(saved){Colors.END}")

            # Save metadata.json (needed for catalog export)
            metadata_path = ctx.output_dir / "metadata.json"
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            import json as json_module
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
            print(f"    📋 {Colors.BOLD}metadata.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

            # Display recommendations in terminal
            data_card = analyze_dataset(str(primary_path))
            recommender = PipelineRecommender(data_card)
            rec = recommender.get_recommendations()

            # Track which recommended steps have been completed in this pipeline run
            completed_steps = set()
            for status, step_name, _, _ in step_results:
                if status == "success":
                    # Map pipeline step names to recommendation step names
                    step_lower = step_name.lower()
                    if "regrid" in step_lower:
                        completed_steps.add("regrid")
                    if "normalize" in step_lower:
                        completed_steps.add("normalize")
                    if "shard" in step_lower or "save" in step_lower or "output" in step_lower:
                        completed_steps.add("shard")
                    if "load" in step_lower or "ingest" in step_lower:
                        completed_steps.add("download")
                    if "featurize" in step_lower or "engineer" in step_lower:
                        completed_steps.add("featurize")
                    if "encode" in step_lower:
                        completed_steps.add("encode")
                    if "label" in step_lower:
                        completed_steps.add("label")

            print_section("Pipeline Recommendations")
            print(f"    {Colors.BOLD}Current Level:{Colors.END} {rec['current_level']}")
            print(f"    {Colors.BOLD}Target Level:{Colors.END} {rec['target_level']}")

            # Show completed steps
            completed_rec_steps = [step for step in rec['needed_steps'] if step['name'] in completed_steps]
            if completed_rec_steps:
                print()
                print(f"    {Colors.BOLD}Completed This Run:{Colors.END}")
                for step in completed_rec_steps:
                    print(f"      {Colors.GREEN}✓{Colors.END} {step['name']}: {step['description']}")

            # Show pending steps
            pending_steps = [step for step in rec['needed_steps']
                           if step['status'] == 'pending' and step['name'] not in completed_steps]
            if pending_steps:
                print()
                print(f"    {Colors.BOLD}Next Steps:{Colors.END}")
                for step in pending_steps:
                    print(f"      {Colors.YELLOW}○{Colors.END} {step['name']}: {step['description']}")

            print()
            print(f"    {Colors.BOLD}Recommended Tools:{Colors.END} {', '.join(rec['recommended_tools'])}")
            print(f"    {Colors.BOLD}Output Formats:{Colors.END} {', '.join(rec['output_formats'])}")

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

        # Save metadata.json and DATA_CARD.md for simple/json modes too
        try:
            from .analyze import generate_report
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            # Save DATA_CARD.md
            report = generate_report(str(primary_path))
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))

            # Save metadata.json
            metadata_path = ctx.output_dir / "metadata.json"
            import json as json_module
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
        except Exception:
            pass  # Silent in simple mode

        if ctx.metadata.get("saved_files"):
            for f in ctx.metadata["saved_files"]:
                print(f"  Output: {f}")

    elif style == "json":
        # Save metadata.json and DATA_CARD.md for json mode too
        try:
            from .analyze import generate_report
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            report = generate_report(str(primary_path))
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))

            metadata_path = ctx.output_dir / "metadata.json"
            import json as json_module
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
        except Exception:
            pass

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


def run_catalog_export(
    metadata_path: str,
    catalog_type: str = "huggingface",
    host: str = None,
    token: str = None,
    token_env: str = None,
    service_name: str = "redi",
    repo_id: str = None,
    private: bool = True,
    style: str = "rich",
) -> bool:
    """Export REDI metadata to a data catalog.

    Args:
        metadata_path: Path to REDI metadata.json file
        catalog_type: Type of catalog (openmetadata, mlflow, intake, huggingface)
        host: Catalog server URL (for OpenMetadata)
        token: Authentication token
        token_env: Environment variable name for token
        service_name: Service name in the catalog
        repo_id: Repository ID for HuggingFace
        private: Make HuggingFace dataset private
        style: Output style (rich, simple, json)

    Returns:
        True if export succeeded
    """
    import os

    try:
        from .catalog import export_to_catalog, list_catalogs

        # Determine token env var based on catalog type
        if token_env is None:
            if catalog_type == "huggingface":
                token_env = "HF_TOKEN"
            else:
                token_env = "OPENMETADATA_TOKEN"

        # Get token from environment if not provided
        if not token:
            token = os.environ.get(token_env, "")
            # Also try alternative HF token env var
            if not token and catalog_type == "huggingface":
                token = os.environ.get("HUGGINGFACE_TOKEN", "")

        # Validation depends on catalog type
        if catalog_type == "huggingface":
            # HuggingFace doesn't need host, but needs token
            if not token:
                if style == "rich":
                    print_section("Catalog Export")
                    print(f"    {Colors.YELLOW}⚠ HuggingFace token not configured{Colors.END}")
                    print(f"    {Colors.DIM}Set HF_TOKEN env var or use --catalog-token{Colors.END}")
                return False

            if style == "rich":
                print_section("Catalog Export")
                print(f"    {Colors.DIM}Type:{Colors.END} {catalog_type}")
                print(f"    {Colors.DIM}Repo:{Colors.END} {repo_id or '(auto-generated)'}")
                print(f"    {Colors.DIM}Private:{Colors.END} {private}")

            # Run HuggingFace export
            result = export_to_catalog(
                metadata_path=metadata_path,
                catalog_type=catalog_type,
                token=token,
                repo_id=repo_id,
                private=private,
            )
        else:
            # OpenMetadata and others need host
            if not host:
                if style == "rich":
                    print_section("Catalog Export")
                    print(f"    {Colors.YELLOW}⚠ Catalog host URL not configured{Colors.END}")
                    print(f"    {Colors.DIM}Set 'catalog.host' in config or use --catalog-host{Colors.END}")
                return False

            if not token:
                if style == "rich":
                    print_section("Catalog Export")
                    print(f"    {Colors.YELLOW}⚠ Catalog token not configured{Colors.END}")
                    print(f"    {Colors.DIM}Set {token_env} env var or use --catalog-token{Colors.END}")
                return False

            if style == "rich":
                print_section("Catalog Export")
                print(f"    {Colors.DIM}Type:{Colors.END} {catalog_type}")
                print(f"    {Colors.DIM}Host:{Colors.END} {host}")
                print(f"    {Colors.DIM}Service:{Colors.END} {service_name}")

            # Run export for non-HF catalogs
            result = export_to_catalog(
                metadata_path=metadata_path,
                catalog_type=catalog_type,
                host=host,
                token=token,
                service_name=service_name,
            )

        if result.success:
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Exported to catalog")
                print(f"    {Colors.DIM}Entity:{Colors.END} {Colors.BOLD}{result.entity_id}{Colors.END}")
                if result.entity_url:
                    print(f"    {Colors.DIM}URL:{Colors.END} {result.entity_url}")
            elif style == "simple":
                print(f"Catalog export: {result.entity_id}")
            elif style == "json":
                import json
                print(json.dumps(result.to_dict(), indent=2))
            return True
        else:
            if style == "rich":
                print(f"    {Colors.RED}✗{Colors.END} Export failed: {result.message}")
            elif style == "simple":
                print(f"Catalog export failed: {result.message}")
            return False

    except ImportError as e:
        if style == "rich":
            print(f"    {Colors.YELLOW}⚠ Catalog module error: {e}{Colors.END}")
        return False
    except Exception as e:
        if style == "rich":
            print(f"    {Colors.RED}✗ Catalog export error: {e}{Colors.END}")
        return False


def run_fair_assessment(
    input_path: str,
    domain: Optional[str] = None,
    fair_config_path: Optional[str] = None,
    style: str = "rich",
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Run FAIR principles assessment.

    Args:
        input_path: Path to data file
        domain: Scientific domain
        fair_config_path: Path to FAIR metadata YAML
        style: Output style (rich, simple, json)
        output_dir: Directory to save FAIR report

    Returns:
        Path to saved FAIR report or None
    """
    try:
        from .fair import FAIRAssessor, FAIRMetadata, FAIRRecommender, load_fair_metadata
        from .analyze import DataAnalyzer

        # Load DataCard for context
        data_card = None
        detected_domain = domain
        try:
            analyzer = DataAnalyzer(input_path)
            data_card = analyzer.analyze()
            if not detected_domain:
                detected_domain = data_card.detected_domain
        except Exception:
            pass

        detected_domain = detected_domain or "general"

        # Load external FAIR metadata if provided
        fair_metadata = None
        if fair_config_path:
            try:
                fair_metadata = load_fair_metadata(fair_config_path)
            except Exception as e:
                if style == "rich":
                    print(f"    {Colors.YELLOW}Warning: Could not load FAIR config: {e}{Colors.END}")

        # Run assessment
        assessor = FAIRAssessor(domain=detected_domain)
        fair_score = assessor.assess(
            file_path=input_path,
            data_card=data_card,
            metadata=fair_metadata,
        )

        # Display results
        if style == "rich":
            print_section("FAIR Assessment")
            print(f"    {Colors.BOLD}Overall Score:{Colors.END} {fair_score.overall_score:.0%}")
            print()

            # Category scores
            for category, score in fair_score.category_scores.items():
                if score >= 0.8:
                    color = Colors.GREEN
                    status = "Excellent"
                elif score >= 0.6:
                    color = Colors.CYAN
                    status = "Good"
                elif score >= 0.4:
                    color = Colors.YELLOW
                    status = "Needs Work"
                else:
                    color = Colors.RED
                    status = "Poor"
                print(f"    {category.capitalize():15} {color}{score:>5.0%}{Colors.END} ({status})")

            # Quick recommendations
            recommender = FAIRRecommender(fair_score, domain=detected_domain)
            quick_wins = recommender.get_quick_wins()
            if quick_wins:
                print()
                print(f"    {Colors.BOLD}Quick Wins:{Colors.END}")
                for win in quick_wins[:3]:
                    print(f"      {Colors.CYAN}→{Colors.END} {win}")

        elif style == "simple":
            print(f"FAIR Score: {fair_score.overall_score:.0%}")
            for category, score in fair_score.category_scores.items():
                print(f"  {category}: {score:.0%}")

        elif style == "json":
            import json
            print(json.dumps(fair_score.to_dict(), indent=2))

        # Save report if output_dir provided
        saved_path = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "FAIR_ASSESSMENT.md"
            report_path.write_text(fair_score.to_markdown())
            saved_path = str(report_path)

            if style == "rich":
                print()
                print(f"    📋 {Colors.BOLD}FAIR_ASSESSMENT.md{Colors.END} {Colors.DIM}(saved){Colors.END}")

        return saved_path

    except ImportError as e:
        if style == "rich":
            print(f"    {Colors.YELLOW}⚠ FAIR module not available: {e}{Colors.END}")
        return None
    except Exception as e:
        if style == "rich":
            print(f"    {Colors.RED}✗ FAIR assessment failed: {e}{Colors.END}")
        return None


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
  %(prog)s pipeline.yaml                 # Run from YAML (input_path in config)

Multi-file examples:
  %(prog)s data_*.nc                     # Process multiple NetCDF files (glob)
  %(prog)s file1.nc file2.nc file3.nc    # Process explicit file list
  %(prog)s "/path/to/data/*.nc"          # Quoted glob pattern

Catalog export examples:
  %(prog)s data.csv --catalog                    # Export to HuggingFace Hub
  %(prog)s data.csv --catalog --catalog-repo org/my-dataset
  %(prog)s data.csv --catalog --catalog-type openmetadata --catalog-host http://localhost:8585/api
        """
    )

    parser.add_argument("input", nargs="*", help="Input file path(s), glob pattern, or YAML config")
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
    parser.add_argument("-c", "--config", help="YAML configuration file path")
    parser.add_argument("--fair", action="store_true", help="Run FAIR principles assessment")
    parser.add_argument("--fair-config", help="YAML file with FAIR metadata (DOI, license, etc.)")
    parser.add_argument("--catalog", action="store_true", help="Export to data catalog after processing")
    parser.add_argument("--catalog-type", choices=["openmetadata", "mlflow", "intake", "huggingface"],
                       default="huggingface", help="Data catalog type (default: huggingface)")
    parser.add_argument("--catalog-host", help="Catalog server URL (e.g., http://localhost:8585/api)")
    parser.add_argument("--catalog-token", help="Catalog authentication token (or set HF_TOKEN for HuggingFace)")
    parser.add_argument("--catalog-service", default="redi", help="Service name in catalog (default: redi)")
    parser.add_argument("--catalog-repo", help="Repository ID for HuggingFace (e.g., your-org/dataset-name)")
    parser.add_argument("--catalog-private", action="store_true", default=True,
                       help="Make HuggingFace dataset private (default: True)")

    args = parser.parse_args()

    config_path = args.config
    input_args = args.input  # Now a list

    # Check if first input is a YAML config file
    if not config_path and input_args and len(input_args) == 1:
        if input_args[0].endswith((".yml", ".yaml")):
            config_path = input_args[0]
            input_args = []

    # Expand glob patterns in input arguments
    input_paths = []
    if input_args:
        input_paths = expand_input_paths(input_args)

    # Load YAML config if provided
    config = {}
    if config_path:
        from .config import load_yaml_config, merge_with_defaults, validate_config
        try:
            yaml_config = load_yaml_config(config_path)
            config = merge_with_defaults(yaml_config)

            # Validate configuration
            issues = validate_config(config)
            if issues:
                for issue in issues:
                    print(f"{Colors.YELLOW}Warning: {issue}{Colors.END}")
        except FileNotFoundError as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI arguments override YAML config
    if args.domain:
        config['domain'] = args.domain
    if args.output != './output':
        config['output_dir'] = args.output
    if args.format != 'npz':
        config['output_format'] = args.format
    if args.verbose:
        config['verbose'] = True
    if args.stages:
        config['stages'] = args.stages.split(',')

    # Set default output_format if not in config
    if 'output_format' not in config:
        config['output_format'] = args.format

    verbose = args.verbose or config.get('verbose', False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle input from config if not provided on CLI
    if not input_paths:
        config_input = config.get('input_path') or config.get('input')
        config_input_paths = config.get('input_paths')

        if config_input_paths:
            # Multi-file from config
            input_paths = expand_input_paths(config_input_paths)
        elif config_input:
            # Single file from config (may be a glob pattern)
            input_paths = expand_input_paths([config_input])

    if not input_paths:
        print(f"{Colors.RED}Error: input file path(s) required. Provide on CLI or set "
              f"'input_path' or 'input_paths' in the YAML config.{Colors.END}")
        sys.exit(1)

    # For analyze mode, use first file
    if args.analyze:
        result = analyze_dataset(str(input_paths[0]))
        print(result)
        if len(input_paths) > 1:
            print(f"\n{Colors.DIM}Note: Analyzed first file only. "
                  f"{len(input_paths) - 1} additional files not shown.{Colors.END}")

        # Run FAIR assessment if requested
        if args.fair:
            run_fair_assessment(
                str(input_paths[0]),
                domain=config.get('domain', args.domain),
                fair_config_path=args.fair_config,
                style=args.style,
            )
        return

    # For FAIR-only mode (no analyze, no pipeline)
    if args.fair and not args.analyze:
        # Check if user wants FAIR-only (no other stages specified)
        if args.stages is None and config.get('stages') is None:
            # Run FAIR assessment only if --fair is the main action
            pass  # Continue to pipeline, FAIR will be run at end

    # Run pipeline
    stages = config.get('stages')
    if args.stages:
        stages = args.stages.split(",")
    ctx = run_pipeline(
        output_dir=config.get('output_dir', args.output),
        domain=config.get('domain', args.domain),
        stages=stages,
        config=config,
        verbose=verbose,
        style=args.style,
        input_paths=[str(p) for p in input_paths]
    )

    # Run FAIR assessment after pipeline if requested (CLI flag or YAML config)
    fair_enabled = args.fair or config.get('fair', {}).get('enabled', False)
    fair_config_path = args.fair_config or config.get('fair', {}).get('metadata_file')

    if fair_enabled and ctx:
        fair_path = run_fair_assessment(
            str(input_paths[0]),
            domain=config.get('domain', args.domain),
            fair_config_path=fair_config_path,
            style=args.style,
            output_dir=config.get('output_dir', args.output),
        )
        if fair_path and ctx.metadata.get("saved_files") is not None:
            ctx.metadata["saved_files"].append(fair_path)

    # Run catalog export after pipeline if requested (CLI flag or YAML config)
    catalog_config = config.get('catalog', {})
    catalog_enabled = args.catalog or catalog_config.get('enabled', False)

    if catalog_enabled and ctx:
        # Get metadata.json path from output
        output_dir = Path(config.get('output_dir', args.output))
        metadata_path = output_dir / "metadata.json"

        if metadata_path.exists():
            run_catalog_export(
                metadata_path=str(metadata_path),
                catalog_type=args.catalog_type or catalog_config.get('type', 'huggingface'),
                host=args.catalog_host or catalog_config.get('host'),
                token=args.catalog_token or catalog_config.get('token'),
                token_env=catalog_config.get('token_env'),
                service_name=args.catalog_service or catalog_config.get('service_name', 'redi'),
                repo_id=args.catalog_repo or catalog_config.get('repo_id'),
                private=args.catalog_private if hasattr(args, 'catalog_private') else catalog_config.get('private', True),
                style=args.style,
            )
        else:
            if args.style == "rich":
                print(f"    {Colors.YELLOW}⚠ metadata.json not found, skipping catalog export{Colors.END}")


if __name__ == "__main__":
    main()
