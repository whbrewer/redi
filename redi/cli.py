#!/usr/bin/env python3
"""
REDI CLI - Readiness Engine for Data Integration

Unified entry point for running REDI in different modes:
  - run (orchestrator): Execute the full pipeline on data
  - inspect: Inspect data files or observe existing workflows (alias: analyze)
  - discover: Analyze unknown datasets and propose pipelines
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
                elif ext == ".json":
                    # Read JSON directly - pd.read_csv hangs on single-line JSON
                    import json, re as _re
                    with open(input_path, 'r') as f:
                        # Read a small chunk to extract keys without loading the full file
                        chunk = f.read(64 * 1024)
                    try:
                        # Try to parse as complete JSON (small files)
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        # Chunk was truncated; try to parse first record from array
                        data = None
                        if chunk.lstrip().startswith('['):
                            # Find first complete object in array
                            inner = chunk.lstrip()[1:].lstrip()
                            if inner.startswith('{'):
                                depth = 0
                                for i, ch in enumerate(inner):
                                    if ch == '{': depth += 1
                                    elif ch == '}': depth -= 1
                                    if depth == 0:
                                        try:
                                            data = json.loads(inner[:i+1])
                                        except json.JSONDecodeError:
                                            pass
                                        break
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        columns = [k.lower() for k in data[0].keys()]
                    elif isinstance(data, dict):
                        columns = [k.lower() for k in data.keys()]
                    else:
                        # Could not parse complete record; extract all JSON keys
                        # from the chunk as potential column names for domain detection
                        columns = _re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"(?:\s*:)', chunk)
                        columns = list(dict.fromkeys(k.lower() for k in columns))
                else:
                    df = pd.read_csv(input_path, nrows=0)
                    columns = list(df.columns)
            except Exception:
                columns = []

        detected = _detect_domain(columns=columns, file_extension=ext)
        return detected.value

    # Check for AMReX plotfile directories (HyperCLaw-V1.1)
    if input_path.is_dir():
        header_path = input_path / "Header"
        if header_path.exists():
            try:
                with open(header_path) as _f:
                    _first = _f.readline().strip()
                if _first.startswith("HyperCLaw") or _first.startswith("AMReX"):
                    return "combustion"
            except Exception:
                pass

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


def generate_output_dir(
    input_paths: List[Path],
    base_dir: str = "./output",
    unique: bool = False,
    hash_content: bool = False,
    name_override: Optional[str] = None,
) -> str:
    """Generate a hash-based output directory name.

    Creates deterministic output directories based on input file characteristics.
    Same inputs will produce the same output directory (unless --unique is used).

    Args:
        input_paths: List of input file paths
        base_dir: Base output directory (default: ./output)
        unique: If True, append timestamp to ensure uniqueness
        hash_content: If True, hash file content (slower but more accurate)
        name_override: If set, use this as the base name instead of the input
            filename (e.g. pass the YAML config stem when one is used)

    Returns:
        Path string like "./output/frontier_a3f2b1c" or "./output/mydata_a3f2b1c_0204_1430"
    """
    import hashlib
    from datetime import datetime

    if not input_paths:
        return base_dir

    # Base name: prefer the override (e.g. YAML config stem), else first input file
    primary = input_paths[0]
    basename = name_override if name_override else primary.stem

    # Build hash input: path + size for each file
    hash_input = []
    for p in input_paths:
        try:
            if hash_content and p.stat().st_size < 100 * 1024 * 1024:  # < 100MB
                # Hash first 64KB of content
                with open(p, 'rb') as f:
                    content_sample = f.read(65536)
                hash_input.append(f"{p}:{p.stat().st_size}:{hashlib.md5(content_sample).hexdigest()}")
            else:
                # Just path + size (fast)
                hash_input.append(f"{p}:{p.stat().st_size}")
        except (OSError, IOError):
            # File doesn't exist yet or can't be read - use path only
            hash_input.append(str(p))

    # Generate short hash
    combined = "|".join(hash_input)
    hash_digest = hashlib.sha256(combined.encode()).hexdigest()[:7]

    # Build output directory name
    dir_name = f"{basename}_{hash_digest}"

    if unique:
        # Append timestamp for uniqueness: MMDD_HHMM
        timestamp = datetime.now().strftime("%m%d_%H%M")
        dir_name = f"{dir_name}_{timestamp}"

    return str(Path(base_dir) / dir_name)


# =============================================================================
# ORCHESTRATOR MODE (redi run)
# =============================================================================

def _generate_enriched_datacard(primary_path, domain, config, ctx, step_results) -> str:
    """Build a DATA_CARD.md enriched with actual pipeline metadata from ctx."""
    from .assess import analyze_dataset, PipelineRecommender
    from .pipeline import PipelineStage

    data_card = analyze_dataset(str(primary_path))
    data_card.detected_domain = domain
    data_card.characteristics.domain = domain

    n_records = (
        ctx.metadata.get("n_structures") or
        ctx.metadata.get("n_proteins") or
        ctx.metadata.get("n_samples") or
        ctx.metadata.get("n_timesteps") or
        ctx.metadata.get("n_records") or
        ctx.metadata.get("assembled_structures") or
        data_card.overview.num_records
    )
    if n_records:
        data_card.overview.num_records = int(n_records)

    out_fmt = config.get("output_format") if isinstance(config, dict) else getattr(config, "get", lambda k, d=None: d)("output_format")
    if out_fmt:
        data_card.overview.file_format = out_fmt.upper()

    norm = ctx.metadata.get("normalization") or ctx.metadata.get("normalization_method")
    if norm:
        data_card.transformations.normalization_method = str(norm)

    split_info = ctx.metadata.get("split")
    if split_info and isinstance(split_info, dict):
        tr = split_info.get("train_size", "")
        va = split_info.get("val_size", "")
        te = split_info.get("test_size", "")
        data_card.transformations.train_val_test_split = f"{tr}/{va}/{te}"

    completed_stage_names = [
        step_name for status, step_name, _, _ in step_results if status == "success"
    ]
    if completed_stage_names:
        data_card.transformations.preprocessing_steps = completed_stage_names

    if ctx.completed_stages and PipelineStage.OUTPUT in ctx.completed_stages:
        data_card.readiness_level = 5
    elif ctx.completed_stages and PipelineStage.STRUCTURE in ctx.completed_stages:
        data_card.readiness_level = 4
    elif ctx.completed_stages and PipelineStage.TRANSFORM in ctx.completed_stages:
        data_card.readiness_level = 3

    recommender = PipelineRecommender(data_card)
    return "\n".join([
        "=" * 60,
        "REDI - Data Readiness Analysis",
        "=" * 60 + "\n",
        data_card.to_markdown(),
        "\n" + "=" * 60 + "\n",
        recommender.get_recommendations_markdown(),
    ])


def run_pipeline(
    input_path: Optional[str] = None,
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None,
    verbose: bool = False,
    style: str = "rich",
    input_paths: Optional[List[str]] = None,
    array_mode: bool = False,
    normalize_method: Optional[str] = None,
    verify: bool = False,
    raps_jobs: bool = False,
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
        array_mode: If True, optimize pipeline for array data (skip DataFrame steps)
        normalize_method: Normalization method for array data ("minmax" or "zscore")
        verify: If True, run verification checks after each pipeline step
        raps_jobs: If True, extract RAPS Job objects from HPC telemetry data
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
        "geospatial": "🌐",
        "combustion": "🔥",
        "general": "📊",
    }
    icon = domain_icons.get(domain, "📊")

    if style == "rich":
        print_section("Domain Detection")
        if domain == "general" and domain_was_auto:
            print(f"    {Colors.YELLOW}⚠ Could not auto-detect domain{Colors.END}")
            print(f"    {Colors.DIM}Tip: Use -d/--domain to specify: climate, materials, life_sciences, fusion, hpc, geospatial, combustion{Colors.END}")
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

    # Enable step verification if requested
    ctx.verify_steps = verify

    # Load domain-specific pipeline
    if style == "rich":
        print_section("Loading Pipeline")

    try:
        if domain == "climate":
            from .domains.climate import get_full_pipeline
        elif domain == "materials":
            from .domains.materials import get_full_pipeline
        elif domain == "life_sciences":
            from .domains.life_sciences import get_full_pipeline as _get_ls_pipeline
            _ls_fmt = config.get("output_format", "npz") if config else "npz"
            _ls_max = config.get("max_proteins") if config else None
            _ls_workers = config.get("num_workers") if config else None
            get_full_pipeline = lambda: _get_ls_pipeline(input_path=primary_path, output_format=_ls_fmt, max_proteins=_ls_max, num_workers=_ls_workers)
        elif domain == "fusion":
            from .domains.fusion import get_full_pipeline
        elif domain == "hpc":
            from .domains.hpc import get_full_pipeline as _get_hpc_pipeline
            from .domains.hpc_job_mapper import load_raps_system_params
            _raps_cfg = config.get("raps_jobs", {}) if config else {}
            _col_overrides = _raps_cfg.get("column_overrides") if isinstance(_raps_cfg, dict) else None
            _out_fmt = config.get("output_format", "npz") if config else "npz"
            _sys_name = _raps_cfg.get("system_name", "unknown") if isinstance(_raps_cfg, dict) else "unknown"
            # Load power + scheduler params from RAPS system config
            _sys_params = load_raps_system_params(_sys_name) if _sys_name != "unknown" else None
            _power_cfg = _sys_params["power_config"] if _sys_params else None
            _trace_quanta = _sys_params["trace_quanta"] if _sys_params else None
            _topology = _sys_params["topology"] if _sys_params else None
            # Fall back to REDI YAML for power config / topology / trace_quanta
            if _power_cfg is None:
                _power_cfg = _raps_cfg.get("power_config") if isinstance(_raps_cfg, dict) else None
            if _topology is None:
                _topology = _raps_cfg.get("topology") if isinstance(_raps_cfg, dict) else None
            if _trace_quanta is None:
                _trace_quanta = _raps_cfg.get("trace_quanta") if isinstance(_raps_cfg, dict) else None
            # Jobprofile path from config
            _jp_path = _raps_cfg.get("jobprofile_path") if isinstance(_raps_cfg, dict) else None
            _priority_policy = _raps_cfg.get("priority_policy") if isinstance(_raps_cfg, dict) else None
            _id_strip_suffix = bool(_raps_cfg.get("id_strip_suffix")) if isinstance(_raps_cfg, dict) else False
            _col_scalars = _raps_cfg.get("column_scalars") if isinstance(_raps_cfg, dict) else None
            get_full_pipeline = lambda: _get_hpc_pipeline(
                raps_jobs=raps_jobs,
                column_overrides=_col_overrides,
                power_config=_power_cfg,
                trace_quanta=_trace_quanta,
                output_format=_out_fmt,
                system_name=_sys_name,
                jobprofile_path=_jp_path,
                topology=_topology,
                priority_policy=_priority_policy,
                id_strip_suffix=_id_strip_suffix,
                column_scalars=_col_scalars,
            )
        elif domain == "geospatial":
            from .domains.geospatial import get_full_pipeline as _get_geo_pipeline
            _split_cfg = config.get("split", {})
            _split = _split_cfg.get("enabled", True) if isinstance(_split_cfg, dict) else bool(_split_cfg)
            get_full_pipeline = lambda: _get_geo_pipeline(split=_split)
        elif domain == "nuclear_physics":
            # Routed through plugin system — install redi-nuclear-physics
            pass  # falls through to plugin/generic lookup below
        elif domain == "combustion":
            from .domains.combustion import get_full_pipeline
        else:
            # Check installed plugins before falling back to generic pipeline
            try:
                from .plugins import get_plugin
                _plugin = get_plugin(domain) if domain else None
            except Exception:
                _plugin = None
            if _plugin is not None:
                _plugin_fmt = config.get("output_format", "npz") if config else "npz"
                import inspect as _inspect
                # Inspect the underlying module function, not the Plugin wrapper method
                _module_fn = getattr(_plugin.module, "get_full_pipeline", None)
                _sig = _inspect.signature(_module_fn) if _module_fn else _inspect.signature(_plugin.get_full_pipeline)
                if "output_format" in _sig.parameters:
                    get_full_pipeline = lambda: _plugin.get_full_pipeline(output_format=_plugin_fmt)
                else:
                    get_full_pipeline = _plugin.get_full_pipeline
            else:
                get_full_pipeline = None

        if get_full_pipeline:
            pipeline = Pipeline(domain)
            steps = get_full_pipeline()

            # Ensure ingest is always included (data must be loaded)
            if stages is not None and "ingest" not in stages:
                stages = ["ingest"] + list(stages)

            for step in steps:
                if stages is None or step.stage.value in stages:
                    pipeline.add_step(step)
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} steps for {domain} domain")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps")
        else:
            # Fall back to generic pipeline
            from .pipeline.ingest import get_ingest_steps
            from .pipeline.preprocess import get_preprocess_steps, NormalizeArray, ComputeArrayStats
            from .pipeline.transform import get_transform_steps
            from .pipeline.structure import get_structure_steps
            from .pipeline.output import get_output_steps

            pipeline = Pipeline(domain)
            all_stages = ["ingest", "preprocess", "transform", "structure", "output"]
            if stages is None:
                stages = all_stages
            else:
                # Ensure ingest is always included (data must be loaded)
                if "ingest" not in stages:
                    stages = ["ingest"] + list(stages)

            if "ingest" in stages:
                for step in get_ingest_steps():
                    pipeline.add_step(step)

            if "preprocess" in stages:
                if array_mode:
                    # Use array-specific preprocessing steps
                    pipeline.add_step(ComputeArrayStats())
                    if normalize_method:
                        pipeline.add_step(NormalizeArray(method=normalize_method))
                else:
                    for step in get_preprocess_steps(domain):
                        pipeline.add_step(step)

            if "transform" in stages:
                if not array_mode:  # Skip DataFrame transforms in array mode
                    for step in get_transform_steps(domain):
                        pipeline.add_step(step)

            if "structure" in stages:
                if not array_mode:  # Skip DataFrame structure in array mode
                    for step in get_structure_steps(domain):
                        pipeline.add_step(step)

            if "output" in stages:
                for step in get_output_steps():
                    pipeline.add_step(step)

            mode_str = " (array mode)" if array_mode else ""
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} generic steps{mode_str}")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps{mode_str}")

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
    current_stage = None
    flowcept_started_here = False
    flowcept_ref = None

    # Start Flowcept once for the CLI-managed execution loop.
    if getattr(ctx, "with_flowcept", False):
        flowcept_started_here = pipeline._start_flowcept(ctx, {"mode": "full_pipeline"})
        flowcept_ref = ctx.flowcept

    for i, step in enumerate(pipeline.steps, 1):
        step_start = time.time()
        if getattr(ctx, "with_flowcept", False) and flowcept_ref is not None and ctx.flowcept is None:
            ctx.flowcept = flowcept_ref

        # Track stage completions
        if step.stage != current_stage:
            if current_stage is not None:
                ctx.mark_complete(current_stage)
            current_stage = step.stage

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
                    print(f"     {Colors.DIM}↳ skipped (not applicable){Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                else:
                    print(f"\r  ⏭️  {Colors.DIM}{step.name}{Colors.END} → skipped (not applicable) {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
            elif style == "simple":
                print(f"  [SKIP] {step.name}")
            step_results.append(("skipped", step.name, "", step_time))
            continue

        # Execute
        try:
            old_metadata = dict(ctx.metadata)
            next_ctx = step.execute(ctx)
            if flowcept_ref is not None and getattr(next_ctx, "flowcept", None) is None:
                try:
                    next_ctx.flowcept = flowcept_ref
                except Exception:
                    pass
            ctx = next_ctx
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

                # Run verification if enabled
                if ctx.verify_steps:
                    verify_result = step.verify(ctx)
                    ctx.verification_results.append(verify_result)
                    if not verify_result.passed:
                        if style == "rich":
                            print(f"       {Colors.YELLOW}⚠ Verification issues:{Colors.END}")
                            for check in verify_result.checks:
                                if not check["passed"]:
                                    print(f"         {Colors.DIM}└─{Colors.END} {check['name']}: {check.get('message', 'FAILED')}")
                        elif style == "simple":
                            print(f"    [VERIFY WARN] {step.name}")

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

    # Mark the final stage complete
    if current_stage is not None:
        ctx.mark_complete(current_stage)
    if flowcept_started_here and flowcept_ref:
        flowcept_ref.stop()
    provenance_result = {"enabled": False, "generated": False, "error": None}
    if getattr(ctx, "with_flowcept", False):
        from redi.provenance_card import finalize_provenance_for_context

        provenance_result = finalize_provenance_for_context(
            ctx=ctx,
            flowcept_ref=flowcept_ref,
            verbose=verbose,
            logger=logger,
        )
        if provenance_result.get("error"):
            err = provenance_result["error"]
            if style == "rich":
                print(f"    {Colors.YELLOW}⚠ Could not generate provenance card: {err}{Colors.END}")
            elif style == "simple":
                print(f"  [WARN] Could not generate provenance card: {err}")

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

        # Verification results
        if ctx.verify_steps and ctx.verification_results:
            print_section("Verification")
            all_passed = all(r.passed for r in ctx.verification_results)
            if all_passed:
                print(f"    {Colors.GREEN}✓ All {len(ctx.verification_results)} verification checks passed{Colors.END}")
            else:
                failed = [r for r in ctx.verification_results if not r.passed]
                print(f"    {Colors.YELLOW}⚠ {len(failed)}/{len(ctx.verification_results)} steps have verification issues{Colors.END}")

            for result in ctx.verification_results:
                if result.passed:
                    print(f"    {Colors.GREEN}✓{Colors.END} {result.step_name}")
                else:
                    print(f"    {Colors.RED}✗{Colors.END} {result.step_name}")
                    for check in result.checks:
                        if not check["passed"]:
                            msg = check.get("message", "FAILED")
                            if check.get("expected") is not None:
                                msg = f"expected {check['expected']}, got {check['actual']}"
                            print(f"      {Colors.DIM}└─{Colors.END} {check['name']}: {msg}")

        # Output files
        if ctx.metadata.get("saved_files"):
            print_section("Output Files")
            for f in ctx.metadata["saved_files"]:
                fpath = Path(f)
                if fpath.name in {
                    "flowcept_buffer.jsonl",
                    "PROVENANCE_CARD.md",
                    "provenance_card.md",
                    "PROVENANCE_REPORT.pdf",
                }:
                    continue
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
            from .assess import analyze_dataset, PipelineRecommender

            report = _generate_enriched_datacard(primary_path, domain, config, ctx, step_results)
            data_card = analyze_dataset(str(primary_path))
            data_card.detected_domain = domain
            data_card.characteristics.domain = domain

            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))
            print(f"    📋 {Colors.BOLD}DATA_CARD.md{Colors.END} {Colors.DIM}(saved){Colors.END}")

            # Save metadata.json for downstream tooling
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
            ctx.output_dir.touch()  # update directory mtime on re-runs
            print(f"    📋 {Colors.BOLD}metadata.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

            if getattr(ctx, "with_flowcept", False):
                from .provenance_card import render_provenance_summary

                render_provenance_summary(
                    ctx=ctx,
                    style="rich",
                    datacard_path=datacard_path,
                    print_section=print_section,
                    printer=print,
                    colors=Colors,
                )

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
                    if "shard" in step_lower:
                        completed_steps.add("shard")
                    if "save" in step_lower and "shard" not in step_lower:
                        # save_parquet/save_npz count as shard only if split was used
                        split_info = ctx.metadata.get("split")
                        if split_info:
                            completed_steps.add("shard")
                    if "load" in step_lower or "ingest" in step_lower:
                        completed_steps.add("ingest")
                        completed_steps.add("download")
                    if "featurize" in step_lower or "engineer" in step_lower:
                        completed_steps.add("featurize")
                    if "encode" in step_lower:
                        completed_steps.add("encode")
                    if "label" in step_lower:
                        completed_steps.add("label")
                    if "hexify" in step_lower:
                        completed_steps.add("hexify")
                    if "aggregate" in step_lower:
                        completed_steps.add("aggregate")
                    if "clean" in step_lower:
                        completed_steps.add("clean")
                    # Combustion (AMReX) step mappings
                    if "amrex" in step_lower or "plotfile" in step_lower:
                        completed_steps.add("ingest")
                    if "clip_species" in step_lower or "validate_physics" in step_lower:
                        completed_steps.add("preprocess")
                    if "derive_combustion" in step_lower or "normalize_combustion" in step_lower:
                        completed_steps.add("transform")
                    if "composite_amr" in step_lower:
                        completed_steps.add("composite")
                    if "combustion_npz" in step_lower or "save_combustion" in step_lower:
                        completed_steps.add("shard")

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

        # Verification results for simple style
        if ctx.verify_steps and ctx.verification_results:
            all_passed = all(r.passed for r in ctx.verification_results)
            if all_passed:
                print(f"Verification: {len(ctx.verification_results)} checks passed")
            else:
                for result in ctx.verification_results:
                    if not result.passed:
                        print(f"Verification FAILED: {result.step_name}")
                        for check in result.checks:
                            if not check["passed"]:
                                print(f"  - {check['name']}: {check.get('message', 'FAILED')}")

        # Save metadata.json and DATA_CARD.md for simple/json modes too
        try:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            # Save DATA_CARD.md
            report = _generate_enriched_datacard(primary_path, domain, config, ctx, step_results)
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

        if getattr(ctx, "with_flowcept", False):
            from .provenance_card import render_provenance_summary

            render_provenance_summary(
                ctx=ctx,
                style="simple",
                datacard_path=datacard_path if 'datacard_path' in locals() else None,
                printer=print,
            )

        if ctx.metadata.get("saved_files"):
            for f in ctx.metadata["saved_files"]:
                if Path(f).name in {
                    "flowcept_buffer.jsonl",
                    "PROVENANCE_CARD.md",
                    "provenance_card.md",
                    "PROVENANCE_REPORT.pdf",
                }:
                    continue
                print(f"  Output: {f}")

    elif style == "json":
        # Save metadata.json and DATA_CARD.md for json mode too
        try:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            report = _generate_enriched_datacard(primary_path, domain, config, ctx, step_results)
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


# =============================================================================
# ANALYZE MODE (redi analyze)
# =============================================================================

def run_analyze(
    workflow_file: str,
    workflow_type: str = "auto",
    output_dir: str = "./output",
    watch: bool = False,
    style: str = "rich",
    config: Optional[dict] = None,
    data_dir: Optional[str] = None,
):
    """Run REDI in analyze mode - observe existing workflows.

    Args:
        workflow_file: Path to workflow file (Snakefile, Nextflow, etc.)
        workflow_type: Type of workflow (auto, snakemake, nextflow, bash)
    output_dir: Output directory for workflow report and analysis outputs
        watch: If True, watch for changes and re-run
        style: Output style (rich, simple, json)
        config: Optional YAML configuration dict
        data_dir: Optional path to data directory to scan for intermediate files
    """
    config = config or {}
    workflow_path = Path(workflow_file)
    data_path = Path(data_dir) if data_dir else None

    if not workflow_path.exists():
        if style == "rich":
            print(f"{Colors.RED}✗ Workflow file not found: {workflow_file}{Colors.END}")
        else:
            print(f"Error: Workflow file not found: {workflow_file}")
        return None

    if style == "rich":
        print_banner()
        print_section("Analyze Mode")
        print_metric("Workflow", str(workflow_path))

    # Auto-detect workflow type
    if workflow_type == "auto":
        if workflow_path.name.lower() in ("snakefile", "snakefile.smk") or workflow_path.suffix == ".smk":
            workflow_type = "snakemake"
        elif workflow_path.suffix in (".nf", ".nextflow"):
            workflow_type = "nextflow"
        elif workflow_path.suffix in (".sh", ".bash"):
            workflow_type = "bash"
        else:
            workflow_type = "unknown"

    if style == "rich":
        print_metric("Type", workflow_type)
        print_metric("Watch", "enabled" if watch else "disabled")
        if data_path:
            print_metric("Data Dir", str(data_path))

    # Parse workflow to understand structure
    if workflow_type == "snakemake":
        return _analyze_snakemake(workflow_path, output_dir, watch, style, data_path)
    elif workflow_type == "nextflow":
        return _analyze_nextflow(workflow_path, output_dir, watch, style, data_path)
    else:
        if style == "rich":
            print(f"\n    {Colors.YELLOW}⚠ Workflow type '{workflow_type}' not yet supported{Colors.END}")
            print(f"    {Colors.DIM}Supported types: snakemake, nextflow{Colors.END}")
        else:
            print(f"Error: Workflow type '{workflow_type}' not supported")
        return None


def _analyze_snakemake(workflow_path: Path, output_dir: str, watch: bool, style: str, data_dir: Optional[Path] = None):
    """Analyze Snakemake workflows."""
    if style == "rich":
        print_section("Parsing Snakemake Workflow")

    # Parse the Snakefile to extract rules
    rules = _parse_snakefile(workflow_path)

    if style == "rich":
        print(f"    Found {Colors.BOLD}{len(rules)}{Colors.END} rules:")
        for rule in rules:
            print(f"      • {rule['name']}: {rule.get('description', 'no description')}")

    # Map rules to REDI pipeline stages
    stage_mapping = _map_rules_to_stages(rules)

    if style == "rich":
        print_section("Stage Mapping")
        for stage, mapped_rules in stage_mapping.items():
            if mapped_rules:
                print(f"    {stage}: {', '.join(mapped_rules)}")

    # Parse Snakemake logs for execution history
    log_report = None
    from .inspect import find_latest_log, parse_snakemake_log

    latest_log = find_latest_log(workflow_path.parent)
    if latest_log:
        if style == "rich":
            print_section("Parsing Execution Logs")
            print(f"    Log: {Colors.DIM}{latest_log.name}{Colors.END}")

        log_report = parse_snakemake_log(latest_log)

        if style == "rich":
            print(f"    Jobs: {Colors.BOLD}{log_report.total_jobs}{Colors.END} total, "
                  f"{Colors.GREEN}{log_report.completed_jobs}{Colors.END} completed, "
                  f"{Colors.RED}{log_report.failed_jobs}{Colors.END} failed")
            print(f"    Executions parsed: {Colors.BOLD}{len(log_report.executions)}{Colors.END}")
            print(f"    File provenance: {Colors.BOLD}{len(log_report.file_provenance)}{Colors.END} files tracked")

            if log_report.errors:
                print(f"    {Colors.RED}Errors: {len(log_report.errors)}{Colors.END}")
    else:
        if style == "rich":
            print_section("Execution Logs")
            print(f"    {Colors.DIM}No Snakemake logs found in .snakemake/log/{Colors.END}")

    # Scan data files if data_dir provided
    snapshots = []
    quality_reports = []

    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Scanning Data Files")

        from .inspect import capture_snapshot, check_snapshot, detect_transformations

        # Find all data files
        data_extensions = {'.nc', '.nc4', '.npz', '.npy', '.parquet', '.h5', '.hdf5', '.zarr'}
        data_files = []
        for ext in data_extensions:
            data_files.extend(data_dir.rglob(f'*{ext}'))

        # Sort by modification time (oldest first = earlier in pipeline)
        data_files = sorted(data_files, key=lambda f: f.stat().st_mtime)

        if style == "rich":
            print(f"    Found {Colors.BOLD}{len(data_files)}{Colors.END} data files")

        # Capture snapshots (limit to first 20 for speed)
        for f in data_files[:20]:
            try:
                if style == "rich":
                    print(f"    📊 Scanning {f.name}...", end="", flush=True)

                snapshot = capture_snapshot(f)
                snapshots.append(snapshot)

                # Run quality checks
                qr = check_snapshot(snapshot)
                quality_reports.append(qr)

                if style == "rich":
                    if qr.passed:
                        print(f"\r    ✅ {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")
                    else:
                        print(f"\r    ⚠️  {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")

            except Exception as e:
                if style == "rich":
                    print(f"\r    ❌ {f.name}: {str(e)[:50]}")

        if len(data_files) > 20:
            if style == "rich":
                print(f"    {Colors.DIM}... and {len(data_files) - 20} more files{Colors.END}")

    # Detect transformation parameters
    transform_report = None
    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Detecting Transformation Parameters")

        transform_report = detect_transformations(data_dir, snapshots)

        if transform_report.normalization:
            norm = transform_report.normalization
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Normalization detected: {Colors.BOLD}{norm.method}{Colors.END}")
                if norm.mean_file:
                    print(f"      Mean file: {Path(norm.mean_file).name}")
                if norm.std_file:
                    print(f"      Std file: {Path(norm.std_file).name}")
                if norm.mean_values:
                    print(f"      Variables: {len(norm.mean_values)}")
                    for var, mean in list(norm.mean_values.items())[:3]:
                        std = norm.std_values.get(var, 'N/A')
                        if isinstance(std, (int, float)):
                            print(f"        {var}: mean={mean:.4g}, std={std:.4g}")
                        else:
                            print(f"        {var}: mean={mean:.4g}")
                    if len(norm.mean_values) > 3:
                        print(f"        {Colors.DIM}... and {len(norm.mean_values) - 3} more{Colors.END}")

        if transform_report.splits:
            splits = transform_report.splits
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Train/val/test split detected: {Colors.BOLD}{splits.method}{Colors.END}")
                if splits.train_size:
                    print(f"      Train: {splits.train_size:,} samples")
                if splits.val_size:
                    print(f"      Val: {splits.val_size:,} samples")
                if splits.test_size:
                    print(f"      Test: {splits.test_size:,} samples")

        if transform_report.regrid:
            regrid = transform_report.regrid
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Regridding detected: {Colors.BOLD}{regrid.method}{Colors.END}")
                if regrid.target_resolution:
                    print(f"      Target resolution: {regrid.target_resolution[0]}° x {regrid.target_resolution[1]}°")

        if not transform_report.normalization and not transform_report.splits and not transform_report.regrid:
            if style == "rich":
                print(f"    {Colors.DIM}No transformation parameters detected{Colors.END}")

    # Check for data quality issues in transformation report
    transform_issues = []
    if transform_report:
        from .inspect import check_transformation_report
        transform_issues = check_transformation_report(transform_report)

        if transform_issues:
            if style == "rich":
                print_section("Transformation Quality Checks")
                for issue in transform_issues:
                    if issue.severity.value == 'error':
                        icon = f"{Colors.RED}✗{Colors.END}"
                    elif issue.severity.value == 'warning':
                        icon = f"{Colors.YELLOW}⚠{Colors.END}"
                    else:
                        icon = f"{Colors.DIM}ℹ{Colors.END}"
                    print(f"    {icon} {Colors.BOLD}{issue.check_name}:{Colors.END} {issue.message}")

    # Generate improvement suggestions
    from .inspect import generate_suggestions, format_suggestions_cli
    all_quality_issues = transform_issues + [issue for qr in quality_reports for issue in qr.issues]
    workflow_info = {'rules': rules, 'stage_mapping': stage_mapping}
    suggestions = generate_suggestions(
        quality_issues=all_quality_issues,
        transformation_report=transform_report,
        workflow_info=workflow_info,
        data_snapshots=snapshots,
    )

    if suggestions and style == "rich":
        print_section("Suggestions for Improvement")
        print(format_suggestions_cli(suggestions, use_color=True))

    # Create output directory and save analysis
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate analyze report
    report = {
        "workflow_file": str(workflow_path),
        "workflow_type": "snakemake",
        "rules": rules,
        "stage_mapping": stage_mapping,
        "mode": "analyze",
    }

    # Add data snapshots if we scanned
    if snapshots:
        report["data_snapshots"] = [s.to_dict() for s in snapshots]
        report["quality_summary"] = {
            "files_scanned": len(snapshots),
            "passed": sum(1 for qr in quality_reports if qr.passed),
            "with_warnings": sum(1 for qr in quality_reports if qr.has_warnings),
            "failed": sum(1 for qr in quality_reports if not qr.passed),
            "issues": [issue.to_dict() for qr in quality_reports for issue in qr.issues]
        }

    # Add transformation parameters if detected
    if transform_report:
        report["transformations"] = transform_report.to_dict()

    # Add transformation quality issues
    if transform_issues:
        report["transformation_quality"] = [issue.to_dict() for issue in transform_issues]

    # Add suggestions
    if suggestions:
        report["suggestions"] = [s.to_dict() for s in suggestions]
        report["suggestions_summary"] = {
            "total": len(suggestions),
            "high_priority": sum(1 for s in suggestions if s.priority == 1),
            "medium_priority": sum(1 for s in suggestions if s.priority == 2),
            "low_priority": sum(1 for s in suggestions if s.priority == 3),
        }

    # Add execution log info if parsed
    if log_report:
        report["execution_log"] = {
            "log_file": str(latest_log),
            "job_stats": log_report.job_stats,
            "total_jobs": log_report.total_jobs,
            "completed_jobs": log_report.completed_jobs,
            "failed_jobs": log_report.failed_jobs,
            "executions_count": len(log_report.executions),
            "file_provenance": log_report.file_provenance,
            "errors": log_report.errors[:5] if log_report.errors else [],
        }

    import json
    report_path = out_path / "analyze_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate workflow markdown if we have snapshots
    if snapshots:
        datacard_path = out_path / "WORKFLOW.md"
        datacard_content = _generate_analyze_datacard(
            workflow_path, rules, stage_mapping, snapshots, quality_reports,
            transform_report, log_report, suggestions
        )
        datacard_path.write_text(datacard_content)

        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}analyze_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
            print(f"    📋 {Colors.BOLD}WORKFLOW.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
    else:
        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}analyze_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

    if style == "rich":
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Analysis complete{Colors.END}")
        if not data_dir:
            print(f"{Colors.DIM}  Tip: Use --data-dir to scan intermediate data files{Colors.END}")
        print()

    # Watch mode: monitor for changes and re-analyze
    if watch:
        _run_watch_mode(workflow_path, data_dir, output_dir, style)

    return report


def _run_watch_mode(workflow_path: Path, data_dir: Optional[Path], output_dir: str, style: str):
    """Run watch mode - monitor for file changes and re-analyze."""
    import time

    # Collect files to watch
    watch_files = [workflow_path]
    if data_dir and data_dir.exists():
        data_extensions = {'.nc', '.nc4', '.npz', '.npy', '.parquet', '.h5', '.hdf5', '.zarr'}
        for ext in data_extensions:
            watch_files.extend(data_dir.rglob(f'*{ext}'))

    # Also watch Snakemake logs
    log_dir = workflow_path.parent / ".snakemake" / "log"
    if log_dir.exists():
        watch_files.extend(log_dir.glob("*.snakemake.log"))

    # Record initial timestamps
    file_mtimes = {}
    for f in watch_files:
        if f.exists():
            try:
                file_mtimes[str(f)] = f.stat().st_mtime
            except OSError:
                pass

    if style == "rich":
        print(f"\n{Colors.CYAN}👁 Watch mode active{Colors.END}")
        print(f"  Monitoring {Colors.BOLD}{len(file_mtimes)}{Colors.END} files for changes")
        print(f"  {Colors.DIM}Press Ctrl+C to stop{Colors.END}\n")

    poll_interval = 2.0  # seconds

    try:
        while True:
            time.sleep(poll_interval)

            # Check for changes
            changed_files = []
            new_files = []

            # Check existing files for modifications
            for path_str, old_mtime in list(file_mtimes.items()):
                path = Path(path_str)
                if path.exists():
                    try:
                        new_mtime = path.stat().st_mtime
                        if new_mtime > old_mtime:
                            changed_files.append(path)
                            file_mtimes[path_str] = new_mtime
                    except OSError:
                        pass

            # Check for new files in data_dir
            if data_dir and data_dir.exists():
                for ext in data_extensions:
                    for f in data_dir.rglob(f'*{ext}'):
                        if str(f) not in file_mtimes:
                            new_files.append(f)
                            try:
                                file_mtimes[str(f)] = f.stat().st_mtime
                            except OSError:
                                pass

            # Check for new log files
            if log_dir.exists():
                for f in log_dir.glob("*.snakemake.log"):
                    if str(f) not in file_mtimes:
                        new_files.append(f)
                        try:
                            file_mtimes[str(f)] = f.stat().st_mtime
                        except OSError:
                            pass

            # Re-run analysis if changes detected
            if changed_files or new_files:
                if style == "rich":
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\n{Colors.YELLOW}[{timestamp}] Changes detected:{Colors.END}")
                    for f in changed_files[:3]:
                        print(f"  {Colors.DIM}modified:{Colors.END} {f.name}")
                    for f in new_files[:3]:
                        print(f"  {Colors.DIM}new:{Colors.END} {f.name}")
                    if len(changed_files) + len(new_files) > 6:
                        print(f"  {Colors.DIM}... and {len(changed_files) + len(new_files) - 6} more{Colors.END}")
                    print()

                # Re-run the analysis
                _analyze_snakemake(workflow_path, output_dir, False, style, data_dir)

    except KeyboardInterrupt:
        if style == "rich":
            print(f"\n{Colors.DIM}Watch mode stopped{Colors.END}")


def _generate_analyze_datacard(workflow_path: Path, rules: list, stage_mapping: dict,
                                snapshots: list, quality_reports: list,
                                transform_report=None, log_report=None,
                                suggestions=None) -> str:
    """Generate a workflow markdown from analyze mode."""
    lines = [
        f"# Workflow Card: {workflow_path.parent.name}",
        "",
        f"**Generated by:** REDI Analyze",
        f"**Workflow:** `{workflow_path.name}`",
        f"**Files Analyzed:** {len(snapshots)}",
    ]

    # Add execution summary if log was parsed
    if log_report:
        lines.extend([
            f"**Jobs:** {log_report.total_jobs} total, {log_report.completed_jobs} completed, {log_report.failed_jobs} failed",
        ])

    lines.extend([
        "",
        "## Pipeline Structure",
        "",
    ])

    # Rules table
    lines.append("| Rule | Description | Stage |")
    lines.append("|------|-------------|-------|")
    for rule in rules:
        stage = next((s for s, r in stage_mapping.items() if rule['name'] in r), '-')
        lines.append(f"| {rule['name']} | {rule.get('description', '-')} | {stage} |")

    lines.append("")

    # File provenance from execution log
    if log_report and log_report.file_provenance:
        lines.extend([
            "## File Provenance",
            "",
            "Files tracked from execution log:",
            "",
            "| Output File | Created By |",
            "|-------------|------------|",
        ])
        # Group by rule
        rule_outputs: dict = {}
        for output_file, rule_name in log_report.file_provenance.items():
            if rule_name not in rule_outputs:
                rule_outputs[rule_name] = []
            rule_outputs[rule_name].append(Path(output_file).name)

        for rule_name, outputs in list(rule_outputs.items())[:10]:
            for output in outputs[:3]:
                lines.append(f"| `{output}` | {rule_name} |")
            if len(outputs) > 3:
                lines.append(f"| *... {len(outputs) - 3} more* | {rule_name} |")

        if len(rule_outputs) > 10:
            lines.append(f"| *... {len(log_report.file_provenance) - sum(min(len(o), 3) for o in list(rule_outputs.values())[:10])} more files* | |")

        lines.append("")

    # Data snapshots summary
    lines.extend([
        "## Data Files",
        "",
        "| File | Format | Size | Variables | Quality |",
        "|------|--------|------|-----------|---------|",
    ])

    for snapshot, qr in zip(snapshots, quality_reports):
        size_mb = snapshot.size_bytes / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{snapshot.size_bytes / 1024:.1f} KB"
        quality = "✓" if qr.passed else "⚠" if qr.has_warnings else "✗"
        lines.append(f"| {snapshot.filename} | {snapshot.format} | {size_str} | {len(snapshot.variables)} | {quality} |")

    lines.append("")

    # Variable statistics
    lines.extend([
        "## Variables Summary",
        "",
    ])

    # Collect all unique variables across snapshots
    all_vars = {}
    for snapshot in snapshots:
        for name, var in snapshot.variables.items():
            if name not in all_vars:
                all_vars[name] = var

    if all_vars:
        lines.append("| Variable | Shape | Dtype | Min | Max | Mean | Std |")
        lines.append("|----------|-------|-------|-----|-----|------|-----|")

        for name, var in list(all_vars.items())[:15]:
            shape_str = str(var.shape)
            min_str = f"{var.min:.4g}" if var.min is not None else "-"
            max_str = f"{var.max:.4g}" if var.max is not None else "-"
            mean_str = f"{var.mean:.4g}" if var.mean is not None else "-"
            std_str = f"{var.std:.4g}" if var.std is not None else "-"
            lines.append(f"| {name} | {shape_str} | {var.dtype} | {min_str} | {max_str} | {mean_str} | {std_str} |")

        if len(all_vars) > 15:
            lines.append(f"| *... {len(all_vars) - 15} more* | | | | | | |")

    lines.append("")

    # Transformation parameters section
    if transform_report:
        transform_md = transform_report.to_markdown()
        if transform_md:
            lines.append(transform_md)
            lines.append("")

    # Quality issues
    all_issues = [issue for qr in quality_reports for issue in qr.issues]
    if all_issues:
        lines.extend([
            "## Quality Issues",
            "",
        ])

        errors = [i for i in all_issues if i.severity.value == 'error']
        warnings = [i for i in all_issues if i.severity.value == 'warning']

        if errors:
            lines.append("### Errors")
            for issue in errors[:5]:
                lines.append(f"- **{issue.check_name}** [{issue.variable}]: {issue.message}")

        if warnings:
            lines.append("")
            lines.append("### Warnings")
            for issue in warnings[:10]:
                lines.append(f"- **{issue.check_name}** [{issue.variable}]: {issue.message}")

    # Suggestions section
    if suggestions:
        from .inspect import format_suggestions_markdown
        lines.append("")
        lines.append(format_suggestions_markdown(suggestions))

    lines.extend([
        "",
        "---",
        "*Generated by REDI Analyze - Observability for Data Pipelines*",
    ])

    return "\n".join(lines)


def _parse_snakefile(path: Path) -> List[dict]:
    """Parse a Snakefile to extract rule information."""
    rules = []
    content = path.read_text()

    import re
    # Simple regex to find rule definitions
    rule_pattern = re.compile(r'rule\s+(\w+):', re.MULTILINE)

    for match in rule_pattern.finditer(content):
        rule_name = match.group(1)
        # Extract rule body (simplified)
        start = match.end()
        # Find next rule or end of file
        next_match = rule_pattern.search(content, start)
        end = next_match.start() if next_match else len(content)
        rule_body = content[start:end]

        # Extract input/output patterns
        input_match = re.search(r'input:\s*\n?\s*(.+?)(?=\n\s*\w+:|$)', rule_body, re.DOTALL)
        output_match = re.search(r'output:\s*\n?\s*(.+?)(?=\n\s*\w+:|$)', rule_body, re.DOTALL)

        rules.append({
            'name': rule_name,
            'input': input_match.group(1).strip() if input_match else None,
            'output': output_match.group(1).strip() if output_match else None,
            'description': _infer_rule_description(rule_name, rule_body),
        })

    return rules


def _infer_rule_description(name: str, body: str) -> str:
    """Infer a description for a rule based on its name and content."""
    name_lower = name.lower()

    if 'download' in name_lower or 'wget' in body or 'curl' in body:
        return "Download data from remote source"
    elif 'regrid' in name_lower:
        return "Regrid data to target resolution"
    elif 'preprocess' in name_lower or 'clean' in name_lower:
        return "Preprocess and clean data"
    elif 'merge' in name_lower or 'concat' in name_lower:
        return "Merge multiple files"
    elif 'normalize' in name_lower:
        return "Normalize data values"
    elif 'split' in name_lower:
        return "Split into train/val/test"
    elif name_lower == 'all':
        return "Final target rule"
    else:
        return "Processing step"


def _map_rules_to_stages(rules: List[dict]) -> dict:
    """Map Snakemake rules to REDI pipeline stages."""
    stage_mapping = {
        'ingest': [],
        'preprocess': [],
        'transform': [],
        'structure': [],
        'output': [],
    }

    for rule in rules:
        name_lower = rule['name'].lower()
        desc_lower = rule.get('description', '').lower()

        if 'download' in name_lower or 'download' in desc_lower:
            stage_mapping['ingest'].append(rule['name'])
        elif 'regrid' in name_lower or 'clean' in name_lower or 'preprocess' in name_lower:
            stage_mapping['preprocess'].append(rule['name'])
        elif 'normalize' in name_lower or 'transform' in name_lower or 'feature' in name_lower:
            stage_mapping['transform'].append(rule['name'])
        elif 'merge' in name_lower or 'concat' in name_lower or 'structure' in name_lower:
            stage_mapping['structure'].append(rule['name'])
        elif 'split' in name_lower or 'save' in name_lower or rule['name'] == 'all':
            stage_mapping['output'].append(rule['name'])

    return stage_mapping


def _analyze_nextflow(workflow_path: Path, output_dir: str, watch: bool, style: str, data_dir: Optional[Path] = None):
    """Analyze Nextflow workflows."""
    if style == "rich":
        print_section("Parsing Nextflow Workflow")

    # Parse the Nextflow file to extract processes and workflows
    from .inspect import (
        parse_nextflow_file,
        map_processes_to_stages as nf_map_processes_to_stages,
        find_nextflow_logs,
        capture_snapshot,
        check_snapshot,
        detect_transformations,
        check_transformation_report,
    )

    parse_result = parse_nextflow_file(workflow_path)

    if parse_result.errors:
        if style == "rich":
            for err in parse_result.errors:
                print(f"    {Colors.RED}✗ {err}{Colors.END}")
        return None

    if style == "rich":
        print(f"    DSL version: {Colors.BOLD}{parse_result.dsl_version}{Colors.END}")
        print(f"    Found {Colors.BOLD}{len(parse_result.processes)}{Colors.END} processes:")
        for proc in parse_result.processes:
            print(f"      • {proc.name}: {proc.description}")
        if parse_result.workflows:
            print(f"    Found {Colors.BOLD}{len(parse_result.workflows)}{Colors.END} workflows:")
            for wf in parse_result.workflows:
                print(f"      • {wf.name}: {len(wf.processes)} process calls")
        if parse_result.includes:
            print(f"    Found {Colors.BOLD}{len(parse_result.includes)}{Colors.END} includes")

    # Map processes to REDI pipeline stages
    stage_mapping = nf_map_processes_to_stages(parse_result.processes)

    if style == "rich":
        print_section("Stage Mapping")
        for stage, mapped_procs in stage_mapping.items():
            if mapped_procs:
                print(f"    {stage}: {', '.join(mapped_procs)}")

    # Find and parse Nextflow logs
    logs = find_nextflow_logs(workflow_path.parent)
    log_info = None
    if logs:
        if style == "rich":
            print_section("Nextflow Logs Found")
            for log in logs[:5]:
                print(f"    • {log.name}")
            if len(logs) > 5:
                print(f"    {Colors.DIM}... and {len(logs) - 5} more{Colors.END}")
        log_info = {"logs_found": len(logs), "log_files": [str(l) for l in logs[:10]]}
    else:
        if style == "rich":
            print_section("Nextflow Logs")
            print(f"    {Colors.DIM}No Nextflow logs found{Colors.END}")

    # Scan data files if data_dir provided
    snapshots = []
    quality_reports = []

    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Scanning Data Files")

        # Find all data files
        data_extensions = {'.nc', '.nc4', '.npz', '.npy', '.parquet', '.h5', '.hdf5', '.zarr', '.fasta', '.fa', '.pdb'}
        data_files = []
        for ext in data_extensions:
            data_files.extend(data_dir.rglob(f'*{ext}'))

        # Sort by modification time (oldest first = earlier in pipeline)
        data_files = sorted(data_files, key=lambda f: f.stat().st_mtime)

        if style == "rich":
            print(f"    Found {Colors.BOLD}{len(data_files)}{Colors.END} data files")

        # Capture snapshots (limit to first 20 for speed)
        for f in data_files[:20]:
            try:
                if style == "rich":
                    print(f"    📊 Scanning {f.name}...", end="", flush=True)

                snapshot = capture_snapshot(f)
                snapshots.append(snapshot)

                # Run quality checks
                qr = check_snapshot(snapshot)
                quality_reports.append(qr)

                if style == "rich":
                    if qr.passed:
                        print(f"\r    ✅ {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")
                    else:
                        print(f"\r    ⚠️  {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")

            except Exception as e:
                if style == "rich":
                    print(f"\r    ❌ {f.name}: {str(e)[:50]}")

        if len(data_files) > 20:
            if style == "rich":
                print(f"    {Colors.DIM}... and {len(data_files) - 20} more files{Colors.END}")

    # Detect transformation parameters
    transform_report = None
    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Detecting Transformation Parameters")

        transform_report = detect_transformations(data_dir, snapshots)

        if transform_report.normalization:
            norm = transform_report.normalization
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Normalization detected: {Colors.BOLD}{norm.method}{Colors.END}")
                if norm.mean_file:
                    print(f"      Mean file: {Path(norm.mean_file).name}")
                if norm.std_file:
                    print(f"      Std file: {Path(norm.std_file).name}")
                if norm.mean_values:
                    print(f"      Variables: {len(norm.mean_values)}")
                    for var, mean in list(norm.mean_values.items())[:3]:
                        std = norm.std_values.get(var, 'N/A')
                        if isinstance(std, (int, float)):
                            print(f"        {var}: mean={mean:.4g}, std={std:.4g}")
                        else:
                            print(f"        {var}: mean={mean:.4g}")
                    if len(norm.mean_values) > 3:
                        print(f"        {Colors.DIM}... and {len(norm.mean_values) - 3} more{Colors.END}")

        if transform_report.splits:
            splits = transform_report.splits
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Train/val/test split detected: {Colors.BOLD}{splits.method}{Colors.END}")
                if splits.train_size:
                    print(f"      Train: {splits.train_size:,} samples")
                if splits.val_size:
                    print(f"      Val: {splits.val_size:,} samples")
                if splits.test_size:
                    print(f"      Test: {splits.test_size:,} samples")

        if transform_report.regrid:
            regrid = transform_report.regrid
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Regridding detected: {Colors.BOLD}{regrid.method}{Colors.END}")
                if regrid.target_resolution:
                    print(f"      Target resolution: {regrid.target_resolution[0]}° x {regrid.target_resolution[1]}°")

        if not transform_report.normalization and not transform_report.splits and not transform_report.regrid:
            if style == "rich":
                print(f"    {Colors.DIM}No transformation parameters detected{Colors.END}")

    # Check for data quality issues in transformation report
    transform_issues = []
    if transform_report:
        transform_issues = check_transformation_report(transform_report)

        if transform_issues:
            if style == "rich":
                print_section("Transformation Quality Checks")
                for issue in transform_issues:
                    if issue.severity.value == 'error':
                        icon = f"{Colors.RED}✗{Colors.END}"
                    elif issue.severity.value == 'warning':
                        icon = f"{Colors.YELLOW}⚠{Colors.END}"
                    else:
                        icon = f"{Colors.DIM}ℹ{Colors.END}"
                    print(f"    {icon} {Colors.BOLD}{issue.check_name}:{Colors.END} {issue.message}")

    # Generate improvement suggestions
    from .inspect import generate_suggestions, format_suggestions_cli
    all_quality_issues = transform_issues + [issue for qr in quality_reports for issue in qr.issues]
    # Convert processes to rules-like format for workflow_info
    rules_like = [{'name': p.name, 'description': p.description} for p in parse_result.processes]
    workflow_info = {'rules': rules_like, 'stage_mapping': stage_mapping}
    suggestions = generate_suggestions(
        quality_issues=all_quality_issues,
        transformation_report=transform_report,
        workflow_info=workflow_info,
        data_snapshots=snapshots,
    )

    if suggestions and style == "rich":
        print_section("Suggestions for Improvement")
        print(format_suggestions_cli(suggestions, use_color=True))

    # Create output directory and save analysis
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate analyze report
    report = {
        "workflow_file": str(workflow_path),
        "workflow_type": "nextflow",
        "dsl_version": parse_result.dsl_version,
        "processes": [p.to_dict() for p in parse_result.processes],
        "workflows": [w.to_dict() for w in parse_result.workflows],
        "includes": [{"module": i.module_name, "from": i.source_path, "alias": i.alias} for i in parse_result.includes],
        "params": parse_result.params,
        "stage_mapping": stage_mapping,
        "mode": "analyze",
    }

    # Add data snapshots if we scanned
    if snapshots:
        report["data_snapshots"] = [s.to_dict() for s in snapshots]
        report["quality_summary"] = {
            "files_scanned": len(snapshots),
            "passed": sum(1 for qr in quality_reports if qr.passed),
            "with_warnings": sum(1 for qr in quality_reports if qr.has_warnings),
            "failed": sum(1 for qr in quality_reports if not qr.passed),
            "issues": [issue.to_dict() for qr in quality_reports for issue in qr.issues]
        }

    # Add transformation parameters if detected
    if transform_report:
        report["transformations"] = transform_report.to_dict()

    # Add transformation quality issues
    if transform_issues:
        report["transformation_quality"] = [issue.to_dict() for issue in transform_issues]

    # Add suggestions
    if suggestions:
        report["suggestions"] = [s.to_dict() for s in suggestions]
        report["suggestions_summary"] = {
            "total": len(suggestions),
            "high_priority": sum(1 for s in suggestions if s.priority == 1),
            "medium_priority": sum(1 for s in suggestions if s.priority == 2),
            "low_priority": sum(1 for s in suggestions if s.priority == 3),
        }

    # Add log info
    if log_info:
        report["execution_logs"] = log_info

    import json
    report_path = out_path / "analyze_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate workflow markdown if we have snapshots or processes
    if snapshots or parse_result.processes:
        datacard_path = out_path / "WORKFLOW.md"
        datacard_content = _generate_nextflow_datacard(
            workflow_path, parse_result, stage_mapping, snapshots, quality_reports,
            transform_report, suggestions
        )
        datacard_path.write_text(datacard_content)

        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}analyze_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
            print(f"    📋 {Colors.BOLD}WORKFLOW.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
    else:
        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}analyze_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

    if style == "rich":
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Nextflow analysis complete{Colors.END}")
        if not data_dir:
            print(f"{Colors.DIM}  Tip: Use --data-dir to scan intermediate data files{Colors.END}")
        print()

    # Watch mode: monitor for changes and re-analyze
    if watch:
        _run_nextflow_watch_mode(workflow_path, data_dir, output_dir, style)

    return report


def _generate_nextflow_datacard(workflow_path: Path, parse_result, stage_mapping: dict,
                                 snapshots: list, quality_reports: list,
                                 transform_report=None, suggestions=None) -> str:
    """Generate a workflow markdown from Nextflow analyze mode."""
    from datetime import datetime

    lines = [
        f"# Workflow Card: {workflow_path.parent.name}",
        "",
        f"**Generated by:** REDI Analyze",
        f"**Workflow:** `{workflow_path.name}`",
        f"**Workflow Type:** Nextflow DSL{parse_result.dsl_version}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Processes section
    if parse_result.processes:
        lines.extend([
            "## Processes",
            "",
            "| Process | Description | Inputs | Outputs |",
            "|---------|-------------|--------|---------|",
        ])
        for proc in parse_result.processes:
            inputs = ", ".join(proc.inputs[:3]) if proc.inputs else "-"
            if len(proc.inputs) > 3:
                inputs += f" (+{len(proc.inputs) - 3})"
            outputs = ", ".join(proc.outputs[:3]) if proc.outputs else "-"
            if len(proc.outputs) > 3:
                outputs += f" (+{len(proc.outputs) - 3})"
            lines.append(f"| {proc.name} | {proc.description or 'Processing step'} | {inputs} | {outputs} |")
        lines.append("")

    # Stage mapping section
    has_mapping = any(procs for procs in stage_mapping.values())
    if has_mapping:
        lines.extend([
            "## Stage Mapping",
            "",
            "| REDI Stage | Nextflow Processes |",
            "|------------|-------------------|",
        ])
        for stage, procs in stage_mapping.items():
            if procs:
                lines.append(f"| {stage} | {', '.join(procs)} |")
        lines.append("")

    # Workflows section
    if parse_result.workflows:
        lines.extend([
            "## Workflows",
            "",
        ])
        for wf in parse_result.workflows:
            lines.append(f"### {wf.name}")
            if wf.processes:
                lines.append(f"- **Process calls:** {', '.join(wf.processes)}")
            if wf.inputs:
                lines.append(f"- **Inputs:** {', '.join(wf.inputs)}")
            if wf.outputs:
                lines.append(f"- **Outputs:** {', '.join(wf.outputs)}")
            lines.append("")

    # Params section
    if parse_result.params:
        lines.extend([
            "## Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ])
        for name, value in list(parse_result.params.items())[:20]:
            # Truncate long values
            val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            lines.append(f"| `params.{name}` | `{val_str}` |")
        if len(parse_result.params) > 20:
            lines.append(f"| ... | ({len(parse_result.params) - 20} more) |")
        lines.append("")

    # Transformations section
    if transform_report:
        lines.extend([
            "## Transformations",
            "",
        ])

        if transform_report.normalization:
            norm = transform_report.normalization
            lines.extend([
                "### Normalization",
                f"- **Method:** {norm.method}",
            ])
            if norm.mean_file:
                lines.append(f"- **Mean file:** `{Path(norm.mean_file).name}`")
            if norm.std_file:
                lines.append(f"- **Std file:** `{Path(norm.std_file).name}`")
            if norm.computed_on:
                lines.append(f"- **Computed on:** {norm.computed_on}")
            lines.append("")

        if transform_report.splits:
            splits = transform_report.splits
            lines.extend([
                "### Train/Val/Test Split",
                f"- **Method:** {splits.method}",
            ])
            if splits.train_size:
                lines.append(f"- **Train:** {splits.train_size:,} samples")
            if splits.val_size:
                lines.append(f"- **Val:** {splits.val_size:,} samples")
            if splits.test_size:
                lines.append(f"- **Test:** {splits.test_size:,} samples")
            lines.append("")

        if transform_report.regrid:
            regrid = transform_report.regrid
            lines.extend([
                "### Regridding",
                f"- **Method:** {regrid.method}",
            ])
            if regrid.target_resolution:
                lines.append(f"- **Target resolution:** {regrid.target_resolution[0]}° x {regrid.target_resolution[1]}°")
            lines.append("")

    # Data files section
    if snapshots:
        lines.extend([
            "## Data Files",
            "",
            "| File | Variables | Shape | Issues |",
            "|------|-----------|-------|--------|",
        ])
        for i, snap in enumerate(snapshots[:15]):
            fname = Path(snap.path).name
            nvars = len(snap.variables)
            # Get shape from first variable if available
            if snap.variables:
                first_var = list(snap.variables.values())[0]
                shape = str(first_var.shape) if first_var.shape else "-"
            else:
                shape = "-"
            # Get issues from quality report
            if i < len(quality_reports):
                qr = quality_reports[i]
                issues = qr.summary() if not qr.passed else "✓ OK"
            else:
                issues = "-"
            lines.append(f"| `{fname}` | {nvars} | {shape} | {issues} |")
        if len(snapshots) > 15:
            lines.append(f"| ... | | | ({len(snapshots) - 15} more files) |")
        lines.append("")

    # Quality summary
    if quality_reports:
        passed = sum(1 for qr in quality_reports if qr.passed)
        warned = sum(1 for qr in quality_reports if qr.has_warnings)
        failed = sum(1 for qr in quality_reports if not qr.passed)
        lines.extend([
            "## Quality Summary",
            "",
            f"- **Files scanned:** {len(quality_reports)}",
            f"- **Passed:** {passed}",
            f"- **With warnings:** {warned}",
            f"- **Failed:** {failed}",
            "",
        ])

    # Includes section
    if parse_result.includes:
        lines.extend([
            "## Module Includes",
            "",
        ])
        for inc in parse_result.includes:
            if inc.alias:
                lines.append(f"- `{inc.module_name}` as `{inc.alias}` from `{inc.source_path}`")
            else:
                lines.append(f"- `{inc.module_name}` from `{inc.source_path}`")
        lines.append("")

    # Suggestions section
    if suggestions:
        from .inspect import format_suggestions_markdown
        lines.append(format_suggestions_markdown(suggestions))

    lines.extend([
        "",
        "---",
        "*Generated by REDI Analyze - Workflow Observability*",
    ])

    return "\n".join(lines)


def _run_nextflow_watch_mode(workflow_path: Path, data_dir: Optional[Path], output_dir: str, style: str):
    """Run watch mode for Nextflow - monitor for file changes and re-analyze."""
    import time

    # Collect files to watch
    watch_files = [workflow_path]
    if data_dir and data_dir.exists():
        data_extensions = {'.nc', '.nc4', '.npz', '.npy', '.parquet', '.h5', '.hdf5', '.zarr', '.fasta', '.fa', '.pdb'}
        for ext in data_extensions:
            watch_files.extend(data_dir.rglob(f'*{ext}'))

    # Also watch Nextflow logs
    nextflow_log = workflow_path.parent / ".nextflow.log"
    if nextflow_log.exists():
        watch_files.append(nextflow_log)

    # Watch work directory for new outputs
    work_dir = workflow_path.parent / "work"
    if work_dir.exists():
        watch_files.extend(list(work_dir.rglob(".command.log"))[:20])

    # Record initial timestamps
    file_mtimes = {}
    for f in watch_files:
        if f.exists():
            try:
                file_mtimes[str(f)] = f.stat().st_mtime
            except OSError:
                pass

    if style == "rich":
        print(f"\n{Colors.CYAN}👁 Watch mode active{Colors.END}")
        print(f"  Monitoring {Colors.BOLD}{len(file_mtimes)}{Colors.END} files for changes")
        print(f"  {Colors.DIM}Press Ctrl+C to stop{Colors.END}\n")

    poll_interval = 2.0  # seconds

    try:
        while True:
            time.sleep(poll_interval)

            # Check for changes
            changed_files = []
            new_files = []

            # Check existing files for modifications
            for path_str, old_mtime in list(file_mtimes.items()):
                path = Path(path_str)
                if path.exists():
                    try:
                        new_mtime = path.stat().st_mtime
                        if new_mtime > old_mtime:
                            changed_files.append(path)
                            file_mtimes[path_str] = new_mtime
                    except OSError:
                        pass

            # Check for new files in data_dir
            if data_dir and data_dir.exists():
                for ext in data_extensions:
                    for f in data_dir.rglob(f'*{ext}'):
                        if str(f) not in file_mtimes:
                            new_files.append(f)
                            try:
                                file_mtimes[str(f)] = f.stat().st_mtime
                            except OSError:
                                pass

            # Re-run analysis if changes detected
            if changed_files or new_files:
                if style == "rich":
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\n{Colors.YELLOW}[{timestamp}] Changes detected:{Colors.END}")
                    for f in changed_files[:3]:
                        print(f"  {Colors.DIM}modified:{Colors.END} {f.name}")
                    for f in new_files[:3]:
                        print(f"  {Colors.DIM}new:{Colors.END} {f.name}")
                    if len(changed_files) + len(new_files) > 6:
                        print(f"  {Colors.DIM}... and {len(changed_files) + len(new_files) - 6} more{Colors.END}")
                    print()

                # Re-run the analysis
                _analyze_nextflow(workflow_path, output_dir, False, style, data_dir)

    except KeyboardInterrupt:
        if style == "rich":
            print(f"\n{Colors.DIM}Watch mode stopped{Colors.END}")


# =============================================================================
# DISCOVER MODE (redi discover)
# =============================================================================
# Business logic lives in redi/discover/__init__.py
from .discover import run_discover


def run_parallel_pipeline(
    input_paths: List[str],
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None,
    style: str = "rich",
    backend: str = "local",
    workers: Optional[int] = None,
    batch_size: int = 4,
    distribution: str = "round_robin",
    generate_only: bool = False,
    slurm_nodes: int = 4,
    slurm_cores_per_node: int = 56,
    slurm_time: str = "0:30:00",
    slurm_account: Optional[str] = None,
    slurm_partition: str = "batch",
    slurm_final_output: Optional[str] = None,
    dragon_mode: str = "subprocess",
    dragon_nodes: int = 1,
    dragon_launcher: str = "local",
):
    """Run REDI pipeline in parallel mode."""
    from pathlib import Path

    try:
        from .parallel import ParallelExecutor, ParallelConfig
    except ImportError as e:
        if style == "rich":
            print(f"{Colors.RED}Error: Parallel module not available: {e}{Colors.END}")
        else:
            print(f"Error: Parallel module not available: {e}")
        return

    paths = [Path(p) for p in input_paths]

    if style == "rich":
        print_banner()
        print_section("Parallel Execution")
        print_metric("Files", f"{len(paths)}")
        print_metric("Backend", backend)
        print_metric("Workers", str(workers) if workers else "auto")
        print_metric("Batch Size", str(batch_size))
        print_metric("Distribution", distribution)
        if backend in ("gnu", "mpi"):
            print_metric("SLURM Nodes", str(slurm_nodes))
            print_metric("Cores/Node", str(slurm_cores_per_node))
            print_metric("Time Limit", slurm_time)
            if slurm_account:
                print_metric("Account", slurm_account)
        if backend == "dragon":
            print_metric("Dragon Mode", dragon_mode)
            print_metric("Dragon Nodes", str(dragon_nodes))
            print_metric("Dragon Launcher", dragon_launcher)
    elif style == "simple":
        print(f"Parallel: {len(paths)} files, backend={backend}, workers={workers or 'auto'}")

    # Build redi arguments to pass to each worker
    redi_args = []
    if domain:
        redi_args.extend(["-d", domain])
    if stages:
        redi_args.extend(["-s", ",".join(stages)])

    # Add config values that affect pipeline execution
    safe_config = dict(config or {})
    if safe_config.get('output_format'):
        redi_args.extend(["-f", safe_config['output_format']])

    # Create executor
    executor = ParallelExecutor(
        backend=backend,
        workers=workers,
        batch_size=batch_size,
        output_dir=output_dir,
        redi_args=redi_args,
        slurm_nodes=slurm_nodes,
        slurm_cores_per_node=slurm_cores_per_node,
        slurm_time=slurm_time,
        slurm_account=slurm_account,
        slurm_partition=slurm_partition,
        slurm_final_output=slurm_final_output,
        dragon_mode=dragon_mode,
        dragon_nodes=dragon_nodes,
        dragon_launcher=dragon_launcher,
    )

    # Update config distribution
    executor.config.distribution = distribution

    if generate_only:
        # Generate scripts without executing
        script_dir = Path(output_dir) / "parallel_scripts"

        if style == "rich":
            print_section("Generating Scripts")

        scripts = executor.generate_scripts(paths, script_dir)

        if style == "rich":
            print(f"    {Colors.GREEN}✓{Colors.END} Generated scripts in {script_dir}")
            for name, path in scripts.items():
                print(f"      📄 {Colors.BOLD}{name}{Colors.END}")

            if backend in ("gnu", "mpi"):
                print()
                print(f"    {Colors.CYAN}To submit:{Colors.END} cd {script_dir} && sbatch multinodejob.sh")
            elif backend == "dragon":
                print()
                print(f"    {Colors.CYAN}To run:{Colors.END} cd {script_dir} && bash run_dragon.sh")
        elif style == "simple":
            print(f"Scripts generated in: {script_dir}")
            for name, path in scripts.items():
                print(f"  {name}: {path}")
        elif style == "json":
            import json
            print(json.dumps({
                "scripts_dir": str(script_dir),
                "scripts": {k: str(v) for k, v in scripts.items()}
            }, indent=2))
        return

    # Execute
    if style == "rich":
        print_section("Executing")
        print(f"    {Colors.YELLOW}⏳ Running parallel jobs...{Colors.END}")

    result = executor.execute(paths)

    # Display results
    if style == "rich":
        print_section("Results")
        if result.success:
            print(f"    {Colors.GREEN}✓{Colors.END} Parallel execution completed")
        else:
            print(f"    {Colors.RED}✗{Colors.END} Parallel execution failed")

        print_metric("Total Files", str(result.total_files))
        print_metric("Completed", str(result.completed))
        print_metric("Failed", str(result.failed))

        if result.errors:
            print()
            print(f"    {Colors.BOLD}Messages:{Colors.END}")
            for err in result.errors[:5]:
                print(f"      {Colors.DIM}{err}{Colors.END}")

        if result.scripts_dir:
            print()
            print(f"    {Colors.DIM}Scripts: {result.scripts_dir}{Colors.END}")

        if result.output_dirs:
            print_section("Output Directories")
            for d in result.output_dirs[:10]:
                print(f"    📁 {d}")
            if len(result.output_dirs) > 10:
                print(f"    {Colors.DIM}... and {len(result.output_dirs) - 10} more{Colors.END}")

        print()
        if result.success:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Parallel pipeline completed!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Parallel pipeline completed with errors{Colors.END}")
        print()

    elif style == "simple":
        status = "OK" if result.success else "FAILED"
        print(f"Parallel: {status}, {result.completed}/{result.total_files} completed")
        if result.errors:
            for err in result.errors:
                print(f"  {err}")

    elif style == "json":
        import json
        print(json.dumps({
            "success": result.success,
            "total_files": result.total_files,
            "completed": result.completed,
            "failed": result.failed,
            "errors": result.errors,
            "output_dirs": [str(d) for d in result.output_dirs],
            "scripts_dir": str(result.scripts_dir) if result.scripts_dir else None,
        }, indent=2))


# =============================================================================
# MAIN CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and return the REDI argument parser.

    Returns:
        Configured ArgumentParser with all subcommands (run, analyze, discover, validate).
    """
    parser = argparse.ArgumentParser(
        prog='redi',
        description="REDI - Readiness Engine for Data Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:

* ``run`` — Run the pipeline (orchestrator mode) - default
* ``inspect`` — Inspect data files or observe existing workflows
* ``discover`` — Analyze unknown datasets and propose pipelines
* ``validate`` — Validate REDI output against ground truth

Examples::

  redi run data.csv                     # Run pipeline on data
  redi data.csv                         # Same as above (run is default)
  redi inspect data.csv                 # Inspect a data file
  redi inspect Snakefile                # Observe Snakemake workflow
  redi discover /path/to/data/          # Analyze unknown dataset
  redi validate output/ -c config.yaml  # Validate against ground truth

For command-specific help::

  redi run --help
  redi inspect --help
  redi discover --help
  redi validate --help
        """
    )

    try:
        from importlib.metadata import version
        _version = version("redi")
    except Exception:
        _version = "unknown"
    parser.add_argument("--version", action="version", version=f"redi {_version}")

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # =========================================================================
    # redi run (orchestrator)
    # =========================================================================
    run_parser = subparsers.add_parser(
        'run',
        help='Run the REDI pipeline (orchestrator mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples::

  redi run data.csv                      # Auto-detect and run full pipeline
  redi run data.csv --domain hpc         # Specify domain
  redi run data.csv -o ./processed       # Custom output directory
  redi run data.csv --stages ingest,preprocess  # Run specific stages
  redi run "data_*.nc"                   # Process multiple files (glob)

Download examples (climate domain)::

  redi run --download --source esgf --variable tas --model TaiESM1
  redi run --download --source cds --variable 2m_temperature --year 2020
        """
    )
    run_parser.add_argument("input", nargs="*", help="Input file path(s) or glob pattern")
    _builtin_domains = ["climate", "materials", "life_sciences", "fusion", "hpc", "geospatial", "nuclear_physics", "combustion", "general"]
    try:
        from .plugins import get_registry
        _plugin_domains = [p.domain for p in get_registry().values()]
    except Exception:
        _plugin_domains = []
    _all_domains = _builtin_domains + [d for d in _plugin_domains if d not in _builtin_domains]
    run_parser.add_argument("-d", "--domain",
                           choices=_all_domains,
                           help="Scientific domain (auto-detected if not specified)")
    run_parser.add_argument("-o", "--output", help="Output directory (disables hash-based naming)")
    run_parser.add_argument("--unique", action="store_true",
                           help="Append timestamp to output dir for unique runs")
    run_parser.add_argument("--hash-content", action="store_true", dest="hash_content",
                           help="Hash file content (not just path+size) for output dir naming")
    run_parser.add_argument("-s", "--stages",
                           help="Comma-separated list of stages to run. "
                                "Available: ingest, preprocess, transform, structure, output. "
                                "Prerequisites are auto-included (e.g., -s preprocess runs ingest,preprocess)")
    run_parser.add_argument("-f", "--format", choices=["npz", "parquet", "csv", "hdf5", "zarr", "pt"],
                           default="npz", help="Output format")
    run_parser.add_argument("--shards", type=int, default=1, metavar="N",
                           help="Number of output shards per split (e.g. match your training node count)")
    run_parser.add_argument("--array", action="store_true",
                           help="Optimize pipeline for array data (skip DataFrame-specific steps)")
    run_parser.add_argument("--normalize", choices=["minmax", "zscore"],
                           help="Normalization method for array data")
    run_parser.add_argument("--provenance", action="store_true",
                           help="Enable FlowCept provenance tracking and generate PROVENANCE_CARD.md")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    run_parser.add_argument("--verify", action="store_true",
                           help="Run verification checks after each pipeline step")
    run_parser.add_argument("--raps-jobs", action="store_true", dest="raps_jobs",
                           help="Extract RAPS Job objects from HPC telemetry (HPC domain only)")
    run_parser.add_argument("--start",
                           help="Filter jobs from this datetime, inclusive "
                                "(ISO format: 2017-10-03T00:00 or YYYYMMDD). "
                                "Overrides 'start' key in YAML config.")
    run_parser.add_argument("--end",
                           help="Filter jobs until this datetime, exclusive "
                                "(ISO format: 2017-10-04T00:00 or YYYYMMDD). "
                                "Overrides 'end' key in YAML config.")
    run_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                           help="Output style")
    run_parser.add_argument("-c", "--config", help="YAML configuration file path")

    # Download options (for climate domain)
    download_group = run_parser.add_argument_group("download options",
        "Download data from remote sources (ESGF for CMIP6, CDS for ERA5)")
    download_group.add_argument("--download", action="store_true",
                               help="Download data before processing")
    download_group.add_argument("--source", choices=["esgf", "cds"], default="esgf",
                               help="Data source: esgf (CMIP6) or cds (ERA5)")
    download_group.add_argument("--variable",
                               help="Variable to download (e.g., tas, pr, t2m)")
    download_group.add_argument("--model",
                               help="Climate model/source_id (e.g., TaiESM1, MPI-ESM1-2-HR)")
    download_group.add_argument("--experiment", default="historical",
                               help="Experiment ID (e.g., historical, ssp585)")
    download_group.add_argument("--frequency", default="6hr",
                               help="Time frequency (e.g., 6hr, day, mon)")
    download_group.add_argument("--year", type=int,
                               help="Year to download (for CDS/ERA5)")

    # Parallel execution group
    parallel_group = run_parser.add_argument_group("parallel execution")
    parallel_group.add_argument("-p", "--parallel", action="store_true",
                                help="Enable parallel execution across input files")
    parallel_group.add_argument("-w", "--workers", type=int, default=None,
                                help="Number of parallel workers (default: auto-detect CPU count)")
    parallel_group.add_argument("--backend", choices=["local", "gnu", "mpi", "dragon"],
                                default="local",
                                help="Parallel backend: local (GNU parallel, no SLURM), "
                                     "gnu (SLURM + GNU parallel), mpi (SLURM + mpi4py), "
                                     "dragon (Dragon HPC distributed runtime)")
    parallel_group.add_argument("--batch-size", type=int, default=4,
                                help="Files per worker batch (default: 4)")
    parallel_group.add_argument("--generate-only", action="store_true",
                                help="Generate parallel scripts without executing (for SLURM)")
    parallel_group.add_argument("--distribution", choices=["round_robin", "balanced"],
                                default="round_robin",
                                help="File distribution strategy (default: round_robin)")

    # Dragon-specific options
    dragon_group = run_parser.add_argument_group("Dragon options (for --backend dragon)")
    dragon_group.add_argument("--dragon-mode", choices=["subprocess", "inprocess"],
                              default="subprocess",
                              help="subprocess: each worker calls 'redi run' as a subprocess; "
                                   "inprocess: pipeline runs inside the Dragon worker (experimental)")
    dragon_group.add_argument("--dragon-nodes", type=int, default=1,
                              help="Number of nodes for multi-node Dragon launch (default: 1)")
    dragon_group.add_argument("--dragon-launcher", choices=["local", "slurm"], default="local",
                              help="local: call 'dragon' directly; "
                                   "slurm: wrap in a SLURM batch script via --generate-only")

    # SLURM-specific options
    slurm_group = run_parser.add_argument_group("SLURM options (for --backend gnu, mpi, or dragon)")
    slurm_group.add_argument("--slurm-nodes", type=int, default=4,
                             help="Number of compute nodes (default: 4)")
    slurm_group.add_argument("--slurm-cores", type=int, default=56,
                             help="CPU cores per node (default: 56)")
    slurm_group.add_argument("--slurm-time", type=str, default="0:30:00",
                             help="Time limit HH:MM:SS (default: 0:30:00)")
    slurm_group.add_argument("--slurm-account", type=str,
                             help="SLURM account/allocation")
    slurm_group.add_argument("--slurm-partition", type=str, default="batch",
                             help="SLURM partition (default: batch)")
    slurm_group.add_argument("--slurm-output", type=str,
                             help="Final output directory (for copying from burst buffer)")


    # =========================================================================
    # redi inspect (formerly analyze)
    # =========================================================================
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect data files or observe existing workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  redi inspect data.csv                       # Inspect a data file
  redi inspect /path/to/data.nc               # Inspect a NetCDF file
  redi inspect Snakefile                      # Auto-detect and observe workflow
  redi inspect pipeline.nf --type nextflow    # Observe Nextflow workflow
  redi inspect --workflow Snakefile --watch   # Watch for changes
  redi inspect -c inspect_config.yaml
        """
    )
    inspect_parser.add_argument("input", nargs="?",
                                help="Path to data file, workflow file, or directory")
    inspect_parser.add_argument("-w", "--workflow",
                                dest="workflow_flag",
                                help="Path to workflow file (overrides positional input)")
    inspect_parser.add_argument("-t", "--type", dest="workflow_type", default="auto",
                                choices=["auto", "snakemake", "nextflow", "bash"],
                                help="Workflow type override (forces workflow mode)")
    inspect_parser.add_argument("-o", "--output", help="Output directory (disables hash-based naming)")
    inspect_parser.add_argument("--unique", action="store_true",
                                help="Append timestamp to output dir for unique runs")
    inspect_parser.add_argument("--data-dir", dest="data_dir",
                                help="Directory containing data files to scan for statistics")
    inspect_parser.add_argument("--watch", action="store_true",
                                help="Watch for changes and re-run")
    inspect_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                                help="Output style")
    inspect_parser.add_argument("-c", "--config", help="YAML configuration file")

    # =========================================================================
    # redi discover
    # =========================================================================
    discover_parser = subparsers.add_parser(
        'discover',
        help='Analyze unknown datasets and propose pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files written to the output directory:

- ``discovery_report.json`` -- full inspection/inference/proposal data
- ``DISCOVERY_REPORT.md`` -- human-readable domain-level summary
- ``processing_plan.md`` -- per-file discoveries and preprocessing suggestions

Examples::

  redi discover /path/to/data/
  redi discover dataset.csv -o ./analysis
  redi discover /data --no-interactive
  redi discover -c discover_config.yaml
        """
    )
    discover_parser.add_argument("input", nargs="?", help="Path to data file or directory")
    discover_parser.add_argument("-o", "--output", help="Output directory (disables hash-based naming)")
    discover_parser.add_argument("--unique", action="store_true",
                                help="Append timestamp to output dir for unique runs")
    discover_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                                help="Output style")
    discover_parser.add_argument("--no-interactive", action="store_true",
                                help="Don't prompt for confirmation")
    discover_parser.add_argument("-c", "--config", help="YAML configuration file")

    # =========================================================================
    # redi validate
    # =========================================================================
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate REDI output against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples::

  redi validate output/usa_0_data_5c79b84 -c examples/nasa_power_validate.yaml
  redi validate output/usa_0_data_5c79b84 --ground-truth /path/to/truth.pt
  redi validate output/usa_0_data_5c79b84 -c config.yaml --aidrin
  redi validate output/usa_0_data_5c79b84 -c config.yaml --style json
        """
    )
    validate_parser.add_argument("input", help="Path to REDI output directory or a direct output file (npz, npy, parquet, csv)")
    validate_parser.add_argument("-g", "--ground-truth", dest="ground_truth",
                                 help="Path to ground truth file (overrides config)")
    validate_parser.add_argument("--gt-variable", dest="gt_variable",
                                 help="Variable name to extract from NetCDF ground truth (default: first variable)")
    validate_parser.add_argument("-c", "--config", help="Validation YAML config file")
    validate_parser.add_argument("-o", "--output", help="Output directory for validation report")
    validate_parser.add_argument("--aidrin", action="store_true",
                                 help="Run AIDRIN metric comparison (slow)")
    validate_parser.add_argument("--per-feature", action="store_true", dest="per_feature",
                                 help="Per-feature similarity breakdown")
    validate_parser.add_argument("--samples", type=int, nargs="?", const=10, default=0,
                                 metavar="N",
                                 help="Show N random job samples side-by-side (default 10)")
    validate_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                                 help="Output style")
    validate_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="Verbose output")

    assess_parser = subparsers.add_parser(
        'assess',
        help='Assess AI-readiness with domain-specific metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples::

  redi assess data.csv                       # auto-detect domain
  redi assess output/frontier/raps_jobs.npz --domain hpc
  redi assess /opt/drai/ClimaX/TaiESM1/tas.nc
  redi assess /opt/drai/aflow/aflow_ml_ready.csv --domain materials
  redi assess --style json data.parquet      # JSON output
        """
    )
    assess_parser.add_argument("input", nargs="?", help="Path to data file")
    assess_parser.add_argument("-d", "--domain",
                               choices=["climate", "materials", "life_sciences",
                                        "fusion", "hpc", "geospatial", "general"],
                               help="Domain (auto-detected if omitted)")
    assess_parser.add_argument("-o", "--output", help="Save JSON report to this path")
    assess_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                               help="Output style (default: rich)")
    assess_parser.add_argument("-c", "--config", help="YAML config file (supports multi-file comparison)")

    # --- install ---
    install_parser = subparsers.add_parser(
        'install',
        help='Install a REDI plugin from a URL, local path, or PyPI package',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples::

  redi install git+https://github.com/org/redi-xgc1
  redi install ./redi-xgc1            # local directory
  redi install redi-xgc1==1.0.0       # PyPI package
  redi install ./plugin.tar.gz --upgrade
        """
    )
    install_parser.add_argument("target", help="pip-compatible install target")
    install_parser.add_argument("--upgrade", action="store_true",
                                help="Upgrade if already installed")

    # --- plugins ---
    plugins_parser = subparsers.add_parser(
        'plugins',
        help='List installed REDI plugins',
    )
    plugins_parser.add_argument("--json", action="store_true",
                                help="Output as JSON")

    # =========================================================================
    # redi demo
    # =========================================================================
    demo_parser = subparsers.add_parser(
        'demo',
        help='Download a sample dataset and run the pipeline end-to-end',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Downloads a small public dataset (or generates synthetic data) for the chosen
domain, runs the full REDI pipeline, and prints the resulting Data Card.

Available domains::

  general    UCI Wine Quality (~84 KB CSV)              [default]
  hpc        Synthetic HPC job log (5,000 jobs)
  climate    NOAA GHCND Honolulu station (~400 KB CSV)
  geospatial EPA Annual AQI by County 2023 (~250 KB CSV)
  materials  Synthetic crystal structure dataset

Examples::

  redi demo                          # Run general demo (UCI Wine Quality)
  redi demo --domain hpc             # Run HPC telemetry demo
  redi demo --domain climate -o ./out
        """
    )
    demo_parser.add_argument(
        "-d", "--domain",
        choices=["general", "hpc", "climate", "geospatial", "materials"],
        default=None,
        help="Domain to demo (default: general)",
    )
    demo_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: ./demo_output/<domain>)",
    )
    demo_parser.add_argument(
        "--list", action="store_true",
        help="List available demos and exit",
    )
    demo_parser.add_argument(
        "--style", choices=["rich", "simple", "json"], default="rich",
        help="Output style",
    )

    return parser


def main():
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Handle backward compatibility: if no command but has positional args, assume 'run'
    if args.command is None:
        # Check if there are any remaining args that look like input files
        # Re-parse with 'run' as default
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            # First arg is likely a file, treat as 'redi run'
            sys.argv.insert(1, 'run')
            args = parser.parse_args()
        elif len(sys.argv) > 1:
            # Has flags but no command - might be old-style usage
            # Try to handle common flags
            parser.print_help()
            return
        else:
            parser.print_help()
            return

    # Dispatch to appropriate handler
    if args.command == 'run':
        _handle_run_command(args)
    elif args.command == 'inspect':
        _handle_inspect_command(args)
    elif args.command == 'discover':
        _handle_discover_command(args)
    elif args.command == 'validate':
        _handle_validate_command(args)
    elif args.command == 'assess':
        _handle_assess_command(args)
    elif args.command == 'install':
        _handle_install_command(args)
    elif args.command == 'plugins':
        _handle_plugins_command(args)
    elif args.command == 'demo':
        _handle_demo_command(args)
    else:
        parser.print_help()


def _handle_assess_command(args):
    """Handle the 'assess' subcommand."""
    from .assess import run_assess
    import json as _json

    style = args.style
    domain = getattr(args, 'domain', None)

    # Config-file mode: assess one or more files defined in YAML
    config_path = getattr(args, 'config', None)
    if config_path:
        try:
            import yaml
        except ImportError:
            print(f"{Colors.RED}Error: PyYAML is required for config files (pip install pyyaml){Colors.END}")
            sys.exit(1)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        domain = domain or cfg.get("domain")
        style = style or cfg.get("style", "rich")
        files = cfg.get("files", [])
        if not files:
            print(f"{Colors.RED}Error: config must contain a 'files' list{Colors.END}")
            sys.exit(1)
        results = []
        for p in files:
            if not Path(p).exists():
                print(f"{Colors.YELLOW}Warning: file not found, skipping: {p}{Colors.END}")
                continue
            result = run_assess(p, domain=domain)
            results.append(result)
            if style == "json":
                pass  # collected below
            elif style == "simple":
                _print_assess_simple(result)
            else:
                _print_assess_rich(result)
        if style == "json":
            print(_json.dumps(results, indent=2, default=str))
        out_path = getattr(args, 'output', None) or cfg.get("output")
        if out_path:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w') as f:
                _json.dump(results, f, indent=2, default=str)
            if style == "rich":
                print(f"\n    {Colors.BOLD}Report saved:{Colors.END} {Colors.DIM}{out}{Colors.END}")
        return

    # Single-file mode
    path = args.input
    if not path:
        print(f"{Colors.RED}Error: provide a file path or -c/--config{Colors.END}")
        sys.exit(1)
    if not Path(path).exists():
        print(f"{Colors.RED}Error: file not found: {path}{Colors.END}")
        sys.exit(1)

    result = run_assess(path, domain=domain)

    if style == "json":
        print(_json.dumps(result, indent=2, default=str))
    elif style == "simple":
        _print_assess_simple(result)
    else:
        _print_assess_rich(result)

    if getattr(args, 'output', None):
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            _json.dump(result, f, indent=2, default=str)
        if style == "rich":
            print(f"\n    {Colors.BOLD}Report saved:{Colors.END} {Colors.DIM}{out}{Colors.END}")


def _print_assess_rich(result: dict):
    """Display a readiness assessment in rich terminal format."""
    level = result["readiness_level"]
    level_name = result["readiness_name"]
    domain = result["domain"]
    u = result["universal"]
    s = result["domain_specific"]
    gaps = result["gaps"]

    level_colors = {1: Colors.RED, 2: Colors.YELLOW, 3: Colors.YELLOW,
                    4: Colors.CYAN, 5: Colors.GREEN}
    lc = level_colors.get(level, Colors.END)

    print_banner()
    print_section(f"Readiness Assessment: {Path(result['path']).name}")
    print_metric("Domain", domain)
    print(f"    {Colors.BOLD}Readiness Level:{Colors.END} "
          f"{lc}{Colors.BOLD}Level {level} — {level_name}{Colors.END}")

    # Universal metrics
    print_section("Universal Metrics")
    if "rows" in u:
        print_metric("rows", f"{u['rows']:,}")
    if "columns" in u:
        print_metric("columns", str(u["columns"]))
    if u.get("completeness") is not None:
        _print_metric_bar("completeness", u["completeness"], good=0.95, bad=0.80)
    if u.get("duplicate_ratio") is not None:
        _print_metric_bar("duplicate_ratio", u["duplicate_ratio"], good=0.01, bad=0.05, invert=True)
    if u.get("outlier_ratio") is not None:
        _print_metric_bar("outlier_ratio", u["outlier_ratio"], good=0.05, bad=0.15, invert=True)

    if u.get("null_by_column"):
        print(f"    {Colors.DIM}Worst null columns:{Colors.END}")
        for col, frac in list(u["null_by_column"].items())[:3]:
            bar = _mini_bar(frac)
            print(f"      {Colors.DIM}{col}:{Colors.END} {bar} {frac:.1%}")
    if u.get("outlier_by_column"):
        print(f"    {Colors.DIM}Worst outlier columns:{Colors.END}")
        for col, frac in list(u["outlier_by_column"].items())[:3]:
            bar = _mini_bar(frac)
            print(f"      {Colors.DIM}{col}:{Colors.END} {bar} {frac:.1%}")
    if u.get("class_imbalance_ratio") is not None:
        ratio = u["class_imbalance_ratio"]
        col = u.get("class_imbalance_col", "")
        color = Colors.YELLOW if ratio > 5 else Colors.GREEN
        print(f"    {Colors.DIM}class_imbalance ({col}):{Colors.END} {color}{ratio:.1f}:1{Colors.END}")

    # Domain-specific metrics
    if s:
        print_section(f"Domain Metrics ({domain})")

        req = s.get("required_fields", {})
        if req:
            print(f"    {Colors.BOLD}Required Fields:{Colors.END}")
            for field, info in req.items():
                if info.get("present"):
                    cov = info.get("coverage", 1.0)
                    cov_color = Colors.GREEN if cov >= 0.95 else Colors.YELLOW
                    col_hint = f" [{info['column']}]" if info.get("column") else ""
                    print(f"      {Colors.GREEN}✓{Colors.END} {field}{Colors.DIM}{col_hint}{Colors.END} "
                          f"coverage={cov_color}{cov:.1%}{Colors.END}")
                else:
                    print(f"      {Colors.RED}✗{Colors.END} {field} {Colors.DIM}(not found){Colors.END}")

        # OpenFold / life-sciences NPZ metrics
        if s.get("msa_depth") is not None:
            depth = s["msa_depth"]
            color = Colors.GREEN if depth >= 128 else (Colors.YELLOW if depth >= 32 else Colors.RED)
            print_metric("msa depth", f"{color}{depth:,} sequences{Colors.END}")
        if s.get("msa_unique_sequences") is not None:
            uniq = s["msa_unique_sequences"]
            total_msa = s.get("msa_depth", uniq)
            pct = uniq / total_msa if total_msa > 0 else 1.0
            color = Colors.GREEN if pct >= 0.99 else Colors.YELLOW
            print_metric("msa unique", f"{color}{uniq:,} ({pct:.1%}){Colors.END}")
        if s.get("msa_gap_ratio") is not None:
            _print_metric_bar("msa gap ratio", s["msa_gap_ratio"], good=0.05, bad=0.30, invert=True)
        if s.get("atom_completeness") is not None and s.get("msa_depth") is not None:
            _print_metric_bar("atom completeness", s["atom_completeness"], good=0.70, bad=0.50)
        if s.get("backbone_completeness") is not None:
            _print_metric_bar("backbone completeness", s["backbone_completeness"], good=0.99, bad=0.95)
        if s.get("resolution_angstrom") is not None:
            res = s["resolution_angstrom"]
            color = Colors.GREEN if res <= 2.5 else (Colors.YELLOW if res <= 3.5 else Colors.RED)
            print_metric("resolution", f"{color}{res} Å{Colors.END}")
        if s.get("aa_entropy_bits") is not None:
            ent = s["aa_entropy_bits"]
            pct = s.get("aa_entropy_pct_uniform", 0)
            color = Colors.GREEN if pct >= 85 else (Colors.YELLOW if pct >= 70 else Colors.RED)
            imb = s.get("aa_imbalance_ratio", "?")
            print_metric("AA entropy", f"{color}{ent:.3f} bits ({pct:.1f}% of uniform){Colors.END}"
                         f"  {Colors.DIM}max/min ratio: {imb}:1{Colors.END}")
        if s.get("aa_distribution"):
            aa_dist = s["aa_distribution"]
            top5 = sorted(aa_dist.items(), key=lambda x: -x[1])[:5]
            bot5 = sorted(aa_dist.items(), key=lambda x: x[1])[:3]
            print(f"    {Colors.DIM}AA freq (top 5):{Colors.END}  "
                  + "  ".join(f"{aa}={v:.1%}" for aa, v in top5))
            print(f"    {Colors.DIM}AA freq (bot 3):{Colors.END}  "
                  + "  ".join(f"{aa}={v:.1%}" for aa, v in bot5))

        # CIF / PDB-specific
        if s.get("experimental_method") is not None and s.get("msa_depth") is None:
            print_metric("experimental method", s["experimental_method"])
        if s.get("r_factor") is not None:
            rf = s["r_factor"]
            color = Colors.GREEN if rf < 0.20 else (Colors.YELLOW if rf < 0.25 else Colors.RED)
            print_metric("R-factor", f"{color}{rf:.4f}{Colors.END}")
        if s.get("resolution_angstrom") is not None and s.get("msa_depth") is None:
            res = s["resolution_angstrom"]
            color = Colors.GREEN if res <= 2.5 else (Colors.YELLOW if res <= 3.5 else Colors.RED)
            print_metric("resolution", f"{color}{res} Å{Colors.END}")
        if s.get("atom_completeness") is not None and s.get("msa_depth") is None:
            _print_metric_bar("atom completeness", s["atom_completeness"], good=0.75, bad=0.50)

        # Climate NPZ metrics (REDI-processed arrays)
        if s.get("array_shape") is not None:
            shape = s["array_shape"]
            print_metric("array shape", f"{shape}")
            _rendered_climate_npz = True
        if s.get("n_timesteps") is not None:
            print_metric("timesteps", f"{s['n_timesteps']:,}")
        if s.get("spatial_shape") is not None:
            print_metric("spatial grid", f"{s['spatial_shape'][0]} lat × {s['spatial_shape'][1]} lon")
        if s.get("value_mean") is not None:
            mean, std = s["value_mean"], s.get("value_std", "?")
            vmin, vmax = s.get("value_min", "?"), s.get("value_max", "?")
            print_metric("value range", f"[{vmin}, {vmax}]  mean={mean}  std={std}")
        if s.get("appears_normalized") is not None:
            flag = s["appears_normalized"]
            color = Colors.GREEN if flag else Colors.YELLOW
            print_metric("normalized", f"{color}{'yes (z-score)' if flag else 'no'}{Colors.END}")
        if s.get("normalization_mean") is not None:
            for var, mu in s["normalization_mean"].items():
                sigma = s.get("normalization_std", {}).get(var, "?")
                print_metric(f"norm {var}", f"mean={mu}  std={sigma}")
        if s.get("nan_fraction") is not None:
            frac = s["nan_fraction"]
            color = Colors.GREEN if frac == 0 else Colors.RED
            print_metric("NaN fraction", f"{color}{frac:.6f}{Colors.END}")
        if s.get("variable_names") is not None:
            print_metric("variables", ", ".join(s["variable_names"]))

        # Domain-specific extras (generic scalar fallback, skip already-rendered keys)
        _rendered = {
            "msa_depth", "msa_unique_sequences", "msa_gap_ratio",
            "atom_completeness", "backbone_completeness", "resolution_angstrom",
            "aa_entropy_bits", "aa_entropy_pct_uniform", "aa_imbalance_ratio",
            "aa_distribution", "experimental_method", "r_factor", "is_distillation",
            "structure_resolution_flag",
            # climate NPZ keys
            "array_shape", "n_timesteps", "spatial_shape", "value_min", "value_max",
            "value_mean", "value_std", "appears_normalized", "normalization_mean",
            "normalization_std", "nan_fraction", "variable_names", "npz_keys", "dtype",
        }
        for key, val in s.items():
            if key == "required_fields" or key in _rendered:
                continue
            if isinstance(val, dict) or isinstance(val, list):
                continue
            label = key.replace("_", " ")
            print_metric(label, str(val))

        # Temporal range
        for tkey in ("temporal_range", "temporal_days", "temporal_steps"):
            if tkey in s:
                print_metric(tkey.replace("_", " "), str(s[tkey]))

        # Variables list (climate)
        if "variables" in s:
            print_metric("variables", ", ".join(s["variables"][:8])
                         + (f" (+{len(s['variables'])-8} more)" if len(s["variables"]) > 8 else ""))

        # Utilization traces (HPC)
        if "utilization_traces" in s:
            traces = s["utilization_traces"]
            color = Colors.GREEN if traces else Colors.DIM
            print_metric("utilization traces", f"{color}{', '.join(traces) if traces else 'none'}{Colors.END}")

        # NaN by variable (climate)
        if "nan_fraction_by_variable" in s:
            print(f"    {Colors.DIM}NaN fraction by variable:{Colors.END}")
            for var, frac in s["nan_fraction_by_variable"].items():
                bar = _mini_bar(frac)
                print(f"      {Colors.DIM}{var}:{Colors.END} {bar} {frac:.1%}")

    # Gaps
    if gaps:
        print_section("Gaps to Next Level")
        for gap in gaps:
            print(f"    {Colors.YELLOW}○{Colors.END} {gap}")
    else:
        next_level = level + 1
        if next_level <= 5:
            print_section(f"To Reach Level {next_level}")
            print(f"    {Colors.GREEN}✓{Colors.END} {Colors.DIM}No blocking gaps detected at this level{Colors.END}")

    print()


def _print_assess_simple(result: dict):
    u = result["universal"]
    s = result["domain_specific"]
    print(f"file:         {result['path']}")
    print(f"domain:       {result['domain']}")
    print(f"level:        {result['readiness_level']} ({result['readiness_name']})")
    print(f"rows:         {u.get('rows', 'N/A')}")
    print(f"completeness: {u.get('completeness', 'N/A')}")
    print(f"dup_ratio:    {u.get('duplicate_ratio', 'N/A')}")
    print(f"outlier_ratio:{u.get('outlier_ratio', 'N/A')}")
    req = s.get("required_fields", {})
    if req:
        present = [k for k, v in req.items() if v.get("present")]
        missing = [k for k, v in req.items() if not v.get("present")]
        print(f"fields_ok:    {', '.join(present)}")
        if missing:
            print(f"fields_miss:  {', '.join(missing)}")
    if result["gaps"]:
        for g in result["gaps"]:
            print(f"gap:          {g}")


def _print_metric_bar(label: str, value: float, good: float, bad: float, invert: bool = False):
    """Print a metric with a color indicator."""
    if invert:
        color = Colors.GREEN if value <= good else (Colors.YELLOW if value <= bad else Colors.RED)
    else:
        color = Colors.GREEN if value >= good else (Colors.YELLOW if value >= bad else Colors.RED)
    bar = _mini_bar(value if not invert else 1 - value)
    print(f"    {Colors.DIM}{label}:{Colors.END} {bar} {color}{value:.4f}{Colors.END}")


def _mini_bar(fraction: float, width: int = 10) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return f"{Colors.DIM}[{'█' * filled}{'░' * (width - filled)}]{Colors.END}"


def _handle_run_command(args):
    """Handle the 'run' subcommand."""
    config_path = args.config
    input_args = args.input

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
    if args.output is not None:
        config['output_dir'] = args.output
    if args.format != 'npz':
        config['output_format'] = args.format
    if args.shards > 1:
        config['shards'] = args.shards
    if args.verbose:
        config['verbose'] = True
    if args.stages:
        config['stages'] = args.stages.split(',')
    if getattr(args, 'provenance', False):
        config['provenance'] = True

    if 'output_format' not in config:
        config['output_format'] = args.format

    # Build download config from CLI args
    if hasattr(args, 'download') and args.download:
        config['download'] = {
            'enabled': True,
            'source': args.source,
            'variable': args.variable,
            'model': args.model,
            'experiment': args.experiment,
            'frequency': args.frequency,
            'year': args.year,
        }
        # Remove None values
        config['download'] = {k: v for k, v in config['download'].items() if v is not None}

    # --start / --end: CLI takes priority over YAML config
    if getattr(args, 'start', None):
        config['start'] = args.start
    if getattr(args, 'end', None):
        config['end'] = args.end

    verbose = args.verbose or config.get('verbose', False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle input from config if not provided on CLI
    if not input_paths:
        config_input = config.get('input_path') or config.get('input')
        config_input_paths = config.get('input_paths')

        if config_input_paths:
            input_paths = expand_input_paths(config_input_paths)
        elif config_input:
            input_paths = expand_input_paths([config_input])

    # Check for input paths (not required if download mode is enabled)
    download_enabled = config.get('download', {}).get('enabled', False)
    if not input_paths and not download_enabled:
        print(f"{Colors.RED}Error: input file path(s) required. Provide on CLI or set "
              f"'input_path' or 'input_paths' in the YAML config.{Colors.END}")
        print(f"{Colors.DIM}Tip: Use --download to fetch data from ESGF or CDS{Colors.END}")
        sys.exit(1)

    # Determine output directory
    # If -o is explicitly set, use it directly (no hash)
    # Otherwise, generate hash-based output dir
    if args.output:
        output_dir = args.output
    elif input_paths:
        base_dir = config.get('output_dir') or './output'
        # If a YAML config was supplied, name the output after the config file
        # (e.g. "frontier") rather than the obscure input data filename.
        _name = Path(config_path).stem if config_path else None
        output_dir = generate_output_dir(
            input_paths=input_paths,
            base_dir=base_dir,
            unique=getattr(args, 'unique', False),
            hash_content=getattr(args, 'hash_content', False),
            name_override=_name,
        )
    else:
        # Download mode without input files - use config or default
        output_dir = config.get('output_dir') or './output'

    config['output_dir'] = output_dir

    # Log command for reproducibility
    from .repro import log_pipeline_run
    log_pipeline_run(
        output_dir=output_dir,
        input_paths=[str(p) for p in input_paths],
        domain=config.get('domain', args.domain),
        stages=stages if args.stages else config.get('stages'),
        array_mode=args.array if hasattr(args, 'array') else config.get('array_mode', False),
        normalize_method=args.normalize if hasattr(args, 'normalize') else config.get('normalize_method'),
        config_file=config_path,
    )

    # Check for parallel execution mode
    parallel_enabled = getattr(args, 'parallel', False) or config.get('parallel', {}).get('enabled', False)

    if parallel_enabled:
        run_parallel_pipeline(
            input_paths=[str(p) for p in input_paths],
            output_dir=config.get('output_dir', args.output),
            domain=config.get('domain', args.domain),
            stages=args.stages.split(",") if args.stages else config.get('stages'),
            config=config,
            style=args.style,
            backend=getattr(args, 'backend', None) or config.get('parallel', {}).get('backend', 'local'),
            workers=getattr(args, 'workers', None) or config.get('parallel', {}).get('workers'),
            batch_size=getattr(args, 'batch_size', None) or config.get('parallel', {}).get('batch_size', 4),
            distribution=getattr(args, 'distribution', None) or config.get('parallel', {}).get('distribution', 'round_robin'),
            generate_only=getattr(args, 'generate_only', False),
            slurm_nodes=getattr(args, 'slurm_nodes', None) or config.get('parallel', {}).get('slurm', {}).get('nodes', 4),
            slurm_cores_per_node=getattr(args, 'slurm_cores', None) or config.get('parallel', {}).get('slurm', {}).get('cores_per_node', 56),
            slurm_time=getattr(args, 'slurm_time', None) or config.get('parallel', {}).get('slurm', {}).get('time_limit', '0:30:00'),
            slurm_account=getattr(args, 'slurm_account', None) or config.get('parallel', {}).get('slurm', {}).get('account'),
            slurm_partition=getattr(args, 'slurm_partition', None) or config.get('parallel', {}).get('slurm', {}).get('partition', 'batch'),
            slurm_final_output=getattr(args, 'slurm_output', None) or config.get('parallel', {}).get('slurm', {}).get('final_output'),
            dragon_mode=getattr(args, 'dragon_mode', None) or config.get('parallel', {}).get('dragon', {}).get('mode', 'subprocess'),
            dragon_nodes=getattr(args, 'dragon_nodes', None) or config.get('parallel', {}).get('dragon', {}).get('nodes', 1),
            dragon_launcher=getattr(args, 'dragon_launcher', None) or config.get('parallel', {}).get('dragon', {}).get('launcher', 'local'),
        )
        return  # Exit after parallel execution

    # Run pipeline
    stages = config.get('stages')
    if args.stages:
        stages = args.stages.split(",")

    # Array mode settings
    array_mode = args.array if hasattr(args, 'array') else config.get('array_mode', False)
    normalize_method = args.normalize if hasattr(args, 'normalize') else config.get('normalize_method')

    verify = args.verify if hasattr(args, 'verify') else config.get('verify', False)
    # --raps-jobs flag takes priority; if not set, check YAML config
    raps_jobs = getattr(args, 'raps_jobs', False)
    if not raps_jobs:
        raps_cfg = config.get('raps_jobs', {})
        if isinstance(raps_cfg, dict):
            raps_jobs = raps_cfg.get('enabled', False)
        else:
            raps_jobs = bool(raps_cfg)

    ctx = run_pipeline(
        output_dir=config.get('output_dir', args.output),
        domain=config.get('domain', args.domain),
        stages=stages,
        config=config,
        verbose=verbose,
        style=args.style,
        input_paths=[str(p) for p in input_paths],
        array_mode=array_mode,
        normalize_method=normalize_method,
        verify=verify,
        raps_jobs=raps_jobs,
    )



_WORKFLOW_FILENAMES = {'Snakefile', 'Snakefile.smk'}
_WORKFLOW_EXTENSIONS = {'.smk', '.nf', '.nextflow'}


def _is_workflow(path: str, type_hint: str = 'auto') -> bool:
    """Return True if path should be treated as a workflow file."""
    if type_hint in ('snakemake', 'nextflow', 'bash'):
        return True
    p = Path(path)
    return p.name in _WORKFLOW_FILENAMES or p.suffix in _WORKFLOW_EXTENSIONS


def _run_data_inspect(path: str, output_dir: Optional[str], style: str):
    """Inspect a data file using DataAnalyzer and print/save the report."""
    from .assess import generate_report

    try:
        report = generate_report(path)
    except Exception as e:
        print(f"{Colors.RED}Error inspecting {path}: {e}{Colors.END}")
        sys.exit(1)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "inspection_report.md"
        report_path.write_text(report)
        print(report)
        print(f"\n{Colors.GREEN}Report saved to {report_path}{Colors.END}")
    else:
        print(report)


def _handle_inspect_command(args):
    """Handle the 'inspect' subcommand."""
    import yaml

    config = {}

    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # Resolve path: -w/--workflow wins over positional, config is fallback
    path = args.workflow_flag or getattr(args, 'input', None) or config.get('workflow') or config.get('input')
    if not path:
        print(f"{Colors.RED}Error: input required. Provide a file path or use -w/--workflow.{Colors.END}")
        sys.exit(1)

    workflow_type = args.workflow_type if args.workflow_type != 'auto' else config.get('workflow_type', 'auto')

    if _is_workflow(path, workflow_type):
        # Workflow analysis path
        if args.output:
            output_dir = args.output
        else:
            base_dir = config.get('output_dir') or './output'
            output_dir = generate_output_dir(
                input_paths=[Path(path)],
                base_dir=base_dir,
                unique=getattr(args, 'unique', False),
            )

        watch = args.watch or config.get('watch', False)
        style = args.style if args.style != 'rich' else config.get('style', 'rich')
        data_dir = args.data_dir or config.get('data_dir')

        from .repro import log_analyze_run
        log_analyze_run(
            output_dir=output_dir,
            workflow_file=path,
            workflow_type=workflow_type,
        )

        run_analyze(
            workflow_file=path,
            workflow_type=workflow_type,
            output_dir=output_dir,
            watch=watch,
            style=style,
            config=config,
            data_dir=data_dir,
        )
    else:
        # Data file / directory inspection
        style = args.style if args.style != 'rich' else config.get('style', 'rich')
        _run_data_inspect(path, args.output, style)


def _handle_discover_command(args):
    """Handle the 'discover' subcommand."""
    import yaml

    config = {}

    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI args override config
    input_path = args.input or config.get('input')
    if not input_path:
        print(f"{Colors.RED}Error: input path required. Provide on CLI or set in config.{Colors.END}")
        sys.exit(1)

    # Determine output directory
    # If -o is explicitly set, use it directly (no hash)
    # Otherwise, generate hash-based output dir
    if args.output:
        output_dir = args.output
    else:
        base_dir = config.get('output_dir') or './output'
        output_dir = generate_output_dir(
            input_paths=[Path(input_path)],
            base_dir=base_dir,
            unique=getattr(args, 'unique', False),
        )

    style = args.style if args.style != 'rich' else config.get('style', 'rich')
    interactive = not args.no_interactive if args.no_interactive else config.get('interactive', True)

    # Log command for reproducibility
    from .repro import log_discover_run
    log_discover_run(
        output_dir=output_dir,
        input_path=input_path,
    )

    run_discover(
        input_path=input_path,
        output_dir=output_dir,
        style=style,
        interactive=interactive,
        config=config,
    )


def _handle_validate_command(args):
    """Handle the 'validate' subcommand."""
    import yaml

    config = {}

    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI --ground-truth and --gt-variable override config
    if args.ground_truth:
        config["ground_truth"] = args.ground_truth
    if args.gt_variable:
        config["gt_variable"] = args.gt_variable

    # Must have ground truth from config or CLI
    if not config.get("ground_truth"):
        print(f"{Colors.RED}Error: ground truth path required. "
              f"Use --ground-truth or set 'ground_truth' in config YAML.{Colors.END}")
        sys.exit(1)

    verbose = args.verbose
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_validate(
        input_dir=args.input,
        config=config,
        aidrin=args.aidrin,
        per_feature=args.per_feature,
        style=args.style,
        output_dir=args.output,
        samples=args.samples,
    )


def run_validate(
    input_dir: str,
    config: dict,
    aidrin: bool = False,
    per_feature: bool = False,
    style: str = "rich",
    output_dir: Optional[str] = None,
    samples: int = 0,
):
    """Run validation and display results.

    Args:
        input_dir: Path to REDI output directory.
        config: Validation config dict (must include 'ground_truth').
        aidrin: If True, run AIDRIN metric comparison.
        per_feature: If True, compute per-feature breakdown.
        style: Output style: rich, simple, json.
        output_dir: Optional output directory for saving report files.
    """
    from .validate import run_validation

    t0 = time.time()

    if style == "rich":
        print_banner()
        print_section("Validation")
        print_metric("REDI output", input_dir)
        print_metric("Ground truth", config.get("ground_truth", ""))
        if config.get("description"):
            print_metric("Description", config["description"])

    try:
        result = run_validation(
            redi_dir=input_dir,
            config=config,
            aidrin=aidrin,
            per_feature=per_feature,
        )
    except Exception as e:
        if style == "json":
            import json as _json
            print(_json.dumps({"error": str(e)}, indent=2))
        elif style == "rich":
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        else:
            print(f"Error: {e}")
        sys.exit(1)

    elapsed = time.time() - t0

    if style == "json":
        import json as _json
        print(_json.dumps(result.to_dict(), indent=2, default=str))
        return result

    if style == "rich":
        print_metric("REDI shape", str(result.redi_shape))
        print_metric("Truth shape", str(result.ground_truth_shape))
        if result.aligned:
            # Infer the aligned shape from the target or ground truth shape
            alignment_cfg = config.get("alignment", {})
            target_shape = alignment_cfg.get("target_shape")
            if target_shape:
                print_metric("Aligned shape", str(tuple(target_shape)))
            else:
                print_metric("Aligned shape", str(result.ground_truth_shape))

        # Try per-field RAPS stats first; fall back to aggregate stats
        from .validate import raps_field_stats
        gt_path_for_stats = config.get("ground_truth", "")
        field_stats = raps_field_stats(input_dir, gt_path_for_stats)

        if field_stats:
            print_section("Per-Field Statistics")
            header = (f"    {'field':<22s}  {'REDI mean':>10s}  {'GT mean':>10s}"
                      f"  {'REDI std':>10s}  {'GT std':>10s}  {'MAE':>10s}")
            print(header)
            for fs in field_stats:
                mark = (f"{Colors.GREEN}=={Colors.END}" if fs["match"]
                        else f"{Colors.YELLOW}!={Colors.END}")
                print(f"    {fs['field']:<22s}  {fs['redi_mean']:>10.2f}  {fs['gt_mean']:>10.2f}"
                      f"  {fs['redi_std']:>10.2f}  {fs['gt_std']:>10.2f}  {fs['mae']:>10.2f}  {mark}")
        elif result.redi_stats and result.ground_truth_stats:
            print_section("Basic Statistics")
            header = f"    {'':20s} {'REDI':>14s}  {'Ground Truth':>14s}"
            print(header)
            for key in ("mean", "std", "min", "max"):
                redi_val = result.redi_stats.get(key, float("nan"))
                gt_val = result.ground_truth_stats.get(key, float("nan"))
                print(f"    {key:20s} {redi_val:>14.4f}  {gt_val:>14.4f}")

        if not field_stats and result.similarity:
            print_section("Similarity")
            print_metric("MAE", f"{result.similarity.get('mae', float('nan')):.6f}")
            print_metric("RMSE", f"{result.similarity.get('rmse', float('nan')):.6f}")
            print_metric("Correlation", f"{result.similarity.get('corr', float('nan')):.6f}")
            print_metric("KL Divergence", f"{result.similarity.get('kl', float('nan')):.6f}")

        if result.per_feature:
            print_section("Per-Feature Similarity")
            header = f"    {'Feature':>8s}  {'MAE':>10s}  {'RMSE':>10s}  {'Corr':>10s}  {'KL':>10s}"
            print(header)
            for feat in result.per_feature:
                idx = feat.get("feature_index", "?")
                print(f"    {idx:>8}  {feat['mae']:>10.6f}  {feat['rmse']:>10.6f}  "
                      f"{feat['corr']:>10.6f}  {feat['kl']:>10.6f}")

        if result.aidrin:
            print_section("AIDRIN Comparison")
            if "error" in result.aidrin:
                print(f"    {Colors.YELLOW}Warning: {result.aidrin['error']}{Colors.END}")
            else:
                if result.aidrin.get("similarity"):
                    sim = result.aidrin["similarity"]
                    print_metric("Overlap keys", str(sim.get("overlap_count", 0)))
                    if sim.get("mae") is not None:
                        print_metric("MAE", f"{sim['mae']:.6f}")
                    if sim.get("corr") is not None:
                        print_metric("Correlation", f"{sim['corr']:.6f}")

        if samples > 0:
            from .validate import raps_sample_comparison
            gt_path = config.get("ground_truth", "")
            job_samples = raps_sample_comparison(input_dir, gt_path, n=samples)
            if job_samples:
                print_section(f"Samples ({len(job_samples)} jobs)")
                for job in job_samples:
                    print(f"\n    {Colors.BOLD}job {job['id']}{Colors.END}")
                    print(f"    {'field':<22s}  {'REDI':>14s}  {'GT':>14s}  {'':>2s}")
                    for f in job["fields"]:
                        rv = f["redi"]
                        gv = f["gt"]
                        mark = f"{Colors.GREEN}=={Colors.END}" if f["match"] else f"{Colors.YELLOW}!={Colors.END}"
                        rv_s = f"{rv:.2f}" if isinstance(rv, float) else str(rv)
                        gv_s = f"{gv:.2f}" if isinstance(gv, float) else str(gv)
                        print(f"    {f['field']:<22s}  {rv_s:>14s}  {gv_s:>14s}  {mark}")

        print()
        print(f"{Colors.GREEN}{Colors.BOLD}v Validation complete{Colors.END} "
              f"{Colors.DIM}({elapsed:.1f}s){Colors.END}")
        print()

    else:
        # simple style
        print(f"REDI shape:   {result.redi_shape}")
        print(f"Truth shape:  {result.ground_truth_shape}")
        print(f"Aligned:      {result.aligned}")
        if result.redi_stats:
            print(f"REDI stats:   {result.redi_stats}")
        if result.ground_truth_stats:
            print(f"Truth stats:  {result.ground_truth_stats}")
        if result.similarity:
            print(f"Similarity:   {result.similarity}")
        if result.per_feature:
            print(f"Per-feature:  {len(result.per_feature)} features")
        if result.aidrin:
            print(f"AIDRIN:       {result.aidrin}")
        print(f"Elapsed: {elapsed:.1f}s")

    # Save report if output dir requested
    if output_dir:
        import json as _json
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report_json_path = out_path / "validation_report.json"
        with open(report_json_path, "w") as f:
            _json.dump(result.to_dict(), f, indent=2, default=str)

        report_md = _generate_validation_markdown(result, elapsed)
        report_md_path = out_path / "validation_report.md"
        report_md_path.write_text(report_md)

        if style == "rich":
            print_section("Output")
            print(f"    {Colors.BOLD}validation_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
            print(f"    {Colors.BOLD}validation_report.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
            print()
        elif style == "simple":
            print(f"Saved: {report_json_path}")
            print(f"Saved: {report_md_path}")

    return result


def _generate_validation_markdown(result, elapsed: float) -> str:
    """Generate a markdown validation report."""
    lines = [
        "# REDI Validation Report",
        "",
        f"**REDI output:** `{result.redi_dir}`",
        f"**Ground truth:** `{result.ground_truth_path}`",
        "",
        "## Shapes",
        "",
        f"| | Shape |",
        f"|---|---|",
        f"| REDI | `{result.redi_shape}` |",
        f"| Ground truth | `{result.ground_truth_shape}` |",
        f"| Aligned | `{result.aligned}` |",
        "",
    ]

    if result.redi_stats and result.ground_truth_stats:
        lines.extend([
            "## Basic Statistics",
            "",
            "| Metric | REDI | Ground Truth |",
            "|---|---|---|",
        ])
        for key in ("count", "mean", "std", "min", "max"):
            rv = result.redi_stats.get(key, "")
            gv = result.ground_truth_stats.get(key, "")
            if isinstance(rv, float):
                rv = f"{rv:.6f}"
            if isinstance(gv, float):
                gv = f"{gv:.6f}"
            lines.append(f"| {key} | {rv} | {gv} |")
        lines.append("")

    if result.similarity:
        lines.extend([
            "## Similarity",
            "",
            "| Metric | Value |",
            "|---|---|",
        ])
        for key in ("mae", "rmse", "corr", "kl"):
            val = result.similarity.get(key, "")
            if isinstance(val, float):
                val = f"{val:.6f}"
            lines.append(f"| {key} | {val} |")
        lines.append("")

    if result.per_feature:
        lines.extend([
            "## Per-Feature Similarity",
            "",
            "| Feature | MAE | RMSE | Corr | KL |",
            "|---|---|---|---|---|",
        ])
        for feat in result.per_feature:
            idx = feat.get("feature_index", "?")
            lines.append(
                f"| {idx} | {feat['mae']:.6f} | {feat['rmse']:.6f} | "
                f"{feat['corr']:.6f} | {feat['kl']:.6f} |"
            )
        lines.append("")

    lines.append(f"*Elapsed: {elapsed:.1f}s*")
    return "\n".join(lines) + "\n"


def _handle_install_command(args):
    """Handle the 'install' subcommand."""
    from .plugins import install, reload_registry
    import re as _re

    pip_args = ["--upgrade"] if args.upgrade else []

    target = args.target
    # Convert SCP-style SSH URLs (git@host:path) to pip's git+ssh://git@host/path
    if _re.match(r'^git@[^:]+:[^/]', target):
        target = _re.sub(r'^git@([^:]+):(.*)', r'git+ssh://git@\1/\2', target)

    print(f"{Colors.BOLD}Installing plugin:{Colors.END} {args.target}")
    rc = install(target, pip_args=pip_args)

    if rc != 0:
        print(f"{Colors.RED}Installation failed (pip exit code {rc}).{Colors.END}")
        sys.exit(rc)

    print(f"\n{Colors.GREEN}Installation successful.{Colors.END}")

    # Show newly available plugins
    plugins = reload_registry()
    if plugins:
        print(f"\n{Colors.BOLD}Installed REDI plugins:{Colors.END}")
        for p in plugins.values():
            print(f"  {Colors.CYAN}{p.domain}{Colors.END}  {p.version}  — {p.description}")
    else:
        print(f"\n{Colors.DIM}No redi.plugins entry points found in the installed package.{Colors.END}")
        print(f"{Colors.DIM}Make sure the package declares [project.entry-points.\"redi.plugins\"] in its pyproject.toml.{Colors.END}")


def _handle_plugins_command(args):
    """Handle the 'plugins' subcommand."""
    from .plugins import list_plugins
    import json as _json

    plugins = list_plugins()

    if getattr(args, "json", False):
        data = [
            {"name": p.name, "domain": p.domain,
             "version": p.version, "description": p.description}
            for p in plugins
        ]
        print(_json.dumps(data, indent=2))
        return

    if not plugins:
        print(f"{Colors.DIM}No REDI plugins installed.{Colors.END}")
        print(f"\nInstall a plugin with:  {Colors.CYAN}redi install <url|path|package>{Colors.END}")
        return

    print(f"{Colors.BOLD}Installed REDI plugins:{Colors.END}\n")
    for p in plugins:
        print(f"  {Colors.CYAN}{Colors.BOLD}{p.domain}{Colors.END}  "
              f"{Colors.DIM}v{p.version}{Colors.END}")
        if p.description:
            print(f"    {p.description}")
        steps = p.get_full_pipeline()
        if steps:
            print(f"    {Colors.DIM}{len(steps)} pipeline steps{Colors.END}")
        print()


def _handle_demo_command(args):
    """Handle the 'demo' subcommand."""
    import tempfile
    from .downloaders.demo import list_demos, download_demo, DEFAULT_DOMAIN

    demos = list_demos()

    if getattr(args, 'list', False):
        print(f"\n{Colors.BOLD}Available redi demo datasets:{Colors.END}\n")
        for key, meta in demos.items():
            tag = f"{Colors.DIM}[{meta['source']}]{Colors.END}"
            print(f"  {Colors.CYAN}{Colors.BOLD}{key:<12}{Colors.END}  {meta['name']}")
            print(f"              {Colors.DIM}{meta['description']}{Colors.END}")
            print(f"              {meta['size']}  {tag}")
            print()
        return

    domain = args.domain or DEFAULT_DOMAIN
    style = getattr(args, 'style', 'rich')
    meta = demos[domain]

    print_banner()
    print(f"{Colors.BOLD}Demo:{Colors.END} {meta['name']}")
    print(f"{Colors.DIM}{meta['description']}{Colors.END}")
    print(f"{Colors.DIM}Source: {meta['source']}  |  Size: {meta['size']}{Colors.END}\n")

    output_dir = args.output or f"./demo_output/{domain}"

    # Download / generate into a temp dir, then run the pipeline
    with tempfile.TemporaryDirectory(prefix="redi_demo_") as tmpdir:
        if meta['source'] == 'generated':
            print(f"  {Colors.YELLOW}⚙{Colors.END}  Generating synthetic {domain} dataset…")
        else:
            print(f"  {Colors.YELLOW}⬇{Colors.END}  Downloading {meta['name']} ({meta['size']})…")

        try:
            data_path, _ = download_demo(domain, Path(tmpdir))
        except Exception as exc:
            print(f"\n{Colors.RED}Error fetching demo data: {exc}{Colors.END}")
            print(f"{Colors.DIM}Check your network connection and try again.{Colors.END}")
            sys.exit(1)

        size_kb = data_path.stat().st_size // 1024
        print(f"  {Colors.GREEN}✓{Colors.END}  Ready — {size_kb} KB  →  {Colors.DIM}{data_path.name}{Colors.END}\n")

        run_pipeline(
            input_path=str(data_path),
            output_dir=output_dir,
            domain=meta['domain'],
            style=style,
        )

    print(f"\n{Colors.BOLD}Output written to:{Colors.END} {Colors.CYAN}{output_dir}{Colors.END}")
    print(f"{Colors.DIM}Run 'redi demo --list' to see all available demos.{Colors.END}\n")


if __name__ == "__main__":
    main()
