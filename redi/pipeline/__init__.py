"""
DRAI Pipeline Module

Modular 5-stage pipeline for data preprocessing:
1. ingest - acquire, stream, format harmonize
2. preprocess - regrid/align/anonymize/clean
3. transform - feature engineer, normalize, encode
4. structure - tensorize, graph/sequence structure
5. output - shard, split, save
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _safe_json_value(value: Any) -> Any:
    """Convert common runtime objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    return value


def missing_dependency_error(package: str, extra: Optional[str] = None) -> str:
    """Create a consistent missing-dependency error with an install hint."""
    if extra:
        install_hint = f'pip install -e ".[{extra}]"'
    else:
        install_hint = "pip install -e ."
    return f"Missing dependency: {package}. Install with: {install_hint}"


class PipelineStage(Enum):
    """Pipeline stages matching the abstracted 5-level pattern"""
    INGEST = "ingest"
    PREPROCESS = "preprocess"
    TRANSFORM = "transform"
    STRUCTURE = "structure"
    OUTPUT = "output"


@dataclass
class PipelineContext:
    """Context passed through pipeline stages"""
    # Input/output paths - support both single and multiple files
    input_path: Path  # Primary input path (first file or single file)
    output_dir: Path
    input_paths: List[Path] = field(default_factory=list)  # All input paths for multi-file

    # Domain detection
    domain: str = "general"

    # Data containers (populated by stages)
    data: Any = None  # Main data object (DataFrame, xarray, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Pipeline state
    current_stage: PipelineStage = PipelineStage.INGEST
    completed_stages: List[PipelineStage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    verification_results: List[Any] = field(default_factory=list)  # List[VerificationResult]

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    verify_steps: bool = False  # Whether to run step verification

    with_flowcept: bool = True  # TODO do it right
    flowcept: Any = None

    @property
    def is_multi_file(self) -> bool:
        """Check if this context has multiple input files"""
        return len(self.input_paths) > 1

    @property
    def num_input_files(self) -> int:
        """Number of input files"""
        return len(self.input_paths) if self.input_paths else 1

    def mark_complete(self, stage: PipelineStage):
        """Mark a stage as completed"""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        # Advance to next stage
        stages = list(PipelineStage)
        idx = stages.index(stage)
        if idx + 1 < len(stages):
            self.current_stage = stages[idx + 1]

    def add_error(self, error: str):
        """Record an error"""
        self.errors.append(error)
        logger.error(error)


@dataclass
class VerificationResult:
    """Result of a step verification check"""
    passed: bool
    step_name: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def add_check(self, name: str, passed: bool, expected: Any = None, actual: Any = None, message: str = ""):
        """Add a verification check result"""
        self.checks.append({
            "name": name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "message": message
        })
        if not passed:
            self.passed = False

    def __bool__(self):
        return self.passed


class PipelineStep:
    """Base class for pipeline steps"""

    name: str = "base_step"
    stage: PipelineStage = PipelineStage.INGEST
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        execute = cls.__dict__.get("execute")
        if execute is None or getattr(execute, "_redi_wrapped", False):
            return

        @wraps(execute)
        def wrapped_execute(self, ctx: PipelineContext) -> PipelineContext:
            flowcept_task = None
            try:
                if (
                    ctx.with_flowcept
                    and ctx.flowcept is not None
                    and getattr(ctx.flowcept, "is_started", False)
                ):
                    from flowcept.instrumentation.task_capture import FlowceptTask

                    input_path_str = str(ctx.input_path)
                    input_paths_str = [str(p) for p in ctx.input_paths]
                    unique_input_paths = list(dict.fromkeys(input_paths_str))
                    used_dict = {
                        "input_path": input_path_str,
                        "output_dir": str(ctx.output_dir),
                    }
                    if not (len(unique_input_paths) == 1 and unique_input_paths[0] == input_path_str):
                        used_dict["input_paths"] = unique_input_paths
                    custom_metadata = {
                        "current_stage": ctx.current_stage.value,
                        "completed_stages": [s.value for s in ctx.completed_stages],
                        "metadata": _safe_json_value(ctx.metadata),
                    }

                    flowcept_task = FlowceptTask(
                        activity_id=self.__class__.__name__,
                        used=used_dict,
                        custom_metadata=custom_metadata,
                    )

                return execute(self, ctx)
            finally:
                if flowcept_task:
                    flowcept_task.end()

        wrapped_execute._redi_wrapped = True
        cls.execute = wrapped_execute

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this step - override in subclasses"""
        raise NotImplementedError

    def validate(self, ctx: PipelineContext) -> bool:
        """Validate prerequisites for this step"""
        return True

    def verify(self, ctx: PipelineContext) -> VerificationResult:
        """Verify step output - override in subclasses to add checks.

        Returns a VerificationResult with pass/fail status and details.
        Default implementation passes if no errors in context.
        """
        result = VerificationResult(passed=True, step_name=self.name)
        result.add_check(
            name="no_errors",
            passed=len(ctx.errors) == 0,
            message=f"Step completed without errors" if not ctx.errors else f"Errors: {ctx.errors}"
        )
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, stage={self.stage.value})"


class Pipeline:
    """Orchestrates pipeline execution"""

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.steps: List[PipelineStep] = []
        self._domain_handlers: Dict[str, Callable] = {}

    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self

    def register_domain_handler(self, domain: str, handler: Callable):
        """Register a domain-specific handler"""
        self._domain_handlers[domain] = handler

    def _start_flowcept(self, ctx: PipelineContext, workflow_args: Dict[str, Any]) -> bool:
        """Start Flowcept once for this execution path. Returns True if started here."""
        if not ctx.with_flowcept:
            return False
        if ctx.flowcept is None:
            from flowcept import Flowcept

            workflow_name = ctx.config.get("name") if isinstance(ctx.config, dict) else None
            ctx.flowcept = Flowcept(
                start_persistence=False,
                workflow_name=workflow_name,
                workflow_args=workflow_args,
            )
        if getattr(ctx.flowcept, "is_started", False):
            return False
        ctx.flowcept.start()
        return True

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all pipeline steps"""
        flowcept_started_here = self._start_flowcept(ctx, {"mode": "full_pipeline"})
        flowcept_ref = ctx.flowcept

        logger.info(f"Starting pipeline for domain: {ctx.domain}")

        current_stage = None
        for step in self.steps:
            if ctx.with_flowcept and flowcept_ref is not None and ctx.flowcept is None:
                ctx.flowcept = flowcept_ref
            logger.info(f"Executing step: {step.name}")

            # Track stage transitions
            if step.stage != current_stage:
                if current_stage is not None:
                    ctx.mark_complete(current_stage)
                current_stage = step.stage

            if not step.validate(ctx):
                ctx.add_error(f"Validation failed for step: {step.name}")
                continue

            try:
                next_ctx = step.execute(ctx)
                if (
                    next_ctx.with_flowcept
                    and flowcept_ref is not None
                    and getattr(next_ctx, "flowcept", None) is None
                ):
                    next_ctx.flowcept = flowcept_ref
                ctx = next_ctx
                logger.info(f"Completed step: {step.name}")

                # Run verification if enabled
                if ctx.verify_steps:
                    result = step.verify(ctx)
                    ctx.verification_results.append(result)
                    if result.passed:
                        logger.info(f"Verification passed for step: {step.name}")
                    else:
                        logger.warning(f"Verification FAILED for step: {step.name}")
                        for check in result.checks:
                            if not check["passed"]:
                                logger.warning(f"  - {check['name']}: {check.get('message', 'FAILED')}")

            except Exception as e:
                ctx.add_error(f"Error in step {step.name}: {str(e)}")
                logger.exception(f"Step {step.name} failed")

        # Mark the final stage complete
        if current_stage is not None:
            ctx.mark_complete(current_stage)

        if flowcept_started_here and flowcept_ref:
            flowcept_ref.stop()

        return ctx

    def execute_stage(self, ctx: PipelineContext, stage: PipelineStage) -> PipelineContext:
        """Execute only steps for a specific stage"""
        flowcept_started_here = self._start_flowcept(
            ctx,
            {"mode": "stage", "stage": stage.value},
        )
        flowcept_ref = ctx.flowcept
        stage_steps = [s for s in self.steps if s.stage == stage]

        for step in stage_steps:
            if ctx.with_flowcept and flowcept_ref is not None and ctx.flowcept is None:
                ctx.flowcept = flowcept_ref
            if not step.validate(ctx):
                ctx.add_error(f"Validation failed for step: {step.name}")
                continue

            try:
                next_ctx = step.execute(ctx)
                if (
                    next_ctx.with_flowcept
                    and flowcept_ref is not None
                    and getattr(next_ctx, "flowcept", None) is None
                ):
                    next_ctx.flowcept = flowcept_ref
                ctx = next_ctx
            except Exception as e:
                ctx.add_error(f"Error in step {step.name}: {str(e)}")

        ctx.mark_complete(stage) # TODO: are marking as complete only the last stage?

        if flowcept_started_here and flowcept_ref:
            flowcept_ref.stop()

        return ctx


# Convenience function to create a pipeline context
def create_context(
    input_path: Optional[str] = None,
    output_dir: str = "./output",
    domain: str = "general",
    input_paths: Optional[List[str]] = None,
    **config
) -> PipelineContext:
    """Create a new pipeline context

    Args:
        input_path: Single input file path (for backward compatibility)
        output_dir: Output directory path
        domain: Scientific domain
        input_paths: List of input file paths for multi-file processing
        **config: Additional configuration options

    Returns:
        PipelineContext configured for single or multi-file processing
    """
    # Handle input paths
    if input_paths:
        # Multi-file mode
        paths = [Path(p) for p in input_paths]
        primary_path = paths[0]
    elif input_path:
        # Single file mode (backward compatible)
        primary_path = Path(input_path)
        paths = [primary_path]
    else:
        raise ValueError("Either input_path or input_paths must be provided")

    with_flowcept = bool(config.get("provenance", False))

    return PipelineContext(
        input_path=primary_path,
        output_dir=Path(output_dir),
        input_paths=paths,
        domain=domain,
        config=config,
        with_flowcept=with_flowcept,
    )


# Convenience exports for commonly used steps
# These are lazy-loaded to avoid circular imports
def __getattr__(name):
    """Lazy load pipeline step classes"""
    if name in ('LoadNPY', 'LoadNPZ', 'LoadManifest', 'LoadCSV', 'LoadJSON',
                'LoadParquet', 'LoadNetCDF', 'LoadHDF5', 'LoadSICKLE',
                'DetectFormat', 'ValidateInput', 'get_ingest_steps'):
        from . import ingest
        return getattr(ingest, name)
    elif name in ('NormalizeArray', 'ComputeArrayStats', 'HandleMissingValues',
                  'RemoveDuplicates', 'ValidateDataTypes', 'HandleOutliers',
                  'AnonymizePII', 'RegridClimate', 'get_preprocess_steps'):
        from . import preprocess
        return getattr(preprocess, name)
    elif name in ('TrainValTestSplit', 'ShardData', 'SaveNPZ', 'SaveHDF5',
                  'SaveParquet', 'SaveMetadata', 'get_output_steps'):
        from . import output
        return getattr(output, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
