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
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

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


class PipelineStep:
    """Base class for pipeline steps"""

    name: str = "base_step"
    stage: PipelineStage = PipelineStage.INGEST
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this step - override in subclasses"""
        raise NotImplementedError

    def validate(self, ctx: PipelineContext) -> bool:
        """Validate prerequisites for this step"""
        return True

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

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all pipeline steps"""
        logger.info(f"Starting pipeline for domain: {ctx.domain}")

        for step in self.steps:
            logger.info(f"Executing step: {step.name}")

            if not step.validate(ctx):
                ctx.add_error(f"Validation failed for step: {step.name}")
                continue

            try:
                ctx = step.execute(ctx)
                logger.info(f"Completed step: {step.name}")
            except Exception as e:
                ctx.add_error(f"Error in step {step.name}: {str(e)}")
                logger.exception(f"Step {step.name} failed")

        return ctx

    def execute_stage(self, ctx: PipelineContext, stage: PipelineStage) -> PipelineContext:
        """Execute only steps for a specific stage"""
        stage_steps = [s for s in self.steps if s.stage == stage]

        for step in stage_steps:
            if not step.validate(ctx):
                ctx.add_error(f"Validation failed for step: {step.name}")
                continue

            try:
                ctx = step.execute(ctx)
            except Exception as e:
                ctx.add_error(f"Error in step {step.name}: {str(e)}")

        ctx.mark_complete(stage)
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

    return PipelineContext(
        input_path=primary_path,
        output_dir=Path(output_dir),
        input_paths=paths,
        domain=domain,
        config=config
    )
