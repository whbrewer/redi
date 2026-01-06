"""
Unit tests for the REDI pipeline core module.
"""

import pytest
from pathlib import Path


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_create_context(self, temp_dir):
        """Test creating a pipeline context."""
        from redi.pipeline import create_context, PipelineStage

        ctx = create_context(
            input_path=str(temp_dir / "test.csv"),
            output_dir=str(temp_dir / "output"),
            domain="hpc"
        )

        assert ctx.input_path == temp_dir / "test.csv"
        assert ctx.output_dir == temp_dir / "output"
        assert ctx.domain == "hpc"
        assert ctx.current_stage == PipelineStage.INGEST
        assert len(ctx.completed_stages) == 0
        assert len(ctx.errors) == 0

    def test_context_mark_complete(self, temp_dir):
        """Test marking stages as complete."""
        from redi.pipeline import create_context, PipelineStage

        ctx = create_context(str(temp_dir / "test.csv"), str(temp_dir), "general")
        ctx.mark_complete(PipelineStage.INGEST)

        assert PipelineStage.INGEST in ctx.completed_stages
        assert ctx.current_stage == PipelineStage.PREPROCESS

    def test_context_add_error(self, temp_dir):
        """Test adding errors to context."""
        from redi.pipeline import create_context

        ctx = create_context(str(temp_dir / "test.csv"), str(temp_dir), "general")
        ctx.add_error("Test error message")

        assert len(ctx.errors) == 1
        assert "Test error" in ctx.errors[0]


class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        from redi.pipeline import Pipeline

        pipeline = Pipeline("hpc")
        assert pipeline.domain == "hpc"
        assert len(pipeline.steps) == 0

    def test_pipeline_add_step(self):
        """Test adding steps to pipeline."""
        from redi.pipeline import Pipeline, PipelineStep, PipelineStage

        class DummyStep(PipelineStep):
            name = "dummy"
            stage = PipelineStage.INGEST

            def execute(self, ctx):
                return ctx

        pipeline = Pipeline("general")
        step = DummyStep()
        pipeline.add_step(step)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "dummy"

    def test_pipeline_execute_empty(self, temp_dir, sample_csv):
        """Test executing empty pipeline."""
        from redi.pipeline import Pipeline, create_context

        pipeline = Pipeline("general")
        ctx = create_context(str(sample_csv), str(temp_dir), "general")

        result = pipeline.execute(ctx)
        assert result is not None
        assert len(result.errors) == 0


class TestPipelineStages:
    """Tests for pipeline stage enum."""

    def test_stage_values(self):
        """Test that all stages have correct values."""
        from redi.pipeline import PipelineStage

        assert PipelineStage.INGEST.value == "ingest"
        assert PipelineStage.PREPROCESS.value == "preprocess"
        assert PipelineStage.TRANSFORM.value == "transform"
        assert PipelineStage.STRUCTURE.value == "structure"
        assert PipelineStage.OUTPUT.value == "output"

    def test_stage_ordering(self):
        """Test that stages are in correct order."""
        from redi.pipeline import PipelineStage

        stages = list(PipelineStage)
        assert stages[0] == PipelineStage.INGEST
        assert stages[-1] == PipelineStage.OUTPUT


class TestIngestSteps:
    """Tests for ingest pipeline steps."""

    def test_detect_format_csv(self, sample_csv, temp_dir):
        """Test format detection for CSV files."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import DetectFormat

        ctx = create_context(str(sample_csv), str(temp_dir), "general")
        step = DetectFormat()

        result = step.execute(ctx)
        assert result.metadata.get("file_format") == "csv"

    def test_load_csv(self, sample_csv, temp_dir):
        """Test CSV loading."""
        from redi.pipeline import create_context
        from redi.pipeline.ingest import LoadCSV

        ctx = create_context(str(sample_csv), str(temp_dir), "general")
        ctx.metadata["file_format"] = "csv"

        step = LoadCSV()
        if step.validate(ctx):
            result = step.execute(ctx)
            assert result.data is not None
            assert len(result.data) == 10


class TestOutputSteps:
    """Tests for output pipeline steps."""

    def test_train_val_test_split(self, temp_dir):
        """Test data splitting."""
        import numpy as np
        from redi.pipeline import create_context
        from redi.pipeline.output import TrainValTestSplit

        ctx = create_context(str(temp_dir / "test.csv"), str(temp_dir), "general")
        ctx.data = {"X": np.random.rand(100, 10)}

        step = TrainValTestSplit()
        result = step.execute(ctx)

        assert "train" in result.data
        assert "val" in result.data
        assert "test" in result.data
        assert result.metadata["split"]["train_size"] == 80
