"""
Smoke tests for redi.discover processors and processing_plan renderer.
"""

import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def csv_file(tmp_path):
    """CSV with numeric, categorical, boolean, and missing-value columns."""
    content = textwrap.dedent("""\
        age,city,score,active,joined
        25,NYC,88.5,True,2020-01-15
        32,LA,72.0,False,2019-06-01
        ,NYC,91.0,True,2021-03-22
        45,Chicago,,False,2018-11-05
        28,NYC,65.5,True,2022-07-30
        31,LA,80.0,False,2020-04-10
        38,NYC,77.5,True,2019-09-14
        22,Chicago,95.0,False,2023-01-01
        50,LA,60.0,True,2017-05-20
        29,NYC,83.5,False,2021-11-11
    """)
    p = tmp_path / "sample.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def markdown_file(tmp_path):
    """Markdown file with headings, links, lists, and bare filenames."""
    content = textwrap.dedent("""\
        # Project Overview

        This project processes climate data.

        ## Setup

        Install dependencies listed in [requirements.txt](requirements.txt).
        Also see environment.yml for conda setup.

        ## Data

        Input files:
        - [raw temperature data](data/raw/temp.csv)
        - [raw precipitation data](data/raw/precip.csv)

        ## Results

        The figures below summarise the output:

        results/fig1.png
        results/fig2.png
        results/fig3.png
    """)
    p = tmp_path / "README.md"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# CsvProcessor tests
# ---------------------------------------------------------------------------

class TestCsvProcessor:
    def test_returns_file_discovery(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        assert d.file_type == "CSV"
        assert d.file_path == str(csv_file)
        assert d.file_size_bytes > 0

    def test_shape(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        shape = d.metadata["shape"]
        assert shape[0] == 10
        assert shape[1] == 5

    def test_column_titles(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        assert d.metadata["column_titles"] == ["age", "city", "score", "active", "joined"]

    def test_column_type_classification(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        types = d.metadata["column_types"]
        assert types["age"] == "numeric"
        assert types["city"] == "categorical"
        assert types["score"] == "numeric"

    def test_numeric_stats_present(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        age_stats = d.metadata["column_stats"]["age"]
        assert "mean" in age_stats
        assert "min" in age_stats
        assert "max" in age_stats
        assert "pct_missing" in age_stats

    def test_categorical_stats_present(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        city_stats = d.metadata["column_stats"]["city"]
        assert "n_unique" in city_stats
        assert "top_values" in city_stats

    def test_missing_value_triggers_impute_step(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        step_names = [s.name for s in d.preprocessing_steps]
        assert any("Impute" in name and "age" in name for name in step_names)
        assert any("Impute" in name and "score" in name for name in step_names)

    def test_numeric_column_gets_normalization_step(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        step_names = [s.name for s in d.preprocessing_steps]
        assert any("Normalize" in name and "age" in name for name in step_names)

    def test_categorical_column_gets_encode_step(self, csv_file):
        from redi.discover.processors.csv_processor import CsvProcessor
        d = CsvProcessor().analyze(csv_file)
        step_names = [s.name for s in d.preprocessing_steps]
        assert any("Encode" in name and "city" in name for name in step_names)

    def test_error_on_bad_file(self, tmp_path):
        from redi.discover.processors.csv_processor import CsvProcessor
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03" * 100)
        d = CsvProcessor().analyze(bad)
        # Should not raise; may return error metadata or a parsed result
        assert d.file_type == "CSV"


# ---------------------------------------------------------------------------
# MarkdownProcessor tests
# ---------------------------------------------------------------------------

class TestMarkdownProcessor:
    def test_returns_file_discovery(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        assert d.file_type == "Markdown"
        assert d.file_size_bytes > 0

    def test_no_preprocessing_steps(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        assert d.preprocessing_steps == []

    def test_headings_extracted(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        headings = d.metadata["headings"]
        assert any(h["text"] == "Project Overview" and h["level"] == 1 for h in headings)
        assert any(h["text"] == "Setup" and h["level"] == 2 for h in headings)
        assert any(h["text"] == "Data" and h["level"] == 2 for h in headings)

    def test_file_references_extracted(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        refs = d.metadata["file_references"]
        assert any("requirements.txt" in r for r in refs)
        assert any("temp.csv" in r for r in refs)

    def test_by_heading_groups(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        by_heading = d.metadata["file_groups"]["by_heading"]
        assert "Setup" in by_heading or any("Setup" in k for k in by_heading)

    def test_adjacency_groups(self, markdown_file):
        from redi.discover.processors.markdown_processor import MarkdownProcessor
        d = MarkdownProcessor().analyze(markdown_file)
        by_adj = d.metadata["file_groups"]["by_adjacency"]
        # fig1/fig2/fig3 should be grouped together
        flat = [ref for group in by_adj for ref in group]
        assert any("fig" in r for r in flat)


# ---------------------------------------------------------------------------
# ProcessorRegistry tests
# ---------------------------------------------------------------------------

class TestProcessorRegistry:
    def test_csv_dispatched(self, csv_file):
        from redi.discover.processors import ProcessorRegistry
        d = ProcessorRegistry.analyze_file(csv_file)
        assert d.file_type == "CSV"

    def test_md_dispatched(self, markdown_file):
        from redi.discover.processors import ProcessorRegistry
        d = ProcessorRegistry.analyze_file(markdown_file)
        assert d.file_type == "Markdown"

    def test_unknown_extension_uses_generic(self, tmp_path):
        from redi.discover.processors import ProcessorRegistry
        f = tmp_path / "data.xyz123"
        f.write_text("hello")
        d = ProcessorRegistry.analyze_file(f)
        assert d.preprocessing_steps == []
        assert d.file_size_bytes > 0


# ---------------------------------------------------------------------------
# render_processing_plan tests
# ---------------------------------------------------------------------------

class TestRenderProcessingPlan:
    def test_creates_file(self, tmp_path, csv_file, markdown_file):
        from redi.discover.processors import ProcessorRegistry
        from redi.discover.report import render_processing_plan

        discoveries = [
            ProcessorRegistry.analyze_file(csv_file),
            ProcessorRegistry.analyze_file(markdown_file),
        ]
        out = tmp_path / "processing_plan.md"
        render_processing_plan(discoveries, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contains_expected_sections(self, tmp_path, csv_file, markdown_file):
        from redi.discover.processors import ProcessorRegistry
        from redi.discover.report import render_processing_plan

        discoveries = [
            ProcessorRegistry.analyze_file(csv_file),
            ProcessorRegistry.analyze_file(markdown_file),
        ]
        out = tmp_path / "processing_plan.md"
        render_processing_plan(discoveries, out)
        text = out.read_text()

        assert "# Processing Plan" in text
        assert "## Summary" in text
        assert "sample.csv" in text
        assert "README.md" in text

    def test_csv_section_has_columns(self, tmp_path, csv_file):
        from redi.discover.processors import ProcessorRegistry
        from redi.discover.report import render_processing_plan

        discoveries = [ProcessorRegistry.analyze_file(csv_file)]
        out = tmp_path / "processing_plan.md"
        render_processing_plan(discoveries, out)
        text = out.read_text()

        assert "### Columns" in text
        assert "age" in text
        assert "city" in text

    def test_markdown_section_has_toc(self, tmp_path, markdown_file):
        from redi.discover.processors import ProcessorRegistry
        from redi.discover.report import render_processing_plan

        discoveries = [ProcessorRegistry.analyze_file(markdown_file)]
        out = tmp_path / "processing_plan.md"
        render_processing_plan(discoveries, out)
        text = out.read_text()

        assert "Table of Contents" in text
        assert "Project Overview" in text

    def test_empty_discoveries_produces_valid_file(self, tmp_path):
        from redi.discover.report import render_processing_plan

        out = tmp_path / "processing_plan.md"
        render_processing_plan([], out)
        assert out.exists()
        text = out.read_text()
        assert "# Processing Plan" in text
