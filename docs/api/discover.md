# redi.discover

Three-phase discovery engine for analyzing unknown datasets and proposing
pipelines. Used by the `redi discover` command.

Phases:

1. **Inspect** — enumerate files, detect formats, run per-file processors
2. **Infer** — determine modality, likely domain, label/split presence
3. **Propose** — recommend pipeline steps and output format

Outputs written to the output directory:

* `discovery_report.json` — full inspection/inference/proposal data
* `DISCOVERY_REPORT.md` — human-readable domain-level summary
* `processing_plan.md` — per-file discoveries and preprocessing suggestions
  (produced by the processor registry; see [`redi.discover.processors`](../../redi/discover/processors/__init__.py))

**Source:** [`redi/discover/__init__.py`](../../redi/discover/__init__.py)

## redi.discover.processors

Plug-in registry for per-file-type analysis. Each processor inspects a single
file and returns a `FileDiscovery` containing
file metadata and a list of `PreprocessingStep`
recommendations.

To add support for a new file type, subclass
`BaseProcessor`, set its `extension`
attribute, implement `analyze()`,
and register an instance with
`ProcessorRegistry.register()`.

**Source:** [`redi/discover/processors/__init__.py`](../../redi/discover/processors/__init__.py)

| Processor | Handles | Source |
|---|---|---|
| `csv_processor` | CSV/TSV — classifies each column as *numeric*, *categorical*, *boolean*, *datetime*, or *text*; computes per-column statistics; suggests normalization, encoding, and imputation steps | [`csv_processor.py`](../../redi/discover/processors/csv_processor.py) |
| `markdown_processor` | Markdown — extracts a heading table of contents and all referenced file paths, grouped by heading, list membership, and adjacency. No preprocessing steps suggested (reference material only) | [`markdown_processor.py`](../../redi/discover/processors/markdown_processor.py) |
| `mmcif_processor` | mmCIF structure files | [`mmcif_processor.py`](../../redi/discover/processors/mmcif_processor.py) |
| `fasta_processor` | FASTA sequence files | [`fasta_processor.py`](../../redi/discover/processors/fasta_processor.py) |
| `a3m_processor` | A3M multiple sequence alignments | [`a3m_processor.py`](../../redi/discover/processors/a3m_processor.py) |
| `netcdf_processor` | NetCDF climate/geospatial data | [`netcdf_processor.py`](../../redi/discover/processors/netcdf_processor.py) |
| `extxyz_processor` | Extended XYZ materials structure files | [`extxyz_processor.py`](../../redi/discover/processors/extxyz_processor.py) |
| `adios2_processor` | ADIOS2 simulation output | [`adios2_processor.py`](../../redi/discover/processors/adios2_processor.py) |
| `lmdb_processor` | LMDB key-value datasets | [`lmdb_processor.py`](../../redi/discover/processors/lmdb_processor.py) |

## redi.discover.report

Renders the `processing_plan.md` output file from a list of
`FileDiscovery` objects.

**Source:** [`redi/discover/report.py`](../../redi/discover/report.py)
