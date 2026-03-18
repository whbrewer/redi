redi.discover
=============

Three-phase discovery engine for analyzing unknown datasets and proposing
pipelines. Used by the ``redi discover`` command.

Phases:

1. **Inspect** — enumerate files, detect formats, run per-file processors
2. **Infer** — determine modality, likely domain, label/split presence
3. **Propose** — recommend pipeline steps and output format

Outputs written to the output directory:

* ``discovery_report.json`` — full inspection/inference/proposal data
* ``DISCOVERY_REPORT.md`` — human-readable domain-level summary
* ``processing_plan.md`` — per-file discoveries and preprocessing suggestions
  (produced by the processor registry; see :mod:`redi.discover.processors`)

.. automodule:: redi.discover
   :members:
   :undoc-members:
   :show-inheritance:

----

redi.discover.processors
------------------------

Plug-in registry for per-file-type analysis. Each processor inspects a single
file and returns a :class:`~redi.discover.processors.FileDiscovery` containing
file metadata and a list of :class:`~redi.discover.processors.PreprocessingStep`
recommendations.

To add support for a new file type, subclass
:class:`~redi.discover.processors.BaseProcessor`, set its ``extension``
attribute, implement :meth:`~redi.discover.processors.BaseProcessor.analyze`,
and register an instance with
:meth:`~redi.discover.processors.ProcessorRegistry.register`.

.. automodule:: redi.discover.processors
   :members:
   :undoc-members:
   :show-inheritance:

redi.discover.processors.csv\_processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes CSV and TSV files: classifies each column as *numeric*,
*categorical*, *boolean*, *datetime*, or *text*; computes per-column
statistics; and suggests normalization, encoding, and imputation steps.

.. automodule:: redi.discover.processors.csv_processor
   :members:
   :undoc-members:
   :show-inheritance:

redi.discover.processors.markdown\_processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes Markdown files: extracts a heading table of contents and all
referenced file paths, then groups those references by heading, list
membership, and adjacency. No preprocessing steps are suggested — Markdown
files are treated as reference material only.

.. automodule:: redi.discover.processors.markdown_processor
   :members:
   :undoc-members:
   :show-inheritance:

----

redi.discover.report
--------------------

Renders the ``processing_plan.md`` output file from a list of
:class:`~redi.discover.processors.FileDiscovery` objects.

.. automodule:: redi.discover.report
   :members:
   :undoc-members:
   :show-inheritance:
