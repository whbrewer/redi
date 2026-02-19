Glossary
========

.. glossary::
   :sorted:

   PipelineContext
      The central mutable dataclass (``ctx``) that flows through every
      pipeline step.  It carries ``ctx.data`` (the actual data object),
      ``ctx.metadata`` (accumulated facts), input/output paths, domain,
      config, error list, and pipeline-state bookkeeping.
      See :class:`redi.pipeline.PipelineContext`.

   PipelineStep
      Base class for every processing step.  Subclasses set class
      attributes ``name``, ``stage``, and ``description``, and implement
      ``execute(ctx) → ctx``.  Optional ``validate(ctx)`` and
      ``verify(ctx)`` hooks provide pre- and post-checks.
      See :class:`redi.pipeline.PipelineStep`.

   PipelineStage
      Enum of the five sequential stages: ``INGEST``, ``PREPROCESS``,
      ``TRANSFORM``, ``STRUCTURE``, ``OUTPUT``.
      See :class:`redi.pipeline.PipelineStage`.

   Domain
      One of the seven supported scientific areas: ``climate``,
      ``materials``, ``life_sciences``, ``fusion``, ``hpc``,
      ``geospatial``, ``general``.  Determines which specialised
      ``PipelineStep`` implementations are loaded.
      See :class:`redi.datacard.ScientificDomain`.

   MaturityLevel
      Five-level data readiness taxonomy: ``RAW`` (1), ``CLEANED`` (2),
      ``LABELED`` (3), ``FEATURE_ENGINEERED`` (4), ``FULLY_READY`` (5).
      See :class:`redi.datacard.MaturityLevel` and
      :doc:`concepts/readiness-levels`.

   Data Card
      Structured metadata document with eight sections (overview,
      provenance, characteristics, sensitive attributes, transformations,
      annotations, limitations, readiness) written to ``data_card.json``
      in the output directory after each pipeline run.
      See :class:`redi.datacard.DataCard`.

   Array mode
      Pipeline mode enabled with ``--array`` that skips DataFrame-centric
      steps and applies direct NumPy normalisation.  Intended for
      high-dimensional data such as climate grids or simulation outputs.

   Flowcept
      Optional provenance-tracking library.  When enabled (``provenance:
      true`` in config), every ``PipelineStep`` execution is recorded as
      a lineage task via automatic ``__init_subclass__`` instrumentation —
      no changes to step implementations required.

   RAPS Job
      Standardised HPC job representation used by the ExaDigiT framework.
      The :mod:`redi.domains.hpc_job_mapper` module auto-maps arbitrary
      HPC telemetry columns to this format using a four-phase fuzzy column
      matching strategy.

   HPC Job Mapper
      Standalone subsystem in :mod:`redi.domains.hpc_job_mapper` for
      extracting structured RAPS Job objects from raw HPC telemetry.
      Uses exact match → token overlap → fuzzy matching with confidence
      scores, then derives missing fields and normalises timestamps.
      Activated with the ``--raps-jobs`` CLI flag.

   ClimaX output
      Output format compatible with the ClimaX climate foundation model.
      Enabled via ``climax_output.enabled: true`` in the YAML config.
      Produces yearly shards, coordinate arrays (``lat.npy``,
      ``lon.npy``), and normalisation statistics.

   ESGF
      Earth System Grid Federation — federated data infrastructure for
      CMIP6 climate model data.  REDI can download from ESGF via
      ``--download --source esgf``.  See :mod:`redi.downloaders.esgf`.

   CDS
      Copernicus Climate Data Store — source of ERA5 reanalysis data.
      REDI can download from CDS via ``--download --source cds``.
      Requires an API key in ``~/.cdsapirc``.
      See :mod:`redi.downloaders.cds`.

   repro.sh
      Shell script written to every output directory by :mod:`redi.repro`.
      Records the exact ``redi`` invocation with full environment metadata
      (host, user, Python version, working directory) for reproducibility.
