Architecture
============

Core abstractions
-----------------

REDI's pipeline is built on three classes defined in :mod:`redi.pipeline`:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Role
   * - :class:`~redi.pipeline.PipelineContext`
     - Mutable dataclass that flows through every step, carrying data and
       all accumulated state.
   * - :class:`~redi.pipeline.PipelineStep`
     - Base class for every processing step.  Subclasses implement
       ``execute(ctx) → ctx``.
   * - :class:`~redi.pipeline.Pipeline`
     - Orchestrator: sequences steps, manages stage transitions, and runs
       optional verification.

PipelineContext
~~~~~~~~~~~~~~~

``PipelineContext`` (``ctx``) is the single object that travels through the
entire pipeline.  Key fields:

.. code-block:: python

   @dataclass
   class PipelineContext:
       # Paths
       input_path: Path          # primary input (first file)
       output_dir: Path
       input_paths: List[Path]   # all files in multi-file mode

       # Domain
       domain: str               # e.g. "climate", "hpc"

       # Data (populated by stages)
       data: Any                 # DataFrame, xarray.Dataset, np.ndarray, …
       metadata: Dict[str, Any]  # accumulated key→value facts about the data

       # Pipeline state
       current_stage: PipelineStage
       completed_stages: List[PipelineStage]
       errors: List[str]
       verification_results: List[VerificationResult]

       # Configuration
       config: Dict[str, Any]
       verify_steps: bool        # run step.verify() after each step?

       # Provenance
       with_flowcept: bool       # enable Flowcept lineage capture
       flowcept: Any             # active Flowcept handle (or None)

Steps record what they did via ``ctx.metadata["key"] = {...}`` and errors via
``ctx.add_error("message")``.  Steps must **not** raise exceptions that halt
the pipeline — errors are collected and reported at the end.

PipelineStep
~~~~~~~~~~~~

Every processing step is a subclass of ``PipelineStep``:

.. code-block:: python

   class MyStep(PipelineStep):
       name        = "my_step"
       stage       = PipelineStage.TRANSFORM
       description = "Does X to the data"

       def execute(self, ctx: PipelineContext) -> PipelineContext:
           # ... transform ctx.data ...
           ctx.metadata["my_step"] = {"rows_before": n_before, "rows_after": n_after}
           return ctx

       # Optional:
       def validate(self, ctx: PipelineContext) -> bool:
           return ctx.data is not None

       def verify(self, ctx: PipelineContext) -> VerificationResult:
           result = VerificationResult(passed=True, step_name=self.name)
           result.add_check("no_nulls", ctx.data.notna().all().all())
           return result

Required class attributes: ``name``, ``stage``, ``description``.
``validate()`` runs before the step; ``verify()`` runs after (only when
``--verify`` is passed on the CLI).

Flowcept auto-instrumentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PipelineStep`` uses ``__init_subclass__`` to automatically wrap every
subclass's ``execute()`` with a Flowcept task-capture decorator at
*class-definition time* — no changes needed in step implementations.  When
``ctx.with_flowcept`` is ``True`` and a Flowcept handle is active, each step
execution is recorded as a lineage task with input paths, output directory,
and pipeline metadata.

Data flow (run mode)
--------------------

::

    User Input (file path + optional YAML config)
      │
      ├─ expand_input_paths()     ← glob expansion, multi-file support
      ├─ detect_domain()          ← file extension → column keyword scoring
      ├─ generate_output_dir()    ← deterministic hash-based dir naming
      │
      ▼
    PipelineContext created
      │
      ├─ Domain module selected   ← get_domain_pipeline(domain)
      ├─ Steps assembled          ← domain.get_{stage}_steps() per stage
      │
      ▼
    Pipeline.execute(ctx)
      │
      ├─ For each step:
      │    1. step.validate(ctx)  ← pre-check (skip if returns False)
      │    2. step.execute(ctx)   ← transform data, update metadata
      │    3. step.verify(ctx)    ← post-check (only if --verify enabled)
      │    4. Flowcept capture    ← automatic provenance (if enabled)
      │
      ▼
    Output directory contains:
      ├─ train/val/test splits    (NPZ, HDF5, Parquet, …)
      ├─ data_card.json           ← 8-section metadata document
      ├─ metadata.json            ← pipeline run details
      └─ repro.sh                 ← full reproducibility script

Domain system
-------------

Each domain is a module under ``redi/domains/`` that exports five
step-factory functions:

.. code-block:: python

   def get_ingest_steps()     -> List[PipelineStep]: ...
   def get_preprocess_steps() -> List[PipelineStep]: ...
   def get_transform_steps()  -> List[PipelineStep]: ...
   def get_structure_steps()  -> List[PipelineStep]: ...
   def get_output_steps()     -> List[PipelineStep]: ...

Domain auto-detection in :func:`redi.domains.detect_domain` works in two
passes:

1. **File extension** — ``.nc`` → climate, ``.cif`` → materials,
   ``.fasta`` → life sciences, etc.
2. **Column keyword scoring** — counts keyword hits across column names;
   e.g. ``temperature``, ``lat``, ``lon`` all score for climate.

Adding a new domain requires one new file under ``redi/domains/`` plus
registering its keywords and extensions in ``redi/domains/__init__.py``.

Key design decisions
--------------------

Context-passing architecture
    A single ``PipelineContext`` is mutated and returned by each step,
    carrying both data and metadata through the entire pipeline.  This avoids
    global state and makes the data lineage explicit.

Domain as a plugin
    Each domain module exports step-factory functions.  The core pipeline
    has no domain-specific logic; it simply calls the factories and sequences
    the returned steps.

Lazy imports
    Domain-specific dependencies (xarray, pymatgen, biopython, …) are
    imported inside ``execute()`` methods, not at module level.  The core
    framework remains lightweight and importable without any scientific stack
    installed.

Array mode
    The ``--array`` flag skips DataFrame-centric steps for high-dimensional
    data (e.g. climate grids) and uses direct NumPy normalisation instead.
    Toggle with ``ctx.config.get("array_mode", False)``.

Reproducibility
    Every ``redi run`` invocation logs the full command to ``repro.sh`` in the
    output directory, including host, user, Python version, and working
    directory, via :mod:`redi.repro`.
