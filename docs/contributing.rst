Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/wesbrewer/redi
   cd redi
   pip install -e ".[dev,docs]"
   pytest tests/ -v

Coding conventions
------------------

Imports
~~~~~~~

Group imports: stdlib → third-party → local (relative).  Domain-specific
heavy dependencies (``xarray``, ``pymatgen``, ``torch``, etc.) must be
imported **lazily inside** ``execute()`` methods, wrapped in
``try/except ImportError``.  Domain factory functions import from
``..pipeline`` inside the function body to avoid circular imports.

.. code-block:: python

   def execute(self, ctx):
       try:
           import xarray as xr
       except ImportError:
           ctx.add_error(missing_dependency_error("xarray", extra="climate"))
           return ctx
       ...

Error handling
~~~~~~~~~~~~~~

Steps must **not** raise exceptions.  Catch all errors and record them with
``ctx.add_error()``.  For missing optional dependencies use the helper:

.. code-block:: python

   from redi.pipeline import missing_dependency_error

   ctx.add_error(missing_dependency_error("pymatgen", extra="materials"))

Metadata tracking
~~~~~~~~~~~~~~~~~

Each step should record what it did:

.. code-block:: python

   ctx.metadata["my_step"] = {
       "rows_before": n_before,
       "rows_after": n_after,
       "parameters": {"method": self.config.get("method", "default")},
   }

Use snake_case keys.  Include before/after values and the parameters used.

Config access
~~~~~~~~~~~~~

Read config through the context to stay in sync with YAML overrides:

.. code-block:: python

   method = ctx.config.get("normalize", {}).get("method", "zscore")

Naming
~~~~~~

- Classes: PascalCase, verb-noun — ``LoadHPCData``, ``NormalizeNumeric``
- Functions/methods: ``snake_case``
- Private helpers: leading underscore ``_helper()``
- Step factory functions: ``get_<stage>_steps()`` returning a list of
  step instances

Type hints
~~~~~~~~~~

Use type hints on all function parameters and return types.  ``Optional[T]``
for nullable fields; ``Any`` for flexible data containers (``ctx.data``).

Docstrings
~~~~~~~~~~

Google-style.  One-liner for simple functions; ``Args``/``Returns`` sections
for anything non-trivial.

Logging
~~~~~~~

.. code-block:: python

   logger = logging.getLogger(__name__)   # module-level

   logger.info("High-level progress message")
   logger.debug("Detailed debug information")

Writing a new pipeline step
----------------------------

.. code-block:: python

   from redi.pipeline import PipelineStep, PipelineStage, PipelineContext

   class MyStep(PipelineStep):
       name        = "my_step"
       stage       = PipelineStage.TRANSFORM
       description = "Short description of what this step does"

       def validate(self, ctx: PipelineContext) -> bool:
           """Return False to skip this step."""
           return ctx.data is not None

       def execute(self, ctx: PipelineContext) -> PipelineContext:
           try:
               import some_optional_dep
           except ImportError:
               ctx.add_error(missing_dependency_error("some_optional_dep"))
               return ctx

           n_before = len(ctx.data)
           # ... transform ctx.data ...
           ctx.metadata["my_step"] = {"n_before": n_before, "n_after": len(ctx.data)}
           return ctx

       def verify(self, ctx: PipelineContext) -> VerificationResult:
           result = VerificationResult(passed=True, step_name=self.name)
           result.add_check("data_not_empty", len(ctx.data) > 0)
           return result

Writing a new domain
--------------------

1. Create ``redi/domains/<name>.py`` exporting all five factory functions:

   .. code-block:: python

      def get_ingest_steps()     -> list: ...
      def get_preprocess_steps() -> list: ...
      def get_transform_steps()  -> list: ...
      def get_structure_steps()  -> list: ...
      def get_output_steps()     -> list: ...

2. Register file extensions and column keywords in
   ``redi/domains/__init__.py``.

3. Add the domain value to :class:`redi.datacard.ScientificDomain`.

4. Add a smoke test in ``tests/test_smoke_<name>.py``.

Testing
-------

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Domain-specific smoke tests
   pytest tests/test_smoke_hpc.py -v
   pytest tests/test_smoke_climate.py -v

Test conventions:

- Use ``@pytest.fixture`` for setup; prefer ``tmp_path`` for temp
  directories.
- Group related tests in classes (``class TestClimateSmoke``).
- Use ``pytest.importorskip("xarray")`` to skip tests when optional deps
  are absent.
- Integration tests that require real data files should call
  ``pytest.skip("data file not found")`` when the file is missing.
