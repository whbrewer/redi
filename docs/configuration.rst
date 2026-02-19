Configuration Reference
=======================

REDI can be configured via a YAML file passed with ``-c``/``--config``.
CLI flags always override the corresponding YAML keys.

.. code-block:: bash

   redi run data.nc -c my_pipeline.yaml
   redi run data.nc -c my_pipeline.yaml --domain hpc   # CLI overrides YAML

Top-level keys
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``domain``
     - ``null``
     - Scientific domain. One of ``climate``, ``materials``,
       ``life_sciences``, ``fusion``, ``hpc``, ``geospatial``,
       ``general``. Auto-detected from file extension + column names if
       omitted.
   * - ``input_path``
     - ``null``
     - Input file path. Can also be provided via ``input_paths`` (list)
       for multi-file runs.  Overridden by positional CLI argument.
   * - ``output_dir``
     - ``./output``
     - Base output directory.  When no explicit ``-o`` flag is given, a
       hash-based subdirectory is created here.
   * - ``output_format``
     - ``npz``
     - Output format.  One of ``npz``, ``parquet``, ``hdf5``, ``csv``,
       ``zarr``, ``pt``.
   * - ``stages``
     - ``null``
     - List of stages to run, e.g. ``[ingest, preprocess]``.  ``null``
       runs all five stages.
   * - ``verbose``
     - ``false``
     - Enable verbose logging.
   * - ``start``
     - ``null``
     - Earliest job datetime to include (HPC domain). ISO format or
       ``YYYYMMDD``.  Overridden by ``--start`` CLI flag.
   * - ``end``
     - ``null``
     - Exclusive upper bound for job datetime filter (HPC domain).
       Overridden by ``--end`` CLI flag.
   * - ``provenance``
     - ``false``
     - Enable Flowcept provenance tracking.

variables
---------

Applies to the **climate** domain.

.. code-block:: yaml

   variables:
     select: null             # null = all; or a list: [tas, pr, psl]
     pressure_levels: null    # null = all; or a list: [500, 700, 850]

regrid
------

Applies to the **climate** domain.

.. code-block:: yaml

   regrid:
     enabled: true
     target_resolution: 1.0  # degrees; must be > 0 and ≤ 10
     method: bilinear         # bilinear | conservative | nearest_s2d | patch
     lat_range: [-90, 90]
     lon_range: [0, 360]

normalize
---------

.. code-block:: yaml

   normalize:
     enabled: true
     method: zscore           # zscore | minmax | none
     two_pass: false          # compute stats across all data before normalising
     per_variable: true       # normalise each variable independently
     per_level: false         # normalise each pressure level independently
     exclude_variables: []    # list of variable names to skip

split
-----

.. code-block:: yaml

   # Random split (default)
   split:
     method: random
     ratios:
       train: 0.8
       val:   0.1
       test:  0.1
     shuffle: true
     random_seed: 42

   # Temporal split
   split:
     method: temporal
     train:
       start: "1979-01-01"
       end:   "2015-12-31"
     val:
       start: "2016-01-01"
       end:   "2017-12-31"
     test:
       start: "2018-01-01"
       end:   "2020-12-31"

.. note::

   For ``method: random``, ratios must sum to 1.0.
   For ``method: temporal``, ``start`` and ``end`` are required for every
   split name.

shard
-----

.. code-block:: yaml

   shard:
     enabled: false
     by: year                  # sharding dimension
     samples_per_shard: 10000  # maximum samples per shard file

climax_output
-------------

Produces output compatible with the `ClimaX <https://microsoft.github.io/ClimaX/>`_
foundation model.  Climate domain only.

.. code-block:: yaml

   climax_output:
     enabled: false
     structure: yearly_shards
     variable_naming: name_level     # e.g. temperature_500
     save_coordinates: true          # writes lat.npy, lon.npy
     save_normalization: true        # writes normalize_mean.npz, normalize_std.npz

raps_jobs
---------

Controls RAPS job extraction for the **HPC** domain (``--raps-jobs``).

.. code-block:: yaml

   raps_jobs:
     enabled: false

download
--------

Trigger remote data acquisition before processing.  Equivalent to the
``--download`` CLI flag group.

.. code-block:: yaml

   download:
     enabled: true
     source: esgf            # esgf | cds
     variable: tas
     model: TaiESM1
     experiment: historical
     frequency: 6hr
     year: null              # used by CDS/ERA5 only

Full example
------------

.. code-block:: yaml

   # REDI configuration — climate pipeline

   domain: climate
   output_dir: ./output
   output_format: npz

   variables:
     select: [tas, pr]
     pressure_levels: [500, 850]

   regrid:
     enabled: true
     target_resolution: 1.0
     method: bilinear

   normalize:
     enabled: true
     method: zscore
     per_variable: true

   split:
     method: temporal
     train:
       start: "1979-01-01"
       end:   "2015-12-31"
     val:
       start: "2016-01-01"
       end:   "2017-12-31"
     test:
       start: "2018-01-01"
       end:   "2020-12-31"

   climax_output:
     enabled: true
     save_coordinates: true
     save_normalization: true
