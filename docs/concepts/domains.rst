Scientific Domains
==================

REDI supports seven scientific domains, defined in :class:`redi.datacard.ScientificDomain`.
The correct domain can be specified with ``--domain`` or left to auto-detection.

.. list-table::
   :header-rows: 1
   :widths: 18 20 30 32

   * - Domain
     - Enum value
     - Detected by
     - Key dependencies / output formats
   * - Climate
     - ``climate``
     - ``.nc``, ``.grib`` extensions; columns containing ``temperature``, ``lat``, ``lon``
     - xarray, xesmf → NPZ, Zarr
   * - Materials
     - ``materials``
     - ``.cif``, ``.poscar`` extensions; columns containing ``formula``, ``lattice``
     - pymatgen, matminer → HDF5, NPZ
   * - Life Sciences
     - ``life_sciences``
     - ``.fasta``, ``.pdb`` extensions; columns containing ``sequence``, ``protein``
     - biopython → HDF5
   * - Fusion
     - ``fusion``
     - Columns containing ``plasma``, ``disruption``, ``tokamak``
     - h5py → HDF5, NPZ
   * - HPC
     - ``hpc``
     - Columns containing ``job``, ``node``, ``walltime``, ``queue``
     - pandas, scikit-learn → NPZ, Parquet
   * - Geospatial
     - ``geospatial``
     - ``.shp``, ``.geojson`` extensions; columns containing ``h3``, ``hex``, ``pm25``, ``smoke``
     - h3, geopandas → Parquet, NPZ
   * - General
     - ``general``
     - Fallback when no domain-specific signals are detected
     - pandas, numpy → NPZ, CSV

Domain auto-detection
---------------------

When ``--domain`` is not supplied, REDI inspects:

1. The file extension (e.g. ``.nc`` → climate).
2. Column names from tabular data (CSV, TSV, Parquet, JSON) — a small sample is read
   to extract header names without loading the full file.

The detection logic lives in :func:`redi.cli.detect_domain` (CLI layer) and
:func:`redi.domains.detect_domain` (domain layer).

Domain-specific processors
---------------------------

Each domain has a dedicated processor module under ``redi/domains/``:

* :mod:`redi.domains.climate`
* :mod:`redi.domains.materials`
* :mod:`redi.domains.life_sciences`
* :mod:`redi.domains.fusion`
* :mod:`redi.domains.hpc`
* :mod:`redi.domains.geospatial`

The HPC domain additionally includes :mod:`redi.domains.hpc_job_mapper` for
constructing structured job objects from raw telemetry (``--raps-jobs`` flag).
