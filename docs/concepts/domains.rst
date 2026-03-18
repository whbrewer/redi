Scientific Domains
==================

REDI supports eight scientific domains, defined in :class:`redi.datacard.ScientificDomain`.
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
     - ``.fasta``, ``.pdb`` extensions; columns containing ``sequence``, ``protein``;
       OpenFold dataset layout (``fasta/``, ``pdb/``, ``alignments/``)
     - biopython → HDF5, NPZ (per-protein features or OpenFoldDataModule layout)
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
   * - Nuclear Physics
     - ``nuclear_physics``
     - ``train.txt`` / ``test.txt`` manifests with ``.npy`` wedge files
       (sPHENIX TPC outer detector layout)
     - numpy → NPZ (log2-normalized, NeuralCompression_v3-compatible)
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
* :mod:`redi.domains.openfold` — OpenFold specialization within ``life_sciences``
* :mod:`redi.domains.fusion`
* :mod:`redi.domains.hpc`
* :mod:`redi.domains.geospatial`
* :mod:`redi.domains.sphenix` — sPHENIX TPC processor for ``nuclear_physics``

The HPC domain additionally includes :mod:`redi.domains.hpc_job_mapper` for
constructing structured job objects from raw telemetry (``--raps-jobs`` flag).

OpenFold (Life Sciences specialization)
----------------------------------------

The ``life_sciences`` domain automatically activates the OpenFold pipeline when
it detects an OpenFold dataset layout — a directory containing ``fasta/``,
``pdb/``, and ``alignments/`` subdirectories.

Two output modes are supported via ``output_format``:

* **npz** (default) — Full IPTSO feature tensors saved as per-protein ``.npz``
  files in ``train/``, ``val/``, and ``test/`` subdirectories.  Use this to
  study the preprocessing pipeline or load features without an OpenFold
  installation.
* **openfold_native** — Mirrors the directory layout expected by
  ``OpenFoldDataModule``, enabling drop-in use with the upstream OpenFold
  training code.

Each NPZ file contains the full set of features computed by OpenFold's IPTSO
pipeline: ``aatype``, ``residue_index``, ``seq_length``,
``between_segment_residues``, ``msa``, ``deletion_matrix_int``,
``num_alignments``, ``all_atom_positions``, ``all_atom_mask``, ``resolution``,
and ``is_distillation``.

Nuclear Physics / sPHENIX
---------------------------

The ``nuclear_physics`` domain processes sPHENIX Time Projection Chamber (TPC)
outer detector data for use with the NeuralCompression_v3 framework.

**Input layout** (``/opt/drai/sphenix/outer/`` on the drai cluster)::

    outer/
        train.txt          # relative paths to 25,152 training wedge files
        test.txt           # relative paths to 6,288 test wedge files
        <wedge_dir>/       # e.g. 12-2_0-0/, 12-2_0-1/, ...
            *.npy           # uint16 arrays, shape (192, 249, 16)

Each wedge array has axes **azimuth × beam × layer** at the full TPC resolution.

**Transform applied:**

1. Zero-suppression: ADC < 64 → 0.0 (removes simulation noise floor)
2. ``log2(ADC + 1)`` normalization (compresses the [0, 1023] range to ≈ [0, 10])

**NeuralCompression_v3 compatibility** — the output is loaded directly with::

    DatasetTPC(data_root='./sphenix_ready', split='train', dimension=2)

Set ``log: false`` in the model config since REDI has already applied ``log2``.
