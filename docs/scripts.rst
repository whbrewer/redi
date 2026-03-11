Research Scripts
================

The ``scripts/`` directory contains utilities used for benchmarking, validation,
and SC26 reproducibility.  These are not part of the ``redi`` package but are
checked in alongside the codebase for reference.

Benchmarking
------------

Measure end-to-end pipeline throughput for each domain.  Each script processes
the corresponding dataset and reports timing in JSON format.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Domain / dataset
   * - ``bench_climate_timing.py``
     - Climate (CMIP6 / ERA5)
   * - ``bench_materials_timing.py``
     - Materials (AFLOW)
   * - ``bench_openfold_timing.py``
     - Life sciences (OpenFold)
   * - ``bench_sphenix_timing.py``
     - Nuclear physics (sPHENIX TPC)

Results are written to ``validate_results/bench_<domain>_timing.json``.

Validation
----------

Compare REDI output against ground-truth data or reference implementations.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Purpose
   * - ``validate_materials.py``
     - Check AFLOW output against reference structures
   * - ``validate_openfold.py``
     - Verify 11 OpenFold feature tensors match the upstream pipeline
   * - ``validate_sphenix.py``
     - Confirm sPHENIX output is loadable by ``DatasetTPC``
   * - ``validate_climate_2m_all.sh``
     - Validate ERA5 2 m-temperature runs end-to-end
   * - ``validate_climate_3d.sh``
     - Validate 3-D pressure-level climate runs
   * - ``run_hpc_validate.sh``
     - Run HPC domain smoke tests and validation

Results are written to ``validate_results/``.

AIDRIN Analysis
---------------

Run the AIDRIN AI-readiness scoring tool on REDI output for each domain.
Results are used to populate the paper tables.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Dataset
   * - ``run_aidrin_climate.py``
     - Climate (ERA5 / CMIP6)
   * - ``run_aidrin_materials.py``
     - Materials (AFLOW)
   * - ``run_aidrin_openfold.py``
     - Life sciences (OpenFold)
   * - ``run_aidrin_sphenix.py``
     - Nuclear physics (sPHENIX)
   * - ``run_aidrin_adastra.py``
     - HPC (Adastra MI250 dataset)

Scored results are written to ``validate_results/aidrin_<domain>.json``.

Table Generation
----------------

``gen_paper_tables.py``
   Reads all ``validate_results/*.json`` files and renders the paper's
   LaTeX tables (``table_v_validate_aidrin.tex``, ``table_vi_timing.tex``)
   and a plain-text summary (``paper_tables.txt``).

Data Acquisition
----------------

``download_cmip6_subset.py``
   Downloads a small CMIP6 subset from ESGF for testing.  Requires
   ESGF credentials.  For full download options see :doc:`cli`.
