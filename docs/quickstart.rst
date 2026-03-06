Quick Start
===========

Installation
------------

REDI requires Python 3.9 or later. Install the core package::

    pip install -e .

Install with optional domain-specific dependencies::

    # All scientific domains
    pip install -e ".[all-domains]"

    # Individual domains
    pip install -e ".[climate]"
    pip install -e ".[materials]"
    pip install -e ".[life_sciences]"
    pip install -e ".[fusion]"
    pip install -e ".[hpc]"
    pip install -e ".[geospatial]"

    # Output format support
    pip install -e ".[parquet]"
    pip install -e ".[hdf5]"

    # Documentation
    pip install -e ".[docs]"

Activating the bundled environment
-----------------------------------

If you are using the pre-built ``drai`` virtual environment::

    source /opt/venvs/drai/bin/activate

The environment automatically sets ``ESMFMKFILE`` for climate domain regridding.

Running the pipeline
--------------------

The most common usage is ``redi run``. The ``run`` subcommand is the default, so
these two forms are equivalent::

    redi run data.csv -o ./output
    redi data.csv -o ./output

**Auto-detect domain and run the full pipeline**::

    redi run data.csv

**Specify the scientific domain**::

    redi run data.csv --domain hpc

**Custom output directory**::

    redi run data.csv -o ./processed

**Run specific pipeline stages only**::

    redi run data.csv --stages ingest,preprocess

**Process multiple files (glob pattern)**::

    redi run "data_*.nc"

**Optimize for array data**::

    redi run data.npy --array --normalize minmax

Downloading climate data
------------------------

REDI can download data from remote sources before processing.

CMIP6 via ESGF::

    redi run --download --source esgf \
        --variable tas --model TaiESM1 --experiment historical \
        -o ./cmip6_output

ERA5 via Copernicus CDS::

    redi run --download --source cds \
        --variable 2m_temperature --year 2020 \
        -o ./era5_output

.. note::

   CDS access requires an API key. Register at https://cds.climate.copernicus.eu/
   and configure ``~/.cdsapirc``.

Observing an existing workflow
-------------------------------

::

    redi inspect --workflow Snakefile
    redi inspect --workflow pipeline.nf --type nextflow
    redi inspect --workflow Snakefile --watch   # watch for changes

Analyzing an unknown dataset
------------------------------

::

    redi discover /path/to/data/
    redi discover dataset.csv -o ./analysis
    redi discover /data --no-interactive

Validating REDI output
-----------------------

::

    redi validate output/myrun_a3f2b1c -c config.yaml
    redi validate output/myrun_a3f2b1c --ground-truth /path/to/truth.pt

Using a YAML configuration file
---------------------------------

All options can be supplied via a YAML config file::

    redi run -c my_pipeline.yaml

CLI flags override corresponding YAML keys. For the full list of supported
configuration keys see :doc:`cli`.
