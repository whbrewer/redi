CLI Reference
=============

REDI exposes four subcommands: ``run``, ``analyze``, ``discover``, and ``validate``.
The ``run`` subcommand is the default — passing a file directly without a subcommand
is equivalent to ``redi run <file>``.

The reference below is generated automatically from the live argparse definition, so
it always reflects the current set of flags.

.. argparse::
   :module: redi.cli
   :func: create_parser
   :prog: redi
