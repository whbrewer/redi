"""Auto-detection only names a domain when the format identifies it unambiguously."""

from pathlib import Path

import pytest

from redi.cli import detect_domain


@pytest.mark.parametrize("name,expected", [
    ("a.nc", "climate"), ("a.grib", "climate"),
    ("a.cif", "materials"), ("a.poscar", "materials"),
    ("a.fasta", "life_sciences"), ("a.vcf", "life_sciences"),
    ("a.shp", "geospatial"), ("a.geojson", "geospatial"),
])
def test_unambiguous_extensions_are_detected(name, expected):
    assert detect_domain(Path(name)) == expected


@pytest.mark.parametrize("name", [
    "a.h5", "a.H5", "a.hdf5", "a.adios", "a.bp", "a.pdb",
    "a.csv", "a.tsv", "a.parquet", "a.json", "a.npz", "a",
])
def test_ambiguous_or_generic_inputs_are_general(name):
    assert detect_domain(Path(name)) == "general"


def test_column_names_no_longer_decide(tmp_path):
    f = tmp_path / "jobs.csv"
    f.write_text("job_id,node,walltime,queue\n1,n1,10,debug\n")
    assert detect_domain(f) == "general"
    assert detect_domain(f, columns=["job", "node", "walltime", "queue"]) == "general"


def test_amrex_plotfile_directory_is_combustion(tmp_path):
    (tmp_path / "Header").write_text("HyperCLaw-V1.1\n")
    assert detect_domain(tmp_path) == "combustion"


def test_plain_directory_is_general(tmp_path):
    assert detect_domain(tmp_path) == "general"
