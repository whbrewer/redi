"""
Pytest configuration and shared fixtures for REDI tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp(prefix="redi_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "sample.csv"
    csv_path.write_text("""id,temperature,pressure,velocity,timestamp,category
1,25.3,101.2,15.7,2024-01-01 10:00:00,A
2,26.1,100.8,16.2,2024-01-01 10:05:00,B
3,,99.5,14.8,2024-01-01 10:10:00,A
4,24.9,101.5,,2024-01-01 10:15:00,C
5,27.2,100.1,17.1,2024-01-01 10:20:00,B
6,25.8,99.8,15.9,,A
7,26.5,101.0,16.5,2024-01-01 10:30:00,
8,25.0,100.5,15.3,2024-01-01 10:35:00,C
9,999.9,100.2,16.0,2024-01-01 10:40:00,A
10,26.3,-999,15.5,2024-01-01 10:45:00,B
""")
    return csv_path


@pytest.fixture
def sample_hpc_csv(temp_dir):
    """Create a sample HPC job accounting CSV."""
    csv_path = temp_dir / "hpc_jobs.csv"
    csv_path.write_text("""JOB_ID,USER,PROJECT,QUEUE,NODES_REQUESTED,WALLTIME_SECONDS,RUNTIME_SECONDS,EXIT_CODE,QUEUED_TIMESTAMP,START_TIMESTAMP,END_TIMESTAMP
1001,user_a,proj_1,batch,4,3600,3200,0,2024-01-01 08:00:00,2024-01-01 08:30:00,2024-01-01 09:23:20
1002,user_b,proj_2,debug,1,1800,1500,0,2024-01-01 09:00:00,2024-01-01 09:05:00,2024-01-01 09:30:00
1003,user_a,proj_1,batch,8,7200,7100,0,2024-01-01 10:00:00,2024-01-01 11:00:00,2024-01-01 12:58:20
1004,user_c,proj_3,batch,16,14400,14000,1,2024-01-01 12:00:00,2024-01-01 14:00:00,2024-01-01 17:53:20
1005,user_b,proj_2,batch,2,3600,3400,0,2024-01-01 13:00:00,2024-01-01 13:15:00,2024-01-01 14:11:40
""")
    return csv_path


@pytest.fixture
def sample_materials_csv(temp_dir):
    """Create a sample materials science CSV (AFLOW-style)."""
    csv_path = temp_dir / "materials.csv"
    csv_path.write_text("""compound,formula,spacegroup,band_gap,formation_energy,density,lattice_a,lattice_b,lattice_c
Silicon,Si,Fd-3m,1.12,-0.0,2.33,5.43,5.43,5.43
Germanium,Ge,Fd-3m,0.67,-0.0,5.32,5.66,5.66,5.66
GaAs,GaAs,F-43m,1.42,-0.74,5.32,5.65,5.65,5.65
AlN,AlN,P63mc,6.2,-3.13,3.26,3.11,3.11,4.98
ZnO,ZnO,P63mc,3.37,-3.63,5.61,3.25,3.25,5.21
TiO2,TiO2,P42/mnm,3.0,-9.73,4.23,4.59,4.59,2.96
""")
    return csv_path


@pytest.fixture
def sample_bio_fasta(temp_dir):
    """Create a sample FASTA file for life sciences testing."""
    fasta_path = temp_dir / "sample.fasta"
    fasta_path.write_text(""">protein_1 Sample protein sequence
MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT
LSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQA
>protein_2 Another sample protein
MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPT
NAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALR
>protein_3 Short protein
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK
""")
    return fasta_path


@pytest.fixture
def sample_json(temp_dir):
    """Create a sample JSON file."""
    json_path = temp_dir / "sample.json"
    json_path.write_text("""[
    {"id": 1, "name": "item_1", "value": 10.5, "category": "A"},
    {"id": 2, "name": "item_2", "value": 20.3, "category": "B"},
    {"id": 3, "name": "item_3", "value": 15.7, "category": "A"}
]""")
    return json_path
