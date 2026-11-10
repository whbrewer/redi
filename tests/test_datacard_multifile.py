"""The enriched data card describes all input files of a multi-file run, not just the first."""

from pathlib import Path
from types import SimpleNamespace

from redi.cli import _build_enriched_datacard


def _ctx(paths, **metadata):
    return SimpleNamespace(
        input_paths=[Path(p) for p in paths], metadata=metadata, completed_stages=[],
    )


def _csv(path, rows):
    path.write_text("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(rows)))
    return path


def test_multi_file_card_covers_every_file(tmp_path):
    files = [_csv(tmp_path / f"d{k}.csv", 5) for k in range(3)]
    card = _build_enriched_datacard(
        files[0], "general", {}, _ctx(files, n_structures_raw=15), [],
    )
    ov = card.overview
    assert ov.name == f"{tmp_path.name} (3 files)"
    assert ov.size_bytes == sum(f.stat().st_size for f in files)
    assert "3 files" in ov.description and "d0.csv" in ov.description
    assert ov.num_records == 15


def test_directory_input_size_is_sum_of_files(tmp_path):
    files = [_csv(tmp_path / f"d{k}.csv", 5) for k in range(2)]
    card = _build_enriched_datacard(tmp_path, "general", {}, _ctx([tmp_path]), [])
    assert card.overview.size_bytes == sum(f.stat().st_size for f in files)


def test_single_file_unchanged(tmp_path):
    f = _csv(tmp_path / "one.csv", 5)
    card = _build_enriched_datacard(f, "general", {}, _ctx([f]), [])
    assert card.overview.name == "one.csv"
    assert card.overview.size_bytes == f.stat().st_size
