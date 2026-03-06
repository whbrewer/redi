"""Tests for provenance card generation."""

import json


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_generate_provenance_card_omits_unknown_workflow_name(temp_dir, monkeypatch):
    import redi.provenance_card as provenance_card

    input_jsonl = temp_dir / "flowcept_buffer.jsonl"
    output_md = temp_dir / "PROVENANCE_CARD.md"

    records = [
        {
            "type": "workflow",
            "workflow_id": "wf-123",
            "name": None,
            "campaign_id": "camp-1",
        },
        {
            "type": "task",
            "workflow_id": "wf-123",
            "activity_id": "LoadCSV",
            "status": "COMPLETED",
            "started_at": 1700000000.0,
            "ended_at": 1700000001.0,
            "used": {"input_path": "/tmp/in.csv", "output_dir": "/tmp/out"},
            "custom_metadata": {"current_stage": "ingest", "metadata": {}},
        },
    ]
    _write_jsonl(input_jsonl, records)

    monkeypatch.setattr(provenance_card, "DUMP_BUFFER_PATH", str(input_jsonl))
    provenance_card.generate_provenance_artifacts(output_dir=temp_dir)
    content = output_md.read_text(encoding="utf-8")

    assert "# Workflow Provenance Card:" in content
    assert "- **Workflow Name:** `unknown`" not in content


def test_generate_provenance_card_keeps_workflow_name_when_present(temp_dir, monkeypatch):
    import redi.provenance_card as provenance_card

    input_jsonl = temp_dir / "flowcept_buffer.jsonl"
    output_md = temp_dir / "PROVENANCE_CARD.md"

    records = [
        {
            "type": "workflow",
            "workflow_id": "wf-234",
            "name": "redi-run",
            "campaign_id": "camp-2",
        },
        {
            "type": "task",
            "workflow_id": "wf-234",
            "activity_id": "LoadCSV",
            "status": "COMPLETED",
            "started_at": 1700000000.0,
            "ended_at": 1700000001.0,
            "used": {"input_path": "/tmp/in.csv", "output_dir": "/tmp/out"},
            "custom_metadata": {"current_stage": "ingest", "metadata": {}},
        },
    ]
    _write_jsonl(input_jsonl, records)

    monkeypatch.setattr(provenance_card, "DUMP_BUFFER_PATH", str(input_jsonl))
    provenance_card.generate_provenance_artifacts(output_dir=temp_dir)
    content = output_md.read_text(encoding="utf-8")

    assert "# Workflow Provenance Card: redi-run" in content
    assert "- **Workflow Name:** `redi-run`" in content

def test_link_datacard_to_provenance_inserts_under_section(temp_dir):
    from redi.provenance_card import link_datacard_to_provenance

    datacard = temp_dir / "DATA_CARD.md"
    datacard.write_text(
        "# Data Card\n\n## 1. Summary\ntext\n\n## 2. Provenance\n\n## 3. Notes\n",
        encoding="utf-8",
    )

    changed = link_datacard_to_provenance(datacard)
    content = datacard.read_text(encoding="utf-8")

    assert changed is True
    assert "## 2. Provenance\n\n- [Data Transformation Pipeline Provenance Card](PROVENANCE_CARD.md)" in content
    assert "## Provenance" not in content
