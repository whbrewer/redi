"""Tests for redi.aidrin_manifest and `redi run --aidrin-report`.

The consumer must never overstate what REDI does: every plan item says whether
REDI acts on a finding (advice) or cannot (unsupported), and `applied` stays
empty until a pipeline setting actually reads a finding.
"""

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from redi.aidrin_manifest import (
    ADVICE,
    APPLIED,
    UNSUPPORTED,
    WARNING,
    AidrinReportError,
    build_plan,
    check_input,
    load_report,
    provenance_block,
)

REPO = Path(__file__).resolve().parent.parent
RGB1_META = Path.home() / "drai/aidrin-reports/RGB1-images/manifest/aidrin_assessment_meta.json"


def finding(name, value, scope="dataset", metric="m", **extra):
    return {"metric": metric, "scope": scope, "name": name, "value": value,
            "direction": "informational", **extra}


def make_manifest(data_path="/data/table.csv", findings=None, roles=None, **over):
    m = {
        "schema_version": "aidrin.assessment/v1",
        "tool": {"name": "aidrin", "version": "2026.08.2"},
        "created_at": "2026-10-14T12:00:00Z",
        "dataset": {"path": str(data_path), "format": "csv", "rows": 3, "columns": 2, "schema": []},
        "redaction": "redacted",
        "roles": {"target": None, "sensitive": [], "quasi_identifiers": [], "id": None,
                  "batch": None, "confirmed_by_user": False, **(roles or {})},
        "metrics": [{"metric": "completeness", "status": "ok", "args": {}, "result": {}, "error": None}],
        "findings": findings or [],
    }
    m.update(over)
    return m


def write(tmp_path, manifest, name="m.json"):
    p = tmp_path / name
    p.write_text(json.dumps(manifest))
    return p


def report_for(tmp_path, **kw):
    return load_report(write(tmp_path, make_manifest(**kw)))


def kinds(plan):
    return {i.topic: i.kind for i in plan}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_valid_manifest_loads(self, tmp_path):
        r = report_for(tmp_path, findings=[finding("aidrin.quality.completeness", 0.9)])
        assert r.value("aidrin.quality.completeness") == 0.9
        assert r.value("aidrin.quality.duplicity") is None
        assert len(r.sha256) == 64

    def test_sha256_tracks_file_bytes(self, tmp_path):
        a = write(tmp_path, make_manifest(), "a.json")
        b = write(tmp_path, make_manifest(intended_use="other"), "b.json")
        assert load_report(a).sha256 != load_report(b).sha256

    def test_column_findings_keyed_by_column(self, tmp_path):
        r = report_for(tmp_path, findings=[
            finding("aidrin.column.completeness", 0.5, scope="column", column="a"),
            finding("aidrin.column.completeness", 1.0, scope="column", column="b"),
            finding("aidrin.column.outliers", 0.1, scope="column", column="a"),
        ])
        assert r.columns("aidrin.column.completeness") == {"a": 0.5, "b": 1.0}

    def test_missing_file(self, tmp_path):
        with pytest.raises(AidrinReportError, match="cannot read"):
            load_report(tmp_path / "nope.json")

    def test_not_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(AidrinReportError, match="not valid JSON"):
            load_report(p)

    def test_unknown_version_names_both_versions(self, tmp_path):
        p = write(tmp_path, make_manifest(schema_version="aidrin.assessment/v2"))
        with pytest.raises(AidrinReportError, match="v2.*aidrin.assessment/v1"):
            load_report(p)

    @pytest.mark.parametrize("key", ["dataset", "roles", "metrics"])
    def test_missing_required_section(self, tmp_path, key):
        m = make_manifest()
        del m[key]
        with pytest.raises(AidrinReportError, match="malformed"):
            load_report(write(tmp_path, m))

    def test_non_numeric_finding_rejected(self, tmp_path):
        m = make_manifest(findings=[finding("aidrin.quality.completeness", "high")])
        with pytest.raises(AidrinReportError, match="numeric"):
            load_report(write(tmp_path, m))

    def test_unknown_extra_fields_ignored(self, tmp_path):
        m = make_manifest(future_field={"x": 1})
        assert load_report(write(tmp_path, m)).data["future_field"] == {"x": 1}


# ---------------------------------------------------------------------------
# does the manifest match the data being run
# ---------------------------------------------------------------------------

class TestCheckInput:
    def test_exact_path_matches(self, tmp_path):
        data = tmp_path / "t.csv"
        data.write_text("a\n1\n")
        old = datetime.now(timezone.utc) + timedelta(days=1)
        r = report_for(tmp_path, data_path=data, created_at=old.strftime("%Y-%m-%dT%H:%M:%SZ"))
        assert check_input(r, [str(data)]) == []

    def test_input_inside_source_dir_matches(self, tmp_path):
        src = tmp_path / "images"
        src.mkdir()
        m = make_manifest(data_path=tmp_path / "features.csv")
        m["dataset"]["source_path"] = str(src)
        r = load_report(write(tmp_path, m))
        assert check_input(r, [str(src / "a.npy")]) == []

    def test_different_data_warns_with_both_paths(self, tmp_path):
        r = report_for(tmp_path, data_path=tmp_path / "assessed.csv")
        items = check_input(r, [str(tmp_path / "other.csv")])
        assert kinds(items) == {"input-mismatch": WARNING}
        assert any("assessed.csv" in e for e in items[0].evidence)
        assert any("other.csv" in e for e in items[0].evidence)

    def test_file_modified_after_assessment_warns(self, tmp_path):
        data = tmp_path / "t.csv"
        data.write_text("a\n1\n")
        r = report_for(tmp_path, data_path=data, created_at="2000-01-01T00:00:00Z")
        assert kinds(check_input(r, [str(data)])) == {"stale-assessment": WARNING}

    def test_unreadable_timestamp_is_not_an_error(self, tmp_path):
        data = tmp_path / "t.csv"
        data.write_text("a\n1\n")
        r = report_for(tmp_path, data_path=data, created_at="yesterday")
        assert check_input(r, [str(data)]) == []


# ---------------------------------------------------------------------------
# findings -> plan
# ---------------------------------------------------------------------------

class TestPlan:
    def test_clean_report_has_no_notes(self, tmp_path):
        r = report_for(tmp_path, findings=[
            finding("aidrin.quality.completeness", 1.0),
            finding("aidrin.quality.duplicity", 0.0),
            finding("aidrin.structure.constant_feature_count", 0.0),
        ])
        assert build_plan(r) == []

    def test_missing_values_names_worst_columns(self, tmp_path):
        r = report_for(tmp_path, findings=[
            finding("aidrin.quality.completeness", 0.9),
            finding("aidrin.column.completeness", 0.7, scope="column", column="bad"),
            finding("aidrin.column.completeness", 1.0, scope="column", column="fine"),
        ])
        item = build_plan(r)[0]
        assert (item.topic, item.kind) == ("missing-values", ADVICE)
        assert "bad: 0.7 complete" in item.evidence
        assert not any("fine" in e for e in item.evidence)

    def test_outliers_say_clip_not_remove(self, tmp_path):
        r = report_for(tmp_path, findings=[finding("aidrin.quality.outliers", 0.05)])
        item = build_plan(r)[0]
        assert item.kind == ADVICE and "clips" in item.message and "does not remove" in item.message

    def test_whole_row_vs_column_duplicates(self, tmp_path):
        r = report_for(tmp_path, findings=[
            finding("aidrin.quality.duplicity", 0.02),
            finding("aidrin.quality.duplicate_rows", 1500),
        ])
        got = kinds(build_plan(r))
        assert got["duplicate-rows"] == ADVICE
        assert got["duplicates-on-columns"] == UNSUPPORTED

    def test_constant_features_unsupported(self, tmp_path):
        r = report_for(tmp_path, findings=[finding("aidrin.structure.constant_feature_count", 6)])
        assert kinds(build_plan(r)) == {"constant-features": UNSUPPORTED}

    def test_imbalance_with_unconfirmed_target_is_not_applied_and_says_why(self, tmp_path):
        r = report_for(tmp_path, roles={"target": "income"},
                       findings=[finding("aidrin.fairness.class_imbalance", 0.93)])
        item = [i for i in build_plan(r) if i.topic == "stratified-split"][0]
        assert item.kind == UNSUPPORTED and "income" in item.message
        assert "not confirmed" in item.message and "plain shuffled random split" in item.message

    def test_imbalance_is_applied_only_when_the_pipeline_was_configured_to(self, tmp_path):
        r = report_for(tmp_path, roles={"target": "income", "confirmed_by_user": True},
                       findings=[finding("aidrin.fairness.class_imbalance", 0.93)])
        not_wired = [i for i in build_plan(r) if i.topic == "stratified-split"][0]
        assert not_wired.kind == UNSUPPORTED and "not confirmed" not in not_wired.message
        wired = build_plan(r, applied={"target-role", "stratified-split"})
        assert kinds(wired)["stratified-split"] == APPLIED
        assert kinds(wired)["target-role"] == APPLIED

    def test_imbalance_without_target_is_silent(self, tmp_path):
        r = report_for(tmp_path, findings=[finding("aidrin.fairness.class_imbalance", 0.93)])
        assert "stratified-split" not in kinds(build_plan(r))

    def test_batch_role_says_no_group_split(self, tmp_path):
        r = report_for(tmp_path, roles={"batch": "group"})
        assert kinds(build_plan(r))["group-split"] == UNSUPPORTED

    def test_privacy_findings_warn_that_redi_does_not_anonymize(self, tmp_path):
        r = report_for(tmp_path, findings=[
            finding("aidrin.governance.k_anonymity", 1),
            finding("aidrin.governance.hipaa_total_flags", 4),
        ])
        got = kinds(build_plan(r))
        assert got["privacy-k-anonymity"] == WARNING
        assert got["privacy-hipaa"] == WARNING

    def test_k_above_one_is_not_flagged(self, tmp_path):
        r = report_for(tmp_path, findings=[finding("aidrin.governance.k_anonymity", 5)])
        assert build_plan(r) == []

    def test_failed_metrics_listed(self, tmp_path):
        m = make_manifest()
        m["metrics"].append({"metric": "k-anonymity", "status": "error", "args": {}, "result": None,
                             "error": {"type": "KeyError", "message": "<redacted>"}})
        item = [i for i in build_plan(load_report(write(tmp_path, m))) if i.topic == "failed-metrics"][0]
        assert item.kind == WARNING and item.evidence == ["k-anonymity"]

    def test_unconfirmed_roles_warn_only_when_roles_exist(self, tmp_path):
        assert "roles-unconfirmed" not in kinds(build_plan(report_for(tmp_path)))
        r = report_for(tmp_path, roles={"id": "stem"})
        assert kinds(build_plan(r))["roles-unconfirmed"] == WARNING
        r = report_for(tmp_path, roles={"id": "stem", "confirmed_by_user": True})
        assert "roles-unconfirmed" not in kinds(build_plan(r))

    def test_raw_results_warn(self, tmp_path):
        r = report_for(tmp_path, redaction="raw")
        assert kinds(build_plan(r))["raw-results"] == WARNING

    def test_domain_pipeline_caveat(self, tmp_path):
        r = report_for(tmp_path)
        assert build_plan(r, "general") == []
        assert kinds(build_plan(r, "hpc"))["domain-pipeline"] == ADVICE


class TestProvenance:
    def test_block_records_the_link_and_applies_nothing(self, tmp_path):
        r = report_for(tmp_path, roles={"id": "stem"}, findings=[finding("aidrin.structure.constant_feature_count", 2)])
        block = provenance_block(r, build_plan(r))
        assert block["manifest"] == str(r.path)
        assert block["manifest_sha256"] == r.sha256
        assert block["file_assessed"] == "/data/table.csv"
        assert block["roles"]["id"] == "stem"
        assert block["applied"] == []
        assert {p["topic"] for p in block["plan"]} >= {"constant-features"}
        json.dumps(block)  # must be serialisable into metadata.json


@pytest.mark.skipif(not RGB1_META.exists(), reason="RGB1 manifest not present")
class TestRealRgb1Manifest:
    def test_plan_for_the_rgb1_assessment(self):
        r = load_report(RGB1_META)
        got = kinds(build_plan(r))
        assert got["duplicates-on-columns"] == UNSUPPORTED   # 1,500 blank frames
        assert got["constant-features"] == UNSUPPORTED
        assert got["group-split"] == UNSUPPORTED             # batch role = group
        assert got["roles-unconfirmed"] == WARNING
        assert "missing-values" not in got                   # completeness is 1.0
        assert "stratified-split" not in got                 # no target role


# ---------------------------------------------------------------------------
# end to end: redi run --aidrin-report
# ---------------------------------------------------------------------------

def run_redi(*argv):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONWARNINGS="ignore")
    return subprocess.run([sys.executable, "-m", "redi.cli", *argv], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=300)


@pytest.fixture
def csv_and_manifest(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b,c\n" + "\n".join(f"{i},{i * 2},{'x' if i % 2 else 'y'}" for i in range(40)) + "\n")
    m = make_manifest(
        data_path=data,
        roles={"target": "c", "confirmed_by_user": True},
        findings=[finding("aidrin.quality.completeness", 0.98),
                  finding("aidrin.fairness.class_imbalance", 0.4),
                  finding("aidrin.structure.constant_feature_count", 1)],
        created_at=(datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return data, write(tmp_path, m)


class TestCliEndToEnd:
    def test_run_records_the_report_and_prints_the_plan(self, tmp_path, csv_and_manifest):
        data, manifest = csv_and_manifest
        out = tmp_path / "out"
        proc = run_redi("run", str(data), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[applied]" in proc.stdout and "stratified" in proc.stdout
        meta = json.loads((out / "metadata.json").read_text())
        block = meta["aidrin_assessment"]
        assert block["manifest_sha256"] == load_report(manifest).sha256
        assert block["mode"] == "apply"
        assert {a["topic"] for a in block["applied"]} == {"target-role", "stratified-split", "constant-features"}
        assert all(a["outcome"] for a in block["applied"])  # outcomes recorded, not just intentions

    def test_run_without_the_flag_is_unchanged(self, tmp_path, csv_and_manifest):
        data, _ = csv_and_manifest
        out = tmp_path / "out"
        proc = run_redi("run", str(data), "-o", str(out), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "aidrin_assessment" not in json.loads((out / "metadata.json").read_text())

    def test_bad_report_fails_before_any_output(self, tmp_path, csv_and_manifest):
        data, _ = csv_and_manifest
        out = tmp_path / "out"
        proc = run_redi("run", str(data), "-o", str(out), "--aidrin-report", str(tmp_path / "nope.json"))
        assert proc.returncode == 1
        assert "cannot read AIDRIN report" in proc.stdout
        assert not out.exists()

    def test_mismatched_input_warns_but_still_runs(self, tmp_path, csv_and_manifest):
        _, manifest = csv_and_manifest
        other = tmp_path / "other.csv"
        other.write_text("a,b,c\n" + "\n".join(f"{i},{i},{i}" for i in range(20)) + "\n")
        out = tmp_path / "out"
        proc = run_redi("run", str(other), "-o", str(out), "--aidrin-report", str(manifest), "--style", "simple")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[warning]" in proc.stdout and "different data" in proc.stdout
