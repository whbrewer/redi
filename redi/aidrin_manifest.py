"""
Consume an AIDRIN assessment manifest (``aidrin_assessment.json``).

AIDRIN assesses a dataset and writes a manifest of the column roles used, the
metrics run and their headline numbers. ``redi run --aidrin-report <manifest>``
reads it, checks it describes the data being run, tells the user what REDI will
and will not do about each finding, and records the link in ``metadata.json``
so SetGo and later readers can trace it.

What gets applied: only things the generic DataFrame pipeline can do and that the
report supports unambiguously (see :func:`derive_settings`):

* a *confirmed* target role: kept out of normalization/encoding, label-encoded if it is
  text, and passed on as ``y``;
* class imbalance on that target: the train/val/test split is stratified on ``y``;
* constant columns: numeric columns with a single value are dropped.

Everything else an assessment suggests is advice or unsupported (group-aware splits,
removal of rows duplicated on specific columns, anonymization). Domain pipelines have
their own steps, so nothing is applied to them. Each plan item says which it is, and after
the run :func:`record_outcomes` checks what actually happened, so ``metadata.json`` records
results, not intentions.

Format: ``aidrin.assessment/v1``. Full schema validation is
``aidrin manifest validate``; this loader checks only what REDI reads, so REDI
does not need AIDRIN installed. Unknown extra fields are ignored.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SUPPORTED_SCHEMA = "aidrin.assessment/v1"

# Plan item kinds.
APPLIED = "applied"          # REDI acts on this finding in this run
WARNING = "warning"          # needs the user's attention before trusting the output
ADVICE = "advice"            # REDI already does something relevant, or the user can act
UNSUPPORTED = "unsupported"  # the finding calls for something REDI cannot do today

_ROLE_FIELDS = ("target", "id", "batch")
_ROLE_LIST_FIELDS = ("sensitive", "quasi_identifiers")


class AidrinReportError(ValueError):
    """The manifest cannot be read, or is not one REDI understands."""


@dataclass
class PlanItem:
    kind: str
    topic: str
    message: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class AidrinReport:
    path: Path
    sha256: str
    data: Dict[str, Any]

    @property
    def roles(self) -> Dict[str, Any]:
        return self.data["roles"]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.data["dataset"]

    @property
    def findings(self) -> List[Dict[str, Any]]:
        return self.data.get("findings") or []

    @property
    def metrics(self) -> List[Dict[str, Any]]:
        return self.data["metrics"]

    def value(self, name: str) -> Optional[float]:
        """A dataset-scope finding by name, or None if AIDRIN did not report it."""
        for f in self.findings:
            if f["name"] == name and f["scope"] == "dataset":
                return f["value"]
        return None

    def columns(self, name: str) -> Dict[str, float]:
        """Per-column findings by name, keyed by column."""
        return {
            f["column"]: f["value"]
            for f in self.findings
            if f["name"] == name and f["scope"] == "column" and "column" in f
        }

    def has_roles(self) -> bool:
        return any(self.roles.get(k) for k in _ROLE_FIELDS + _ROLE_LIST_FIELDS)


def load_report(path) -> AidrinReport:
    """Read and sanity-check a manifest. Raises :class:`AidrinReportError`."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AidrinReportError(f"cannot read AIDRIN report {path}: {exc.strerror or exc}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AidrinReportError(f"AIDRIN report is not valid JSON: {path} ({exc})") from None

    if not isinstance(data, dict):
        raise AidrinReportError(f"AIDRIN report must be a JSON object: {path}")
    version = data.get("schema_version")
    if version != SUPPORTED_SCHEMA:
        raise AidrinReportError(
            f"unsupported AIDRIN report version {version!r} in {path}; "
            f"this REDI reads {SUPPORTED_SCHEMA}"
        )

    problems = _structural_problems(data)
    if problems:
        raise AidrinReportError(
            f"AIDRIN report {path} is malformed:\n  " + "\n  ".join(problems)
            + "\nRun 'aidrin manifest validate' on it for the full schema check."
        )
    return AidrinReport(path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest(), data=data)


def _structural_problems(data: Dict[str, Any]) -> List[str]:
    problems = []
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
        problems.append("dataset.path is missing")
    if not isinstance(data.get("roles"), dict):
        problems.append("roles is missing")
    if not isinstance(data.get("metrics"), list):
        problems.append("metrics is missing")
    findings = data.get("findings", [])
    if not isinstance(findings, list):
        problems.append("findings must be a list")
    else:
        for i, f in enumerate(findings):
            ok = (
                isinstance(f, dict)
                and isinstance(f.get("name"), str)
                and f.get("scope") in ("dataset", "column")
                and isinstance(f.get("value"), (int, float))
                and not isinstance(f.get("value"), bool)
            )
            if not ok:
                problems.append(f"findings[{i}] needs a string name, a scope, and a numeric value")
                break
    return problems


# ---------------------------------------------------------------------------
# Does the manifest describe the data being run?
# ---------------------------------------------------------------------------

def _resolve(p) -> Optional[Path]:
    try:
        return Path(p).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _related(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_input(report: AidrinReport, input_paths: Iterable[str]) -> List[PlanItem]:
    """Warn when the run's input is not what AIDRIN assessed, or has changed since."""
    items = []
    assessed = [_resolve(report.dataset["path"])]
    if report.dataset.get("source_path"):
        assessed.append(_resolve(report.dataset["source_path"]))
    assessed = [p for p in assessed if p is not None]

    inputs = [p for p in (_resolve(i) for i in input_paths) if p is not None]
    if inputs and assessed and not any(_related(i, a) for i in inputs for a in assessed):
        shown = report.dataset["path"]
        if report.dataset.get("source_path"):
            shown += f" (source: {report.dataset['source_path']})"
        items.append(PlanItem(
            WARNING, "input-mismatch",
            "The AIDRIN report describes different data than this run's input, so its "
            "findings may not apply.",
            [f"assessed: {shown}", f"input: {', '.join(str(i) for i in inputs)}"],
        ))

    items.extend(_stale_check(report))
    return items


def _stale_check(report: AidrinReport) -> List[PlanItem]:
    created = report.data.get("created_at")
    path = _resolve(report.dataset["path"])
    if not created or path is None or not path.exists():
        return []
    try:
        made = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except (ValueError, OSError):
        return []
    if modified > made:
        return [PlanItem(
            WARNING, "stale-assessment",
            "The assessed file was modified after the assessment ran.",
            [f"assessed at {created}", f"file modified {modified.strftime('%Y-%m-%dT%H:%M:%SZ')}"],
        )]
    return []


# ---------------------------------------------------------------------------
# Findings -> what REDI will and will not do
# ---------------------------------------------------------------------------

def _ev(name: str, value: float) -> str:
    return f"{name} = {value:g}"


@dataclass
class AidrinSettings:
    """Pipeline settings a report supports. Defaults leave the pipeline unchanged."""
    target_column: Optional[str] = None
    stratify: bool = False
    drop_constant: bool = False


def derive_settings(report: AidrinReport) -> AidrinSettings:
    """What the report licenses the pipeline to do.

    The target role is used only when the user confirmed it: it decides what the model
    learns to predict, so a role inferred from a metric argument is not enough.
    """
    s = AidrinSettings()
    target = report.roles.get("target")
    if target and report.roles.get("confirmed_by_user"):
        s.target_column = str(target)
        s.stratify = bool(report.value("aidrin.fairness.class_imbalance"))
    s.drop_constant = bool(report.value("aidrin.structure.constant_feature_count"))
    return s


def build_plan(
    report: AidrinReport,
    domain: Optional[str] = None,
    applied: Iterable[str] = (),
) -> List[PlanItem]:
    """Translate findings into plan items.

    ``applied`` names the topics the pipeline was actually configured to act on; those
    items are reported as ``applied``, the rest as advice or unsupported.
    """
    applied = set(applied)
    items: List[PlanItem] = []

    if domain and domain != "general":
        items.append(PlanItem(
            ADVICE, "domain-pipeline",
            f"The '{domain}' domain has its own pipeline steps. The notes below describe "
            "REDI's generic DataFrame pipeline and may not apply to this run.",
        ))

    failed = [m["metric"] for m in report.metrics if m.get("status") == "error"]
    if failed:
        items.append(PlanItem(
            WARNING, "failed-metrics",
            "Some AIDRIN metrics failed, so the report is incomplete for those dimensions.",
            failed,
        ))

    if report.has_roles() and not report.roles.get("confirmed_by_user"):
        items.append(PlanItem(
            WARNING, "roles-unconfirmed",
            "Column roles in the report were inferred, not confirmed by the user. "
            "Treat them as suggestions.",
        ))

    if report.data.get("redaction") == "raw":
        items.append(PlanItem(
            WARNING, "raw-results",
            "The report contains raw metric results, which can include cell values. "
            "Do not share or commit it without checking.",
        ))

    comp = report.value("aidrin.quality.completeness")
    if comp is not None and comp < 1:
        worst = sorted(
            ((c, v) for c, v in report.columns("aidrin.column.completeness").items() if v < 1),
            key=lambda cv: cv[1],
        )[:3]
        items.append(PlanItem(
            ADVICE, "missing-values",
            "Preprocessing handles missing values (default strategy: drop rows). "
            "Check the row count after the run.",
            [_ev("aidrin.quality.completeness", comp)] + [f"{c}: {v:g} complete" for c, v in worst],
        ))

    dup = report.value("aidrin.quality.duplicity")
    if dup:
        items.append(PlanItem(
            ADVICE, "duplicate-rows",
            "Preprocessing removes fully identical rows by default.",
            [_ev("aidrin.quality.duplicity", dup)],
        ))
    dup_rows = report.value("aidrin.quality.duplicate_rows")
    if dup_rows:
        items.append(PlanItem(
            UNSUPPORTED, "duplicates-on-columns",
            "AIDRIN found rows duplicated on specific columns. REDI only removes rows that "
            "are identical in every column, so these may remain. Decide which to drop, "
            "and watch for leakage across train/test.",
            [_ev("aidrin.quality.duplicate_rows", dup_rows)],
        ))

    out = report.value("aidrin.quality.outliers")
    if out:
        items.append(PlanItem(
            ADVICE, "outliers",
            "Preprocessing clips outliers by default (IQR, 1.5x). It does not remove rows.",
            [_ev("aidrin.quality.outliers", out)],
        ))

    const = report.value("aidrin.structure.constant_feature_count")
    if const:
        if "constant-features" in applied:
            items.append(PlanItem(
                APPLIED, "constant-features",
                "Dropping numeric columns with a single value. Constant text columns are not removed.",
                [_ev("aidrin.structure.constant_feature_count", const)],
            ))
        else:
            items.append(PlanItem(
                UNSUPPORTED, "constant-features",
                "Constant columns carry no information, and this run does not remove them.",
                [_ev("aidrin.structure.constant_feature_count", const)],
            ))

    imbalance = report.value("aidrin.fairness.class_imbalance")
    target = report.roles.get("target")
    if target and "target-role" in applied:
        items.append(PlanItem(
            APPLIED, "target-role",
            f"Using '{target}' as the target: kept out of normalization and encoding "
            "(label-encoded if it is text) and written as y.",
        ))
    if imbalance and target:
        if "stratified-split" in applied:
            items.append(PlanItem(
                APPLIED, "stratified-split",
                f"The target '{target}' is imbalanced, so the train/val/test split is stratified on it.",
                [_ev("aidrin.fairness.class_imbalance", imbalance)],
            ))
        else:
            why = ("" if report.roles.get("confirmed_by_user")
                   else " The target role is not confirmed, so it was not used.")
            items.append(PlanItem(
                UNSUPPORTED, "stratified-split",
                f"The target '{target}' is imbalanced, and this run's split is a plain shuffled "
                f"random split.{why} Check class balance in each split.",
                [_ev("aidrin.fairness.class_imbalance", imbalance)],
            ))

    batch = report.roles.get("batch")
    if batch:
        items.append(PlanItem(
            UNSUPPORTED, "group-split",
            f"The report names '{batch}' as the grouping column. REDI cannot split by group, "
            "so related rows can land on both sides of the split.",
        ))

    k = report.value("aidrin.governance.k_anonymity")
    if k is not None and k <= 1:
        items.append(PlanItem(
            WARNING, "privacy-k-anonymity",
            "k-anonymity is 1, meaning some rows are unique on the quasi-identifiers. "
            "REDI does not anonymize.",
            [_ev("aidrin.governance.k_anonymity", k)],
        ))
    flags = report.value("aidrin.governance.hipaa_total_flags")
    if flags:
        items.append(PlanItem(
            WARNING, "privacy-hipaa",
            "AIDRIN flagged possible HIPAA identifiers. REDI does not remove them.",
            [_ev("aidrin.governance.hipaa_total_flags", flags)],
        ))

    return items


def provenance_block(
    report: AidrinReport,
    plan: List[PlanItem],
    applied: Iterable[str] = (),
) -> Dict[str, Any]:
    """What ``metadata.json`` records about the AIDRIN report that informed a run.

    ``applied`` entries start with no outcome; :func:`record_outcomes` fills them in after
    the run.
    """
    return {
        "manifest": str(report.path),
        "manifest_sha256": report.sha256,
        "schema_version": report.data["schema_version"],
        "aidrin_version": (report.data.get("tool") or {}).get("version"),
        "assessed_at": report.data.get("created_at"),
        "file_assessed": report.dataset["path"],
        "source_path": report.dataset.get("source_path"),
        "roles": report.roles,
        "redaction": report.data.get("redaction"),
        "findings": len(report.findings),
        "plan": [asdict(i) for i in plan],
        "applied": [{"topic": topic, "outcome": None, "detail": None} for topic in sorted(set(applied))],
    }


def record_outcomes(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fill in what each applied action actually did, from the run's own metadata.

    Returns the entries that did not take effect so the caller can say so. An action can be
    wired in and still not happen (for example the target column is not in the data, or a
    stage was skipped), so the recorded outcome comes from what the steps wrote.
    """
    block = metadata.get("aidrin_assessment")
    if not block:
        return []

    def outcome(topic):
        if topic == "target-role":
            # numpy_structure records the *requested* target even if the column does not
            # exist, so success is read from target_encoding, which EncodeTarget writes
            # only after finding the column.
            struct = metadata.get("numpy_structure") or {}
            enc = metadata.get("target_encoding") or {}
            column = struct.get("target_column")
            if column and enc.get("column") == column:
                detail = f"y written from '{column}'"
                if enc.get("method") == "label":
                    detail += f"; label-encoded ({len(enc.get('mapping', {}))} classes)"
                return True, detail
            return False, "the target column was not found in the data, or the transform/structure stages did not run"
        if topic == "stratified-split":
            split = metadata.get("split") or {}
            if split.get("stratified"):
                return True, "split stratified on y"
            return False, split.get("stratify_note") or "the split did not run"
        if topic == "constant-features":
            sel = metadata.get("feature_selection")
            if sel is None:
                return False, "the feature-selection step did not run"
            removed = sel.get("removed", [])
            return True, f"dropped {len(removed)} column(s): {', '.join(map(str, removed[:10]))}" if removed \
                else "no numeric constant columns were found"
        return False, "unknown action"

    failed = []
    for entry in block.get("applied", []):
        ok, detail = outcome(entry["topic"])
        entry["outcome"] = "applied" if ok else "not applied"
        entry["detail"] = detail
        if not ok:
            failed.append(entry)
    return failed
