"""
Snakemake log parser for REDI observability.

Parses Snakemake execution logs to extract:
- Rule execution details (inputs, outputs, timing)
- Job statistics
- Errors and warnings
- File provenance (which rule created which file)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleExecution:
    """A single rule execution from the log."""
    rule_name: str
    jobid: int
    timestamp: Optional[datetime] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    wildcards: Dict[str, str] = field(default_factory=dict)
    reason: Optional[str] = None
    status: str = "started"  # started, finished, error
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "jobid": self.jobid,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "wildcards": self.wildcards,
            "reason": self.reason,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class SnakemakeLogReport:
    """Parsed Snakemake log report."""
    log_path: str
    job_stats: Dict[str, int] = field(default_factory=dict)
    executions: List[RuleExecution] = field(default_factory=list)
    file_provenance: Dict[str, str] = field(default_factory=dict)  # output_file -> rule_name
    errors: List[str] = field(default_factory=list)
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    def to_dict(self) -> dict:
        return {
            "log_path": self.log_path,
            "job_stats": self.job_stats,
            "executions": [e.to_dict() for e in self.executions],
            "file_provenance": self.file_provenance,
            "errors": self.errors,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
        }

    def get_rule_files(self, rule_name: str) -> Tuple[List[str], List[str]]:
        """Get all input and output files for a rule."""
        inputs = []
        outputs = []
        for exec in self.executions:
            if exec.rule_name == rule_name:
                inputs.extend(exec.inputs)
                outputs.extend(exec.outputs)
        return list(set(inputs)), list(set(outputs))


def parse_snakemake_log(log_path: Path) -> SnakemakeLogReport:
    """
    Parse a Snakemake log file.

    Args:
        log_path: Path to the .snakemake.log file

    Returns:
        SnakemakeLogReport with parsed information
    """
    report = SnakemakeLogReport(log_path=str(log_path))

    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return report

    content = log_path.read_text()
    lines = content.split("\n")

    # Patterns
    timestamp_pattern = re.compile(r'\[([A-Za-z]{3} [A-Za-z]{3} \d+ \d+:\d+:\d+ \d+)\]')
    rule_pattern = re.compile(r'^(?:local)?rule (\w+):')
    input_pattern = re.compile(r'^\s+input:\s*(.+)$')
    output_pattern = re.compile(r'^\s+output:\s*(.+)$')
    jobid_pattern = re.compile(r'^\s+jobid:\s*(\d+)')
    reason_pattern = re.compile(r'^\s+reason:\s*(.+)$')
    wildcards_pattern = re.compile(r'^\s+wildcards:\s*(.+)$')
    # Match both formats: "Finished job X." and "Finished jobid: X (Rule: name)"
    finished_pattern = re.compile(r'^Finished job(?:id)?[:\s]+(\d+)')
    # Match various error patterns
    error_pattern = re.compile(r'^(RuleException|Error in rule|Error:|CalledProcessError|WorkflowError)')
    job_stats_pattern = re.compile(r'^(\w+)\s+(\d+)$')

    # Parse job stats section
    in_job_stats = False
    for line in lines:
        if "Job stats:" in line:
            in_job_stats = True
            continue
        if in_job_stats:
            if line.strip() == "" or line.startswith("Select"):
                in_job_stats = False
                continue
            if line.startswith("job") or line.startswith("-"):
                continue
            match = job_stats_pattern.match(line)
            if match:
                rule_name, count = match.groups()
                report.job_stats[rule_name] = int(count)
                if rule_name == "total":
                    report.total_jobs = int(count)

    # Parse rule executions
    current_exec: Optional[RuleExecution] = None
    current_timestamp: Optional[datetime] = None
    executions_by_jobid: Dict[int, RuleExecution] = {}

    i = 0
    while i < len(lines):
        line = lines[i]

        # Parse timestamp
        ts_match = timestamp_pattern.match(line)
        if ts_match:
            try:
                current_timestamp = datetime.strptime(
                    ts_match.group(1), "%a %b %d %H:%M:%S %Y"
                )
            except ValueError:
                pass

        # Parse rule start
        rule_match = rule_pattern.match(line)
        if rule_match:
            if current_exec:
                report.executions.append(current_exec)
                if current_exec.outputs:
                    for output in current_exec.outputs:
                        report.file_provenance[output] = current_exec.rule_name

            current_exec = RuleExecution(
                rule_name=rule_match.group(1),
                jobid=-1,
                timestamp=current_timestamp,
            )
            i += 1
            continue

        # Parse rule details
        if current_exec:
            input_match = input_pattern.match(line)
            if input_match:
                # Handle multiple inputs (comma or space separated)
                inputs_str = input_match.group(1).strip()
                inputs = _parse_file_list(inputs_str)
                current_exec.inputs.extend(inputs)
                i += 1
                continue

            output_match = output_pattern.match(line)
            if output_match:
                outputs_str = output_match.group(1).strip()
                outputs = _parse_file_list(outputs_str)
                current_exec.outputs.extend(outputs)
                i += 1
                continue

            jobid_match = jobid_pattern.match(line)
            if jobid_match:
                current_exec.jobid = int(jobid_match.group(1))
                executions_by_jobid[current_exec.jobid] = current_exec
                i += 1
                continue

            reason_match = reason_pattern.match(line)
            if reason_match:
                current_exec.reason = reason_match.group(1)
                i += 1
                continue

            wildcards_match = wildcards_pattern.match(line)
            if wildcards_match:
                wildcards_str = wildcards_match.group(1)
                current_exec.wildcards = _parse_wildcards(wildcards_str)
                i += 1
                continue

        # Parse job finished
        finished_match = finished_pattern.match(line)
        if finished_match:
            jobid = int(finished_match.group(1))
            if jobid in executions_by_jobid:
                executions_by_jobid[jobid].status = "finished"
                report.completed_jobs += 1

        # Parse errors
        if error_pattern.match(line):
            # Collect error context
            error_lines = [line]
            j = i + 1
            while j < len(lines) and j < i + 10:
                if lines[j].strip() and not timestamp_pattern.match(lines[j]):
                    error_lines.append(lines[j])
                else:
                    break
                j += 1
            error_msg = "\n".join(error_lines)
            report.errors.append(error_msg)

            # Mark current job as failed
            if current_exec:
                current_exec.status = "error"
                current_exec.error_message = error_msg
                report.failed_jobs += 1

        i += 1

    # Add last execution
    if current_exec:
        report.executions.append(current_exec)
        if current_exec.outputs:
            for output in current_exec.outputs:
                report.file_provenance[output] = current_exec.rule_name

    return report


def _parse_file_list(file_str: str) -> List[str]:
    """Parse a file list from Snakemake log format."""
    files = []
    # Handle comma-separated
    if ", " in file_str:
        parts = file_str.split(", ")
    else:
        parts = [file_str]

    for part in parts:
        # Remove quotes if present
        part = part.strip().strip("'\"")
        if part:
            files.append(part)

    return files


def _parse_wildcards(wildcards_str: str) -> Dict[str, str]:
    """Parse wildcards string like 'dataset=/path, res=1.40625'."""
    wildcards = {}
    pairs = wildcards_str.split(", ")
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            wildcards[key.strip()] = value.strip()
    return wildcards


def find_latest_log(workflow_dir: Path) -> Optional[Path]:
    """Find the most recent Snakemake log in a workflow directory."""
    log_dir = workflow_dir / ".snakemake" / "log"
    if not log_dir.exists():
        return None

    logs = list(log_dir.glob("*.snakemake.log"))
    if not logs:
        return None

    # Sort by modification time, newest first
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0]


def find_all_logs(workflow_dir: Path) -> List[Path]:
    """Find all Snakemake logs in a workflow directory."""
    log_dir = workflow_dir / ".snakemake" / "log"
    if not log_dir.exists():
        return []

    logs = list(log_dir.glob("*.snakemake.log"))
    # Sort by modification time, newest first
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs
