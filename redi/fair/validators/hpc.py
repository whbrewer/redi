"""HPC domain FAIR validators.

Validates HPC data against Slurm, PBS, and related job schemas.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..principles import DomainStandardResult


# Common Slurm sacct fields
SLURM_REQUIRED_FIELDS = {
    "jobid", "user", "state", "submit", "start", "end",
}
SLURM_RECOMMENDED_FIELDS = {
    "partition", "account", "nnodes", "ncpus", "elapsed",
    "timelimit", "exitcode", "nodelist", "jobname",
}

# Common PBS accounting fields
PBS_REQUIRED_FIELDS = {
    "job_id", "user", "queue", "ctime", "qtime",
}
PBS_RECOMMENDED_FIELDS = {
    "resources_used", "exit_status", "start", "end",
    "exec_host", "resource_list",
}

# Generic HPC job fields
GENERIC_JOB_FIELDS = {
    "job", "user", "node", "cpu", "memory", "runtime",
    "walltime", "queue", "status", "exit",
}


class SlurmValidator:
    """Validates data against Slurm accounting format."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate against Slurm schema.

        Args:
            file_path: Path to data file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".csv", ".parquet", ".json"):
            return None

        columns = self._get_columns(file_path, data_card)
        if not columns:
            return None

        # Normalize column names
        columns_lower = {c.lower().replace("_", "").replace("-", "") for c in columns}

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        # Check required fields
        req_found = 0
        for field in SLURM_REQUIRED_FIELDS:
            field_normalized = field.lower().replace("_", "")
            if field_normalized in columns_lower or self._fuzzy_match(field, columns):
                req_found += 1

        if req_found >= len(SLURM_REQUIRED_FIELDS) * 0.8:
            passed.append(f"Has {req_found}/{len(SLURM_REQUIRED_FIELDS)} required Slurm fields")
        elif req_found > 0:
            passed.append(f"Has {req_found} Slurm-like fields")
        else:
            # Not Slurm data
            return None

        # Check recommended fields
        rec_found = 0
        for field in SLURM_RECOMMENDED_FIELDS:
            if self._fuzzy_match(field, columns):
                rec_found += 1

        if rec_found > 0:
            passed.append(f"Has {rec_found} recommended Slurm fields")

        # Check for job state values
        if data_card:
            # Could check for valid Slurm states: COMPLETED, FAILED, TIMEOUT, etc.
            pass

        # Calculate compliance
        total = len(passed) + len(failed)
        compliance = len(passed) / max(total, 1)

        return DomainStandardResult(
            standard_name="Slurm Accounting",
            compliance_percentage=compliance * 100,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
        )

    def _get_columns(
        self, file_path: Path, data_card: Optional[Any]
    ) -> Set[str]:
        """Get column names from file or DataCard."""
        columns = set()

        # Try DataCard first
        if data_card and hasattr(data_card, "characteristics"):
            features = getattr(data_card.characteristics, "features", [])
            columns = {f.get("name", "") for f in features if f.get("name")}

        # If no columns from DataCard, try reading file
        if not columns:
            try:
                ext = file_path.suffix.lower()
                if ext == ".csv":
                    with open(file_path) as f:
                        header = f.readline().strip()
                        columns = set(header.split(","))
                elif ext == ".parquet":
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(file_path)
                    columns = set(schema.names)
                elif ext == ".json":
                    import json
                    with open(file_path) as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            columns = set(data[0].keys())
                        elif isinstance(data, dict):
                            columns = set(data.keys())
            except Exception:
                pass

        return columns

    def _fuzzy_match(self, field: str, columns: Set[str]) -> bool:
        """Check if field matches any column with fuzzy matching."""
        field_lower = field.lower().replace("_", "").replace("-", "")
        for col in columns:
            col_lower = col.lower().replace("_", "").replace("-", "")
            if field_lower in col_lower or col_lower in field_lower:
                return True
        return False


class PBSValidator:
    """Validates data against PBS accounting format."""

    def validate(
        self, file_path: Path, data_card: Optional[Any] = None
    ) -> Optional[DomainStandardResult]:
        """Validate against PBS schema.

        Args:
            file_path: Path to data file
            data_card: Optional DataCard

        Returns:
            DomainStandardResult or None if not applicable
        """
        ext = file_path.suffix.lower()
        if ext not in (".csv", ".parquet", ".json"):
            return None

        columns = self._get_columns(file_path, data_card)
        if not columns:
            return None

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []

        # Check for PBS-specific fields
        pbs_indicators = {"ctime", "qtime", "exec_host", "resource_list", "pbs"}
        pbs_found = 0

        for indicator in pbs_indicators:
            if self._fuzzy_match(indicator, columns):
                pbs_found += 1

        if pbs_found == 0:
            # Not PBS data
            return None

        # Check required fields
        req_found = 0
        for field in PBS_REQUIRED_FIELDS:
            if self._fuzzy_match(field, columns):
                req_found += 1

        if req_found > 0:
            passed.append(f"Has {req_found}/{len(PBS_REQUIRED_FIELDS)} required PBS fields")

        # Check recommended fields
        rec_found = 0
        for field in PBS_RECOMMENDED_FIELDS:
            if self._fuzzy_match(field, columns):
                rec_found += 1

        if rec_found > 0:
            passed.append(f"Has {rec_found} recommended PBS fields")

        # Calculate compliance
        total = len(passed) + len(failed)
        compliance = len(passed) / max(total, 1)

        return DomainStandardResult(
            standard_name="PBS Accounting",
            compliance_percentage=compliance * 100,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
        )

    def _get_columns(
        self, file_path: Path, data_card: Optional[Any]
    ) -> Set[str]:
        """Get column names from file or DataCard."""
        # Same implementation as SlurmValidator
        columns = set()

        if data_card and hasattr(data_card, "characteristics"):
            features = getattr(data_card.characteristics, "features", [])
            columns = {f.get("name", "") for f in features if f.get("name")}

        if not columns:
            try:
                ext = file_path.suffix.lower()
                if ext == ".csv":
                    with open(file_path) as f:
                        header = f.readline().strip()
                        columns = set(header.split(","))
                elif ext == ".parquet":
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(file_path)
                    columns = set(schema.names)
            except Exception:
                pass

        return columns

    def _fuzzy_match(self, field: str, columns: Set[str]) -> bool:
        """Check if field matches any column with fuzzy matching."""
        field_lower = field.lower().replace("_", "").replace("-", "")
        for col in columns:
            col_lower = col.lower().replace("_", "").replace("-", "")
            if field_lower in col_lower or col_lower in field_lower:
                return True
        return False
