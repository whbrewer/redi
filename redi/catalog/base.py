"""
Base classes for REDI catalog exporters.

Provides abstract base class and data structures for implementing
catalog integrations.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CatalogType(Enum):
    """Supported catalog types."""
    OPENMETADATA = "openmetadata"
    MLFLOW = "mlflow"
    INTAKE = "intake"
    HUGGINGFACE = "huggingface"


@dataclass
class CatalogConfig:
    """Configuration for catalog export.

    Attributes:
        host: Catalog server URL
        token: Authentication token (JWT, API key, etc.)
        service_name: Name of the service/connection in the catalog
        extra: Additional catalog-specific configuration
    """
    host: str = ""
    token: str = ""
    service_name: str = "redi"
    verify_ssl: bool = True
    timeout: int = 30
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CatalogConfig":
        """Create config from dictionary."""
        return cls(
            host=data.get("host", ""),
            token=data.get("token", ""),
            service_name=data.get("service_name", "redi"),
            verify_ssl=data.get("verify_ssl", True),
            timeout=data.get("timeout", 30),
            extra={k: v for k, v in data.items()
                   if k not in ("host", "token", "service_name", "verify_ssl", "timeout")}
        )


@dataclass
class CatalogExportResult:
    """Result of a catalog export operation.

    Attributes:
        success: Whether the export succeeded
        catalog_type: Type of catalog exported to
        entity_id: ID/FQN of the created entity in the catalog
        entity_url: URL to view the entity in the catalog UI
        message: Status message or error description
        metadata: Additional result metadata
    """
    success: bool
    catalog_type: str
    entity_id: str = ""
    entity_url: str = ""
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "catalog_type": self.catalog_type,
            "entity_id": self.entity_id,
            "entity_url": self.entity_url,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class REDIMetadata:
    """Parsed REDI metadata for catalog export.

    Extracts and normalizes metadata from REDI's metadata.json
    into a format suitable for catalog registration.
    """
    # Basic info
    name: str = ""
    description: str = ""
    domain: str = ""
    input_file: str = ""

    # Data characteristics
    num_rows: int = 0
    num_columns: int = 0
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None

    # Split information
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0

    # Normalization
    normalization_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Output files
    output_format: str = ""
    saved_files: List[str] = field(default_factory=list)

    # Processing info
    completed_stages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Raw metadata
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, metadata_path: str) -> "REDIMetadata":
        """Load REDI metadata from a JSON file.

        Args:
            metadata_path: Path to metadata.json file

        Returns:
            REDIMetadata instance

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "REDIMetadata":
        """Create from dictionary."""
        # Extract split info
        split = data.get("split", {})

        # Generate name from input file
        input_file = data.get("input_file", "")
        name = Path(input_file).stem if input_file else "redi_dataset"
        # Make name safe for catalogs (alphanumeric + underscore)
        name = "".join(c if c.isalnum() else "_" for c in name)

        return cls(
            name=name,
            description=f"REDI-processed {data.get('domain', 'general')} dataset",
            domain=data.get("domain", "general"),
            input_file=input_file,
            num_rows=data.get("num_rows", 0),
            num_columns=data.get("num_columns", 0),
            feature_columns=data.get("feature_columns", []),
            target_column=data.get("target_column"),
            train_size=split.get("train_size", 0),
            val_size=split.get("val_size", 0),
            test_size=split.get("test_size", 0),
            normalization_stats=data.get("normalization_stats", {}),
            output_format=data.get("output_format", ""),
            saved_files=data.get("saved_files", []),
            completed_stages=data.get("completed_stages", []),
            errors=data.get("errors", []),
            raw=data,
        )

    def get_feature_types(self) -> Dict[str, str]:
        """Infer feature types from normalization stats.

        Returns:
            Dict mapping feature name to type (numerical/categorical)
        """
        types = {}
        for col in self.feature_columns:
            # If we have normalization stats, it's numerical
            if col in self.normalization_stats:
                types[col] = "numerical"
            else:
                types[col] = "categorical"
        return types


class CatalogExporter(ABC):
    """Abstract base class for catalog exporters.

    Subclasses must implement:
        - export(): Export REDI metadata to the catalog
        - validate_connection(): Check catalog connectivity
        - _create_entity(): Create the catalog entity

    Example implementation:
        class MyExporter(CatalogExporter):
            catalog_type = CatalogType.OPENMETADATA

            def export(self, metadata_path: str) -> CatalogExportResult:
                metadata = REDIMetadata.from_file(metadata_path)
                # ... export logic ...
                return CatalogExportResult(success=True, ...)
    """

    catalog_type: CatalogType = None

    def __init__(self):
        """Initialize the exporter."""
        self.config = CatalogConfig()
        self._client = None

    def configure(self, **kwargs) -> "CatalogExporter":
        """Configure the exporter.

        Args:
            **kwargs: Configuration options (host, token, service_name, etc.)

        Returns:
            self for method chaining
        """
        self.config = CatalogConfig.from_dict(kwargs)
        return self

    @abstractmethod
    def export(self, metadata_path: str) -> CatalogExportResult:
        """Export REDI metadata to the catalog.

        Args:
            metadata_path: Path to REDI metadata.json file

        Returns:
            CatalogExportResult with export status
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to the catalog.

        Returns:
            True if connection is valid
        """
        pass

    def _log(self, message: str, level: str = "info"):
        """Log a message.

        Args:
            message: Message to log
            level: Log level (debug, info, warning, error)
        """
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{self.__class__.__name__}] {message}")

    def _create_error_result(self, message: str) -> CatalogExportResult:
        """Create an error result.

        Args:
            message: Error message

        Returns:
            CatalogExportResult with success=False
        """
        return CatalogExportResult(
            success=False,
            catalog_type=self.catalog_type.value if self.catalog_type else "unknown",
            message=message,
        )
