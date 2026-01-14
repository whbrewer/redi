"""
REDI Catalog Module

Provides integration with data catalogs for registering REDI-processed datasets.
Supports multiple catalog backends through a pluggable architecture.

Supported Catalogs:
- OpenMetadata: Open-source metadata platform
- MLflow: ML experiment tracking (planned)
- Intake: Lightweight Python catalog (planned)

Example:
    from redi.catalog import export_to_catalog, get_catalog_exporter

    # Export using registry
    result = export_to_catalog(
        metadata_path="output/metadata.json",
        catalog_type="openmetadata",
        host="http://localhost:8585/api",
        token="your-jwt-token"
    )

    # Or use exporter directly
    exporter = get_catalog_exporter("openmetadata")
    exporter.configure(host="http://localhost:8585/api", token="...")
    result = exporter.export(metadata_path)
"""

from .base import (
    CatalogExporter,
    CatalogExportResult,
    CatalogConfig,
    CatalogType,
)

# Registry of available catalog exporters
_CATALOG_REGISTRY: dict = {}


def register_catalog(name: str, exporter_class: type):
    """Register a catalog exporter class.

    Args:
        name: Catalog type name (e.g., "openmetadata", "mlflow")
        exporter_class: CatalogExporter subclass
    """
    _CATALOG_REGISTRY[name.lower()] = exporter_class


def get_catalog_exporter(catalog_type: str, **config) -> CatalogExporter:
    """Get a catalog exporter instance.

    Args:
        catalog_type: Type of catalog ("openmetadata", "mlflow", etc.)
        **config: Configuration options passed to the exporter

    Returns:
        Configured CatalogExporter instance

    Raises:
        ValueError: If catalog type is not supported
    """
    catalog_type = catalog_type.lower()

    if catalog_type not in _CATALOG_REGISTRY:
        # Try to import and register on-demand
        if catalog_type == "openmetadata":
            from .openmetadata import OpenMetadataExporter
            register_catalog("openmetadata", OpenMetadataExporter)
        elif catalog_type == "mlflow":
            from .mlflow import MLflowExporter
            register_catalog("mlflow", MLflowExporter)
        elif catalog_type == "intake":
            from .intake import IntakeExporter
            register_catalog("intake", IntakeExporter)
        elif catalog_type == "huggingface":
            from .huggingface import HuggingFaceExporter
            register_catalog("huggingface", HuggingFaceExporter)
        else:
            available = list(_CATALOG_REGISTRY.keys()) or ["openmetadata", "mlflow", "intake", "huggingface"]
            raise ValueError(
                f"Unknown catalog type: '{catalog_type}'. "
                f"Available: {', '.join(available)}"
            )

    exporter_class = _CATALOG_REGISTRY[catalog_type]
    exporter = exporter_class()

    if config:
        exporter.configure(**config)

    return exporter


def list_catalogs() -> list:
    """List available catalog types.

    Returns:
        List of supported catalog type names
    """
    # Include both registered and known importable catalogs
    known = {"openmetadata", "mlflow", "intake", "huggingface"}
    registered = set(_CATALOG_REGISTRY.keys())
    return sorted(known | registered)


def export_to_catalog(
    metadata_path: str,
    catalog_type: str = "openmetadata",
    **config
) -> CatalogExportResult:
    """Export REDI metadata to a data catalog.

    Convenience function that gets an exporter and runs the export.

    Args:
        metadata_path: Path to REDI metadata.json file
        catalog_type: Type of catalog to export to
        **config: Catalog-specific configuration

    Returns:
        CatalogExportResult with export status and details

    Example:
        result = export_to_catalog(
            "output/metadata.json",
            catalog_type="openmetadata",
            host="http://localhost:8585/api",
            token="your-jwt-token"
        )
        if result.success:
            print(f"Exported as: {result.entity_id}")
    """
    exporter = get_catalog_exporter(catalog_type, **config)
    return exporter.export(metadata_path)


__all__ = [
    # Base classes
    "CatalogExporter",
    "CatalogExportResult",
    "CatalogConfig",
    "CatalogType",
    # Registry functions
    "register_catalog",
    "get_catalog_exporter",
    "list_catalogs",
    "export_to_catalog",
]
