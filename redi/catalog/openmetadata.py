"""
OpenMetadata catalog exporter for REDI.

Exports REDI-processed dataset metadata to OpenMetadata as ML Model entities,
including features, hyperparameters, and lineage information.

Requirements:
    pip install openmetadata-ingestion

Example:
    from redi.catalog.openmetadata import OpenMetadataExporter

    exporter = OpenMetadataExporter()
    exporter.configure(
        host="http://localhost:8585/api",
        token="your-jwt-token",
        service_name="redi"
    )

    result = exporter.export("output/metadata.json")
    print(f"Created: {result.entity_id}")
"""

import logging
from typing import Any, Dict, List, Optional

from .base import (
    CatalogExporter,
    CatalogExportResult,
    CatalogType,
    REDIMetadata,
)

logger = logging.getLogger(__name__)


class OpenMetadataExporter(CatalogExporter):
    """Export REDI metadata to OpenMetadata.

    Creates ML Model entities in OpenMetadata with:
    - Dataset features as MlFeatures
    - Processing parameters as MlHyperParameters
    - Split sizes and normalization info
    - Links to source data (if available)

    Attributes:
        catalog_type: CatalogType.OPENMETADATA
    """

    catalog_type = CatalogType.OPENMETADATA

    def __init__(self):
        """Initialize the OpenMetadata exporter."""
        super().__init__()
        self._metadata_client = None

    def _get_client(self):
        """Get or create OpenMetadata client.

        Returns:
            OpenMetadata client instance

        Raises:
            ImportError: If openmetadata-ingestion is not installed
            ConnectionError: If connection to server fails
        """
        if self._metadata_client is not None:
            return self._metadata_client

        try:
            from metadata.ingestion.ometa.ometa_api import OpenMetadata
            from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
                OpenMetadataConnection,
                AuthProvider,
            )
            from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
                OpenMetadataJWTClientConfig,
            )
        except ImportError:
            raise ImportError(
                "OpenMetadata SDK not installed. Install with:\n"
                "  pip install openmetadata-ingestion"
            )

        if not self.config.host:
            raise ValueError("OpenMetadata host URL is required")

        if not self.config.token:
            raise ValueError("OpenMetadata JWT token is required")

        # Create connection config
        server_config = OpenMetadataConnection(
            hostPort=self.config.host,
            authProvider=AuthProvider.openmetadata,
            securityConfig=OpenMetadataJWTClientConfig(
                jwtToken=self.config.token
            ),
            verifySSL=self.config.verify_ssl,
        )

        self._metadata_client = OpenMetadata(server_config)
        return self._metadata_client

    def validate_connection(self) -> bool:
        """Validate connection to OpenMetadata server.

        Returns:
            True if connection is valid and authenticated
        """
        try:
            client = self._get_client()
            # Try to get server health or version
            health = client.health_check()
            return health
        except Exception as e:
            self._log(f"Connection validation failed: {e}", "error")
            return False

    def export(self, metadata_path: str) -> CatalogExportResult:
        """Export REDI metadata to OpenMetadata.

        Creates an ML Model entity representing the REDI-processed dataset.

        Args:
            metadata_path: Path to REDI metadata.json file

        Returns:
            CatalogExportResult with export status and entity details
        """
        try:
            # Load REDI metadata
            redi_meta = REDIMetadata.from_file(metadata_path)
            self._log(f"Loaded metadata for: {redi_meta.name}")

            # Get OpenMetadata client
            client = self._get_client()

            # Ensure ML Model service exists
            service_fqn = self._ensure_ml_service(client)

            # Create ML Model entity
            ml_model = self._create_ml_model(client, redi_meta, service_fqn)

            # Build entity URL
            entity_url = ""
            if self.config.host and ml_model:
                base_url = self.config.host.replace("/api", "")
                entity_url = f"{base_url}/mlmodel/{ml_model.fullyQualifiedName.__root__}"

            return CatalogExportResult(
                success=True,
                catalog_type=self.catalog_type.value,
                entity_id=str(ml_model.fullyQualifiedName.__root__),
                entity_url=entity_url,
                message=f"Successfully exported '{redi_meta.name}' to OpenMetadata",
                metadata={
                    "name": redi_meta.name,
                    "domain": redi_meta.domain,
                    "features_count": len(redi_meta.feature_columns),
                    "service": service_fqn,
                },
            )

        except ImportError as e:
            return self._create_error_result(str(e))
        except FileNotFoundError as e:
            return self._create_error_result(f"Metadata file not found: {e}")
        except Exception as e:
            self._log(f"Export failed: {e}", "error")
            return self._create_error_result(f"Export failed: {str(e)}")

    def _ensure_ml_service(self, client) -> str:
        """Ensure ML Model service exists in OpenMetadata.

        Args:
            client: OpenMetadata client

        Returns:
            Fully qualified name of the ML service
        """
        from metadata.generated.schema.api.services.createMlModelService import (
            CreateMlModelServiceRequest,
        )
        from metadata.generated.schema.entity.services.mlmodelService import (
            MlModelService,
            MlModelServiceType,
            MlModelConnection,
        )
        from metadata.generated.schema.entity.services.connections.mlmodel.customMlModelConnection import (
            CustomMlModelConnection,
            CustomMlModelType,
        )

        service_name = self.config.service_name

        # Check if service exists
        try:
            existing = client.get_by_name(
                entity=MlModelService,
                fqn=service_name,
            )
            if existing:
                self._log(f"Using existing ML service: {service_name}")
                return service_name
        except Exception:
            pass  # Service doesn't exist, create it

        # Create the service
        self._log(f"Creating ML service: {service_name}")

        create_request = CreateMlModelServiceRequest(
            name=service_name,
            serviceType=MlModelServiceType.CustomMlModel,
            description="REDI AI-Ready Dataset Service",
            connection=MlModelConnection(
                config=CustomMlModelConnection(
                    type=CustomMlModelType.CustomMlModel,
                    sourcePythonClass="redi.catalog.openmetadata.OpenMetadataExporter",
                )
            ),
        )

        client.create_or_update(create_request)
        return service_name

    def _create_ml_model(self, client, redi_meta: REDIMetadata, service_fqn: str):
        """Create ML Model entity in OpenMetadata.

        Args:
            client: OpenMetadata client
            redi_meta: REDI metadata
            service_fqn: ML service FQN

        Returns:
            Created MlModel entity
        """
        from metadata.generated.schema.api.data.createMlModel import (
            CreateMlModelRequest,
        )
        from metadata.generated.schema.entity.data.mlmodel import (
            MlFeature,
            MlHyperParameter,
            FeatureType,
        )

        # Build ML features from REDI feature columns
        ml_features = self._build_ml_features(redi_meta)

        # Build hyperparameters from processing config
        ml_hyperparams = self._build_hyperparameters(redi_meta)

        # Create description
        description = self._build_description(redi_meta)

        # Create the ML Model request
        create_request = CreateMlModelRequest(
            name=redi_meta.name,
            displayName=f"REDI: {redi_meta.name}",
            description=description,
            algorithm="AI-Ready Dataset",
            target=redi_meta.target_column,
            mlFeatures=ml_features if ml_features else None,
            mlHyperParameters=ml_hyperparams if ml_hyperparams else None,
            service=service_fqn,
        )

        self._log(f"Creating ML Model: {redi_meta.name}")
        ml_model = client.create_or_update(create_request)

        return ml_model

    def _build_ml_features(self, redi_meta: REDIMetadata) -> List:
        """Build MlFeature list from REDI metadata.

        Args:
            redi_meta: REDI metadata

        Returns:
            List of MlFeature objects
        """
        from metadata.generated.schema.entity.data.mlmodel import (
            MlFeature,
            FeatureType,
        )

        features = []
        feature_types = redi_meta.get_feature_types()

        for col in redi_meta.feature_columns[:100]:  # Limit to 100 features
            # Determine feature type
            col_type = feature_types.get(col, "numerical")
            feature_type = (
                FeatureType.numerical
                if col_type == "numerical"
                else FeatureType.categorical
            )

            # Get normalization info for description
            norm_info = ""
            if col in redi_meta.normalization_stats:
                stats = redi_meta.normalization_stats[col]
                norm_info = f" (mean={stats.get('mean', 0):.4f}, std={stats.get('std', 1):.4f})"

            features.append(
                MlFeature(
                    name=col,
                    dataType=feature_type,
                    description=f"{col_type.capitalize()} feature{norm_info}",
                )
            )

        return features

    def _build_hyperparameters(self, redi_meta: REDIMetadata) -> List:
        """Build MlHyperParameter list from REDI metadata.

        Args:
            redi_meta: REDI metadata

        Returns:
            List of MlHyperParameter objects
        """
        from metadata.generated.schema.entity.data.mlmodel import MlHyperParameter

        params = []

        # Add domain
        params.append(MlHyperParameter(name="domain", value=redi_meta.domain))

        # Add split sizes
        if redi_meta.train_size:
            params.append(
                MlHyperParameter(name="train_size", value=str(redi_meta.train_size))
            )
        if redi_meta.val_size:
            params.append(
                MlHyperParameter(name="val_size", value=str(redi_meta.val_size))
            )
        if redi_meta.test_size:
            params.append(
                MlHyperParameter(name="test_size", value=str(redi_meta.test_size))
            )

        # Add data shape
        if redi_meta.num_rows:
            params.append(
                MlHyperParameter(name="num_samples", value=str(redi_meta.num_rows))
            )
        if redi_meta.num_columns:
            params.append(
                MlHyperParameter(name="num_features", value=str(redi_meta.num_columns))
            )

        # Add output format
        if redi_meta.output_format:
            params.append(
                MlHyperParameter(name="output_format", value=redi_meta.output_format)
            )

        # Add completed stages
        if redi_meta.completed_stages:
            params.append(
                MlHyperParameter(
                    name="pipeline_stages",
                    value=", ".join(redi_meta.completed_stages),
                )
            )

        return params

    def _build_description(self, redi_meta: REDIMetadata) -> str:
        """Build markdown description for the ML Model.

        Args:
            redi_meta: REDI metadata

        Returns:
            Markdown description string
        """
        lines = [
            f"## REDI-Processed Dataset",
            "",
            f"**Domain:** {redi_meta.domain}",
            f"**Source:** `{redi_meta.input_file}`",
            "",
            "### Data Summary",
            f"- **Samples:** {redi_meta.num_rows:,}",
            f"- **Features:** {redi_meta.num_columns}",
        ]

        if redi_meta.target_column:
            lines.append(f"- **Target:** {redi_meta.target_column}")

        if redi_meta.train_size:
            lines.extend([
                "",
                "### Train/Val/Test Split",
                f"- **Train:** {redi_meta.train_size:,}",
                f"- **Validation:** {redi_meta.val_size:,}",
                f"- **Test:** {redi_meta.test_size:,}",
            ])

        if redi_meta.saved_files:
            lines.extend([
                "",
                "### Output Files",
            ])
            for f in redi_meta.saved_files[:5]:
                lines.append(f"- `{f}`")
            if len(redi_meta.saved_files) > 5:
                lines.append(f"- ... and {len(redi_meta.saved_files) - 5} more")

        lines.extend([
            "",
            "---",
            "*Generated by [REDI](https://github.com/your-org/redi)*",
        ])

        return "\n".join(lines)


def export_to_openmetadata(
    metadata_path: str,
    host: str,
    token: str,
    service_name: str = "redi",
    **kwargs,
) -> CatalogExportResult:
    """Convenience function to export to OpenMetadata.

    Args:
        metadata_path: Path to REDI metadata.json
        host: OpenMetadata server URL (e.g., "http://localhost:8585/api")
        token: JWT authentication token
        service_name: Name for the ML service
        **kwargs: Additional configuration options

    Returns:
        CatalogExportResult

    Example:
        result = export_to_openmetadata(
            "output/metadata.json",
            host="http://localhost:8585/api",
            token="eyJ...",
        )
    """
    exporter = OpenMetadataExporter()
    exporter.configure(
        host=host,
        token=token,
        service_name=service_name,
        **kwargs,
    )
    return exporter.export(metadata_path)
