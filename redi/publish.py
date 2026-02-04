"""
REDI Catalog Publishing Module.

Integrates with SetGo to publish processed datasets to data catalogs
(HuggingFace, OpenMetadata, etc.) while maintaining separation of concerns:
- REDI handles data transformation
- SetGo handles metadata and catalog publishing
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    catalog_type: str
    entity_url: Optional[str] = None
    entity_id: Optional[str] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "catalog_type": self.catalog_type,
            "entity_url": self.entity_url,
            "entity_id": self.entity_id,
            "message": self.message,
        }


def check_setgo_available() -> bool:
    """Check if SetGo is installed and available."""
    try:
        import setgo
        return True
    except ImportError:
        return False


def get_available_catalogs() -> list:
    """Get list of available catalog types from SetGo."""
    if not check_setgo_available():
        return []

    try:
        from setgo.catalog import list_catalogs
        return list_catalogs()
    except Exception as e:
        logger.warning(f"Could not list catalogs: {e}")
        return ["huggingface", "openmetadata"]  # Known defaults


def publish_to_catalog(
    output_dir: str,
    catalog_type: str = "huggingface",
    repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> PublishResult:
    """
    Publish a REDI output directory to a data catalog.

    This function delegates to SetGo for the actual publishing,
    maintaining separation of concerns between data transformation (REDI)
    and metadata/catalog management (SetGo).

    Args:
        output_dir: Path to REDI output directory containing metadata.json
        catalog_type: Type of catalog ("huggingface", "openmetadata", etc.)
        repo_id: Repository/dataset ID (required for HuggingFace)
        private: Whether to make the dataset private
        token: Authentication token (or use HF_TOKEN env var)
        **kwargs: Additional catalog-specific options

    Returns:
        PublishResult with success status and details

    Example:
        >>> result = publish_to_catalog(
        ...     "./output",
        ...     catalog_type="huggingface",
        ...     repo_id="my-org/climate-data",
        ...     private=True
        ... )
        >>> if result.success:
        ...     print(f"Published to: {result.entity_url}")
    """
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"

    # Check if SetGo is available
    if not check_setgo_available():
        return PublishResult(
            success=False,
            catalog_type=catalog_type,
            message="SetGo is not installed. Install with: pip install setgo"
        )

    # Check if metadata.json exists
    if not metadata_path.exists():
        return PublishResult(
            success=False,
            catalog_type=catalog_type,
            message=f"metadata.json not found in {output_dir}. Run REDI pipeline first."
        )

    # Import SetGo catalog module
    try:
        from setgo.catalog import export_to_catalog
    except ImportError as e:
        return PublishResult(
            success=False,
            catalog_type=catalog_type,
            message=f"Failed to import SetGo catalog module: {e}"
        )

    # Get token from environment if not provided
    if token is None:
        if catalog_type == "huggingface":
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        elif catalog_type == "openmetadata":
            token = os.environ.get("OPENMETADATA_TOKEN")

    # Validate required parameters
    if catalog_type == "huggingface" and not repo_id:
        return PublishResult(
            success=False,
            catalog_type=catalog_type,
            message="repo_id is required for HuggingFace. Use --repo <org/dataset-name>"
        )

    # Build config for SetGo
    config = {
        "private": private,
        **kwargs
    }

    if repo_id:
        config["repo_id"] = repo_id
    if token:
        config["token"] = token

    # Call SetGo to export
    try:
        logger.info(f"Publishing to {catalog_type}...")
        result = export_to_catalog(
            str(metadata_path),
            catalog_type=catalog_type,
            **config
        )

        return PublishResult(
            success=result.success,
            catalog_type=catalog_type,
            entity_url=result.entity_url,
            entity_id=result.entity_id,
            message=result.message
        )

    except Exception as e:
        logger.error(f"Failed to publish to {catalog_type}: {e}")
        return PublishResult(
            success=False,
            catalog_type=catalog_type,
            message=str(e)
        )


def publish_to_huggingface(
    output_dir: str,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
) -> PublishResult:
    """
    Convenience function to publish to HuggingFace Hub.

    Args:
        output_dir: Path to REDI output directory
        repo_id: HuggingFace repo ID (e.g., "org/dataset-name")
        private: Whether to make the dataset private
        token: HuggingFace token (or use HF_TOKEN env var)

    Returns:
        PublishResult with success status and URL
    """
    return publish_to_catalog(
        output_dir=output_dir,
        catalog_type="huggingface",
        repo_id=repo_id,
        private=private,
        token=token
    )


def run_fair_assessment(output_dir: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Run FAIR assessment on REDI output using SetGo.

    Args:
        output_dir: Path to REDI output directory
        domain: Optional domain hint (climate, materials, etc.)

    Returns:
        FAIR score dictionary or None if SetGo not available
    """
    if not check_setgo_available():
        logger.warning("SetGo not installed, skipping FAIR assessment")
        return None

    try:
        from setgo import assess_fair
        score = assess_fair(output_dir, domain=domain)
        return {
            "overall_score": score.overall_score,
            "findable": score.findable,
            "accessible": score.accessible,
            "interoperable": score.interoperable,
            "reusable": score.reusable,
            "recommendations": score.recommendations if hasattr(score, 'recommendations') else []
        }
    except Exception as e:
        logger.error(f"FAIR assessment failed: {e}")
        return None


def format_publish_result(result: PublishResult, use_color: bool = True) -> str:
    """Format publish result for CLI output."""
    if use_color:
        GREEN = '\033[92m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        END = '\033[0m'
    else:
        GREEN = RED = BOLD = DIM = END = ''

    lines = []

    if result.success:
        lines.append(f"{GREEN}✓ Published to {result.catalog_type}{END}")
        if result.entity_url:
            lines.append(f"  {BOLD}URL:{END} {result.entity_url}")
        if result.entity_id:
            lines.append(f"  {DIM}ID: {result.entity_id}{END}")
    else:
        lines.append(f"{RED}✗ Failed to publish to {result.catalog_type}{END}")
        lines.append(f"  {result.message}")

    return "\n".join(lines)
