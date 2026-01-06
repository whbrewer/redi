"""
Smoke tests for Climate domain pipeline.

These tests verify that the climate pipeline handles NetCDF files
and climate-specific processing.
"""

import pytest
from pathlib import Path
import tempfile


# Check if xarray is available
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# Check if xesmf is available
try:
    import xesmf
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


@pytest.fixture
def sample_netcdf(temp_dir):
    """Create a sample NetCDF file for testing."""
    if not HAS_XARRAY:
        pytest.skip("xarray not installed")

    import numpy as np

    # Create a simple climate-like dataset
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    time = np.arange(10)

    temperature = 15 + 8 * np.random.randn(10, 37, 72)

    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temperature),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "Sample Climate Data",
            "Conventions": "CF-1.6",
        }
    )
    ds["temperature"].attrs = {"units": "degC", "long_name": "Surface Temperature"}

    nc_path = temp_dir / "sample_climate.nc"
    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def sample_climate_csv(temp_dir):
    """Create a sample climate CSV for testing without xarray."""
    csv_path = temp_dir / "climate.csv"
    csv_path.write_text("""lat,lon,temperature,precipitation,pressure,humidity,timestamp
-45.0,170.0,12.5,5.2,1013.2,65,2024-01-01
-30.0,150.0,22.3,2.1,1015.5,55,2024-01-01
0.0,0.0,28.1,15.3,1010.0,80,2024-01-01
30.0,-90.0,18.7,8.5,1012.8,60,2024-01-01
45.0,-120.0,8.2,12.0,1018.3,70,2024-01-01
60.0,10.0,-2.5,3.5,1020.1,45,2024-01-01
""")
    return csv_path


class TestClimateSmoke:
    """Smoke tests for Climate domain."""

    def test_climate_csv_analyze(self, sample_climate_csv):
        """Test analyzing climate CSV data."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_climate_csv))

        assert data_card.detected_domain == "climate"
        assert data_card.overview.num_records == 6

    @pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
    def test_climate_netcdf_analyze(self, sample_netcdf):
        """Test analyzing NetCDF climate data."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_netcdf))

        assert data_card.detected_domain == "climate"
        assert data_card.overview.num_features > 0
        assert "netcdf" in data_card.characteristics.data_types
        assert data_card.characteristics.spatial_coverage  # Should have lat/lon info

    @pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
    def test_climate_netcdf_pipeline(self, sample_netcdf, temp_dir):
        """Test running climate pipeline on NetCDF."""
        from redi.cli import run_pipeline

        output_dir = temp_dir / "climate_output"
        ctx = run_pipeline(
            input_path=str(sample_netcdf),
            output_dir=str(output_dir),
            domain="climate"
        )

        assert ctx is not None
        assert ctx.domain == "climate"

        # Should have loaded data
        if ctx.data is not None:
            # Check metadata was extracted
            assert "variables" in ctx.metadata or "dimensions" in ctx.metadata

    def test_climate_domain_detection_csv(self, sample_climate_csv):
        """Test climate domain detection from CSV columns."""
        from redi.cli import detect_domain

        domain = detect_domain(Path(sample_climate_csv))
        assert domain == "climate"

    @pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
    def test_climate_domain_detection_netcdf(self, sample_netcdf):
        """Test climate domain detection from NetCDF extension."""
        from redi.cli import detect_domain

        domain = detect_domain(Path(sample_netcdf))
        assert domain == "climate"


class TestClimateDataCard:
    """Tests for Climate Data Card generation."""

    @pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
    def test_netcdf_dimensions_extracted(self, sample_netcdf):
        """Test that NetCDF dimensions are extracted."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_netcdf))

        # Description should mention dimensions
        assert "dimensions" in data_card.overview.description.lower()

    @pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
    def test_netcdf_spatial_coverage(self, sample_netcdf):
        """Test that spatial coverage is extracted."""
        from redi.analyze import analyze_dataset

        data_card = analyze_dataset(str(sample_netcdf))

        assert data_card.characteristics.spatial_coverage
        assert "Lat" in data_card.characteristics.spatial_coverage

    def test_climate_recommendations(self, sample_climate_csv):
        """Test climate pipeline recommendations."""
        from redi.analyze import get_recommendations

        recs = get_recommendations(str(sample_climate_csv))

        assert recs["domain"] == "climate"
        assert "xarray" in recs["recommended_tools"]
        assert any("Zarr" in fmt or "NPZ" in fmt for fmt in recs["output_formats"])
