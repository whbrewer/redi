#!/usr/bin/env python3
"""
REDI Demo Script

Demonstrates REDI capabilities across different scientific domains.

Usage:
    python demo.py --domain hpc       # HPC job accounting data
    python demo.py --domain climate   # Climate/NetCDF data
    python demo.py --domain materials # Materials science/AFLOW data
    python demo.py --domain bio       # Life sciences/protein data
    python demo.py --all              # Run all domain demos

    # With real data:
    python demo.py -i data.csv -d hpc           # Process HPC job data
    python demo.py -i climate.nc -d climate     # Process NetCDF file
    python demo.py -i proteins.fasta -d bio     # Process FASTA file
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}>>> {text}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def create_hpc_sample(temp_dir: Path) -> Path:
    """Create sample HPC job accounting data."""
    csv_path = temp_dir / "hpc_jobs.csv"
    csv_path.write_text("""JOB_ID,USER,PROJECT,QUEUE,NODES_REQUESTED,WALLTIME_SECONDS,RUNTIME_SECONDS,EXIT_CODE,QUEUED_TIMESTAMP,START_TIMESTAMP,END_TIMESTAMP,CORES_USED,GPU_COUNT
1001,user_alpha,proj_climate,batch,64,14400,13800,0,2024-01-15 08:00:00,2024-01-15 08:30:00,2024-01-15 12:20:00,4096,0
1002,user_beta,proj_ml,gpu,8,7200,7000,0,2024-01-15 09:00:00,2024-01-15 09:15:00,2024-01-15 11:11:40,512,32
1003,user_alpha,proj_climate,batch,128,28800,28500,0,2024-01-15 10:00:00,2024-01-15 12:00:00,2024-01-15 19:55:00,8192,0
1004,user_gamma,proj_physics,debug,4,3600,1200,1,2024-01-15 11:00:00,2024-01-15 11:05:00,2024-01-15 11:25:00,256,0
1005,user_beta,proj_ml,gpu,16,14400,14200,0,2024-01-15 12:00:00,2024-01-15 14:00:00,2024-01-15 17:56:40,1024,64
1006,user_delta,proj_bio,batch,32,10800,10500,0,2024-01-15 13:00:00,2024-01-15 13:30:00,2024-01-15 16:25:00,2048,0
1007,user_alpha,proj_climate,batch,256,43200,42000,0,2024-01-15 14:00:00,2024-01-15 18:00:00,2024-01-16 05:40:00,16384,0
1008,user_epsilon,proj_chem,gpu,4,7200,6800,0,2024-01-15 15:00:00,2024-01-15 15:10:00,2024-01-15 17:03:20,256,16
1009,user_gamma,proj_physics,batch,16,7200,7100,0,2024-01-15 16:00:00,2024-01-15 16:30:00,2024-01-15 18:28:20,1024,0
1010,user_beta,proj_ml,gpu,32,21600,21000,0,2024-01-15 17:00:00,2024-01-15 19:00:00,2024-01-16 00:50:00,2048,128
""")
    return csv_path


def create_climate_sample(temp_dir: Path) -> Path:
    """Create sample climate data."""
    try:
        import xarray as xr
        import numpy as np

        # Create a realistic climate dataset
        lat = np.linspace(-90, 90, 37)
        lon = np.linspace(0, 357.5, 72)
        time = np.arange('2024-01-01', '2024-01-11', dtype='datetime64[D]')

        # Generate realistic temperature patterns
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        base_temp = 15 - 30 * np.abs(lat_grid) / 90  # Colder at poles

        temperature = np.zeros((10, 37, 72))
        precipitation = np.zeros((10, 37, 72))

        for t in range(10):
            temperature[t] = base_temp + 5 * np.random.randn(37, 72)
            precipitation[t] = np.maximum(0, 5 + 10 * np.random.randn(37, 72))

        ds = xr.Dataset(
            {
                "temperature": (["time", "lat", "lon"], temperature),
                "precipitation": (["time", "lat", "lon"], precipitation),
            },
            coords={
                "time": time,
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "title": "Sample Climate Reanalysis Data",
                "Conventions": "CF-1.6",
                "source": "REDI Demo",
            }
        )
        ds["temperature"].attrs = {"units": "degC", "long_name": "2m Temperature"}
        ds["precipitation"].attrs = {"units": "mm/day", "long_name": "Total Precipitation"}

        nc_path = temp_dir / "climate_sample.nc"
        ds.to_netcdf(nc_path)
        return nc_path

    except ImportError:
        # Fallback to CSV if xarray not available
        csv_path = temp_dir / "climate_sample.csv"
        csv_path.write_text("""lat,lon,temperature,precipitation,pressure,humidity,wind_speed,timestamp
-45.0,170.0,8.5,12.2,1013.2,75,8.5,2024-01-01
-30.0,150.0,18.3,5.1,1015.5,60,6.2,2024-01-01
-15.0,120.0,26.1,18.3,1010.0,85,4.1,2024-01-01
0.0,90.0,28.5,22.5,1008.5,90,3.5,2024-01-01
15.0,60.0,25.2,15.8,1011.2,78,5.2,2024-01-01
30.0,30.0,15.7,8.5,1018.3,55,7.8,2024-01-01
45.0,0.0,5.2,10.0,1020.1,65,9.5,2024-01-01
60.0,-30.0,-5.5,6.5,1015.8,50,12.3,2024-01-01
""")
        return csv_path


def create_materials_sample(temp_dir: Path) -> Path:
    """Create sample materials science data (AFLOW-style)."""
    csv_path = temp_dir / "materials_aflow.csv"
    csv_path.write_text("""auid,compound,formula,spacegroup,crystal_system,band_gap,formation_energy,density,volume_atom,lattice_a,lattice_b,lattice_c
aflow:0001,Silicon,Si,Fd-3m,cubic,1.12,-0.00,2.33,20.02,5.43,5.43,5.43
aflow:0002,Germanium,Ge,Fd-3m,cubic,0.67,-0.00,5.32,22.63,5.66,5.66,5.66
aflow:0003,Gallium Arsenide,GaAs,F-43m,cubic,1.42,-0.74,5.32,22.54,5.65,5.65,5.65
aflow:0004,Aluminum Nitride,AlN,P63mc,hexagonal,6.20,-3.13,3.26,10.58,3.11,3.11,4.98
aflow:0005,Zinc Oxide,ZnO,P63mc,hexagonal,3.37,-3.63,5.61,11.92,3.25,3.25,5.21
aflow:0006,Titanium Dioxide,TiO2,P42/mnm,tetragonal,3.00,-9.73,4.23,15.68,4.59,4.59,2.96
aflow:0007,Copper,Cu,Fm-3m,cubic,0.00,-0.00,8.96,11.81,3.61,3.61,3.61
aflow:0008,Gold,Au,Fm-3m,cubic,0.00,-0.00,19.32,16.96,4.08,4.08,4.08
aflow:0009,Magnesium Oxide,MgO,Fm-3m,cubic,7.80,-6.23,3.58,9.39,4.21,4.21,4.21
aflow:0010,Sodium Chloride,NaCl,Fm-3m,cubic,8.50,-4.26,2.16,22.49,5.64,5.64,5.64
""")
    return csv_path


def create_bio_sample(temp_dir: Path) -> Path:
    """Create sample life sciences data."""
    fasta_path = temp_dir / "proteins.fasta"
    fasta_path.write_text(""">sp|P53_HUMAN|P53 Tumor suppressor p53
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
>sp|BRCA1_HUMAN|BRCA1 Breast cancer type 1 susceptibility protein
MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQ
CPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKD
EVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYI
>sp|MYC_HUMAN|MYC Myc proto-oncogene protein
MPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPP
LSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPD
DETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQD
>sp|EGFR_HUMAN|EGFR Epidermal growth factor receptor
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEV
VLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALA
VLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDF
""")
    return fasta_path


def demo_domain(domain: str, temp_dir: Path, output_dir: Path, output_format: str = "npz", input_file: str = None):
    """Run demo for a specific domain."""
    from redi.analyze import analyze_dataset, generate_report, get_recommendations
    from redi.cli import run_pipeline

    print_header(f"REDI Demo: {domain.upper()} Domain")

    # Use provided input file or create sample data
    if input_file:
        sample_file = Path(input_file)
        if not sample_file.exists():
            print_warning(f"Input file not found: {input_file}")
            return
        print_section("Using provided input file...")
        print_success(f"Input: {sample_file}")
    else:
        print_section("Creating sample data...")

        if domain == "hpc":
            sample_file = create_hpc_sample(temp_dir)
            print_success(f"Created HPC job data: {sample_file.name}")
        elif domain == "climate":
            sample_file = create_climate_sample(temp_dir)
            print_success(f"Created climate data: {sample_file.name}")
        elif domain == "materials":
            sample_file = create_materials_sample(temp_dir)
            print_success(f"Created materials data: {sample_file.name}")
        elif domain == "bio":
            sample_file = create_bio_sample(temp_dir)
            print_success(f"Created protein data: {sample_file.name}")
        else:
            print_warning(f"Unknown domain: {domain}")
            return

    # Step 1: Analyze dataset
    print_section("Step 1: Analyzing dataset...")
    data_card = analyze_dataset(str(sample_file))

    print_info(f"File: {data_card.overview.name}")
    print_info(f"Size: {data_card.overview.size_bytes / 1024:.1f} KB")
    print_info(f"Records: {data_card.overview.num_records:,}")
    print_info(f"Features: {data_card.overview.num_features}")
    print_info(f"Detected Domain: {data_card.detected_domain}")
    print_info(f"AI Readiness Level: {data_card.readiness_level}/5")

    # Step 2: Show recommendations
    print_section("Step 2: Getting pipeline recommendations...")
    recs = get_recommendations(str(sample_file))

    print_info(f"Recommended Pipeline: {recs['pipeline_name']}")
    print_info(f"Tools: {', '.join(recs['recommended_tools'])}")
    print_info(f"Output Formats: {', '.join(recs['output_formats'])}")

    # Step 3: Run pipeline
    print_section(f"Step 3: Running preprocessing pipeline (output: {output_format})...")
    domain_output = output_dir / domain

    ctx = run_pipeline(
        input_path=str(sample_file),
        output_dir=str(domain_output),
        domain=data_card.detected_domain,
        config={"output_format": output_format}
    )

    if ctx.errors:
        print_warning(f"Pipeline completed with {len(ctx.errors)} warnings")
        for err in ctx.errors[:3]:
            print(f"   - {err}")
    else:
        print_success("Pipeline completed successfully!")

    # Step 4: Show outputs
    print_section("Step 4: Output files...")

    if ctx.metadata.get("saved_files"):
        for f in ctx.metadata["saved_files"]:
            fpath = Path(f)
            if fpath.exists():
                size = fpath.stat().st_size
                print_success(f"{fpath.name} ({size/1024:.1f} KB)")

    # Show metadata highlights
    if ctx.metadata.get("split"):
        split = ctx.metadata["split"]
        print_info(f"Data split: train={split.get('train_size', 'N/A')}, "
                  f"val={split.get('val_size', 'N/A')}, "
                  f"test={split.get('test_size', 'N/A')}")

    if ctx.metadata.get("engineered_features"):
        feats = ctx.metadata["engineered_features"]
        print_info(f"Engineered features: {len(feats)} ({', '.join(feats[:3])}...)")

    # Step 5: Generate full report
    print_section("Step 5: Data Card Report")
    print("-" * 60)
    report = generate_report(str(sample_file))
    # Print abbreviated report
    lines = report.split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"\n... ({len(lines) - 50} more lines)")
    print("-" * 60)

    print_success(f"Demo complete for {domain.upper()} domain!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="REDI Demo - Demonstrate data readiness capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demos with sample data:
  python demo.py --domain hpc             # HPC job accounting demo
  python demo.py --domain climate         # Climate/NetCDF demo
  python demo.py --domain materials       # Materials science demo
  python demo.py --domain bio             # Life sciences demo
  python demo.py --all                    # All domains

  # Process real data files:
  python demo.py -i /path/to/jobs.csv -d hpc
  python demo.py -i /path/to/climate.nc -d climate
  python demo.py -i /path/to/structures.csv -d materials
  python demo.py -i /path/to/proteins.fasta -d bio

  # Output options:
  python demo.py -d hpc -o ./output --keep   # Save output to ./output
  python demo.py -d hpc --format parquet     # Output as Parquet files
  python demo.py -d hpc --format csv         # Output as CSV files
        """
    )

    parser.add_argument(
        "-d", "--domain",
        choices=["hpc", "climate", "materials", "bio"],
        default=None,
        help="Domain to demonstrate (required unless --all or --input specified)"
    )

    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Input file to process (uses sample data if not specified)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run demos for all domains"
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: temporary directory)"
    )

    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep output files after demo"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["npz", "parquet", "csv", "hdf5", "zarr"],
        default="npz",
        help="Output format (default: npz)"
    )

    args = parser.parse_args()

    # Check if domain or --all is specified
    if not args.all and not args.domain:
        parser.print_help()
        print(f"\n{Colors.YELLOW}Error: Please specify a domain with -d/--domain or use --all{Colors.END}")
        sys.exit(1)

    # Print banner
    print(f"""
{Colors.BOLD}{Colors.CYAN}
 ██████╗ ███████╗██████╗ ██╗
 ██╔══██╗██╔════╝██╔══██╗██║
 ██████╔╝█████╗  ██║  ██║██║
 ██╔══██╗██╔══╝  ██║  ██║██║
 ██║  ██║███████╗██████╔╝██║
 ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝
 Readiness Engine for Data Integration
{Colors.END}""")

    # Setup directories
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        cleanup = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="redi_demo_"))
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        cleanup = not args.keep

    print_info(f"Output directory: {output_dir}")

    try:
        if args.all:
            domains = ["hpc", "climate", "materials", "bio"]
        elif args.domain:
            domains = [args.domain]
        else:
            # This shouldn't happen due to earlier check, but handle gracefully
            parser.print_help()
            sys.exit(1)

        for domain in domains:
            demo_domain(domain, temp_dir, output_dir, args.format, args.input)
            if len(domains) > 1:
                print("\n" + "=" * 70 + "\n")

        print_header("Demo Complete!")

        if cleanup:
            print_info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print_info(f"Output files saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        raise


if __name__ == "__main__":
    main()
