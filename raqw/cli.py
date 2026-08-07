"""Command-line interface for the publication-ready RAQW workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from . import __version__
from .comparison import fetch_riversp_table, run_comparison
from .config import RAQWConfig
from .provenance import write_manifest
from .reaches import build_valid_discharge_reaches, download_south_america_sos
from .workflow import fit_stage, preprocess_stage


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="TOML run configuration.")
    parser.add_argument("--reach-id")
    parser.add_argument("--year", type=int)
    parser.add_argument("--reach-file", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--reach-layer")
    parser.add_argument("--reach-id-field")
    parser.add_argument("--projected-crs")


def config_from_arguments(args: argparse.Namespace) -> RAQWConfig:
    override_names = (
        "reach_id",
        "year",
        "reach_file",
        "run_root",
        "reach_layer",
        "reach_id_field",
        "projected_crs",
    )
    overrides = {
        name: getattr(args, name, None)
        for name in override_names
        if getattr(args, name, None) is not None
    }
    if args.config is not None:
        return RAQWConfig.from_toml(args.config, **overrides)

    required = ("reach_id", "year")
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Provide --config or the required arguments: {flags}")
    overrides.setdefault("run_root", Path("outputs").resolve())
    overrides.setdefault("reach_id_field", "reach_id")
    if "reach_file" in overrides:
        overrides["reach_file"] = Path(overrides["reach_file"]).resolve()
    overrides["run_root"] = Path(overrides["run_root"]).resolve()
    return RAQWConfig(**overrides)


def preprocess_command(args: argparse.Namespace) -> None:
    cfg = config_from_arguments(args)
    if args.raw_dir is None:
        nc_paths = None
    else:
        nc_paths = sorted(Path(args.raw_dir).glob("*.nc"))
        if not nc_paths:
            raise FileNotFoundError(f"No .nc files were found in {args.raw_dir}.")
    summary, slopes = preprocess_stage(
        cfg,
        login=not args.no_login,
        nc_paths=nc_paths,
    )
    print(
        f"Preprocessed {slopes['file'].nunique():,} accepted overpasses "
        f"from {len(summary):,} candidate granules."
    )


def fit_command(args: argparse.Namespace) -> None:
    cfg = config_from_arguments(args)
    reference = getattr(args, "reference", None)
    results = fit_stage(
        cfg,
        make_plots=args.plots,
        reference_path=reference,
    )
    print(f"Wrote {len(results):,} results to {cfg.results_dir / 'raqw_results.csv'}")


def run_command(args: argparse.Namespace) -> None:
    preprocess_command(args)
    fit_command(args)


def export_command(args: argparse.Namespace) -> None:
    from .apply import export_observations

    output_dir = args.output_dir or (
        args.run_root / "exports" / f"reach_{args.reach_id}" / args.date
    )
    paths = export_observations(
        run_root=args.run_root.resolve(),
        reach_id=str(args.reach_id),
        date=args.date,
        output_dir=output_dir.resolve(),
    )
    for path in paths:
        print(f"Wrote {path}")


def reach_list_command(args: argparse.Namespace) -> None:
    sos_file = args.sos_file
    if sos_file is None:
        if args.download_dir is None:
            raise ValueError("Provide --sos-file or --download-dir.")
        sos_file = download_south_america_sos(
            args.download_dir,
            login=not args.no_login,
        )
    result = build_valid_discharge_reaches(
        sword_file=args.sword_file,
        sos_file=sos_file,
        output_csv=args.output,
        reach_layer=args.reach_layer,
        reach_id_field=args.reach_id_field,
        algorithm_group=args.algorithm_group,
        discharge_variable=args.discharge_variable,
    )
    print(f"Wrote {len(result):,} reach records to {args.output}")


def compare_command(args: argparse.Namespace) -> None:
    matched, summary = run_comparison(
        filter_results_path=args.filter_results,
        riversp_path=args.riversp_table,
        output_dir=args.output_dir,
        slope_column=args.slope_column,
        slope_scale=args.slope_scale,
        maximum_time_delta_minutes=args.maximum_time_delta_minutes,
    )
    print(
        f"Matched {matched['riversp_slope_m_per_km'].notna().sum():,}/"
        f"{len(matched):,} observations."
    )
    if not summary.empty:
        print(summary.to_string(index=False))


def fetch_riversp_command(args: argparse.Namespace) -> None:
    results = pd.read_csv(args.filter_results, dtype={"reach_id": str})
    table = fetch_riversp_table(
        results,
        collection_name=args.collection_name,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table):,} RiverSP records to {args.output}")


def manifest_command(args: argparse.Namespace) -> None:
    output = write_manifest(
        args.paths,
        args.output,
        root=args.root,
        role=args.role,
    )
    print(f"Wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raqw",
        description="Reproducible SWOT PIXC reach-slope filtering workflow.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Discover or read PIXC granules and build reach-year quantile slopes.",
    )
    add_run_arguments(preprocess)
    preprocess.add_argument("--raw-dir", type=Path, help="Use existing PIXC NetCDF files.")
    preprocess.add_argument("--no-login", action="store_true")
    preprocess.set_defaults(function=preprocess_command)

    fit = subparsers.add_parser("fit", help="Fit a reach-year reference and select tau windows.")
    add_run_arguments(fit)
    fit.add_argument("--plots", action="store_true")
    fit.set_defaults(function=fit_command, reference=None)

    apply_reference = subparsers.add_parser(
        "apply-reference",
        help="Apply a previously fitted reference envelope to another year.",
    )
    add_run_arguments(apply_reference)
    apply_reference.add_argument("--reference", required=True, type=Path)
    apply_reference.add_argument("--plots", action="store_true")
    apply_reference.set_defaults(function=fit_command)

    run = subparsers.add_parser("run", help="Run preprocessing followed by reference fitting.")
    add_run_arguments(run)
    run.add_argument("--raw-dir", type=Path)
    run.add_argument("--no-login", action="store_true")
    run.add_argument("--plots", action="store_true")
    run.set_defaults(function=run_command, reference=None)

    export = subparsers.add_parser(
        "export",
        help="Export retained-point flags for one processed observation date.",
    )
    export.add_argument("--reach-id", required=True)
    export.add_argument("--date", required=True)
    export.add_argument("--run-root", type=Path, default=Path("outputs"))
    export.add_argument("--output-dir", type=Path)
    export.set_defaults(function=export_command)

    reach_list = subparsers.add_parser(
        "reach-list",
        help="Select SWORD reaches having valid DAWG/SOS discharge.",
    )
    reach_list.add_argument("--sword-file", required=True, type=Path)
    reach_list.add_argument("--sos-file", type=Path)
    reach_list.add_argument("--download-dir", type=Path)
    reach_list.add_argument("--output", required=True, type=Path)
    reach_list.add_argument("--reach-layer")
    reach_list.add_argument("--reach-id-field", default="reach_id")
    reach_list.add_argument("--algorithm-group", default="consensus")
    reach_list.add_argument("--discharge-variable", default="allq")
    reach_list.add_argument("--no-login", action="store_true")
    reach_list.set_defaults(function=reach_list_command)

    fetch_riversp = subparsers.add_parser(
        "fetch-riversp",
        help="Retrieve RiverSP observations for processed PIXC reach-year windows.",
    )
    fetch_riversp.add_argument("--filter-results", required=True, type=Path)
    fetch_riversp.add_argument("--output", required=True, type=Path)
    fetch_riversp.add_argument(
        "--collection-name",
        default="SWOT_L2_HR_RiverSP_D",
    )
    fetch_riversp.add_argument("--timeout-seconds", type=int, default=60)
    fetch_riversp.set_defaults(function=fetch_riversp_command)

    compare = subparsers.add_parser(
        "compare",
        help="Match filtered observations to a RiverSP table and make comparison outputs.",
    )
    compare.add_argument("--filter-results", required=True, type=Path)
    compare.add_argument("--riversp-table", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    compare.add_argument("--slope-column", default="riversp_slope_m_per_km")
    compare.add_argument("--slope-scale", type=float, default=1.0)
    compare.add_argument("--maximum-time-delta-minutes", type=int, default=30)
    compare.set_defaults(function=compare_command)

    manifest = subparsers.add_parser("manifest", help="Write SHA-256 provenance for files.")
    manifest.add_argument("paths", nargs="+", type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("--root", type=Path)
    manifest.add_argument("--role", default="input")
    manifest.set_defaults(function=manifest_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
