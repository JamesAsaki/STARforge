from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atx_tenx_count",
        description="Run the STARforge ATX/10x barcode-counting pipeline end to end.",
    )
    parser.add_argument("--sample-name", required=True, help="Sample name used in output filenames.")
    parser.add_argument("--r1-fastq", required=True, type=Path, help="Path to the R1 FASTQ.gz file.")
    parser.add_argument("--r2-fastq", required=True, type=Path, help="Path to the R2 FASTQ.gz file.")
    parser.add_argument("--cell-barcodes", required=True, type=Path, help="Path to the 10x cell barcode whitelist.")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for output parquet files, plots, and the pipeline log.",
    )
    parser.add_argument("--fb-ref", type=Path, default=None, help="Feature barcode whitelist CSV.")
    parser.add_argument("--mhc-ref", type=Path, default=None, help="MHC barcode whitelist CSV.")
    parser.add_argument("--pep-ref", type=Path, default=None, help="Peptide barcode whitelist CSV.")
    parser.add_argument("--max-reads", type=int, default=None, help="Optional parse cap for validation or sampling.")
    parser.add_argument(
        "--skip-save-intermediates",
        action="store_true",
        help="Skip writing Stage 01-03 parquet outputs; final Stage 04 outputs are still written.",
    )
    parser.add_argument(
        "--skip-graphs",
        action="store_true",
        help="Skip generating all plots while still printing stage metrics.",
    )
    parser.add_argument("--plot-dpi", type=int, default=60, help="DPI for generated plots.")
    parser.add_argument(
        "--require-strict-constants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require fixed R2 constants during parsing (default: true).",
    )
    parser.add_argument("--umi-collapse-max-passes", type=int, default=3, help="Max passes for UMI collapse.")
    parser.add_argument("--umi-min-ratio", type=float, default=5.0, help="Minimum ratio for UMI collapse.")
    parser.add_argument(
        "--collision-min-ratio",
        type=float,
        default=5.0,
        help="Minimum top/second ratio to keep the top feature in a collision.",
    )
    parser.add_argument(
        "--collision-min-top-reads",
        type=int,
        default=3,
        help="Minimum reads for the top feature when resolving collisions.",
    )
    parser.add_argument(
        "--min-avg-q",
        type=float,
        default=99.05,
        help="Minimum average cell and UMI quality required to keep an aggregated observation.",
    )
    return parser


def args_to_config(args: argparse.Namespace):
    from .pipeline import build_pipeline_config

    return build_pipeline_config(
        sample_name=args.sample_name,
        r1_fastq=args.r1_fastq,
        r2_fastq=args.r2_fastq,
        cell_barcodes=args.cell_barcodes,
        out_dir=args.out_dir,
        fb_ref=args.fb_ref,
        mhc_ref=args.mhc_ref,
        pep_ref=args.pep_ref,
        max_reads_parse=args.max_reads,
        save_intermediates=not args.skip_save_intermediates,
        generate_graphs=not args.skip_graphs,
        plot_dpi=args.plot_dpi,
        require_strict_constants=args.require_strict_constants,
        umi_collapse_max_passes=args.umi_collapse_max_passes,
        umi_min_ratio=args.umi_min_ratio,
        collision_min_ratio=args.collision_min_ratio,
        collision_min_top_reads=args.collision_min_top_reads,
        min_avg_q=args.min_avg_q,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    from .pipeline import run_pipeline

    run_pipeline(config)
    return 0
