import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl
from birder.common import cli
from birder.common.lib import format_duration
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


def _delete_image(file_path: str, score: float, backup_dir: Optional[str]) -> dict[str, Any]:
    status = utils.perform_file_deletion_with_backup(file_path, backup_dir)
    return {
        "file_path": file_path,
        "score": score,
        "remediation_status": status,
    }


# pylint: disable=too-many-locals
def _display_statistics(report_df: pl.DataFrame, score_column: str, threshold: float) -> None:
    console = Console()

    total_samples = len(report_df)
    scores = report_df[score_column]
    valid_scores = scores.drop_nulls()
    null_count = total_samples - len(valid_scores)

    if len(valid_scores) == 0:
        console.print("[red]Error: No valid scores found in the data.[/red]")
        return

    # Deletion stats
    samples_to_delete = valid_scores.filter(valid_scores < threshold)
    deletion_count = len(samples_to_delete)
    deletion_percentage = (deletion_count / total_samples) * 100

    # Score summary
    score_min = float(valid_scores.min())  # type: ignore[arg-type]
    score_max = float(valid_scores.max())  # type: ignore[arg-type]
    score_mean = float(valid_scores.mean())  # type: ignore[arg-type]

    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values: list[float] = [valid_scores.quantile(p / 100) for p in percentiles]  # type: ignore[misc]

    # Threshold position in distribution
    threshold_percentile = (valid_scores.filter(valid_scores < threshold).len() / len(valid_scores)) * 100
    threshold_label = f"P{threshold_percentile:.0f}"

    # Output
    console.print("\n[bold]Dataset Summary[/bold]")
    console.print(f"Samples: {total_samples:,} total | {deletion_count:,} below threshold ({deletion_percentage:.1f}%)")
    console.print(f"Scores: {score_min:.3f} to {score_max:.3f} (mean: {score_mean:.3f}) | Threshold: < {threshold}")
    if null_count > 0:
        console.print(f"[yellow]Missing scores: {null_count}[/yellow]")

    # Build percentile table
    percentile_table = Table(show_header=True, header_style="bold", box=None)
    percentile_table.add_column("Percentile", justify="center")
    percentile_table.add_column("Score", justify="right")
    percentile_table.add_column("vs Threshold", justify="center")
    percentile_table.add_column("Action", justify="center")

    # Build rows
    table_rows = []
    replaced = False
    for p, val in zip(percentiles, percentile_values):
        label = f"P{p}"
        if label == threshold_label and not replaced:
            table_rows.append((label, threshold, True))
            replaced = True
        else:
            table_rows.append((label, float(val), False))

    if replaced is False:
        table_rows.append((threshold_label, threshold, True))

    table_rows.sort(key=lambda x: x[1])
    for label, val, is_threshold in table_rows:
        if is_threshold is True:
            percentile_table.add_row(label, f"{val:.4f}", "-", "Threshold")
        else:
            if val < threshold:
                vs_threshold = f"-{threshold - val:.3f}"
                action = "Delete"
                style = "red"
            else:
                vs_threshold = f"+{val - threshold:.3f}"
                action = "Keep"
                style = "green"

            percentile_table.add_row(
                label,
                f"{val:.4f}",
                f"[{style}]{vs_threshold}[/{style}]",
                f"[{style}]{action}[/{style}]",
            )

    console.print("\n[bold]Score Distribution[/bold]")
    console.print(percentile_table)
    console.print()


def apply_score_filter(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    logger.info(f"Applying filter from report: {args.report_csv}")
    logger.info(f"Score column: '{args.score_column}' | Threshold: < {args.threshold}")
    if args.apply_deletion is True:
        logger.info("Running with --apply-deletion, files will be deleted")
        if args.backup_dir is not None:
            logger.info(f"Original files will be backed up to: {args.backup_dir}")
        else:
            logger.warning("WARNING: No backup directory specified, deleted files will be unrecoverable")
    else:
        logger.info("Running in DRY-RUN mode. No files will be modified or deleted")

    logger.info(f"Output report will be saved to: {args.output_csv}")
    logger.info(f"Using {args.num_workers} worker processes")

    report_df = pl.scan_csv(args.report_csv).select("sample", args.score_column).collect()
    _display_statistics(report_df, args.score_column, args.threshold)

    if args.apply_deletion is False:
        logger.info("Dry-run complete, no files were modified or deleted")
        return

    tic = time.time()
    deleted_count = 0
    skipped_count = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["file_path", "score", "remediation_status"])
        csv_writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_delete_image, row["sample"], row[args.score_column], args.backup_dir): row["sample"]
                for row in report_df.filter(pl.col(args.score_column) < args.threshold).iter_rows(named=True)
            }

            with tqdm(desc="Processing images for deletion", leave=False, unit="files") as progress_bar:
                for future in as_completed(futures):
                    original_file_path = futures.pop(future)
                    try:
                        result = future.result()
                        csv_writer.writerow(result)
                        if result["remediation_status"] == "deleted":
                            deleted_count += 1
                        else:
                            skipped_count += 1

                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        if isinstance(exc, KeyboardInterrupt):
                            # Re-raise KeyboardInterrupt so the main program exits
                            raise

                        # This catches exceptions that occurred during the retrieval of the result,
                        # not necessarily within the worker function itself.
                        progress_bar.write(f"An error occurred processing {original_file_path}: {exc}")
                        skipped_count += 1

                    progress_bar.update()

    toc = time.time()
    rate = deleted_count / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {deleted_count:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Samples skipped: {skipped_count:,}")
    logger.info(f"Report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Filter by Score Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Delete images based on scores from a content filter report",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.apply_score_filter --report-csv results/aesthetic_filter_report.csv "
            "--score-column aesthetic_score --threshold 4.25\n"
            "python -m vdc.scripts.apply_score_filter --report-csv results/nsfw_filter_report.csv "
            "--score-column nsfw_score --threshold 0.1 --apply-deletion --backup-dir data/deleted_backups -j 16\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Remediation options
    remediation_group = parser.add_argument_group("Remediation options")
    remediation_group.add_argument(
        "--apply-deletion", action="store_true", help="delete images matching the filter (USE WITH CAUTION)"
    )
    remediation_group.add_argument(
        "--backup-dir", type=str, metavar="DIR", help="backup directory for original files before remediation"
    )
    remediation_group.add_argument(
        "--no-backup", action="store_true", help="disable backup creation (files will be deleted without backup)"
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--force", action="store_true", help="override existing output report")
    parser.add_argument("-j", "--num-workers", type=int, default=8, metavar="N", help="number of workers")
    parser.add_argument(
        "--report-csv",
        type=str,
        required=True,
        metavar="FILE",
        help="path to the input CSV report generated by a content filter",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        required=True,
        metavar="COL",
        help="name of the column in the report CSV containing the score to filter by",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        metavar="TH",
        help="images with scores BELOW this value will be targeted for deletion",
    )
    parser.add_argument("--output-csv", type=str, metavar="FILE", help="output CSV file for the actions taken report")

    return (config_parser, parser)


def parse_args() -> argparse.Namespace:
    (config_parser, parser) = get_args_parser()
    (args_config, remaining) = config_parser.parse_known_args()

    if args_config.config is None:
        logger.debug("No user config file specified. Loading default bundled config")
        config = utils.load_default_bundled_config()
    else:
        config = utils.read_json(args_config.config)

    if args_config.project is not None:
        project_dir = settings.RESULTS_DIR.joinpath(args_config.project)
        backup_dir = settings.DATA_DIR.joinpath("backup").joinpath(args_config.project)
    else:
        project_dir = settings.RESULTS_DIR
        backup_dir = settings.DATA_DIR.joinpath("backup")

    default_paths = {
        "output_csv": str(project_dir.joinpath("score_filter_actions_report.csv")),
        "backup_dir": str(backup_dir),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        apply_config = config.get("apply_score_filter", {})
        parser.set_defaults(**apply_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_backup is True:
        args.backup_dir = None
    if args.apply_deletion is True and args.backup_dir is not None:
        backup_dir = Path(args.backup_dir)
        if backup_dir.exists() is False:
            logger.info(f"Creating {backup_dir} directory...")
            backup_dir.mkdir(parents=True)

    apply_score_filter(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
