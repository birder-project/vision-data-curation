import argparse
import csv
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import polars as pl
from birder.common import cli
from birder.common.lib import format_duration
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings
from vdc.deduplication.dsu import DSU

logger = logging.getLogger(__name__)


def _process_deduplication_action(
    file_path: str, action: Literal["keep", "delete"], backup_dir: Optional[str]
) -> dict[str, Any]:
    remediation_status = "no_action_taken"
    if action == "delete":
        remediation_status = utils.perform_file_deletion_with_backup(file_path, backup_dir)
    elif action == "keep":
        remediation_status = "kept"

    return {
        "file_path": file_path,
        "action": action,
        "remediation_status": remediation_status,
    }


# pylint: disable=too-many-branches,too-many-locals
def apply_deduplication(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    logger.info(f"Applying deduplication from report: {args.deduplication_report_csv}")
    logger.info(f"Selection strategy: {args.selection_strategy} | Threshold: < {args.threshold}")
    if args.apply_deletion is True:
        logger.info("Running with --apply-deletion, files marked for deletion will be processed")
        if args.backup_dir is not None:
            logger.info(f"Original files will be backed up to: {args.backup_dir}")
        else:
            logger.warning("WARNING: No backup directory specified, deleted files will be unrecoverable")
    else:
        logger.info("Running in DRY-RUN mode. No files will be modified or deleted")

    logger.info(f"Output report will be saved to: {args.output_csv}")
    logger.info(f"Using {args.num_workers} worker processes")

    if args.random_seed is not None:
        random.seed(args.random_seed)
        logger.debug(f"Random seed set to: {args.random_seed}")

    tic = time.time()
    dedup_report_df = pl.read_csv(args.deduplication_report_csv)
    logger.info(f"Initial pairs in report: {len(dedup_report_df):,}")
    if args.threshold is not None:
        dedup_report_df = dedup_report_df.filter(pl.col("distance") < args.threshold)

    logger.info(f"Pairs remaining after threshold filtering ({args.threshold}): {len(dedup_report_df):,}")
    if dedup_report_df.is_empty():
        logger.info("No pairs remain after filtering by threshold, no deduplication actions to take")
        return

    # Rebuild the connected components (duplicate groups) using DSU.
    # The 'group_id' from the initial report is from a previous run with a potentially different threshold.
    # By filtering the report with the new --threshold, we create a new set of valid duplicate links.
    # DSU is used here to identify the actual connected components based on these filtered links,
    # ensuring that if a strict threshold splits an original large group into smaller ones, they are treated distinctly.
    all_involved_samples = set(dedup_report_df.get_column("sample_id_1").to_list())
    all_involved_samples.update(dedup_report_df.get_column("sample_id_2").to_list())
    dsu = DSU(list(all_involved_samples))
    for row in dedup_report_df.iter_rows(named=True):
        dsu.union(row["sample_id_1"], row["sample_id_2"])

    all_components = dsu.get_components()

    # Filter for actual duplicate groups (size > 1) after DSU reconstruction
    duplicate_groups: dict[int, list[str]] = {}
    group_id_counter = 0
    for members_list in all_components.values():
        if len(members_list) > 1:
            duplicate_groups[group_id_counter] = sorted(members_list)
            group_id_counter += 1

    logger.info(f"Identified {group_id_counter:,} duplicate groups after threshold and component reconstruction")
    if len(duplicate_groups) == 0:
        logger.info("No duplicate groups found after consolidating report and threshold, no actions to take")
        return

    files_to_process: dict[str, Literal["keep", "delete"]] = {}
    for members in duplicate_groups.values():
        if args.selection_strategy == "alphabetical":
            keeper = members[0]  # Members are already sorted alphabetically
            files_to_process[keeper] = "keep"
            for member in members:
                if member != keeper:
                    files_to_process[member] = "delete"

        elif args.selection_strategy == "random":
            keeper = random.choice(members)
            files_to_process[keeper] = "keep"
            for member in members:
                if member != keeper:
                    files_to_process[member] = "delete"

        else:
            raise ValueError(f"Unknown selection strategy: {args.selection_strategy}")

    if args.apply_deletion is False:
        deleted_count = sum(1 for action in files_to_process.values() if action == "delete")
        kept_count = sum(1 for action in files_to_process.values() if action == "keep")
        logger.info("Dry-run complete, no files were modified or deleted")
        logger.info(f"Would keep {kept_count:,} files and delete {deleted_count:,} files.")
        return

    total_files_deleted = 0
    total_files_skipped = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["file_path", "action", "remediation_status"])
        csv_writer.writeheader()

        files_for_deletion = [
            (file_path, action, args.backup_dir) for file_path, action in files_to_process.items() if action == "delete"
        ]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_process_deduplication_action, file_path, action, backup_dir): file_path
                for file_path, action, backup_dir in files_for_deletion
            }

            with tqdm(
                total=len(files_for_deletion), desc="Applying deduplication actions", leave=False, unit="files"
            ) as progress_bar:
                for future in as_completed(futures):
                    original_file_path = futures.pop(future)
                    try:
                        result = future.result()
                        csv_writer.writerow(result)

                        if result["remediation_status"] == "deleted":
                            total_files_deleted += 1
                        else:  # file_not_found, skipped_backup_exists, backup_failed_error, error
                            total_files_skipped += 1

                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        if isinstance(exc, KeyboardInterrupt):
                            raise
                        progress_bar.write(f"An error occurred processing {original_file_path}: {exc}")
                        total_files_skipped += 1

                    progress_bar.update()

    toc = time.time()
    rate = len(files_to_process) / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {len(files_to_process):,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Files deleted: {total_files_deleted:,}")
    logger.info(f"Files skipped/errored: {total_files_skipped:,}")
    logger.info(f"Deduplication actions report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Deduplication Action Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Apply deduplication actions based on a generated report",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.apply_deduplication --deduplication-report-csv results/deduplication_report.csv "
            "--selection-strategy alphabetical --apply-deletion --backup-dir data/deleted_duplicates\n"
            "python -m vdc.scripts.apply_deduplication --deduplication-report-csv results/deduplication_report.csv "
            "--selection-strategy random --apply-deletion -j 16\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Deduplication action parameters
    action_group = parser.add_argument_group("Deduplication Action Parameters")
    action_group.add_argument(
        "--deduplication-report-csv",
        type=str,
        metavar="FILE",
        help="path to the input CSV report generated by deduplicate_images.py",
    )
    action_group.add_argument(
        "--selection-strategy",
        type=str,
        choices=["alphabetical", "random"],
        help="strategy to select the 'keeper' from a group of duplicates",
    )
    action_group.add_argument(
        "--threshold",
        type=float,
        metavar="TH",
        help="distance threshold, only pairs with distance < this value will be considered duplicates",
    )
    action_group.add_argument("--random-seed", type=int, metavar="SEED", help="random seed for reproducibility")

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
        "output_csv": str(project_dir.joinpath("deduplication_actions_report.csv")),
        "backup_dir": str(backup_dir),
        "deduplication_report_csv": str(project_dir.joinpath("deduplication_report.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        apply_config = config.get("apply_deduplication", {})
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

    apply_deduplication(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
