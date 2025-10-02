import argparse
import csv
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl
from birder.common import cli
from birder.common.lib import format_duration
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)

ROTATION_ANGLES = [0, 90, 180, 270]


def _perform_rotation_with_backup(
    file_path: str, angle: int, confidence: float, backup_dir: Optional[str], apply_rotation: bool
) -> dict[str, Any]:
    """
    Performs image rotation with an optional backup of the original file

    Parameters
    ----------
    file_path
        The path to the image file to be rotated.
    angle
        The rotation angle in degrees (clockwise: 0, 90, 180, 270).
    confidence
        Confidence score for the rotation.
    backup_dir
        An optional path to a directory where the original file should be backed up.
        If None, no backup is performed.
    apply_rotation
        Whether to actually apply the rotation.

    Returns
    -------
    A dictionary with rotation results.
    """

    result = {
        "file_path": file_path,
        "rotation_angle": angle,
        "confidence": confidence,
        "remediation_status": "no_action",
    }

    if apply_rotation is False:
        result["remediation_status"] = "dry_run"
        return result

    original_path = Path(file_path)

    backup_successful = False
    if backup_dir is not None:
        backup_path = utils.build_backup_path(original_path, backup_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            logger.error(f"Backup file already exists: {backup_path}, skipping rotation")
            result["remediation_status"] = "skipped_backup_exists"
            return result
        try:
            shutil.copy2(original_path, backup_path)
            logger.debug(f"Backed up {original_path} to {backup_path}")
            backup_successful = True
        except OSError as e:
            logger.error(
                f"Failed to backup {original_path} to {backup_path} (Error: {type(e).__name__} - {e}). "
                "Rotation will NOT proceed for this file to prevent data loss."
            )
            result["remediation_status"] = "backup_failed_error"
            return result

    try:
        with Image.open(original_path) as img:
            rotated_img = img.rotate(-angle, expand=True)
            rotated_img.save(original_path)

        logger.debug(f"ROTATED: {file_path} by {angle} degrees (backup created: {backup_successful})")
        result["remediation_status"] = "rotated"
        return result
    except (OSError, UnidentifiedImageError) as e:  # pylint: disable=overlapping-except
        logger.error(
            f"Error processing image {file_path} (corruption/format error): {e}. "
            "Original file state is preserved if backup succeeded, otherwise potentially corrupted."
        )
        result["remediation_status"] = "error_rotation_failed"
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            f"An unexpected error occurred during rotation of {file_path} (Error: {type(e).__name__} - {e}). "
            "Original file state is preserved if backup succeeded, otherwise potentially corrupted."
        )
        result["remediation_status"] = "error_rotation_failed"
        return result


# pylint: disable=too-many-branches
def apply_rotation_correction(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) and not args.force:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    logger.info(f"Loading rotation predictions from: {args.predictions_path}")
    logger.info(f"Confidence threshold: {args.threshold}")
    if args.apply_rotation is True:
        logger.info("Running with --apply-correction, files will be modified")
        if args.backup_dir is not None:
            logger.info(f"Original files will be backed up to: {args.backup_dir}")
        else:
            logger.warning("WARNING: No backup directory specified")
    else:
        logger.info("Running in DRY-RUN mode. No files will be modified")

    logger.info(f"Output report will be saved to: {args.output_csv}")
    logger.info(f"Using {args.num_workers} worker processes")

    # Load predictions
    if Path(args.predictions_path).suffix == ".csv":
        predictions_df = pl.read_csv(args.predictions_path)
    elif Path(args.predictions_path).suffix == ".parquet":
        predictions_df = pl.read_parquet(args.predictions_path)
    else:
        raise ValueError(
            f"Unsupported input report format: {Path(args.predictions_path).suffix}, must be .csv or .parquet"
        )

    total_files = len(predictions_df)
    rotated_count = 0
    skipped_count = 0
    error_count = 0

    tic = time.time()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["file_path", "rotation_angle", "confidence", "remediation_status"]
        )
        csv_writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for row in predictions_df.iter_rows(named=True):
                probs = {str(col): row[str(col)] for col in ROTATION_ANGLES}
                max_rotation = max(probs, key=probs.get)  # type: ignore[arg-type]
                max_prob = probs[max_rotation]

                if max_rotation != "0" and max_prob > args.threshold:
                    angle = int(max_rotation)

                    future = executor.submit(
                        _perform_rotation_with_backup,
                        row["sample"],
                        angle,
                        max_prob,
                        args.backup_dir,
                        args.apply_rotation,
                    )
                    futures[future] = row["sample"]
                else:
                    skipped_count += 1

            with tqdm(desc="Processing images", leave=False, unit="files") as progress_bar:
                for future in as_completed(futures):
                    original_file_path = futures.pop(future)
                    try:
                        result = future.result()
                        csv_writer.writerow(result)

                        if result["remediation_status"] == "rotated":
                            rotated_count += 1
                        elif result["remediation_status"] == "dry_run":
                            skipped_count += 1
                        else:
                            error_count += 1

                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        if isinstance(exc, KeyboardInterrupt):
                            # Re-raise KeyboardInterrupt so the main program exits
                            raise

                        progress_bar.write(f"An error occurred processing {original_file_path}: {exc}")
                        error_count += 1
                        csv_writer.writerow(
                            {
                                "file_path": original_file_path,
                                "rotation_angle": 0,
                                "confidence": 0.0,
                                "remediation_status": "error",
                            }
                        )

                    progress_bar.update()

    toc = time.time()
    rate = total_files / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {total_files:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Rotated: {rotated_count:,}")
    logger.info(f"Skipped: {skipped_count:,}")
    logger.info(f"Errors: {error_count:,}")
    logger.info(f"Report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Rotation Correction Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Apply rotation correction to images",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.rotation_correction --threshold 0.8 --apply-correction "
            "--backup-dir data/rotated_backups results/rotation_probs.csv\n"
            "python -m vdc.scripts.rotation_correction --threshold 0.9 -j 16 results/rotation_probs.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Rotation Correction parameters
    correction_group = parser.add_argument_group("Rotation Correction Parameters")
    correction_group.add_argument(
        "--threshold",
        type=float,
        metavar="TH",
        help="only apply rotation if the dominant probability is ABOVE this threshold",
    )

    # Remediation options
    remediation_group = parser.add_argument_group("Remediation options")
    remediation_group.add_argument(
        "--apply-rotation", action="store_true", help="apply rotations to images (USE WITH CAUTION)"
    )
    remediation_group.add_argument(
        "--backup-dir", type=str, metavar="DIR", help="backup directory for original files before rotation"
    )
    remediation_group.add_argument(
        "--no-backup", action="store_true", help="disable backup creation (files will be modified without backup)"
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
    parser.add_argument("predictions_path", help="path to the file containing rotation predictions")

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
        "output_csv": str(project_dir.joinpath("rotation_correction_report.csv")),
        "backup_dir": str(backup_dir),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        rotation_config = config.get("rotation_correction", {})
        parser.set_defaults(**rotation_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if not output_dir.exists():
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_backup is True:
        args.backup_dir = None
    if args.apply_rotation is True and args.backup_dir is not None:
        backup_dir = Path(args.backup_dir)
        if backup_dir.exists() is False:
            logger.info(f"Creating {backup_dir} directory...")
            backup_dir.mkdir(parents=True)

    apply_rotation_correction(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
