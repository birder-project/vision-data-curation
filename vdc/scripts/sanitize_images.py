import argparse
import csv
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Any
from typing import Optional

from birder.common import cli
from birder.common import fs_ops
from birder.common.lib import format_duration
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


class RemediationAction(StrEnum):
    NO_ACTION = "no_action"
    DELETE = "delete"
    RESIZE = "resize"
    RECOMPRESS = "recompress"


ACTION_PRIORITIES = {
    RemediationAction.DELETE: 1,  # Highest priority
    RemediationAction.RESIZE: 2,
    RemediationAction.RECOMPRESS: 3,
    RemediationAction.NO_ACTION: 4,
}


@dataclass
class ValidationResult:
    is_valid: bool
    violation: Optional[str] = None
    action: Optional[RemediationAction] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_priority_value(action: Optional[RemediationAction]) -> int:
    if action is None:
        return ACTION_PRIORITIES[RemediationAction.NO_ACTION]

    return ACTION_PRIORITIES.get(action, ACTION_PRIORITIES[RemediationAction.NO_ACTION])


def resolve_violations(violations: list[ValidationResult]) -> ValidationResult:
    """
    Resolve multiple violations by selecting the highest priority action.

    Returns a single ValidationResult with the highest priority action and combined violation descriptions.
    """

    if len(violations) == 0:
        return ValidationResult(is_valid=True)

    # Sort by priority (lowest number = highest priority)
    violations_by_priority = sorted(violations, key=lambda v: _get_priority_value(v.action))
    primary_violation = violations_by_priority[0]

    # Combine all violation messages
    violation_messages = [v.violation for v in violations if v.violation]
    combined_violation = "; ".join(violation_messages)

    # Merge metadata from all violations
    combined_metadata = {}
    for v in violations:
        combined_metadata.update(v.metadata)

    return ValidationResult(
        is_valid=False, violation=combined_violation, action=primary_violation.action, metadata=combined_metadata
    )


# pylint: disable=too-many-return-statements
def _perform_remediation_action(file_path: str, action: RemediationAction, args: argparse.Namespace) -> bool:
    """
    Executes the specified remediation action on the given file.
    Performs backup if --backup-dir is specified.

    Parameters
    ----------
    file_path
        Path to the image file.
    action
        The remediation action to perform.
    args
        Argparse namespace containing script arguments (e.g., backup_dir, max_width).

    Returns
    -------
    True if the remediation action was successfully performed, False otherwise.
    """

    backup_successful = False
    original_path = Path(file_path)
    if args.backup_dir is not None:
        backup_path = utils.build_backup_path(original_path, args.backup_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists() is True:
            logger.error(f"Backup file already exists: {backup_path}. Skipping remediation.")
            return False

        try:
            shutil.copy2(original_path, backup_path)
            logger.debug(f"Backed up {original_path} to {backup_path}")
            backup_successful = True
        except OSError as e:
            logger.error(
                f"Failed to backup {original_path} to {backup_path} (Error: {type(e).__name__} - {e}). "
                "Remediation will NOT proceed for this file to prevent data loss or unrecoverable state."
            )
            return False

    try:
        if action == RemediationAction.DELETE:
            original_path.unlink()
            logger.info(f"DELETED: {file_path} (backup created: {backup_successful})")
            return True

        if action == RemediationAction.RESIZE:
            with Image.open(original_path) as img:
                (width, height) = img.size
                new_width = width
                new_height = height

                # Calculate new dimensions while maintaining aspect ratio
                if args.max_width is not None and width > args.max_width:
                    new_width = args.max_width
                    new_height = int(new_width * (height / width))

                if args.max_height is not None and new_height > args.max_height:
                    new_height = args.max_height
                    new_width = int(new_height * (width / height))

                resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
                resized_img.save(original_path)
                logger.info(
                    f"RESIZED: {file_path} from ({width}x{height}) to ({new_width}x{new_height}) "
                    f"(backup created: {backup_successful})"
                )

            return True

        if action == RemediationAction.RECOMPRESS:
            # NOTE:
            # This approach does not guarantee that the re-encoded file will meet a specific target size.
            # However, in typical scraping scenarios, images that are significantly larger than expected
            # (given their resolution) are often stored in poorly compressed formats (e.g., BMP, TIFF)
            # or with suboptimal encoding settings.
            # Re-encoding to WebP at a fixed quality level provides a low-cost, high-impact optimization
            # that resolves the majority of such cases without incurring the computational overhead
            # of iterative size targeting.
            with Image.open(original_path) as img:
                new_path = original_path.with_suffix(".webp")
                img.save(new_path, format="WEBP", quality=80, method=6)  # method: 6=slower-better
                new_file_size_mb = os.path.getsize(new_path) / (1024 * 1024)
                logger.info(
                    f"RECOMPRESSED: {file_path} (new size: {new_file_size_mb:.2f}MB, "
                    f"backup created: {backup_successful})"
                )

            if new_path != original_path:
                original_path.unlink()

            return True

        if action == RemediationAction.NO_ACTION:
            logger.debug(f"NO ACTION: {file_path} (backup created: {backup_successful})")
            return True

        raise ValueError(f"Unknown action: {action.value}")

    except (OSError, ValueError) as e:
        logger.error(
            f"Error performing {action.value} on {file_path} (Error: {type(e).__name__} - {e}). "
            "Original file state is now potentially corrupted or partially modified. Check backup if available."
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            f"An unexpected error occurred during {action.value} on {file_path} (Error: {type(e).__name__} - {e}). "
            "Original file state is now potentially corrupted or partially modified. Check backup if available."
        )
        return False


# pylint: disable=too-many-branches
def validate_image(image_path: str, args: argparse.Namespace) -> ValidationResult:
    """
    Validate an image against various criteria including corruption, dimensions,
    aspect ratio, file size, etc.

    Returns ValidationResult with validation status and recommended action.
    """

    collected_violations = []

    # Check file size first
    if args.max_file_size_mb is not None:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > args.max_file_size_mb:
            collected_violations.append(
                ValidationResult(
                    is_valid=False,
                    violation=f"File too large: {file_size_mb:.2f}MB > {args.max_file_size_mb}MB",
                    action=RemediationAction.RECOMPRESS,
                    metadata={"file_size_mb": file_size_mb},
                )
            )

    try:
        with Image.open(image_path) as img:
            if args.check_corruption is True:
                img.verify()
                img = Image.open(image_path)

            # Get image dimensions
            (width, height) = img.size

            # Check dimensions
            if args.min_width is not None:
                if width < args.min_width:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Width too small: {width}px < {args.min_width}px",
                            action=RemediationAction.DELETE,
                            metadata={"current_size": (width, height)},
                        )
                    )

            if args.min_height is not None:
                if height < args.min_height:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Height too small: {height}px < {args.min_height}px",
                            action=RemediationAction.DELETE,
                            metadata={"current_size": (width, height)},
                        )
                    )

            if args.max_width is not None:
                if width > args.max_width:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Width too large: {width}px > {args.max_width}px",
                            action=RemediationAction.RESIZE,
                            metadata={"current_size": (width, height)},
                        )
                    )

            if args.max_height is not None:
                if height > args.max_height:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Height too large: {height}px > {args.max_height}px",
                            action=RemediationAction.RESIZE,
                            metadata={"current_size": (width, height)},
                        )
                    )

            # Check aspect ratio
            aspect_ratio = width / height if height > 0 else float("inf")

            if args.min_aspect_ratio is not None:
                if aspect_ratio < args.min_aspect_ratio:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Aspect ratio too small: {aspect_ratio:.2f} < {args.min_aspect_ratio}",
                            action=RemediationAction.DELETE,
                            metadata={"aspect_ratio": aspect_ratio},
                        )
                    )

            if args.max_aspect_ratio is not None:
                if aspect_ratio > args.max_aspect_ratio:
                    collected_violations.append(
                        ValidationResult(
                            is_valid=False,
                            violation=f"Aspect ratio too large: {aspect_ratio:.2f} > {args.max_aspect_ratio}",
                            action=RemediationAction.DELETE,
                            metadata={"aspect_ratio": aspect_ratio},
                        )
                    )

    except (OSError, UnidentifiedImageError) as e:  # pylint: disable=overlapping-except
        collected_violations.append(
            ValidationResult(
                is_valid=False, violation=f"Corrupted: {type(e).__name__}", action=RemediationAction.DELETE
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        if isinstance(e, KeyboardInterrupt):
            # Re-raise KeyboardInterrupt so the main program exits
            raise

        collected_violations.append(
            ValidationResult(
                is_valid=False, violation=f"Image processing error: {type(e).__name__}", action=RemediationAction.DELETE
            )
        )

    return resolve_violations(collected_violations)


# pylint: disable=too-many-locals
def sanitize_images(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False and args.append is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    if os.path.exists(args.output_csv) is True and args.append is True:
        write_mode = "a"
    else:
        write_mode = "w"

    logger.info(f"Sanitizing images in: {', '.join(args.data_path)}")
    if args.apply_fixes is True:
        logger.info("ATTENTION: Running in destructive mode, '--apply-fixes' is enabled")
    else:
        logger.info("No actions will be taken")

    logger.info(f"Report will be saved to: {args.output_csv}")
    logger.info(f"Using {args.num_workers} worker processes")

    sample_iter = chain.from_iterable(
        fs_ops.file_iter(path, extensions=args.allowed_formats) for path in args.data_path
    )

    total_files_scanned = 0
    total_violations_found = 0
    fixes_applied = 0
    fixes_failed = 0
    total_deleted_file = 0
    tic = time.time()
    with open(args.output_csv, write_mode, newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_mode == "w":
            csv_writer.writerow(["timestamp", "file_name", "violation_summary", "decided_action", "remediation_status"])

        initial_fill_target = args.num_workers * 100
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Phase 1: Initial filling of the worker pool
            # Submit an initial batch of tasks to keep all workers busy immediately.
            initial_fill_count = 0
            active_futures = {}
            try:
                for _ in range(initial_fill_target):
                    image_path = next(sample_iter)
                    future = executor.submit(validate_image, image_path, args)
                    active_futures[future] = image_path
                    initial_fill_count += 1
            except StopIteration:
                logger.debug(
                    f"Fewer than {args.num_workers * 2} files found. Submitted all {initial_fill_count} tasks."
                )

            # Phase 2: Continuous processing
            # Now, we continuously process completed tasks and submit new ones.
            with tqdm(desc="Processing images", leave=False, unit="files") as progress_bar:
                while len(active_futures) > 0:  # pylint: disable=too-many-nested-blocks
                    try:
                        for future in as_completed(active_futures, timeout=1.0):
                            original_file_path = active_futures.pop(future)
                            try:
                                result = future.result()
                                remediation_status = "NoFixAttempted"
                                if result.is_valid is False:
                                    total_violations_found += 1
                                    if args.apply_fixes is True and result.action is not None:
                                        if _perform_remediation_action(original_file_path, result.action, args) is True:
                                            remediation_status = "Applied"
                                            fixes_applied += 1
                                            if result.action == RemediationAction.DELETE:
                                                total_deleted_file += 1
                                        else:
                                            remediation_status = "Failed"
                                            fixes_failed += 1

                                    csv_writer.writerow(
                                        [
                                            datetime.now().isoformat(),
                                            original_file_path,
                                            result.violation,
                                            (
                                                result.action.value
                                                if result.action is not None
                                                else RemediationAction.NO_ACTION.value
                                            ),
                                            remediation_status,
                                        ]
                                    )

                            except Exception as exc:  # pylint: disable=broad-exception-caught
                                if isinstance(exc, KeyboardInterrupt):
                                    # Re-raise KeyboardInterrupt so the main program exits
                                    raise

                                # This catches exceptions that occurred during the retrieval of the result,
                                # not necessarily within the worker function itself.
                                progress_bar.write(f"An error occurred processing {original_file_path}: {exc}")

                            total_files_scanned += 1
                            progress_bar.update()

                            # After processing a completed task, submit a new one if available
                            try:
                                next_image_path = next(sample_iter)
                                new_future = executor.submit(validate_image, next_image_path, args)
                                active_futures[new_future] = next_image_path
                            except StopIteration:
                                pass

                    except TimeoutError:
                        # No futures completed within the timeout, continue loop to re-check
                        pass

            progress_bar.close()

    toc = time.time()
    rate = total_files_scanned / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to scan {total_files_scanned:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Total files scanned: {total_files_scanned:,}")
    logger.info(f"Images with violations found: {total_violations_found:,}")
    if args.apply_fixes is True:
        logger.info(f"Fixes attempted: {fixes_applied + fixes_failed:,}")
        logger.info(f"Fixes applied: {fixes_applied:,} (deleted {total_deleted_file:,})")
        logger.info(f"Fixes failed: {fixes_failed:,}")

    logger.info(f"Report of files with detected violations saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Sanitization Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Sanitize images by removing corrupted files and filtering by size/aspect ratio",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.sanitize_images -j 8 data/raw_data\n"
            "python -m vdc.scripts.sanitize_images --config custom_config.json --apply-fixes data/raw_data\n"
            "python -m vdc.scripts.sanitize_images -j 16 --max-ratio 5.0 data/raw_data /mnt/data/collections\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Image validation overrides
    sanitization_group = parser.add_argument_group("Sanitization parameters")
    sanitization_group.add_argument("--min-width", type=int, metavar="PIXELS", help="minimum image width")
    sanitization_group.add_argument("--min-height", type=int, metavar="PIXELS", help="minimum image height")
    sanitization_group.add_argument("--max-width", type=int, metavar="PIXELS", help="minimum image width")
    sanitization_group.add_argument("--max-height", type=int, metavar="PIXELS", help="minimum image height")
    sanitization_group.add_argument(
        "--min-aspect-ratio", type=float, metavar="RATIO", help="minimum aspect ratio (width/height) allowed"
    )
    sanitization_group.add_argument(
        "--max-aspect-ratio", type=float, metavar="RATIO", help="maximum aspect ratio (width/height) allowed"
    )
    sanitization_group.add_argument("--allowed-formats", nargs="+", metavar="EXT", help="allowed image formats")
    sanitization_group.add_argument("--max-file-size-mb", type=float, metavar="MB", help="maximum file size in MB")
    sanitization_group.add_argument("--check-corruption", action="store_true", help="enable corruption checking")

    # Remediation arguments
    remediation_group = parser.add_argument_group("Remediation options")
    remediation_group.add_argument(
        "--apply-fixes", action="store_true", help="automatically fix issues (USE WITH CAUTION)"
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
    parser.add_argument("--force", action="store_true", help="override existing report")
    parser.add_argument("--append", action="store_true", help="append to an existing report instead of overwriting")
    parser.add_argument("-j", "--num-workers", type=int, default=8, metavar="N", help="number of workers")
    parser.add_argument("--output-csv", type=str, metavar="FILE", help="output CSV file for sanitization report")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")

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
        "output_csv": str(project_dir.joinpath("sanitization_report.csv")),
        "backup_dir": str(backup_dir),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        sanitization_config = config.get("sanitization", {})
        parser.set_defaults(**sanitization_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    if args.force is True and args.append is True:
        raise ValueError("--force cannot be used with --append")

    logger.debug(f"Running with config of: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_backup is True:
        args.backup_dir = None
    if args.apply_fixes is True and args.backup_dir is not None:
        backup_dir = Path(args.backup_dir)
        if backup_dir.exists() is False:
            logger.info(f"Creating {backup_dir} directory...")
            backup_dir.mkdir(parents=True)

    try:
        max_image_pixels = os.environ.get("MAX_IMAGE_PIXELS", None)
        if max_image_pixels is not None:
            Image.MAX_IMAGE_PIXELS = int(max_image_pixels)
    except ValueError:
        pass

    sanitize_images(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
