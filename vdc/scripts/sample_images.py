import argparse
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
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings
from vdc.sampling.allocation import BaseAllocator
from vdc.sampling.allocation import LRMAllocator
from vdc.sampling.allocation import WaterFillingAllocator
from vdc.sampling.base_sampler import BaseSampler
from vdc.sampling.cluster import ClusterInfo
from vdc.sampling.hierarchical_random_sampler import HierarchicalRandomSampler

logger = logging.getLogger(__name__)

SAMPLERS: dict[str, type[BaseSampler]] = {
    "random": HierarchicalRandomSampler,
}
ALLOCATORS: dict[str, type[BaseAllocator]] = {
    "lrm": LRMAllocator,
    "water-filling": WaterFillingAllocator,
}


def _process_deletion_action_for_unselected(file_path: str, backup_dir: Optional[str]) -> dict[str, Any]:
    status = utils.perform_file_deletion_with_backup(file_path, backup_dir)
    return {
        "file_path": file_path,
        "action": "deleted_unselected",
        "remediation_status": status,
    }


# pylint: disable=too-many-locals
def sample_images(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False and args.use_existing_sampled_list is False:
        logger.warning(f"Output CSV already exists at: {args.output_csv}, use --force to overwrite")
        return

    assignments_df = pl.read_csv(args.assignments_csv)
    cluster_info = ClusterInfo(assignments_df)

    logger.info(f"Loading clustering assignments from: {args.assignments_csv}")
    logger.info(f"Total samples in dataset: {cluster_info.total_samples:,}")
    logger.info(
        "Cluster hierarchy: "
        f"{[cluster_info.get_num_clusters_at_level(i) for i in range(cluster_info.get_max_level() + 1)]}"
    )
    if args.total_samples is not None:
        logger.info(
            f"Requested sample size: {args.total_samples:,} "
            f"({100 * args.total_samples / cluster_info.total_samples:.1f}%)"
        )
    logger.info(f"Sampling strategy: {args.sampling_strategy}")
    logger.info(f"Allocation strategy: {args.allocation_strategy}")

    if args.apply_deletion is True:
        logger.info("Running with --apply-deletion, unselected files will be processed")
        if args.backup_dir is not None:
            logger.info(f"Original unselected files will be backed up to: {args.backup_dir}")
        else:
            logger.warning("WARNING: No backup directory specified, deleted files will be unrecoverable")
    else:
        logger.info("Running in DRY-RUN mode. No files will be deleted")

    logger.info(f"Output sampled list will be saved to: {args.output_csv}")

    if args.total_samples is not None and args.total_samples >= cluster_info.total_samples:
        logger.warning(
            f"Requested total samples ({args.total_samples:,}) is greater than or equal to the "
            f"total number of available samples ({cluster_info.total_samples:,}) in the dataset. "
            "No sampling will be performed as this would effectively select all samples. Aborting execution."
        )
        return

    tic = time.time()
    if args.use_existing_sampled_list is True:
        logger.info(f"Loading existing sampled CSV from: {args.output_csv}")
        existing_df = pl.read_csv(args.output_csv)
        selected_samples = existing_df.get_column("sample").to_list()
        total_samples = len(selected_samples)
    else:
        allocator_class = ALLOCATORS[args.allocation_strategy]
        allocator = allocator_class()
        sampler_class = SAMPLERS[args.sampling_strategy]
        sampler = sampler_class(allocator)

        selected_samples = sampler.sample(
            cluster_info=cluster_info, total_samples=args.total_samples, random_seed=args.random_seed
        )

        # Inform if fewer samples were collected than requested
        if len(selected_samples) < args.total_samples:
            logger.warning(
                f"Could only collect {len(selected_samples):,} samples, which is less than "
                f"the requested {args.total_samples:,}"
            )

        output_df = pl.DataFrame({"sample": selected_samples})
        output_df.write_csv(args.output_csv)
        total_samples = args.total_samples

    total_files_deleted = 0
    total_files_skipped = 0
    if args.apply_deletion is True:
        all_available_samples = set(assignments_df.get_column("sample").to_list())
        selected_samples_set = set(selected_samples)
        unselected_samples_for_deletion = list(all_available_samples - selected_samples_set)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_process_deletion_action_for_unselected, file_path, args.backup_dir): file_path
                for file_path in unselected_samples_for_deletion
            }

            with tqdm(
                total=len(unselected_samples_for_deletion),
                desc="Deleting unselected images",
                leave=False,
                unit="files",
            ) as progress_bar:
                for future in as_completed(futures):
                    original_file_path = futures.pop(future)
                    try:
                        result = future.result()
                        if result["remediation_status"] == "deleted":
                            total_files_deleted += 1
                        else:
                            total_files_skipped += 1

                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        if isinstance(exc, KeyboardInterrupt):
                            raise
                        progress_bar.write(f"An error occurred processing {original_file_path}: {exc}")
                        total_files_skipped += 1

                    progress_bar.update()

    toc = time.time()
    rate = len(selected_samples) / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to sample {len(selected_samples):,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Sampling complete. Selected {len(selected_samples):,} samples (requested: {total_samples:,})")
    logger.info(f"Unselected files deleted: {total_files_deleted:,}")
    logger.info(f"Unselected files skipped: {total_files_skipped:,}")
    logger.info(f"Sampled list saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Sampling Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Perform hierarchical sampling from pre-clustered embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.sample_images --total-samples 10000\n"
            "python -m vdc.scripts.sample_images --total-samples 50000 --output-csv "
            "my_sampled_list.csv --assignments-csv results/my_assignments.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Sampling parameters
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=list(SAMPLERS.keys()),
        help="sampling strategy to use",
    )
    sampling_group.add_argument(
        "--allocation-strategy",
        type=str,
        choices=list(ALLOCATORS.keys()),
        help="allocation strategy to use for distributing samples across clusters",
    )
    sampling_group.add_argument(
        "--total-samples",
        type=int,
        metavar="N",
        help="total number of samples to select from the entire dataset",
    )
    sampling_group.add_argument(
        "--random-seed",
        type=int,
        metavar="SEED",
        help="random seed for reproducibility of sampling",
    )
    sampling_group.add_argument(
        "--use-existing-sampled-list",
        action="store_true",
        help="if set, read the sampled images from --output-csv instead of performing new sampling",
    )

    # Deletion parameters
    deletion_group = parser.add_argument_group("Deletion parameters")
    deletion_group.add_argument(
        "--apply-deletion",
        action="store_true",
        help="delete images *not* selected by the sampling process (USE WITH CAUTION)",
    )
    deletion_group.add_argument(
        "--backup-dir", type=str, metavar="DIR", help="backup directory for original unselected files before deletion"
    )
    deletion_group.add_argument("--no-backup", action="store_true", help="disable backup creation for deleted files")

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
        "--output-csv", type=str, metavar="FILE", help="output CSV file containing the list of sampled image paths"
    )
    parser.add_argument(
        "--assignments-csv", type=str, metavar="FILE", help="path to the hierarchical K-Means assignments CSV file"
    )

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
        "backup_dir": str(backup_dir),
        "output_csv": str(project_dir.joinpath("hierarchical_sampled_samples.csv")),
        "assignments_csv": str(project_dir.joinpath("hierarchical_kmeans_assignments.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        sampling_config = config.get("hierarchical_sampling", {})
        parser.set_defaults(**sampling_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    if args.use_existing_sampled_list is False:
        if args.total_samples is None:
            raise ValueError("--total-samples is required when --use-existing-sampled-list is NOT set")
    else:
        if args.total_samples is not None:
            logger.warning("--total-samples will be ignored when using existing sampled list")

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

    sample_images(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
