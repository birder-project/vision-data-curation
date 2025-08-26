import argparse
import logging
import os
import time
from pathlib import Path

import polars as pl
from birder.common import cli
from birder.common.lib import format_duration

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


def sample_images(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False:
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
    logger.info(
        f"Requested sample size: {args.total_samples:,} "
        f"({100 * args.total_samples / cluster_info.total_samples:.1f}%)"
    )
    logger.info(f"Sampling strategy: {args.sampling_strategy}")
    logger.info(f"Allocation strategy: {args.allocation_strategy}")
    logger.info(f"Output sampled list will be saved to: {args.output_csv}")

    if args.total_samples >= cluster_info.total_samples:
        logger.warning(
            f"Requested total samples ({args.total_samples:,}) is greater than or equal to the "
            f"total number of available samples ({cluster_info.total_samples:,}) in the dataset. "
            "No sampling will be performed as this would effectively select all samples. Aborting execution."
        )
        return

    allocator_class = ALLOCATORS[args.allocation_strategy]
    allocator = allocator_class()
    sampler_class = SAMPLERS[args.sampling_strategy]
    sampler = sampler_class(allocator)

    tic = time.time()
    selected_samples = sampler.sample(
        cluster_info=cluster_info, total_samples=args.total_samples, random_seed=args.random_seed
    )
    toc = time.time()

    # Inform if fewer samples were collected than requested
    if len(selected_samples) < args.total_samples:
        logger.warning(
            f"Could only collect {len(selected_samples):,} samples, which is less than "
            f"the requested {args.total_samples:,}"
        )

    output_df = pl.DataFrame({"sample": selected_samples})
    output_df.write_csv(args.output_csv)

    toc = time.time()
    rate = len(selected_samples) / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to sample {len(selected_samples):,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Sampling complete. Selected {len(selected_samples):,} samples (requested: {args.total_samples:,})")
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
        required=True,
        metavar="N",
        help="total number of samples to select from the entire dataset",
    )
    sampling_group.add_argument(
        "--random-seed",
        type=int,
        metavar="SEED",
        help="random seed for reproducibility of sampling",
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--force", action="store_true", help="override existing output report")
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
    else:
        project_dir = settings.RESULTS_DIR

    default_paths = {
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
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    sample_images(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
