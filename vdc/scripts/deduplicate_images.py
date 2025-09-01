import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl
import torch
from birder.common import cli
from birder.common.lib import format_duration
from pt_kmeans import compute_distance  # NOTE: This calculates all combinations and not just "triangle"
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings
from vdc.deduplication.dsu import DSU
from vdc.deduplication.lsh import LSHIndex
from vdc.sampling.cluster import ClusterInfo

logger = logging.getLogger(__name__)


def write_report(
    output_csv: str,
    reported_pairs_with_distances: dict[frozenset[str], float],
    dsu: DSU,
    root_to_group_id: dict[str, int],
) -> int:
    total_pairs = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "sample_id_1", "sample_id_2", "distance"])
        for pair, distance in reported_pairs_with_distances.items():
            id1, id2 = list(pair)
            root = dsu.find(id1)
            if root in root_to_group_id:
                writer.writerow([root_to_group_id[root], id1, id2, distance])
                total_pairs += 1

    return total_pairs


def find_duplicate_pairs(
    distances: torch.Tensor,
    group_member_ids: list[str],
    reported_pairs_with_distances: dict[frozenset[str], float],
    report_threshold: Optional[float] = None,
) -> None:
    if report_threshold is not None:
        mask = distances < report_threshold
    else:
        mask = torch.ones_like(distances, dtype=torch.bool)

    mask = torch.triu(mask, diagonal=1)
    (row_indices, col_indices) = torch.where(mask)

    for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
        sample_id_1 = group_member_ids[row_idx]
        sample_id_2 = group_member_ids[col_idx]
        pair = frozenset({sample_id_1, sample_id_2})
        if pair not in reported_pairs_with_distances:
            distance = distances[row_idx, col_idx].item()
            reported_pairs_with_distances[pair] = distance


def write_duplicate_pairs(csv_writer: Any, pairs: list[tuple[str, str, float]]) -> None:
    for sample_id_1, sample_id_2, distance in pairs:
        csv_writer.writerow([sample_id_1, sample_id_2, distance])


# pylint: disable=too-many-locals
def deduplicate_images(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    logger.info(f"Loading embeddings from: {args.embeddings_path}")
    if args.lsh_index is not None:
        logger.info(f"Using LSH index for candidate generation: {args.lsh_index}")
    elif args.assignments_csv is not None:
        logger.info(f"Using K-Means assignments for candidate generation: {args.assignments_csv}")

    logger.info(f"Distance metric: {args.distance_metric} (with report threshold of {args.report_threshold})")
    logger.info(f"Report will be saved to: {args.output_csv}")
    logger.info(f"Using device: {device}")

    tic = time.time()

    all_sample_ids = utils.get_file_samples(args.embeddings_path)
    all_embeddings = torch.from_numpy(utils.read_vector_file(args.embeddings_path))
    sample_to_index = {sample_id: i for i, sample_id in enumerate(all_sample_ids)}

    reported_pairs_with_distances: dict[frozenset[str], float] = {}
    if args.lsh_index is not None:
        lsh_index = LSHIndex.load(args.lsh_index)
        logger.info("Starting LSH-based deduplication...")

        num_groups = lsh_index.num_buckets()
        with tqdm(desc="Processing LSH buckets", total=num_groups, leave=False, unit="buckets") as progress_bar:
            for i in range(lsh_index.num_hash_tables):
                for group_members in lsh_index.table(i).values():
                    if len(group_members) < 2:
                        progress_bar.update()
                        continue

                    indices = [sample_to_index[m] for m in group_members]
                    bucket_emb = all_embeddings[indices]
                    bucket_emb = lsh_index._preprocess_batch(bucket_emb)  # pylint: disable=protected-access
                    distances = compute_distance(
                        bucket_emb, bucket_emb, args.distance_metric, chunk_size=args.chunk_size
                    )

                    find_duplicate_pairs(distances, group_members, reported_pairs_with_distances, args.report_threshold)
                    progress_bar.update()

    elif args.assignments_csv is not None:
        cluster_info = ClusterInfo(pl.read_csv(args.assignments_csv))
        level0_cluster_ids = list(cluster_info.get_top_level_cluster_ids())
        num_groups = len(level0_cluster_ids)
        logger.info("Starting K-Means cluster-based deduplication...")
        with tqdm(desc="Processing K-Means clusters", total=num_groups, leave=False, unit="clusters") as progress_bar:
            for cluster_id in level0_cluster_ids:
                group_members = cluster_info.get_samples_in_level0_cluster(cluster_id)
                if len(group_members) < 2:
                    progress_bar.update()
                    continue

                indices = [sample_to_index[m] for m in group_members]
                cluster_emb = all_embeddings[indices]
                distances = compute_distance(cluster_emb, cluster_emb, args.distance_metric, chunk_size=args.chunk_size)

                find_duplicate_pairs(distances, group_members, reported_pairs_with_distances, args.report_threshold)
                progress_bar.update()

    logger.info(
        f"Identified {len(reported_pairs_with_distances):,} unique duplicate pairs below threshold. "
        "Building connected components..."
    )

    dsu = DSU(all_sample_ids.to_list())
    for pair_frozenset in reported_pairs_with_distances:
        (id1, id2) = list(pair_frozenset)
        dsu.union(id1, id2)

    all_components = dsu.get_components()
    root_to_group_id: dict[str, int] = {}
    group_id_counter = 0

    valid_duplicate_roots = sorted([root for root, members in all_components.items() if len(members) > 1])
    for root in valid_duplicate_roots:
        root_to_group_id[root] = group_id_counter
        group_id_counter += 1

    total_duplicate_groups = group_id_counter
    total_pairs = write_report(args.output_csv, reported_pairs_with_distances, dsu, root_to_group_id)

    toc = time.time()
    logger.info(f"{format_duration(toc - tic)} to process {len(all_sample_ids):,} samples (from {num_groups:,} groups)")
    logger.info(
        f"Found {total_duplicate_groups:,} duplicate groups and reported {total_pairs:,} individual duplicate pairs"
    )
    logger.info(f"Report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Deduplication Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Deduplicate images based on similarity",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.deduplicate_images --lsh-index results/lsh_index.pkl --distance-metric cosine "
            "--report-threshold 0.1 data/dataset_embeddings.parquet\n"
            "python -m vdc.scripts.deduplicate_images --assignments-csv results/hierarchical_kmeans_assignments.csv "
            "--distance-metric l2 --report-threshold 0.5 data/dataset_embeddings.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Deduplication parameters
    deduplication_group = parser.add_argument_group("Deduplication parameters")
    deduplication_group.add_argument(
        "--distance-metric", choices=["l2", "cosine"], metavar="METRIC", help="distance metric to use"
    )
    deduplication_group.add_argument(
        "--lsh-index", type=str, metavar="FILE", help="path to the pre-built LSH index (.pkl)"
    )
    deduplication_group.add_argument(
        "--assignments-csv", type=str, metavar="FILE", help="path to the K-Means assignments CSV file"
    )
    deduplication_group.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="number of embeddings to process in a single batch during iterative loading and preprocessing",
    )
    deduplication_group.add_argument(
        "--report-threshold",
        type=float,
        metavar="TH",
        help="only include samples with distance below this threshold in the report",
    )
    deduplication_group.add_argument("--random-seed", type=int, metavar="SEED", help="random seed for reproducibility")

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument("--force", action="store_true", help="override existing report")
    parser.add_argument("--output-csv", type=str, metavar="FILE", help="output CSV file for deduplication report")
    parser.add_argument("embeddings_path", help="path to embeddings file")

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
        "output_csv": str(project_dir.joinpath("deduplication_report.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        deduplication_config = config.get("deduplication", {})
        parser.set_defaults(**deduplication_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    if args.lsh_index is not None and args.assignments_csv is not None:
        raise ValueError("--lsh-index cannot be used with --assignments-csv")
    if args.lsh_index is None and args.assignments_csv is None:
        raise ValueError("Either --lsh-index or --assignments-csv must be provided")

    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    deduplicate_images(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
