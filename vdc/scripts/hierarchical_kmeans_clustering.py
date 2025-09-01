import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import polars as pl
import pt_kmeans
import torch
from birder.common import cli
from birder.common.lib import format_duration

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


def _save_hierarchical_clusters_to_csv(hierarchical_clusters: list[dict[str, torch.Tensor]], output_path: str) -> None:
    all_rows = []
    for level, result in enumerate(hierarchical_clusters):
        centers = result["centers"].cpu().numpy()
        n_clusters_at_level = centers.shape[0]
        n_features = centers.shape[1]

        for cluster_idx in range(n_clusters_at_level):
            row = {
                "level": level,
                "cluster_id": cluster_idx,
                **{f"{feature_idx}": centers[cluster_idx, feature_idx] for feature_idx in range(n_features)},
            }
            all_rows.append(row)

    pl.DataFrame(all_rows).write_csv(output_path)


def _save_hierarchical_assignments_to_csv(
    hierarchical_clusters: list[dict[str, torch.Tensor]], output_path: str, sample_names: Optional[list[str]] = None
) -> None:
    n_samples = hierarchical_clusters[0]["assignment"].size(0)
    all_rows = []
    for sample_idx in range(n_samples):
        row = {
            "sample": sample_names[sample_idx] if sample_names is not None else sample_idx,
            **{
                f"level_{level}": result["assignment"][sample_idx].item()
                for level, result in enumerate(hierarchical_clusters)
            },
        }
        all_rows.append(row)

    pl.DataFrame(all_rows).write_csv(output_path)


def hierarchical_kmeans_clustering(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_centers_csv) is True and args.force is False:
        logger.warning(f"Output centers file already exists at: {args.output_centers_csv}, use --force to overwrite")
        return
    if os.path.exists(args.output_assignments_csv) is True and args.force is False:
        logger.warning(
            f"Output assignments file already exists at: {args.output_assignments_csv}, use --force to overwrite"
        )
        return

    if args.method == "resampled":
        if args.n_samples is None:
            logger.error("Error: --n-samples must be provided when --method is 'resampled'")
            return
        if len(args.n_samples) != len(args.n_clusters):
            logger.error(
                f"Error: Length of --n-samples ({len(args.n_samples)}) must match length of --n-clusters "
                f"({len(args.n_clusters)}) when --method is 'resampled'"
            )
            return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    logger.info(f"Loading dataset embeddings from: {args.embeddings_path}")
    logger.info(f"Number of clusters per level: {args.n_clusters}")
    logger.info(f"Distance metric: {args.distance_metric}")
    logger.info(f"Initialization method: {args.init_method}")
    logger.info(f"Hierarchical method: {args.method}")
    if args.method == "resampled":
        logger.info(f"Number of samples per level: {args.n_samples}")
        logger.info(f"Number of resamples: {args.n_resamples}")

    logger.info(f"Centers will be saved to: {args.output_centers_csv}")
    logger.info(f"Assignments will be saved to: {args.output_assignments_csv}")
    logger.info(f"Using device: {device}")

    embeddings = torch.tensor(utils.read_vector_file(args.embeddings_path))
    sample_names = utils.get_file_samples(args.embeddings_path).to_list()

    tic = time.time()
    hierarchical_clusters = pt_kmeans.hierarchical_kmeans(
        embeddings,
        args.n_clusters,
        args.max_iters,
        args.tol,
        args.distance_metric,
        args.init_method,
        args.chunk_size,
        show_progress=True,
        random_seed=args.random_seed,
        device=device,
        method=args.method,
        n_samples=args.n_samples,
        n_resamples=args.n_resamples,
    )
    _save_hierarchical_clusters_to_csv(hierarchical_clusters, args.output_centers_csv)
    logger.info(f"Centers saved to: {args.output_centers_csv}")

    _save_hierarchical_assignments_to_csv(hierarchical_clusters, args.output_assignments_csv, sample_names)
    logger.info(f"Assignments saved to: {args.output_assignments_csv}")

    toc = time.time()
    rate = len(embeddings) / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {len(embeddings):,} samples ({rate:.2f} samples/sec)")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Hierarchical K-Means Clustering Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Perform hierarchical K-Means clustering on image embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.hierarchical_kmeans_clustering --n-clusters 100000 500 20 "
            "--device cuda data/dataset_embeddings.csv\n"
            "python -m vdc.scripts.hierarchical_kmeans_clustering --config local_config.json "
            "--n-clusters 50000 --distance-metric l2 data/large_embeddings.csv\n"
            "python -m vdc.scripts.hierarchical_kmeans_clustering --n-clusters 1000 50 --max-iters 20 "
            "--device cpu results/vit_l14_pn_bioclip-v2_1024_224px_crop1.0_167015_embeddings.csv\n"
            "python -m vdc.scripts.hierarchical_kmeans_clustering --n-clusters 1000 50 10 --method resampled "
            "--n-samples 50 4 0 data/sample_embeddings.parquet\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Clustering parameters
    clustering_group = parser.add_argument_group("Clustering parameters")
    clustering_group.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="number of clusters for each level of hierarchical K-Means",
    )
    clustering_group.add_argument(
        "--max-iters", type=int, metavar="N", help="maximum number of K-Means iterations per level"
    )
    clustering_group.add_argument(
        "--tol", type=float, metavar="TOL", help="tolerance for convergence (normalized change in centers)"
    )
    clustering_group.add_argument(
        "--distance-metric", choices=["l2", "cosine"], metavar="METRIC", help="distance metric to use for clustering"
    )
    clustering_group.add_argument(
        "--init-method", choices=["random", "kmeans++"], metavar="METHOD", help="centers initialization method"
    )
    clustering_group.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="number of data points to process in a single batch during distance computations (to save memory)",
    )
    clustering_group.add_argument(
        "--method",
        choices=["centers", "resampled"],
        metavar="METHOD",
        help="method for building hierarchy: 'centers' (traditional) or 'resampled' (with refinement)",
    )
    clustering_group.add_argument(
        "--n-samples",
        type=int,
        nargs="+",
        metavar="N",
        help="number of samples to resample per cluster at each level (required when --method=resampled)",
    )
    clustering_group.add_argument(
        "--n-resamples",
        type=int,
        metavar="N",
        help="number of resampling steps to perform for each level when --method=resampled",
    )
    clustering_group.add_argument("--random-seed", type=int, metavar="SEED", help="random seed for reproducibility")

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument("--force", action="store_true", help="override existing results")
    parser.add_argument(
        "--output-centers-csv", type=str, metavar="FILE", help="output CSV file for all hierarchical cluster centers"
    )
    parser.add_argument(
        "--output-assignments-csv",
        type=str,
        metavar="FILE",
        help="output CSV file for hierarchical assignments of original samples",
    )
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
        "output_centers_csv": str(project_dir.joinpath("hierarchical_kmeans_centers.csv")),
        "output_assignments_csv": str(project_dir.joinpath("hierarchical_kmeans_assignments.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        kmeans_config = config.get("hierarchical_kmeans", {})
        parser.set_defaults(**kmeans_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_centers_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_assignments_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    hierarchical_kmeans_clustering(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
