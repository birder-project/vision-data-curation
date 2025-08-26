import argparse
import logging
import os
import time
from pathlib import Path

import polars as pl
import torch
from birder.common import cli
from birder.common.lib import format_duration
from pt_kmeans import compute_distance
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


def filter_by_examples(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_csv) is True and args.force is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    logger.info(f"Loading dataset embeddings from: {args.embeddings_path}")
    logger.info(f"Loading example embeddings from: {args.examples_embeddings_file}")
    logger.info(f"Distance metric: {args.distance_metric} (with report threshold of {args.report_threshold})")
    logger.info(f"Report will be saved to: {args.output_csv}")
    logger.info(f"Using device: {device}")

    examples = torch.tensor(utils.read_vector_file(args.examples_embeddings_file), device=device)

    # Write CSV header
    with open(args.output_csv, "w", encoding="utf-8") as handle:
        handle.write("sample,distance\n")

    chunk_size = args.chunk_size or 4096
    total_samples = 0
    tic = time.time()
    with tqdm(desc="Processing embeddings", leave=False, unit="samples") as progress_bar:
        for df in utils.data_file_iter(args.embeddings_path, batch_size=chunk_size):
            sample_names = df.select("sample").to_series()
            df = df.select(pl.exclude(["sample"]))
            if len(df.columns) == 1:
                x = torch.tensor(df.to_series().to_numpy(), device=device)
            else:
                x = torch.tensor(df.to_numpy(), device=device)

            all_distances = compute_distance(x, examples, distance_metric=args.distance_metric)
            min_distances = torch.min(all_distances, dim=1).values.cpu().numpy()

            if args.report_threshold is not None:
                mask = min_distances < args.report_threshold
                sample_names = sample_names.filter(mask)
                min_distances = min_distances[mask]

            batch_results = pl.DataFrame({"sample": sample_names, "distance": min_distances})
            with open(args.output_csv, "a", encoding="utf-8") as handle:
                batch_results.write_csv(handle, include_header=False)

            total_samples += df.height
            progress_bar.update(df.height)

    toc = time.time()
    rate = total_samples / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {total_samples:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Filter by Examples Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Filter images by distance to unwanted examples using pre-computed embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.filter_by_examples --device cuda --report-threshold 0.3 --examples-embeddings-file "
            "data/bad_examples_embeddings.csv data/dataset_embeddings.csv\n"
            "python -m vdc.scripts.filter_by_examples --distance-metric l2 --examples-embeddings-file data/samples.csv "
            "data/dataset_embeddings.parquet\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Filtering parameters
    filtering_group = parser.add_argument_group("Filtering parameters")
    filtering_group.add_argument(
        "--distance-metric", choices=["l2", "cosine"], metavar="METRIC", help="distance metric to use"
    )
    filtering_group.add_argument(
        "--chunk-size", type=int, metavar="N", help="process embeddings in chunks to save memory"
    )
    filtering_group.add_argument(
        "--report-threshold",
        type=float,
        metavar="TH",
        help="only include samples with distance below this threshold in the report",
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument("--force", action="store_true", help="override existing report")
    parser.add_argument(
        "--examples-embeddings-file",
        type=str,
        required=True,
        metavar="FILE",
        help="pre-computed embeddings for unwanted examples",
    )
    parser.add_argument("--output-csv", type=str, metavar="FILE", help="output CSV file for filtering report")
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
        "output_csv": str(project_dir.joinpath("filter_by_examples_report.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        filter_config = config.get("filter_by_examples", {})
        parser.set_defaults(**filter_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    filter_by_examples(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
