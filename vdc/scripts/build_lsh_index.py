import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from birder.common import cli
from birder.common.lib import format_duration
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings
from vdc.deduplication.lsh import LSHIndex

logger = logging.getLogger(__name__)


def _compute_global_mean(embeddings_path: str, chunk_size: int) -> npt.NDArray[np.float64]:
    logger.info("Calculating global mean of embeddings...")
    with tqdm(desc="Global mean calculation", leave=False, unit="samples") as progress_bar:
        data_iterator = utils.data_file_iter(embeddings_path, batch_size=chunk_size)
        first_batch = next(data_iterator)
        batch = first_batch.select(pl.exclude(["sample"])).to_numpy()
        mean_embedding: npt.NDArray[np.float64] = batch.mean(axis=0, dtype=np.float64)
        total_count = batch.shape[0]
        progress_bar.update(total_count)

        for df_batch in data_iterator:
            batch = df_batch.select(pl.exclude(["sample"])).to_numpy()
            batch_size = batch.shape[0]
            total_count_new = total_count + batch_size
            mean_embedding = (mean_embedding * total_count + batch.sum(axis=0)) / total_count_new
            total_count = total_count_new
            progress_bar.update(batch_size)

    logger.info(f"Global mean calculated for {total_count:,} embeddings")
    return mean_embedding


def build_lsh_index(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_file) is True and args.force is False:
        logger.warning(f"LSH index already exists at: {args.output_file}, use --force to overwrite")
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    logger.info(f"Loading dataset from: {args.embeddings_path}")
    logger.info(
        f"LSH parameters: num_hash_tables={args.num_hash_tables}, "
        f"num_hyperplanes_per_table={args.num_hyperplanes_per_table}"
    )
    logger.info(f"Centering embeddings: {args.center_embeddings}, L2 normalize: {args.l2_normalize}")
    logger.info(f"Output index will be saved to: {args.output_file}")
    logger.info(f"Using device: {device}")

    tic = time.time()
    mean_embedding: Optional[torch.Tensor] = None
    if args.center_embeddings is True:
        mean_embedding = torch.from_numpy(
            _compute_global_mean(args.embeddings_path, args.chunk_size).astype(np.float32)
        )

    logger.info("Populating LSH index...")
    data_iterator = utils.data_file_iter(args.embeddings_path, batch_size=1)
    first_df_for_dim = next(data_iterator)
    first_embedding_np_for_dim = first_df_for_dim.select(pl.exclude(["sample"])).to_numpy()
    embedding_dim = first_embedding_np_for_dim.shape[1]

    lsh_index = LSHIndex(
        embedding_dim,
        num_hash_tables=args.num_hash_tables,
        num_hyperplanes_per_table=args.num_hyperplanes_per_table,
        random_seed=args.random_seed,
        center_embedding=mean_embedding,
        l2_normalize=args.l2_normalize,
        device=device,
    )

    total_count = 0
    with tqdm(desc="Building LSH index", leave=False, unit="samples") as progress_bar:
        for df_batch in utils.data_file_iter(args.embeddings_path, batch_size=args.chunk_size):
            sample_ids = df_batch.select("sample").to_series().to_list()
            embeddings_batch = torch.from_numpy(df_batch.select(pl.exclude(["sample"])).to_numpy())

            lsh_index.add_embeddings(sample_ids, embeddings_batch)

            total_count += embeddings_batch.shape[0]
            progress_bar.update(embeddings_batch.shape[0])

    lsh_index.save(args.output_file)

    toc = time.time()
    rate = total_count / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to build LSH index for {total_count:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"LSH index saved to: {args.output_file}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="LSH Index Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Builds an LSH (Locality Sensitive Hashing) index for image embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.build_lsh_index --device cpu data/dataset_embeddings.parquet\n"
            "python -m vdc.scripts.build_lsh_index --num-hash-tables 8 --num-hyperplanes-per-table 32 "
            "--center-embeddings --l2-normalize data/embeddings.csv --output-file results/my_lsh_index.pkl\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # LSH parameters
    lsh_group = parser.add_argument_group("LSH parameters")
    lsh_group.add_argument("--center-embeddings", action="store_true", help="center embeddings by subtracting the mean")
    lsh_group.add_argument(
        "--l2-normalize",
        action="store_true",
        help="L2 normalize embeddings (essential for cosine similarity)",
    )
    lsh_group.add_argument(
        "--num-hash-tables",
        type=int,
        metavar="N",
        help="number of LSH hash tables, more tables increase recall but also memory and query time",
    )
    lsh_group.add_argument(
        "--num-hyperplanes-per-table",
        type=int,
        metavar="N",
        help="number of random hyperplanes per LSH hash table, more hyperplanes increase precision",
    )
    lsh_group.add_argument(
        "--assignments-csv", type=str, metavar="FILE", help="path to the K-Means assignments CSV file"
    )
    lsh_group.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="number of embeddings to process in a single batch during iterative loading and preprocessing",
    )
    lsh_group.add_argument("--random-seed", type=int, metavar="SEED", help="random seed for reproducibility")

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument("--force", action="store_true", help="override existing LSH index file")
    parser.add_argument(
        "--output-file", type=str, metavar="FILE", help="output file for the serialized LSH index (.pkl)"
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
        "output_file": str(project_dir.joinpath("lsh_index.pkl")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        lsh_config = config.get("build_lsh_index", {})
        parser.set_defaults(**lsh_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_file).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    build_lsh_index(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
