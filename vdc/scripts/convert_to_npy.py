import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
from birder.common import cli
from birder.common.lib import format_duration
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings  # noqa: F401 # pylint: disable=unused-import

logger = logging.getLogger(__name__)


def convert_to_npy(args: argparse.Namespace) -> None:
    output_file = str(Path(args.embeddings_path).with_suffix(".npy"))
    if os.path.exists(output_file) is True and args.force is False:
        logger.warning(f"Output file already exists at: {output_file}, use --force to overwrite")
        return

    logger.info(f"Starting conversion of '{args.embeddings_path}' to '{output_file}'")

    tic = time.time()

    # Determine shape
    total_rows = len(utils.get_file_samples(args.embeddings_path))

    first_df_batch = next(utils.data_file_iter(args.embeddings_path, batch_size=1))
    first_batch_data = utils.df_to_numpy(first_df_batch)
    embedding_dim = first_batch_data.shape[1]

    # Create memory-mapped NumPy array
    memmap_array = np.lib.format.open_memmap(
        output_file, dtype=np.float32, mode="w+", shape=(total_rows, embedding_dim)
    )

    # Write to memmap
    current_offset = 0
    with tqdm(desc="Processing batches", leave=True, unit="rows") as progress_bar:
        for batch in utils.data_file_iter(args.embeddings_path, batch_size=args.chunk_size):
            batch_size = batch.height
            batch_data = utils.df_to_numpy(batch)
            if args.l2_normalize is True:
                batch_data = batch_data / (np.linalg.norm(batch_data, axis=1, keepdims=True) + 1e-12)

            memmap_array[current_offset : current_offset + batch_size] = batch_data
            current_offset += batch_size

            progress_bar.update(batch_size)

    memmap_array.flush()

    toc = time.time()
    rate = total_rows / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to convert {total_rows:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Memory-mapped file saved to: {output_file}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Convert to NumPy Conversion Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Converts a data file to a memory-mapped NumPy (.npy) file",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.convert_to_npy data/raw_embeddings.parquet\n"
            "python -m vdc.scripts.convert_to_npy --chunk-size 50000 "
            "results/tol10m_rope_i_vit_l14_pn_aps_c1_pe-core_filtered_embeddings.parquet\n"
            "python -m vdc.scripts.convert_to_npy --chunk-size 50000 data/logits.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Conversion parameters
    conversion_group = parser.add_argument_group("Conversion parameters")
    conversion_group.add_argument(
        "--chunk-size", type=int, default=4096, metavar="N", help="number of rows to read from the input file at a time"
    )
    conversion_group.add_argument(
        "--l2-normalize", action="store_true", help="L2-normalize vectors before saving them to the .npy file"
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output embeddings file")
    parser.add_argument("embeddings_path", help="path to the input embeddings/logits file")

    return (config_parser, parser)


def parse_args() -> argparse.Namespace:
    (config_parser, parser) = get_args_parser()
    (args_config, remaining) = config_parser.parse_known_args()

    if args_config.config is None:
        logger.debug("No user config file specified. Loading default bundled config")
        config = utils.load_default_bundled_config()
    else:
        config = utils.read_json(args_config.config)

    if config is not None:
        convert_config = config.get("convert_to_npy", {})
        parser.set_defaults(**convert_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    convert_to_npy(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
