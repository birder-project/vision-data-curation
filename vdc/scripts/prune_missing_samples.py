import argparse
import logging
import os
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from birder.common import cli
from birder.common.lib import format_duration
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings  # noqa: F401 # pylint: disable=unused-import

logger = logging.getLogger(__name__)


def _get_schema(data_path: str) -> dict[str, pl.DataType]:
    if data_path.endswith(".parquet") is True:
        return pl.read_parquet_schema(data_path)

    return pl.read_csv(data_path, n_rows=1).schema


def prune_missing_samples(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_file) is True and args.force is False:
        logger.warning(f"Output file already exists at: {args.output_file}, use --force to overwrite")
        return

    logger.info(f"Pruning entries from: {args.embeddings_path}")
    logger.info(f"Output will be saved to: {args.output_file}")

    schema = _get_schema(args.embeddings_path)
    output_file_format = Path(args.output_file).suffix[1:]

    total_samples = 0
    total_samples_kept = 0
    total_samples_deleted = 0
    tic = time.time()

    parquet_writer: pq.ParquetWriter | None = None
    csv_output_header_written = False
    with tqdm(desc="Pruning batches", leave=True, unit="rows") as progress_bar:
        for batch in utils.data_file_iter(args.embeddings_path, batch_size=args.chunk_size):
            mask = batch["sample"].map_elements(lambda s: Path(s).exists(), return_dtype=pl.Boolean)
            kept = batch.filter(mask)

            kept_rows = kept.height
            deleted_rows = batch.height - kept_rows
            total_samples += batch.height
            total_samples_kept += kept_rows
            total_samples_deleted += deleted_rows

            if kept_rows == 0:
                continue

            if output_file_format == "parquet":
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(args.output_file, pl.DataFrame(schema=schema).to_arrow().schema)

                parquet_writer.write_table(kept.to_arrow())
            elif output_file_format == "csv":
                if csv_output_header_written is False:
                    kept.write_csv(args.output_file)
                else:
                    with open(args.output_file, "a", encoding="utf-8") as handle:
                        kept.write_csv(handle, include_header=False)
            else:
                raise ValueError(
                    f"Unsupported output file format '{output_file_format}', only .csv and .parquet are supported"
                )

            progress_bar.update(batch.height)

    toc = time.time()
    rate = total_samples / (toc - tic)

    logger.info(f"{format_duration(toc - tic)} to process {total_samples:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Total samples kept: {total_samples_kept:,}")
    logger.info(f"Total samples deleted: {total_samples_deleted:,}")
    logger.info(f"Pruned file saved to: {args.output_file}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Prune Missing Samples Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Prunes entries from embedding/logits files where the corresponding sample image no longer exists",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.prune_missing_samples --output-file data/curated_embeddings.parquet -j 16 "
            "data/raw_embeddings.parquet\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Pruning parameters
    pruning_group = parser.add_argument_group("Pruning parameters")
    pruning_group.add_argument(
        "--chunk-size", type=int, metavar="N", help="number of rows to read from the input file at a time"
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output embeddings file")
    parser.add_argument(
        "--output-file", type=str, metavar="FILE", help="path to the output pruned embeddings/logits file"
    )
    parser.add_argument("embeddings_path", help="path to the input embeddings/logits file")

    return (config_parser, parser)


def parse_args() -> argparse.Namespace:
    config_parser, parser = get_args_parser()
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config is None:
        logger.debug("No user config file specified. Loading default bundled config")
        config = utils.load_default_bundled_config()
    else:
        config = utils.read_json(args_config.config)

    if config is not None:
        prune_config = config.get("prune_missing_samples", {})
        parser.set_defaults(**prune_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_file).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    prune_missing_samples(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
