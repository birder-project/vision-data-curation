import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Literal

import polars as pl
import torch
from birder.common import cli
from birder.common.lib import format_duration
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


def run_filter(args: argparse.Namespace, filter_config: dict[str, Any]) -> None:
    model_name = filter_config["model_name"]
    model_file = filter_config["model_file"]

    if os.path.exists(args.output_csv) is True and args.force is False:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)
    model_path = settings.MODELS_DIR.joinpath(filter_config["model_file"])

    logger.info(f"Loading dataset embeddings from: {args.embeddings_path}")
    logger.info(f"Loading {model_name} model from: {model_path}")
    if args.report_threshold is not None:
        logger.info(f"Report will include samples with score below: {args.report_threshold}")

    logger.info(f"Report will be saved to: {args.output_csv}")
    logger.info(f"Using device: {device}")

    # Load model
    if model_path.exists() is False:
        cli.download_file(
            f"https://huggingface.co/birder-project/{model_name}/resolve/main/{model_file}",
            model_path,
            expected_sha256=filter_config["model_sha256"],
        )

    model = filter_config["model_class"](input_dim=768).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Data
    file_format: Literal["csv", "parquet"] = Path(args.embeddings_path).suffix[1:]  # type: ignore[assignment]
    dataset = utils.InferenceDataset(
        args.embeddings_path, file_format, columns_to_drop=["prediction"], metadata_columns=["sample"]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.inference_batch_size, num_workers=1)

    # Write CSV header
    with open(args.output_csv, "w", encoding="utf-8") as handle:
        handle.write(f"sample,{filter_config['score_label']}\n")

    # Inference
    total_samples = 0
    tic = time.time()
    with torch.inference_mode():
        with tqdm(desc="Processing embeddings", leave=False, unit="samples") as progress_bar:
            for inputs, samples in dataloader:
                inputs = inputs.to(device)
                scores = model(inputs).cpu().numpy().flatten()
                sample_names = pl.Series("sample", samples)
                if args.report_threshold is not None:
                    mask = scores < args.report_threshold
                    sample_names = sample_names.filter(mask)
                    scores = scores[mask]

                batch_results = pl.DataFrame({"sample": sample_names, filter_config["score_label"]: scores})
                with open(args.output_csv, "a", encoding="utf-8") as handle:
                    batch_results.write_csv(handle, include_header=False)

                total_samples += inputs.size(0)
                progress_bar.update(inputs.size(0))

    toc = time.time()
    rate = total_samples / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {total_samples:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Report saved to: {args.output_csv}")
