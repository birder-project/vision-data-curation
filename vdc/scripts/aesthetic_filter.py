import argparse
import logging
import os
import time

import polars as pl
import torch
import torch.nn.functional as F
from birder.common import cli
from birder.common.lib import format_duration
from torch import nn
from tqdm import tqdm

from vdc import utils
from vdc.conf import settings

logger = logging.getLogger(__name__)


class AestheticClassifier(nn.Module):
    """
    Taken from: https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
    Original code licensed under Apache-2.0
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return self.layers(x)


def filter_by_aesthetic(args: argparse.Namespace) -> None:
    model_file = "openai-clip_aesthetic-predictor.pt"

    if os.path.exists(args.output_csv) is True and not args.force:
        logger.warning(f"Report already exists at: {args.output_csv}, use --force to overwrite")
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)
    model_path = settings.MODELS_DIR.joinpath(model_file)

    logger.info(f"Loading dataset embeddings from: {args.embeddings_path}")
    logger.info(f"Loading aesthetic model from: {model_path}")
    if args.report_threshold is not None:
        logger.info(f"Report will include samples with aesthetic score below: {args.report_threshold}")

    logger.info(f"Report will be saved to: {args.output_csv}")
    logger.info(f"Using device: {device}")

    # Load model
    if model_path.exists() is False:
        cli.download_file(
            f"https://huggingface.co/birder-project/aesthetic-predictor/resolve/main/{model_file}",
            model_path,
            expected_sha256="21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0",
        )

    model = AestheticClassifier(input_dim=768).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Data
    dataset = utils.InferenceCSVDataset(
        args.embeddings_path, columns_to_drop=["prediction"], metadata_columns=["sample"]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.inference_batch_size, num_workers=1)

    # Write CSV header
    with open(args.output_csv, "w", encoding="utf-8") as handle:
        handle.write("sample,aesthetic_score\n")

    # Inference
    total_samples = 0
    tic = time.time()
    with torch.inference_mode():
        with tqdm(desc="Processing embeddings", leave=False, unit="samples") as progress_bar:
            for inputs, samples in dataloader:
                inputs = inputs.to(device)
                aesthetic_scores = model(inputs).cpu().numpy().flatten()
                sample_names = pl.Series("sample", samples)
                if args.report_threshold is not None:
                    mask = aesthetic_scores < args.report_threshold
                    sample_names = sample_names.filter(mask)
                    aesthetic_scores = aesthetic_scores[mask]

                batch_results = pl.DataFrame({"sample": sample_names, "aesthetic_score": aesthetic_scores})
                with open(args.output_csv, "a", encoding="utf-8") as handle:
                    batch_results.write_csv(handle, include_header=False)

                total_samples += inputs.size(0)
                progress_bar.update(inputs.size(0))

    toc = time.time()
    rate = total_samples / (toc - tic)
    logger.info(f"{format_duration(toc - tic)} to process {total_samples:,} samples ({rate:.2f} samples/sec)")
    logger.info(f"Report saved to: {args.output_csv}")


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Aesthetic Filter Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Filter images by aesthetic score using pre-computed embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.aesthetic_filter --device cuda --report-threshold 5.0 data/dataset_embeddings.csv\n"
            "python -m vdc.scripts.aesthetic_filter --report-threshold 6.0 "
            "results/vit_l14_pn_quick_gelu_openai-clip_768_224px_crop1.0_201399_iwildcam2022_output.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Filtering parameters
    filtering_group = parser.add_argument_group("Filtering parameters")
    filtering_group.add_argument(
        "--report-threshold",
        type=float,
        metavar="TH",
        help="only include samples with distance below this threshold in the report (aesthetic scores range from 0-10)",
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument(
        "--inference-batch-size", type=int, default=1024, metavar="N", help="batch size for model inference"
    )
    parser.add_argument("--force", action="store_true", help="override existing report")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(settings.RESULTS_DIR.joinpath("aesthetic_filter_report.csv")),
        metavar="FILE",
        help="output CSV file for aesthetic report",
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

    if config is not None:
        filter_config = config.get("aesthetic_filter", {})
        parser.set_defaults(**filter_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    if settings.RESULTS_DIR.exists() is False:
        logger.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    filter_by_aesthetic(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
