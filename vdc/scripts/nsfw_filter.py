import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from birder.common import cli
from torch import nn

from vdc import utils
from vdc.conf import settings
from vdc.scripts import _content_filter

logger = logging.getLogger(__name__)


class Normalization(nn.Module):
    def __init__(self, shape: list[int]) -> None:
        super().__init__()
        self.mean = nn.Buffer(torch.zeros(shape))
        self.variance = nn.Buffer(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.variance.sqrt()


class NSFWClassifier(nn.Module):
    """
    Taken from: https://github.com/LAION-AI/CLIP-based-NSFW-Detector/issues/7
    Original code licensed under MIT license
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.norm = Normalization([input_dim])
        self.linear_1 = nn.Linear(input_dim, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        x = 1 - x  # Make sure higher is better (as in the aesthetic classifier)

        return x


FILTER = {
    "model_class": NSFWClassifier,
    "model_name": "nsfw-predictor",
    "model_file": "openai-clip_nsfw-predictor.pt",
    "model_sha256": "c0b8d7905cc1a315af2f3505222c8d81d339e64eb4cd4011912b645ab8bee4f1",
    "score_label": "nsfw_score",
}


def get_args_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # First parser for config file only
    config_parser = argparse.ArgumentParser(description="Filter Config", add_help=False)
    config_parser.add_argument(
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    config_parser.add_argument("--project", type=str, metavar="NAME", help="name of the project")

    # Main parser
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Filter images by nsfw score using pre-computed embeddings",
        epilog=(
            "Usage examples:\n"
            "python -m vdc.scripts.nsfw_filter --device cuda --report-threshold 0.5 data/dataset_embeddings.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    # Filtering parameters
    filtering_group = parser.add_argument_group("Filtering parameters")
    filtering_group.add_argument(
        "--report-threshold",
        type=float,
        metavar="TH",
        help=("only include samples with score below this threshold in the report (nsfw scores range from 0-1)"),
    )

    # Core arguments
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--config", type=str, metavar="FILE", help="JSON config file specifying default arguments"
    )
    parser.add_argument(  # Does nothing, just so it will show up at the usage message
        "--project", type=str, metavar="NAME", help="name of the project"
    )
    parser.add_argument("--device", default="auto", help="device to use for computations (cpu, cuda, mps, ...)")
    parser.add_argument(
        "--inference-batch-size", type=int, default=2048, metavar="N", help="batch size for model inference"
    )
    parser.add_argument("--force", action="store_true", help="override existing report")
    parser.add_argument("--output-csv", type=str, metavar="FILE", help="output CSV file for nsfw report")
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
        "output_csv": str(project_dir.joinpath("nsfw_filter_report.csv")),
    }
    parser.set_defaults(**default_paths)

    if config is not None:
        filter_config = config.get("nsfw_filter", {})
        parser.set_defaults(**filter_config)

    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger.debug(f"Running with config: {args}")

    output_dir = Path(args.output_csv).parent
    if output_dir.exists() is False:
        logger.info(f"Creating {output_dir} directory...")
        output_dir.mkdir(parents=True, exist_ok=True)

    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    _content_filter.run_filter(args, FILTER)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
