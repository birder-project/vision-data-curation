# Vision Data Curation (VDC)

A lightweight framework for cleaning, filtering, and sampling large-scale image datasets.
Built for computer vision researchers and practitioners who want higher-quality data with less manual effort.

At its core, VDC embraces a data-centric AI philosophy, aiming to enhance the quality and diversity of your datasets with minimal manual intervention.
By providing a structured, iterative pipeline, VDC helps researchers and practitioners build higher-performing models by starting with better data.

## Features

VDC provides modular tools for dataset cleanup:

- **Input validation** - detect corrupt files, invalid formats, low resolution, or extreme aspect ratios
- **Rotation correction** - correct 90°/180°/270° orientation errors
- **Example-based filtering** - remove images similar to a set of unwanted examples
- **Image Quality Filtering** - remove images based on aesthetic score or NSFW classification
- **Duplicate removal** - identify and remove near-duplicate images from your dataset
- **Hierarchical K-Means sampling** - select diverse, representative subsets from large datasets

## The Curation Pipeline

```mermaid
flowchart LR
    A[Raw<br/>Dataset] --> V[Validation]
    V --> R[Rotation]
    R --> D[Dedup]
    D --> E[Example<br/>Filter]
    E --> Q[Quality Filter<br/>Aesthetic/NSFW]
    Q --> S[Cluster-based<br/>Sampling]
    S --> F[Curated<br/>Dataset]

    U[Unwanted<br/>Examples] --> E
```

## Installation

### From PyPI

```sh
pip install vision-data-curation
```

### From Source

```sh
git clone https://gitlab.com/birder/vision-data-curation.git
cd vision-data-curation
pip install -e .
```

Developing directly from the project root allows for script and configuration execution as if fully installed.

## Usage

Each step is a script under `vdc.scripts`.

Examples:

```sh
# Remove corrupt/invalid images
python -m vdc.scripts.sanitize_images data/raw_images/

# Filter based on "Unwanted examples"
python -m vdc.scripts.filter_by_examples data/embeddings.csv --examples bad_examples.csv
```

- Run `python -m vdc.scripts` to see available scripts
- Run `python -m vdc.scripts.<script> --help` for options

**Model Downloads:** Models required for certain features (e.g., SSCD, PE, CLIP, aesthetic/NSFW predictors) are automatically downloaded to and managed within the `models/` directory (defined by `vdc.conf.settings.MODELS_DIR`).

## Configuration

- Default settings live in [vdc/conf/config.json](vdc/conf/config.json)
- A `config.json` in your project root will take precedence (or pass `--config` to any script)

## Visualization & Exploration

VDC provides accompanying R Markdown notebooks (found in the `notebooks/` directory) to visually inspect and understand the impact of various curation steps, aiding in data-driven decision making.

While these notebooks use the `.Rmd` extension, they are entirely written in Python.
This choice was made for their text-based nature, which significantly improves version control by producing clean and meaningful Git diffs compared to traditional `.ipynb` files.

For a seamless interactive experience in a Jupyter environment, we recommend installing `jupytext` (`pip install jupytext`). With `jupytext` installed, you can open and run `.Rmd` files directly in Jupyter as if they were `.ipynb` notebooks.

## Documentation

For detailed walkthroughs, examples, and in-depth guides, please refer to our [full documentation](docs/README.md).

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
