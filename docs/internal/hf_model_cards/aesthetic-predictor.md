---
tags:
- birder
- pytorch
license: apache-2.0
---

# Model Card for CLIP-based Aesthetic Predictor

A simple MLP intended to run on CLIP embeddings to predict the "aesthetic quality" of an image (how much people like it on average).

Trained by Christoph Schuhmann and adapted to suit the [Vision Data Curation](https://gitlab.com/birder/vision-data-curation) project.

For more information see: <https://github.com/christophschuhmann/improved-aesthetic-predictor>

## Model Details

- **Model Type:** Aesthetic score regression model
- **Input:** OpenAI CLIP embeddings ([vit_l14_pn_quick_gelu_openai-clip](https://huggingface.co/birder-project/vit_l14_pn_quick_gelu_openai-clip))
- **Output:** A score between 0 and 10, where higher values correspond to more aesthetic images

Original authorship: Adapted from Christoph Schuhmann's MLP Aesthetic Score Predictor

## Model Usage

This classifier operates on CLIP image embeddings rather than raw pixels. To run inference with the Birder framework:

```sh
# Download the CLIP backbone
python -m birder.tools download-model vit_l14_pn_quick_gelu_openai-clip

# Run prediction on a dataset
python -m birder.scripts.predict \
    -n vit_l14_pn_quick_gelu \
    -t openai-clip \
    --simple-crop \
    --gpu \
    --parallel \
    --batch-size 256 \
    --chunk-size 50000 \
    --amp \
    --amp-dtype bfloat16 \
    --save-logits \
    --suffix optional-dataset-name \
    path/to/dataset

# Can now run the aesthetic predictor on the saved logits
```

## Intended Use

Primary use case: Ranking or filtering images by aesthetic appeal, dataset curation, and training data selection.

Recommended scope: Research, dataset preparation, and large-scale data analysis.

Not intended for: As a measure of artistic merit, cultural value, or taste preferences of specific individuals.

## Citation

```bibtex
@misc{christophschuhmann2022improved-aesthetic-predictor,
  author = {Christoph Schuhmann},
  title = {MLP Aesthetic Score Predictor},
  year = {2022},
  url = {https://github.com/christophschuhmann/improved-aesthetic-predictor},
  note = {Accessed: August 22, 2025},
}
```
