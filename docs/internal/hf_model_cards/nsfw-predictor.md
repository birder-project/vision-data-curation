---
tags:
- birder
- pytorch
license: mit
---

# Model Card for CLIP-based NSFW Classifier

A simple MLP intended to run on CLIP embeddings to classify NSFW images.

Trained by LAION-AI and only adapted to suit the [Vision Data Curation](https://gitlab.com/birder/vision-data-curation) project.

For more information see: <https://github.com/LAION-AI/CLIP-based-NSFW-Detector>

## Model Details

- **Model Type:** NSFW binary classifier
- **Input:** OpenAI CLIP embeddings ([vit_l14_pn_quick_gelu_openai-clip](https://huggingface.co/birder-project/vit_l14_pn_quick_gelu_openai-clip))
- **Output:** A probability score between 0 and 1, where higher values correspond to safer (SFW) content

Original authorship: Adapted from LAION-AI’s CLIP-based-NSFW-Detector

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

# Can now run the NSFW classifier on the saved logits
```

## Intended Use

Primary use case: Filtering and scoring image embeddings for potentially NSFW content.

Recommended scope: Pre-screening in research, data curation, and large-scale dataset processing.

Not intended for: Deployment as a sole moderation tool, enforcement decisions, or safety-critical applications.

## Citation

```bibtex
@misc{LAION-AI2022CLIP-based-NSFW-Detector,
  author = {Christoph Schuhmann},
  title = {CLIP-based-NSFW-Detector},
  year = {2022},
  url = {https://github.com/LAION-AI/CLIP-based-NSFW-Detector},
  note = {Accessed: August 22, 2025},
}
```
