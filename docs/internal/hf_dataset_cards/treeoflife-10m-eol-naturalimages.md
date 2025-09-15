---
license: cc0-1.0
language:
- en
- la
source_datasets:
  - imageomics/TreeOfLife-10M
  - birder-project/TreeOfLife-10M-WEBP
task_categories:
- image-classification
- zero-shot-classification
- image-feature-extraction
pretty_name: TreeOfLife-10M-EOL-NaturalImages
size_categories:
- 1M<n<10M
---

# Dataset Card for TreeOfLife-10M-EOL-NaturalImages

## Dataset Description

This is a curated version of the [TreeOfLife-10M-WEBP](https://huggingface.co/datasets/birder-project/TreeOfLife-10M-WEBP) EOL training split, filtered to contain exclusively natural biological imagery.
The dataset has been systematically cleaned using the [Vision Data Curation (VDC)](https://gitlab.com/birder/vision-data-curation) framework to remove non-natural content while preserving high-quality biological specimens.

### Dataset Summary

This version further refines the dataset from `birder-project/TreeOfLife-10M-WEBP` through a multi-stage curation process:

- **Initial Sanitization:** Corrupted or invalid images were detected and removed/repaired.
- **Deduplication:** Near-duplicate images were identified and removed using SSCD embeddings to prevent data redundancy and improve training efficiency.
- **Filtering for Natural Images:** A key curation step involved extensive example-based filtering using PE-Core embeddings to remove non-natural content such as:
    - Documents (notebook pages, text)
    - Maps and charts
    - Drawings and illustrations
    - Footprint images
    - Specimen photography (where not desired for natural imagery)
    This step ensures the dataset is primarily composed of real-world photographs of natural subjects.
- **Aesthetic Filtering:** Images with very low aesthetic scores (e.g., severely blurred or out-of-focus) were identified and removed using CLIP-derived aesthetic scores, further enhancing overall visual quality.

The result is a high-quality dataset of approximately 5.6 million natural images, ideal for self-supervised learning, natural image classification, and other computer vision tasks requiring a clean and diverse representation of the natural world.
This version is provided *before* hierarchical sampling, allowing users to apply their own sampling strategies to achieve a desired dataset size and diversity.

To facilitate custom sampling and analysis, this dataset also includes the pre-computed **hierarchical K-Means clustering assignments (`hierarchical_kmeans_assignments.csv`)** and **cluster centroids (`hierarchical_kmeans_centers.csv`)**.
These files can be used with the VDC framework's sampling tools (e.g., `sample_images`) or custom scripts to create representative subsets based on the learned cluster structure.

For a detailed walkthrough of the entire curation process that generated this dataset, please refer to the [VDC Real-World Workflow: Cleaning the Tree of Life 10M Dataset](https://gitlab.com/birder/vision-data-curation/-/blob/main/docs/example/tree_of_life_10m.md) documentation.

For original dataset details, licensing information, taxonomy, and annotation processes, please refer to the [original TreeOfLife-10M dataset card](https://huggingface.co/datasets/imageomics/TreeOfLife-10M).

## Limitations

- Taxonomic coverage: Maintains original dataset limitations regarding taxonomic coverage and class imbalance among the remaining natural images
- Image resolution: Some images were resized in the preceding WEBP conversion step, which may still affect fine-grained visual analysis of extremely high-resolution specimens
- Reduced Size: The filtering process has reduced the total number of images
- Targeted Content: This dataset is specifically curated for natural images, it is not suitable for tasks requiring documents, maps, or other non-natural visual content from the original dataset

## Licensing

This curated dataset is distributed under the same licensing terms as the original TreeOfLife-10M dataset.
Please review the [original licensing information](https://huggingface.co/datasets/imageomics/TreeOfLife-10M#licensing-information) before using this dataset, as all terms and restrictions remain applicable.

## Citation

```bibtex
@dataset{treeoflife_10m,
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  title = {TreeOfLife-10M},
  year = {2023},
  url = {https://huggingface.co/datasets/imageomics/TreeOfLife-10M},
  doi = {10.57967/hf/1972},
  publisher = {Hugging Face}
}

@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life},
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
  pages = {19412-19424}
}
```

If you utilize the [Vision Data Curation (VDC)](https://gitlab.com/birder/vision-data-curation) framework, please also consider citing it.

```bibtex
@software{Hasson_Vision_Data_Curation,
  author = {Hasson, Ofer},
  license = {Apache-2.0},
  title = {{Vision Data Curation}},
  url = {https://gitlab.com/birder/vision-data-curation}
}
```

## Acknowledgments

This curated dataset builds upon the exceptional work of the TreeOfLife-10M creators at the Imageomics Institute.
All credit for original data collection, taxonomic labeling, and scientific contributions belongs to the original team.
This curation work aims to enhance the dataset's utility for computer vision research while preserving its scientific integrity.
