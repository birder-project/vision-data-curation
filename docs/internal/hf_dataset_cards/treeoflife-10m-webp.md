---
license: cc0-1.0
language:
- en
- la
source_datasets:
  - imageomics/TreeOfLife-10M
task_categories:
- image-classification
- zero-shot-classification
pretty_name: TreeOfLife-10M WEBP
size_categories:
- 10M<n<100M
---

# Dataset Card for TreeOfLife-10M-WEBP

## Dataset Description

This is an optimized version of the [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) dataset,
containing over 10 million images covering 454 thousand taxa in the tree of life.
This version has been processed to improve usability and reduce storage requirements while maintaining full compatibility with the original dataset structure.

### Dataset Summary

This version modifies the original dataset as follows:

- Corrupted files were repaired.
- Very large images (some >40K pixels in width) were resized so that the total number of pixels < 1,048,576 (=1024×1024), preserving aspect ratio.
- All images were re-encoded to WEBP format.
- The dataset was repacked in the same shard structure as the original to remain fully compatible.

The result is a significantly reduced dataset size (~500GB vs. ~2TB), with lower I/O overhead and fewer extreme image cases that can slow down training pipelines.

For dataset details, licensing information, taxonomy information and annotation process, please see the [original dataset card](https://huggingface.co/datasets/imageomics/TreeOfLife-10M).

## Limitations

- Maintains all original dataset limitations regarding taxonomic coverage and class imbalance
- Some images have been resized, which may affect fine-grained visual analysis of extremely high-resolution specimens

## Licensing

This repackaged dataset is distributed under the same licensing terms as the original TreeOfLife-10M dataset.
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

## Acknowledgments

This optimization builds upon the outstanding work of the original TreeOfLife-10M creators.
All credit for data curation, taxonomic labeling, and scientific contributions belongs to the original team at the Imageomics Institute.
