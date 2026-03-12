#  Texture Anomaly Detection

## Problem interpretation
This project addresses one-class anomaly detection in an unsupervised setting for the texture domain. The model learns normal carpet texture appearance from defect-free training images and identifies deviations during testing at both image level and pixel level.

## Dataset
The dataset is organized into training and testing sets, with the following structure:

```
data/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── color/
│   ├── cut/
│   ├── hole/
│   ├── metal_contamination/
│   └── thread/
├── ground_truth/
    ├── color/
    ├── cut/
    ├── hole/
    ├── metal_contamination/
    └── thread/
```
- **Training Set**: Contains only normal (good) images for the model to learn the distribution of normal texture.
- **Testing Set**: Contains both normal and various types of anomalous images (color,
cut, hole, metal contamination, thread) for evaluation.
- **Ground Truth**: Pixel-level masks for anomalous test images, used for evaluating localization performance.

## Methodology
The approach is based on the PatchCore method, which consists of the following steps:
1. **Feature Extraction**: A pretrained CNN backbone (ResNet) is used to extract patch-level features from the normal training images.
2. **Memory Bank Construction**: A coreset subsampling technique is applied to create a memory bank of representative patch features, which captures the normal texture distribution while being memory efficient.
3. **Anomaly Detection**: During testing, patch features are extracted from test images and compared against the memory bank using nearest neighbor search. An anomaly score is computed based on the distance to the nearest neighbors, allowing for both image-level and pixel-level anomaly detection.


## Training
The training process involves:
1. Extracting features from the normal training images using the pretrained backbone.
2. Constructing the memory bank using coreset subsampling to retain representative features while reducing memory usage.
3. Saving the trained model checkpoint for later evaluation.
The training script can be executed using the following command:

```bash
python train_patchcore.py
```
## Testing and Evaluation
TBD

## Results
TBD

## Conclusion
TBD

## References
- [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)