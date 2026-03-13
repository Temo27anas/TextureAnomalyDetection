#  Texture Anomaly Detection

## Problem interpretation
The presented carpet category is a one-class classification &  pixel-level localization problem. We have only normal samples at training time (defects are unknown). This puts out supervised approaches and points to unsupervised anomaly detection.
The right paradigm is: learn a representation of "normality," then flag deviations at inference time.

## Literature Review:
Given our review, the following dominant approaches for industrial surface anomaly detection are available:
-	Reconstruction-based (early dominant approach): Train an autoencoder or VAE on normal images. Anomalies are pixels where reconstruction error is high. Works in principle, but suffers from the generalization paradox where good autoencoders reconstruct anomalies too, because they learn general image priors.
-	Flow-based / density estimation (DifferNet, FastFlow): Model the distribution of normal patch features using normalizing flows. Anomaly score = negative log-likelihood under the flow. Strong performance but computationally expensive.
-	Feature-embedding methods (current state of the art): Extract features from a pretrained CNN (typically ResNet/WideResNet on ImageNet) and build a compact model of the normal feature distribution. Two dominant variants:
 * PatchCore (Roth et al., 2022) coreset-based memory bank of patch embeddings. Achieves ~98-99% image AUROC and ~98% pixel AUROC on benchmarks.
 * PaDiM (Defard et al., 2021) — fit a multivariate Gaussian per patch position using multi-scale features. It recorded slightly lower performance, but very clean to implement.
-	Student-Teacher approaches (STFPM, RD4AD) are also strong but require more engineering overhead.


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
PatchCore is the right choice because:
(1) MVTec carpet benchmark scores are exceptional
(2) it's conceptually clean and available with anomalib
(3) no training required beyond feature extraction
(4) inference is interpretable
(5) it's been replicated extensively in literature

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
The testing process involves:
1. Loading the trained model checkpoint and memory bank.
2. Extracting features from the test images and computing anomaly scores based on nearest neighbor search against the memory bank. Heatmaps are generated for pixel-level anomaly localization, and can be found in `outputs/patchcore_results/images/` -  for example:
![output_009.png](https://github.com/Temo27anas/TextureAnomalyDetection/blob/main/assets/output_009.png)

3. Evaluating the performance using metrics such as AUROC, AUPR, F1-score.

## Web App
TBC

### Features
- **Configuration Panel (Sidebar)**:
  - Data directory paths (train, test, output)
  - Model architecture: backbone selection (wide_resnet50_2, resnet50, etc.) and feature layers
  - Hyperparameters: coreset sampling ratio, number of neighbors, image size, crop size
  - Training parameters: epochs, batch size

- **Action Buttons**:
  - **Train Model**: Train PatchCore on normal images with live training logs
  - **Run Test**: Evaluate the model on test set and compute metrics
  - **Run Full Pipeline**: Execute training followed by evaluation in one step

- **Results Display**:
  - **Latest Checkpoint**: Shows the path to the current trained model
  - **Execution Logs**: Live streaming logs during training/testing
  - **Evaluation Results**: Metrics from the last test run (AUROC, F1-score, etc.)
  - **Test Results Viewer**: Interactive dropdown to browse and view saved heatmap overlays from test images by defect class

### Color Scheme
The dashboard uses a professional green and white color scheme (#008342 accent color, white backgrounds, black text).

## Results

The results of the PatchCore model are saved in the `outputs/patchcore_results/` directory with the following structure:

```
outputs/patchcore_results/
├── evaluation_results.json     # Metrics (AUROC, F1-score, AUPR, etc.)
└── images/
    ├── color/
    ├── cut/
    ├── good/
    ├── hole/
    ├── metal_contamination/
    └── thread/
```

Each image in the `images/` subdirectories is a **heatmap overlay visualization** showing:
- **Original image** (55% opacity)
- **Anomaly detection heatmap** (45% opacity, red = anomalous, blue = normal)
- **Pixel-level localization** of detected defects

### Performance Metrics
After running the test pipeline, evaluation results are saved to `evaluation_results.json` and include:
- **Image-level AUROC**: Overall classification performance
- **Pixel-level metrics**: Localization accuracy of detected anomalies
- **F1-Score**: Harmonic mean of precision and recall

## Setup & Requirements

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation
1. Create a virtual environment & Install dependencies:
using :
    ``` uv venv .venv && source .venv/bin/activate ```
    then
     ``` uv  sync ```

or 
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   and then
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- **anomalib** (>=1.2.0): Anomaly detection framework
- **torch** / **torchvision**: Deep learning
- **lightning-pytorch**: Training orchestration
- **pandas** (>=2.0, <3): Data handling (pinned to avoid API breakage)
- **scikit-learn**: Metrics computation
- **PIL**: Image processing

## Project Structure

```
TextureAnomalyDetection/
├── src/
│   ├── train_patchcore.py      # Training pipeline
│   ├── test_patchcore.py       # Evaluation pipeline
│   ├── utils.py                # Utility functions (model loading, inference, logging)
│   
├── data/                        # Dataset directory (train, test, ground_truth)
├── outputs/                     # Model checkpoints and results
│   ├── patchcore/              # Trained model checkpoint
│   └── patchcore_results/       # Test results and heatmap overlays
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage Workflow

### 1. Prepare Dataset
Organize your texture images following the MVTec structure (see **Dataset** section above):
- Place normal training images in `data/train/good/`
- Place test images in `data/test/{good,defect_type}/`
- (Optional) Place pixel-level masks in `data/ground_truth/{defect_type}/`


### 3. Configure & Train
- Configuration can be changed manualy (backbone, resize dimensions...)
- Run Training using `python -u src.train_patchcore`

### 4. Evaluate
- Run Evaluation using `python -u src.test_patchcore`
We could report:
´
 #### image_AUROC: 0.998
 #### image_F1Score: 0.953
  

### 5. Iterate
- Modify hyperparameters in the sidebar and retrain
- Compare results across different configurations

## Troubleshooting

**Issue**: "Checkpoint not found"
- Ensure you've run **Train Model** at least once before **Run Test**
- Check that paths in the sidebar match your directory structure

**Issue**: Uniform heatmaps in results
- This may indicate the model needs more training epochs
- Try increasing `max_epochs` in the configuration sidebar

**Issue**: Memory errors during training
- Reduce `batch_size` in the sidebar
- Reduce `coreset_sampling_ratio` to use fewer memory bank features

## Assumptions & Limitations
- Assumed the product's texture is preserverd in the training set (no domain shift)
- Training data set fits a single shot imagery with fixed setup - otherwise object tracking would be needed
- View angle and lighting, focal length conditions are consistent between training and testing
- Locality of the image is preserved (defects are local and not global - image won't include external objects or background)

## Conclusion

This project implements PatchCore, a state-of-the-art unsupervised anomaly detection method, for texture defect detection in industrial quality control. The interactive Streamlit dashboard enables rapid experimentation with different configurations. (TBC)

This approach is production-ready for deployment in quality assurance pipelines where interpretability and performance are critical.

## References
- [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
