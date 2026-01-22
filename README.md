# Attention-Guided U-Net for Cell Nucleus Segmentation in Microscopy Images

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight encoder-decoder architecture for cell nuclei segmentation in microscopy images, achieving state-of-the-art performance with significantly reduced model complexity. This implementation features a custom residual encoder with dilated convolutions, Squeeze-and-Excitation (SE) modules, and Spatial Pyramid Pooling for enhanced multi-scale representation.

## 📋 Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Performance](#performance)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Enhanced Visualizations](#enhanced-visualizations)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 📝 Abstract

Cell nuclei segmentation is a fundamental step in computational pathology and biomedical image analysis, enabling downstream tasks such as disease diagnosis and drug discovery. However, existing deep learning-based approaches often rely on heavy encoders or complex multi-branch designs, leading to large parameter counts and limited practicality in clinical settings. 

We propose a lightweight encoder-decoder architecture that achieves improved segmentation performance with significantly reduced model complexity. Our custom residual encoder leverages dilated convolutions and Squeeze-and-Excitation (SE) modules to capture rich contextual features, while a Spatial Pyramid Pooling bottleneck enhances multi-scale representation.

**Results**: The proposed model consistently outperforms or matches State-of-the-art (SOTA) models while using fewer parameters. Specifically, it achieves Dice scores of **0.9817** on BCS, **0.9262** on DSB, and **86.45** on NuInsSeg, surpassing SOTA models in most cases despite much smaller computational footprint.

## ✨ Features

- **Lightweight Architecture**: Significantly reduced parameters compared to SOTA models
- **Attention Mechanisms**: Enhanced attention gates with CBAM (Convolutional Block Attention Module)
- **Residual Connections**: Custom residual blocks for better gradient flow
- **Multi-scale Features**: Spatial Pyramid Pooling for enhanced representation
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1, HD95, ASD, and more
- **Visualization Tools**: Enhanced visualization scripts for paper-quality figures
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Gradient Clipping**: Stabilized training with gradient clipping
- **TensorBoard Integration**: Real-time training monitoring

## 🎯 Performance

| Dataset | Dice Score | IoU | Parameters |
|---------|------------|-----|------------|
| BCS     | 0.9817     | -   | -          |
| DSB     | 0.9262     | -   | -          |
| NuInsSeg| 86.45      | -   | -          |

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/saqibnaziir/CellSegmentation.git
cd CellSegmentation
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n cellseg python=3.8
conda activate cellseg

# Or using venv
python -m venv cellseg
source cellseg/bin/activate  # On Windows: cellseg\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PyTorch installation, visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to install the appropriate version for your system.

## 📁 Dataset Preparation

### Directory Structure

Organize your dataset in the following structure:

```
data/
├── original/          # Original microscopy images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── mask/              # Corresponding mask images (binary)
    ├── image1.png
    ├── image2.png
    └── ...
```

### Dataset Requirements

- **Image Format**: PNG (recommended) or JPG
- **Image Size**: Any size (will be resized during training)
- **Mask Format**: Binary masks (0 for background, 255 for foreground)
- **Naming**: Image and mask files must have the same filename

### Supported Datasets

The code has been tested on:
- **BCS** (Breast Cancer Segmentation)
- **DSB 2018** (Data Science Bowl 2018)
- **NuInsSeg** (Nuclei Instance Segmentation)

## 💻 Usage

### Training

Train the model with default parameters:

```bash
python main.py \
    --original_dir data/original \
    --mask_dir data/mask \
    --batch_size 4 \
    --epochs 100 \
    --lr 0.0001 \
    --img_size 256 \
    --checkpoint_dir checkpoints
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--original_dir` | `data/original` | Directory with original images |
| `--mask_dir` | `data/mask` | Directory with mask images |
| `--batch_size` | `4` | Batch size for training |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--img_size` | `256` | Image size (will be resized to this) |
| `--seed` | `42` | Random seed for reproducibility |
| `--checkpoint_dir` | `checkpoints` | Directory to save checkpoints |
| `--workers` | `4` | Number of worker threads |
| `--base_channels` | `64` | Number of base channels in model |
| `--grad_clip` | `1.0` | Gradient clipping value (0 to disable) |
| `--bce_weight` | `0.5` | BCE loss weight |
| `--dice_weight` | `0.5` | Dice loss weight |
| `--scheduler` | `reduce` | LR scheduler: `reduce` or `cosine` |
| `--resume` | `None` | Path to checkpoint to resume from |

#### Advanced Training Options

```bash
# Training with gradient clipping and warmup
python main.py \
    --original_dir data/original \
    --mask_dir data/mask \
    --epochs 100 \
    --grad_clip 1.0 \
    --warmup_epochs 5 \
    --scheduler cosine

# Resume training from checkpoint
python main.py \
    --original_dir data/original \
    --mask_dir data/mask \
    --resume checkpoints/checkpoint_epoch50.pth.tar
```

#### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=checkpoints/tensorboard_logs --port=6006
```

Then open your browser and navigate to `http://localhost:6006`

### Testing

Test a trained model:

```bash
python test.py \
    --test_dir data/test/original \
    --mask_dir data/test/mask \
    --checkpoint checkpoints/best_model.pth.tar \
    --save_dir predictions \
    --batch_size 8 \
    --img_size 256
```

#### Testing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_dir` | Required | Directory with test images |
| `--mask_dir` | Required | Directory with test masks |
| `--checkpoint` | Required | Path to model checkpoint |
| `--save_dir` | `predictions` | Directory to save predictions |
| `--batch_size` | `8` | Batch size for testing |
| `--img_size` | `256` | Image size |
| `--threshold` | `0.5` | Threshold for binary segmentation |
| `--base_channels` | `64` | Number of base channels (must match training) |

#### Test Output

The test script generates:
- **Prediction images**: Visual comparison of original, ground truth, and prediction
- **Metrics file**: `test_metrics.txt` with detailed metrics including:
  - Jaccard Index (IoU)
  - Dice Coefficient
  - Precision
  - Recall
  - F1 Score
  - Mean Surface Distance (MSD)

### Enhanced Visualizations

Generate publication-quality visualizations:

```bash
python enhanced_visualizations.py \
    --test_dir data/test/original \
    --mask_dir data/test/mask \
    --checkpoint checkpoints/best_model.pth.tar \
    --save_dir enhanced_results \
    --num_samples 20 \
    --threshold 0.5
```

#### Visualization Output

The script generates:
- **Overlays**: Original image with prediction/GT overlays
- **Error Maps**: TP, FP, FN visualization
- **Confidence Maps**: Prediction probability maps
- **Boundary Overlays**: Boundary visualization
- **Comprehensive Visualizations**: 6-panel figures
- **ROC Curves**: Receiver Operating Characteristic curves
- **Metrics Distribution**: Box plots of metrics

## 📂 Project Structure

```
CellSegmentation/
├── main.py                      # Training script
├── test.py                      # Testing script
├── model.py                     # Model architecture
├── dataset.py                   # Dataset loading and preprocessing
├── enhanced_visualizations.py   # Enhanced visualization tools
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore file
│
├── data/                        # Dataset directory (not included)
│   ├── original/                # Original images
│   └── mask/                    # Mask images
│
├── checkpoints/                 # Model checkpoints
│   ├── best_model.pth.tar      # Best model
│   ├── checkpoint_epoch*.pth.tar
│   └── tensorboard_logs/        # TensorBoard logs
│
├── predictions/                 # Test predictions
│   └── test_metrics.txt         # Test metrics
│
└── enhanced_results/            # Enhanced visualization outputs
    ├── overlays/
    ├── error_maps/
    ├── confidence_maps/
    ├── boundary_overlays/
    ├── comprehensive/
    └── enhanced_metrics.txt
```

## 🏗️ Model Architecture

The model consists of:

1. **Encoder**: Custom residual encoder with dilated convolutions
2. **Bottleneck**: Spatial Pyramid Pooling for multi-scale features
3. **Decoder**: Attention-guided upsampling with residual blocks
4. **Attention Gates**: Enhanced attention gates with CBAM
5. **Output Fusion**: Multi-scale feature fusion

### Key Components

- **Residual Blocks**: Improved gradient flow
- **Dilated Convolutions**: Larger receptive field
- **Squeeze-and-Excitation (SE)**: Channel attention
- **Spatial Attention**: Spatial feature refinement
- **Attention Gates**: Skip connection refinement

## 📊 Metrics

The model is evaluated using multiple metrics:

- **Jaccard Index (IoU)**: Intersection over Union
- **Dice Coefficient**: 2 × Intersection / (Sum of Areas)
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **Hausdorff Distance (HD95)**: 95th percentile boundary distance
- **Average Surface Distance (ASD)**: Average boundary distance
- **Volume Similarity (VS)**: Volume overlap measure
- **Mean Surface Distance (MSD)**: Mean boundary distance

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@article{nazir2024attention,
  title={Attention-Guided U-Net for Cell Nucleus Segmentation in Microscopy Images},
  author={Nazir, Saqib and Behera, Ardhendu},
  journal={Journal Name},
  year={2024},
  institution={Edge Hill University}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Saqib Nazir** - [saqib.nazir@edgehill.ac.uk](mailto:saqib.nazir@edgehill.ac.uk)
- **Ardhendu Behera** - [beheraa@edgehill.ac.uk](mailto:beheraa@edgehill.ac.uk)

**Affiliation**: Department of Computer Science, Edge Hill University, St Helens Road, Ormskirk, UK

## 🙏 Acknowledgments

- Edge Hill University for providing computational resources
- The open-source community for excellent tools and libraries
- Dataset providers: BCS, DSB 2018, and NuInsSeg

## 📧 Contact

For questions, issues, or collaborations, please contact:
- **Saqib Nazir**: saqib.nazir@edgehill.ac.uk
- **Ardhendu Behera**: beheraa@edgehill.ac.uk

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 2`
   - Reduce image size: `--img_size 128`
   - Use gradient accumulation (modify code)

2. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.7+)

3. **Dataset Loading Issues**
   - Ensure image and mask filenames match exactly
   - Check image formats (PNG recommended)
   - Verify masks are binary (0 and 255)

4. **Training Instability**
   - Enable gradient clipping: `--grad_clip 1.0`
   - Reduce learning rate: `--lr 0.00005`
   - Use warmup: `--warmup_epochs 5`

## 🔄 Updates

- **v1.0.0** (2024): Initial release
  - Attention-guided U-Net implementation
  - Training and testing scripts
  - Enhanced visualization tools
  - Comprehensive documentation

---

**Note**: This repository is actively maintained. For the latest updates, please check the repository regularly.
