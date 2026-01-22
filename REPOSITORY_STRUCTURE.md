# Repository Structure

This document describes the clean, professional structure of the CellSegmentation repository.

## 📁 Core Files (Essential)

### Main Code Files
- **`main.py`** - Training script with comprehensive options
- **`test.py`** - Testing/evaluation script
- **`model.py`** - Model architecture (Attention-Guided U-Net)
- **`dataset.py`** - Dataset loading and preprocessing

### Utility Scripts
- **`enhanced_visualizations.py`** - Generate publication-quality visualizations
- **`augmentation.py`** - Data augmentation utilities
- **`measure_efficiency.py`** - Model efficiency analysis (FLOPs, inference time, etc.)

## 📚 Documentation

- **`README.md`** - Comprehensive documentation (installation, usage, examples)
- **`SETUP.md`** - Detailed setup and GitHub upload guide
- **`QUICK_START.md`** - Quick reference for common tasks
- **`LICENSE`** - MIT License

## ⚙️ Configuration

- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - Git ignore rules (excludes data, checkpoints, results, etc.)

## 🚀 Helper Scripts

- **`upload_to_github.bat`** - Windows script for uploading to GitHub
- **`upload_to_github.sh`** - Linux/Mac script for uploading to GitHub

## 📊 Excluded Files (Not in Repository)

The following files are excluded via `.gitignore`:

### Data & Results
- `data/` - Dataset files (too large)
- `Datasets/` - Dataset directories
- `Original/`, `Mask/`, `Augmented/` - Image directories
- `Predictions/` - Test predictions
- `enhanced_results/` - Visualization outputs
- `checkpoints/` - Model checkpoints
- `Models/` - Saved models

### Temporary & Analysis Files
- `Backup/` - Backup directories
- `1.Results/` - Results directory
- `*.log` - Log files
- `*.png`, `*.jpg`, `*.tif` - Image files
- `*.pth.tar`, `*.pth` - Model weights
- `*.csv`, `*.txt` (except requirements.txt) - Data files

### Development Files
- `__pycache__/` - Python cache
- `*.tex` - LaTeX files (paper writing)
- Analysis markdown files (QUICK_REFERENCE.md, RESULTS_ANALYSIS.md, etc.)
- Utility scripts (add_noise.py, binary.py) - Optional utilities

## 🎯 Repository Goals

This structure ensures:
1. **Clean & Professional** - Only essential code and documentation
2. **Easy to Use** - Clear documentation and examples
3. **Lightweight** - No large data files or checkpoints
4. **Reproducible** - All code needed to run the project
5. **Well-Documented** - Comprehensive README and guides

## 📝 File Organization Principles

- **Core functionality** → Main repository
- **Data & results** → Excluded (users provide their own)
- **Development files** → Excluded (not needed for users)
- **Documentation** → Included (helps users understand and use)
- **Configuration** → Included (needed to run the project)
