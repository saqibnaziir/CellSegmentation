# Setup Guide for GitHub Repository

This guide will help you upload your code to the GitHub repository.

## Prerequisites

1. Git installed on your system
2. GitHub account with access to the repository
3. Repository URL: `https://github.com/saqibnaziir/CellSegmentation.git`

## Step-by-Step Setup

### Step 1: Initialize Git Repository (if not already initialized)

```bash
git init
```

### Step 2: Add Remote Repository

```bash
git remote add origin https://github.com/saqibnaziir/CellSegmentation.git
```

If the remote already exists, update it:
```bash
git remote set-url origin https://github.com/saqibnaziir/CellSegmentation.git
```

### Step 3: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Or add specific files
git add README.md
git add requirements.txt
git add .gitignore
git add LICENSE
git add *.py
```

### Step 4: Commit Changes

```bash
git commit -m "Initial commit: Attention-Guided U-Net for Cell Nucleus Segmentation"
```

### Step 5: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

If you encounter authentication issues, you may need to:
- Use a Personal Access Token (PAT) instead of password
- Set up SSH keys
- Use GitHub CLI

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create saqibnaziir/CellSegmentation --public --source=. --remote=origin --push
```

## File Structure to Upload

Make sure these files are included:
- ✅ `README.md` - Comprehensive documentation
- ✅ `requirements.txt` - All dependencies
- ✅ `.gitignore` - Git ignore rules
- ✅ `LICENSE` - MIT License
- ✅ `main.py` - Training script
- ✅ `test.py` - Testing script
- ✅ `model.py` - Model architecture
- ✅ `dataset.py` - Dataset loading
- ✅ `enhanced_visualizations.py` - Visualization tools

Files that should NOT be uploaded (handled by .gitignore):
- ❌ `__pycache__/` - Python cache
- ❌ `checkpoints/` - Model checkpoints (too large)
- ❌ `data/` - Dataset files (too large)
- ❌ `*.pth.tar` - Model weights (too large)
- ❌ `*.log` - Log files
- ❌ `Backup/` - Backup directories

## Verify Upload

After pushing, verify by visiting:
https://github.com/saqibnaziir/CellSegmentation

## Troubleshooting

### Issue: Authentication Failed

**Solution**: Use a Personal Access Token (PAT)
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Use the token as password when pushing

### Issue: Large Files

**Solution**: Use Git LFS for large files (if needed)
```bash
git lfs install
git lfs track "*.pth.tar"
git add .gitattributes
```

### Issue: Remote Already Exists

**Solution**: Remove and re-add
```bash
git remote remove origin
git remote add origin https://github.com/saqibnaziir/CellSegmentation.git
```

## Next Steps

After successful upload:
1. Add a repository description on GitHub
2. Add topics/tags (e.g., `cell-segmentation`, `deep-learning`, `pytorch`, `biomedical-imaging`)
3. Create a release tag (e.g., `v1.0.0`)
4. Add a repository banner/logo (optional)
