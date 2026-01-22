#!/bin/bash
# Script to upload code to GitHub repository
# Usage: bash upload_to_github.sh [--auto]

echo "========================================"
echo "Uploading to GitHub Repository"
echo "========================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add remote (will update if exists)
echo "Adding/updating remote repository..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/saqibnaziir/CellSegmentation.git

# Add all files
echo "Adding files to git..."
git add .

# Show status
echo ""
echo "Files to be committed:"
git status --short

echo ""
echo "========================================"
echo "Ready to commit and push!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review the files above"
echo "2. Run: git commit -m 'Initial commit: Attention-Guided U-Net for Cell Nucleus Segmentation'"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""

if [ "$1" == "--auto" ]; then
    echo "Auto-committing and pushing..."
    git commit -m "Initial commit: Attention-Guided U-Net for Cell Nucleus Segmentation"
    git branch -M main
    git push -u origin main
    echo ""
    echo "Done! Check your repository at: https://github.com/saqibnaziir/CellSegmentation"
fi
