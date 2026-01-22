import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from functools import lru_cache
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import cv2
import warnings

# Suppress iCCP profile warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# Ensure NumPy is properly initialized
np.set_printoptions(threshold=np.inf)

def load_image_safely(image_path):
    """Load image safely handling color profiles"""
    try:
        # First try with PIL to handle color profiles
        with Image.open(image_path) as img:
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            # Convert to numpy array
            img_array = np.array(img)
    except Exception as e:
        print(f"Warning: PIL failed to load {image_path}, falling back to cv2: {e}")
        # Fallback to cv2 if PIL fails
        img_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            raise ValueError(f"Failed to load image: {image_path}")
    
    return img_array
class CellSegmentationDataset(Dataset):
    def __init__(self, original_dir, mask_dir, img_size=256, is_training=True):
        """Dataset for cell segmentation - no augmentation since data is pre-augmented"""
        self.original_dir = Path(original_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.is_training = is_training
        
        # Get all image files that exist in both directories
        original_files = set(f.name for f in self.original_dir.glob('*.png'))
        mask_files = set(f.name for f in self.mask_dir.glob('*.png'))
        self.image_files = sorted(list(original_files.intersection(mask_files)))
        
        print(f"Found {len(self.image_files)} valid image pairs")
        
        # Simple transforms - only resize and normalize (no augmentation)
        self.transform = A.Compose([
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ], is_check_shapes=False)
        
        # Print sample image info
        if len(self.image_files) > 0:
            sample_img_path = self.original_dir / self.image_files[0]
            sample_mask_path = self.mask_dir / self.image_files[0]
            
            sample_img = load_image_safely(sample_img_path)
            sample_mask = load_image_safely(sample_mask_path)
            
            print(f"Sample image size: {sample_img.shape}")
            print(f"Sample mask size: {sample_mask.shape}")
            print(f"Using image size: {img_size}x{img_size}")
            print(f"Training mode: {is_training} (pre-augmented data)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image-mask pair (no augmentation)"""
        img_name = self.image_files[idx]
        
        # Load image and mask using safe loading function
        img_path = self.original_dir / img_name
        mask_path = self.mask_dir / img_name
        
        image = load_image_safely(img_path)
        mask = load_image_safely(mask_path)
        
        # Apply only resize and normalization
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to binary
        mask = (mask > 0.5).float()
        
        # Add channel dimension to mask to match model output
        mask = mask.unsqueeze(0)
        
        return image, mask
