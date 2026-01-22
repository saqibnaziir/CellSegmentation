# import os
# import cv2
# import numpy as np
# from pathlib import Path
# import albumentations as A
# from tqdm import tqdm
# import logging
# from datetime import datetime
# import shutil
# import random

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f"augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def create_augmented_dataset(
#     original_dir,
#     mask_dir,
#     output_dir,
#     num_augmentations=5,
#     img_size=256,
#     seed=42
# ):
#     """
#     Create augmented dataset by applying various augmentations to original images and masks
    
#     Args:
#         original_dir (str): Directory containing original images
#         mask_dir (str): Directory containing mask images
#         output_dir (str): Directory to save augmented images
#         num_augmentations (int): Number of augmented versions to create per image
#         img_size (int): Size to resize images to
#         seed (int): Random seed for reproducibility
#     """
#     # Set random seed
#     random.seed(seed)
#     np.random.seed(seed)
    
#     # Create output directories with consistent structure
#     output_original_dir = Path(output_dir) / "Original"
#     output_mask_dir = Path(output_dir) / "Mask"
#     output_original_dir.mkdir(parents=True, exist_ok=True)
#     output_mask_dir.mkdir(parents=True, exist_ok=True)
    
#     # Copy original files to output directory
#     logger.info("Copying original files to output directory...")
#     for img_path in Path(original_dir).glob("*.png"):
#         shutil.copy2(img_path, output_original_dir / img_path.name)
#         mask_path = Path(mask_dir) / img_path.name
#         if mask_path.exists():
#             shutil.copy2(mask_path, output_mask_dir / img_path.name)
    
#     # Define augmentations
#     transform = A.Compose([
#         # Geometric transforms
#         A.RandomRotate90(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.ShiftScaleRotate(
#             shift_limit=0.0625,
#             scale_limit=0.1,
#             rotate_limit=45,
#             p=0.5
#         ),
#         A.ElasticTransform(
#             alpha=120,
#             sigma=120 * 0.05,
#             alpha_affine=120 * 0.03,
#             p=0.3
#         ),
#         A.GridDistortion(p=0.3),
        
#         # Intensity transforms
#         A.OneOf([
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.2,
#                 contrast_limit=0.2,
#                 p=1.0
#             ),
#             A.RandomGamma(gamma_limit=(80, 120), p=1.0),
#             A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
#         ], p=0.5),
        
#         # Blur and sharpening
#         A.OneOf([
#             A.GaussianBlur(blur_limit=3, p=1.0),
#             A.MedianBlur(blur_limit=3, p=1.0),
#             A.MotionBlur(blur_limit=3, p=1.0),
#         ], p=0.3),
        
#         # Resize
#         A.Resize(img_size, img_size, always_apply=True),
#     ], is_check_shapes=False)
    
#     # Get all image files
#     image_files = list(Path(original_dir).glob("*.png"))
#     logger.info(f"Found {len(image_files)} original images")
    
#     # Perform augmentation
#     logger.info(f"Performing {num_augmentations} augmentations per image...")
#     for img_path in tqdm(image_files, desc="Augmenting images"):
#         # Load image and mask
#         image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
#         mask_path = Path(mask_dir) / img_path.name
#         if not mask_path.exists():
#             logger.warning(f"Mask not found for {img_path.name}, skipping...")
#             continue
        
#         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
#         # Create augmented versions
#         for i in range(num_augmentations):
#             # Apply augmentation
#             augmented = transform(image=image, mask=mask)
#             aug_image = augmented['image']
#             aug_mask = augmented['mask']
            
#             # Save augmented image and mask with consistent naming
#             aug_img_name = f"{img_path.stem}_aug_{i+1}{img_path.suffix}"
#             aug_image_path = output_original_dir / aug_img_name
#             aug_mask_path = output_mask_dir / aug_img_name
            
#             cv2.imwrite(str(aug_image_path), aug_image)
#             cv2.imwrite(str(aug_mask_path), aug_mask)
    
#     # Print summary
#     total_original = len(list(output_original_dir.glob("*.png")))
#     total_augmented = len(list(output_original_dir.glob("*_aug_*.png")))
#     logger.info(f"Dataset augmentation complete!")
#     logger.info(f"Original images: {total_original - total_augmented}")
#     logger.info(f"Augmented images: {total_augmented}")
#     logger.info(f"Total images: {total_original}")
#     logger.info(f"Augmented images saved to:")
#     logger.info(f"  Original images: {output_original_dir}")
#     logger.info(f"  Mask images: {output_mask_dir}")

# def main():
#     # Define paths
#     original_dir = r"D:\EdgeHill\Data\BBBC041Seg\Final_data\archive\BCCD Dataset with mask\train\original"
#     mask_dir = r"D:\EdgeHill\Data\BBBC041Seg\Final_data\archive\BCCD Dataset with mask\train\mask"
#     output_dir = "D:/EdgeHill/Data/BBBC041Seg/Augmented"
    
#     # Create augmented dataset
#     create_augmented_dataset(
#         original_dir=original_dir,
#         mask_dir=mask_dir,
#         output_dir=output_dir,
#         num_augmentations=5,  # Create 5 augmented versions of each image
#         img_size=256,
#         seed=42
#     )

# if __name__ == "__main__":
#     main() 


################ updated
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_simple_augmented_dataset(
    original_dir,
    mask_dir,
    output_dir,
    num_augmentations=2,  # Reduced from 5 to 2
    target_width=1600,
    target_height=1200,
    seed=42
):
    """
    Create a simply augmented dataset with minimal, essential transformations
    
    Args:
        original_dir (str): Directory containing original images
        mask_dir (str): Directory containing mask images
        output_dir (str): Directory to save augmented images
        num_augmentations (int): Number of augmented versions to create per image
        target_width (int): Target width for output images
        target_height (int): Target height for output images
        seed (int): Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_original_dir = Path(output_dir) / "Original"
    output_mask_dir = Path(output_dir) / "Mask"
    output_original_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and resize original files to output directory
    logger.info("Copying and resizing original files to output directory...")
    for img_path in Path(original_dir).glob("*.png"):
        # Load and resize original image (width, height for cv2.resize)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(output_original_dir / img_path.name), resized_image)
        
        # Load and resize corresponding mask (width, height for cv2.resize)
        mask_path = Path(mask_dir) / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_mask_dir / img_path.name), resized_mask)
    
    # Define SIMPLE augmentations - only the essentials
    transform = A.Compose([
        # Essential geometric transforms (conservative settings)
        A.OneOf([
            A.HorizontalFlip(p=1.0),           # Simple horizontal flip
            A.VerticalFlip(p=1.0),             # Simple vertical flip
            A.RandomRotate90(p=1.0),           # 90-degree rotations only
        ], p=0.8),  # 80% chance of one of these
        
        # Light rotation (much smaller range)
        A.Rotate(
            limit=15,  # Only ±15 degrees (reduced from ±180)
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4  # Lower probability
        ),
        
        # Minor brightness/contrast adjustment only
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # Very conservative ±10%
            contrast_limit=0.1,    # Very conservative ±10%
            p=0.3  # Low probability
        ),
        
        # Final resize (height, width for Albumentations)
        A.Resize(target_height, target_width, interpolation=cv2.INTER_LANCZOS4),
    ], is_check_shapes=False)
    
    # Get all image files
    image_files = list(Path(original_dir).glob("*.png"))
    logger.info(f"Found {len(image_files)} original images")
    logger.info(f"Target output size: {target_width}x{target_height}")
    
    # Print simple augmentation summary
    logger.info("SIMPLE Augmentations being applied:")
    logger.info("- Basic flips/90° rotations: 80% chance")
    logger.info("- Light rotation (±15°): 40% chance")
    logger.info("- Minor brightness/contrast: 30% chance")
    logger.info("- NO elastic transforms, NO noise, NO blur, NO heavy distortions")
    
    # Perform augmentation
    logger.info(f"Performing {num_augmentations} simple augmentations per image...")
    for img_path in tqdm(image_files, desc="Augmenting images"):
        # Load image and mask
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_path = Path(mask_dir) / img_path.name
        if not mask_path.exists():
            logger.warning(f"Mask not found for {img_path.name}, skipping...")
            continue
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size first (width, height for cv2.resize)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # Create augmented versions
        for i in range(num_augmentations):
            # Apply simple augmentation
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Verify output size (height, width for OpenCV format)
            assert aug_image.shape == (target_height, target_width), f"Wrong size: {aug_image.shape}, expected: ({target_height}, {target_width})"
            assert aug_mask.shape == (target_height, target_width), f"Wrong size: {aug_mask.shape}, expected: ({target_height}, {target_width})"
            
            # Save augmented image and mask
            aug_img_name = f"{img_path.stem}_aug_{i+1}{img_path.suffix}"
            aug_image_path = output_original_dir / aug_img_name
            aug_mask_path = output_mask_dir / aug_img_name
            
            cv2.imwrite(str(aug_image_path), aug_image)
            cv2.imwrite(str(aug_mask_path), aug_mask)
    
    # Print summary
    total_original = len(list(output_original_dir.glob("*.png")))
    total_augmented = len(list(output_original_dir.glob("*_aug_*.png")))
    original_count = total_original - total_augmented
    logger.info(f"SIMPLE Dataset augmentation complete!")
    logger.info(f"Original images: {original_count}")
    logger.info(f"Augmented images: {total_augmented}")
    logger.info(f"Total images: {total_original}")
    logger.info(f"Multiplier: {total_original / original_count:.1f}x (reduced from 6x)")
    logger.info(f"All images are sized: {target_width}x{target_height}")
    logger.info(f"Images saved to:")
    logger.info(f"  Original images: {output_original_dir}")
    logger.info(f"  Mask images: {output_mask_dir}")

def main():
    # Define paths
    original_dir = r"D:\EdgeHill\Data\BBBC041Seg\Final_data\archive\BCCD Dataset with mask\train\original"
    mask_dir = r"D:\EdgeHill\Data\BBBC041Seg\Final_data\archive\BCCD Dataset with mask\train\mask"
    output_dir = "D:/EdgeHill/Data/BBBC041Seg/SimpleAugmented"  # Different output folder
    
    # Create simply augmented dataset
    create_simple_augmented_dataset(
        original_dir=original_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        num_augmentations=2,    # Only 2 augmented versions instead of 5
        target_width=1600,      # Your required width
        target_height=1200,     # Your required height
        seed=42
    )

if __name__ == "__main__":
    main()