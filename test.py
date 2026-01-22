import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
from scipy.ndimage import distance_transform_edt

from dataset import CellSegmentationDataset
from model import get_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def dice_coef(pred, target, threshold=0.5, smooth=1.0):
    """Dice coefficient for evaluation"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2. * intersection + smooth) / 
            (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()

def save_prediction(image, mask, pred, save_path, threshold=0.5):
    """Save prediction visualization"""
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow((pred > threshold).float().squeeze(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate various segmentation metrics"""
    # Convert to binary
    pred = (pred > threshold).float()
    target = target.float()
    
    # Calculate intersection and union for Jaccard Index (IoU)
    # Jaccard Index = Σ(ŷy) / Σ(y + ̂y - ŷy)
    intersection = (pred * target).sum(dim=(2, 3))  # Σ(ŷy)
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection  # Σ(y + ̂y - ŷy)
    
    # Jaccard Index (IoU)
    jaccard = (intersection + 1e-6) / (union + 1e-6)
    
    # Precision and Recall
    true_positives = intersection
    false_positives = pred.sum(dim=(2, 3)) - intersection
    false_negatives = target.sum(dim=(2, 3)) - intersection
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Dice Coefficient
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)
    
    # Mean Surface Distance
    def compute_msd(pred_mask, target_mask):
        pred_dist = distance_transform_edt(pred_mask.cpu().numpy())
        target_dist = distance_transform_edt(target_mask.cpu().numpy())
        return np.mean(np.abs(pred_dist - target_dist))
    
    msd = torch.tensor([compute_msd(p, t) for p, t in zip(pred, target)])
    
    return {
        'jaccard': jaccard.mean().item(),  # Renamed from 'iou' to be more explicit
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'dice': dice.mean().item(),
        'msd': msd.mean().item()
    }

def test_model(model, test_loader, device, save_dir, threshold=0.5):
    """Test model on test set with multiple metrics"""
    model.eval()
    metrics_list = []
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Calculate all metrics
            metrics = calculate_metrics(outputs, masks, threshold)
            metrics_list.append(metrics)
            
            # Save predictions
            for j in range(images.size(0)):
                save_path = save_dir / f"prediction_{i}_{j}.png"
                save_prediction(
                    images[j].cpu(),
                    masks[j].cpu(),
                    outputs[j].cpu(),
                    save_path,
                    threshold
                )
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in metrics_list])
        for metric in metrics_list[0].keys()
    }
    
    # Log all metrics with detailed explanations
    logger.info("\nTest Results:")
    logger.info("=" * 50)
    logger.info("Segmentation Metrics:")
    logger.info("-" * 30)
    logger.info(f"Jaccard Index (IoU): {avg_metrics['jaccard']:.4f} (Intersection over Union)")
    logger.info(f"Dice Coefficient: {avg_metrics['dice']:.4f} (2 * Intersection / (Sum of Areas))")
    logger.info(f"Precision: {avg_metrics['precision']:.4f} (True Positives / (True Positives + False Positives))")
    logger.info(f"Recall: {avg_metrics['recall']:.4f} (True Positives / (True Positives + False Negatives))")
    logger.info(f"F1 Score: {avg_metrics['f1']:.4f} (Harmonic mean of Precision and Recall)")
    logger.info(f"Mean Surface Distance: {avg_metrics['msd']:.4f} (Average distance between boundaries)")
    logger.info("=" * 50)
    
    # Save metrics to file with detailed explanations using UTF-8 encoding
    metrics_file = save_dir / "test_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Test Results:\n")
        f.write("=" * 50 + "\n")
        f.write("Segmentation Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Jaccard Index (IoU): {avg_metrics['jaccard']:.4f}\n")
        f.write("Formula: sum(y_pred * y_true) / sum(y_pred + y_true - y_pred * y_true)\n")
        f.write("Where:\n")
        f.write("- y_pred * y_true: Intersection of prediction and ground truth\n")
        f.write("- y_pred + y_true - y_pred * y_true: Union of prediction and ground truth\n\n")
        
        f.write(f"Dice Coefficient: {avg_metrics['dice']:.4f}\n")
        f.write("Formula: 2 * sum(y_pred * y_true) / (sum(y_pred) + sum(y_true))\n\n")
        
        f.write(f"Precision: {avg_metrics['precision']:.4f}\n")
        f.write("Formula: True Positives / (True Positives + False Positives)\n\n")
        
        f.write(f"Recall: {avg_metrics['recall']:.4f}\n")
        f.write("Formula: True Positives / (True Positives + False Negatives)\n\n")
        
        f.write(f"F1 Score: {avg_metrics['f1']:.4f}\n")
        f.write("Formula: 2 * (Precision * Recall) / (Precision + Recall)\n\n")
        
        f.write(f"Mean Surface Distance: {avg_metrics['msd']:.4f}\n")
        f.write("Formula: Mean absolute distance between prediction and ground truth boundaries\n")
        f.write("=" * 50 + "\n")
    
    return avg_metrics

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Cell Segmentation Model')
    
    # Data paths
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with test images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory with test masks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Number of base channels in the model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for dataloader')
    parser.add_argument('--save_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    
    return parser.parse_args()

def main():
    # Get arguments
    args = get_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    test_dataset = CellSegmentationDataset(
        args.test_dir,
        args.mask_dir,
        img_size=args.img_size,
        is_training=False  # Explicitly set to False to prevent augmentations
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create model
    model = get_model(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Test model
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main() 