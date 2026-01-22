"""
Enhanced Visualization Script for Cell Segmentation Results
Generates comprehensive visualizations commonly shown in cell segmentation papers
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from scipy.spatial.distance import directed_hausdorff
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import cv2
import argparse
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

from dataset import CellSegmentationDataset
from model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def calculate_hausdorff_distance(pred_mask, gt_mask):
    """Calculate Hausdorff Distance and HD95"""
    pred_boundary = get_boundary(pred_mask)
    gt_boundary = get_boundary(gt_mask)
    
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return np.inf, np.inf
    
    pred_coords = np.column_stack(np.where(pred_boundary))
    gt_coords = np.column_stack(np.where(gt_boundary))
    
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
    hd = max(hd1, hd2)
    
    # HD95: 95th percentile of distances
    distances = []
    for p in pred_coords:
        dists = np.sqrt(((gt_coords - p) ** 2).sum(axis=1))
        distances.append(np.min(dists))
    for g in gt_coords:
        dists = np.sqrt(((pred_coords - g) ** 2).sum(axis=1))
        distances.append(np.min(dists))
    
    hd95 = np.percentile(distances, 95) if len(distances) > 0 else np.inf
    
    return hd, hd95

def get_boundary(mask):
    """Extract boundary from binary mask"""
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    return boundary.astype(np.uint8)

def calculate_average_surface_distance(pred_mask, gt_mask):
    """Calculate Average Surface Distance"""
    pred_boundary = get_boundary(pred_mask)
    gt_boundary = get_boundary(gt_mask)
    
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return np.inf
    
    # Distance from pred boundary to GT boundary
    pred_coords = np.column_stack(np.where(pred_boundary))
    gt_coords = np.column_stack(np.where(gt_boundary))
    
    distances = []
    for p in pred_coords:
        dists = np.sqrt(((gt_coords - p) ** 2).sum(axis=1))
        distances.append(np.min(dists))
    
    return np.mean(distances) if len(distances) > 0 else np.inf

def calculate_volume_similarity(pred_mask, gt_mask):
    """Calculate Volume Similarity"""
    vol_pred = pred_mask.sum()
    vol_gt = gt_mask.sum()
    
    if vol_pred + vol_gt == 0:
        return 1.0
    
    vs = 1 - abs(vol_pred - vol_gt) / (vol_pred + vol_gt)
    return vs

def create_overlay_visualization(image, mask, pred, threshold=0.5, save_path=None):
    """Create overlay visualization: original with prediction/GT overlaid"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = (torch.sigmoid(pred) > threshold).float().squeeze().cpu().numpy()
    
    # Normalize image to [0, 1]
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    # Original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(image_np, cmap='gray')
    axes[1].imshow(mask_np, alpha=0.5, cmap='Greens')
    axes[1].set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(image_np, cmap='gray')
    axes[2].imshow(pred_np, alpha=0.5, cmap='Blues')
    axes[2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_error_map(image, mask, pred, threshold=0.5, save_path=None):
    """Create error map showing TP, FP, FN regions"""
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = (torch.sigmoid(pred) > threshold).float().squeeze().cpu().numpy()
    
    # Normalize image
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    # Create error map: 0=TN, 1=TP, 2=FP, 3=FN
    error_map = np.zeros_like(mask_np, dtype=np.uint8)
    error_map[(mask_np > 0.5) & (pred_np > 0.5)] = 1  # TP (green)
    error_map[(mask_np <= 0.5) & (pred_np > 0.5)] = 2  # FP (red)
    error_map[(mask_np > 0.5) & (pred_np <= 0.5)] = 3  # FN (blue)
    
    # Create colormap: transparent, green, red, blue
    colors = ['black', 'green', 'red', 'blue']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original with error overlay
    axes[0].imshow(image_np, cmap='gray')
    im = axes[0].imshow(error_map, alpha=0.6, cmap=cmap, vmin=0, vmax=3)
    axes[0].set_title('Error Map Overlay', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Legend
    patches = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative')
    ]
    axes[0].legend(handles=patches, loc='upper right', fontsize=10)
    
    # Error map only
    axes[1].imshow(error_map, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title('Error Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_confidence_map(image, pred, save_path=None):
    """Create confidence/probability map"""
    image_np = image.squeeze().cpu().numpy()
    pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    # Normalize image
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Confidence map
    im = axes[1].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Prediction Confidence Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_boundary_overlay(image, mask, pred, threshold=0.5, save_path=None):
    """Create boundary overlay visualization"""
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = (torch.sigmoid(pred) > threshold).float().squeeze().cpu().numpy()
    
    # Normalize image
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    # Get boundaries
    pred_boundary = get_boundary(pred_np)
    gt_boundary = get_boundary(mask_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # GT boundary
    axes[1].imshow(image_np, cmap='gray')
    axes[1].contour(gt_boundary, colors='green', linewidths=2)
    axes[1].set_title('Ground Truth Boundary', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction boundary
    axes[2].imshow(image_np, cmap='gray')
    axes[2].contour(pred_boundary, colors='red', linewidths=2)
    axes[2].set_title('Prediction Boundary', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_comprehensive_visualization(image, mask, pred, threshold=0.5, metrics=None, save_path=None):
    """Create comprehensive 6-panel visualization"""
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = (torch.sigmoid(pred) > threshold).float().squeeze().cpu().numpy()
    pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    # Normalize image
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    fig = plt.figure(figsize=(20, 12))
    # Minimal spacing: wspace=0.01 for tight columns, hspace=0.08 to prevent label overlap
    gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.01, left=0.005, right=0.995, top=0.96, bottom=0.04)
    
    # Row 1: Basic visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=11, fontweight='bold', pad=2)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_np, cmap='gray')
    ax2.set_title('Ground Truth', fontsize=11, fontweight='bold', pad=2)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_np, cmap='gray')
    title = 'Prediction'
    if metrics:
        title += f'\nIoU: {metrics.get("jaccard", 0):.3f}, Dice: {metrics.get("dice", 0):.3f}'
    ax3.set_title(title, fontsize=11, fontweight='bold', pad=2)
    ax3.axis('off')
    
    # Row 2: Advanced visualizations
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(image_np, cmap='gray')
    ax4.imshow(pred_np, alpha=0.5, cmap='Blues')
    ax4.set_title('Prediction Overlay', fontsize=11, fontweight='bold', pad=2)
    ax4.axis('off')
    
    # Error map
    ax5 = fig.add_subplot(gs[1, 1])
    error_map = np.zeros_like(mask_np, dtype=np.uint8)
    error_map[(mask_np > 0.5) & (pred_np > 0.5)] = 1  # TP
    error_map[(mask_np <= 0.5) & (pred_np > 0.5)] = 2  # FP
    error_map[(mask_np > 0.5) & (pred_np <= 0.5)] = 3  # FN
    colors = ['black', 'green', 'red', 'blue']
    cmap = ListedColormap(colors)
    ax5.imshow(image_np, cmap='gray')
    ax5.imshow(error_map, alpha=0.6, cmap=cmap, vmin=0, vmax=3)
    ax5.set_title('Error Map', fontsize=11, fontweight='bold', pad=2)
    ax5.axis('off')
    
    # Confidence map
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Confidence Map', fontsize=11, fontweight='bold', pad=2)
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.01)
    
    if save_path:
        # Minimal padding: pad_inches reduced from 0.1 to 0.01
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, facecolor='white')
    plt.close()

def calculate_enhanced_metrics(pred, target, threshold=0.5):
    """Calculate enhanced metrics including HD95, ASD, VS"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    metrics = {}
    
    # Convert to numpy for scipy operations
    pred_np = pred_binary.squeeze().cpu().numpy()
    target_np = target_binary.squeeze().cpu().numpy()
    
    # Basic metrics (from original calculate_metrics)
    intersection = (pred_binary * target_binary).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3)) - intersection
    
    jaccard = (intersection + 1e-6) / (union + 1e-6)
    true_positives = intersection
    false_positives = pred_binary.sum(dim=(2, 3)) - intersection
    false_negatives = target_binary.sum(dim=(2, 3)) - intersection
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3)) + 1e-6)
    
    metrics['jaccard'] = jaccard.mean().item()
    metrics['precision'] = precision.mean().item()
    metrics['recall'] = recall.mean().item()
    metrics['f1'] = f1.mean().item()
    metrics['dice'] = dice.mean().item()
    
    # Enhanced metrics
    hd_list = []
    hd95_list = []
    asd_list = []
    vs_list = []
    
    for i in range(pred_np.shape[0]):
        p = pred_np[i]
        t = target_np[i]
        
        hd, hd95 = calculate_hausdorff_distance(p, t)
        asd = calculate_average_surface_distance(p, t)
        vs = calculate_volume_similarity(p, t)
        
        hd_list.append(hd)
        hd95_list.append(hd95)
        asd_list.append(asd)
        vs_list.append(vs)
    
    metrics['hausdorff'] = np.mean(hd_list)
    metrics['hausdorff95'] = np.mean(hd95_list)
    metrics['asd'] = np.mean(asd_list)
    metrics['volume_similarity'] = np.mean(vs_list)
    
    # MSD (from original)
    def compute_msd(pred_mask, target_mask):
        pred_dist = distance_transform_edt(pred_mask.cpu().numpy())
        target_dist = distance_transform_edt(target_mask.cpu().numpy())
        return np.mean(np.abs(pred_dist - target_dist))
    
    msd = torch.tensor([compute_msd(p, t) for p, t in zip(pred_binary, target_binary)])
    metrics['msd'] = msd.mean().item()
    
    return metrics

def create_metrics_distribution_plot(metrics_list, save_path=None):
    """Create box plots showing distribution of metrics"""
    df = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metric_names = ['jaccard', 'dice', 'precision', 'recall', 'f1', 'hausdorff95']
    metric_labels = ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'HD95']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        if metric in df.columns:
            axes[idx].boxplot(df[metric], vert=True)
            axes[idx].set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(label)
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_roc_curve(y_true, y_pred_probs, save_path=None):
    """Create ROC curve"""
    y_true_flat = y_true.flatten().cpu().numpy()
    y_pred_flat = torch.sigmoid(y_pred_probs).flatten().cpu().numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_enhanced_visualizations(model, test_loader, device, save_dir, threshold=0.5, num_samples=10):
    """Generate all enhanced visualizations"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    (save_dir / 'overlays').mkdir(exist_ok=True)
    (save_dir / 'error_maps').mkdir(exist_ok=True)
    (save_dir / 'confidence_maps').mkdir(exist_ok=True)
    (save_dir / 'boundary_overlays').mkdir(exist_ok=True)
    (save_dir / 'comprehensive').mkdir(exist_ok=True)
    
    all_metrics = []
    all_preds = []
    all_targets = []
    all_probs = []
    
    sample_count = 0
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Generating visualizations")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            for j in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                image = images[j]
                mask = masks[j]
                output = outputs[j]
                prob = probs[j]
                
                # Calculate metrics
                metrics = calculate_enhanced_metrics(output.unsqueeze(0), mask.unsqueeze(0), threshold)
                all_metrics.append(metrics)
                
                # Save visualizations
                idx = f"{i}_{j}"
                
                create_overlay_visualization(
                    image, mask, output, threshold,
                    save_dir / 'overlays' / f'overlay_{idx}.png'
                )
                
                create_error_map(
                    image, mask, output, threshold,
                    save_dir / 'error_maps' / f'error_{idx}.png'
                )
                
                create_confidence_map(
                    image, output,
                    save_dir / 'confidence_maps' / f'confidence_{idx}.png'
                )
                
                create_boundary_overlay(
                    image, mask, output, threshold,
                    save_dir / 'boundary_overlays' / f'boundary_{idx}.png'
                )
                
                create_comprehensive_visualization(
                    image, mask, output, threshold, metrics,
                    save_dir / 'comprehensive' / f'comprehensive_{idx}.png'
                )
                
                all_preds.append(output.cpu())
                all_targets.append(mask.cpu())
                all_probs.append(prob.cpu())
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Create summary visualizations
    logger.info("Creating summary visualizations...")
    
    # Metrics distribution
    create_metrics_distribution_plot(
        all_metrics,
        save_dir / 'metrics_distribution.png'
    )
    
    # ROC curve (using all data)
    if len(all_probs) > 0:
        all_probs_tensor = torch.stack(all_probs)
        all_targets_tensor = torch.stack(all_targets)
        create_roc_curve(
            all_targets_tensor, all_probs_tensor,
            save_dir / 'roc_curve.png'
        )
    
    # Save enhanced metrics
    df_metrics = pd.DataFrame(all_metrics)
    avg_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()
    
    metrics_file = save_dir / 'enhanced_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Enhanced Segmentation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write("Average Metrics:\n")
        f.write("-" * 30 + "\n")
        for key, value in avg_metrics.items():
            std_val = std_metrics.get(key, 0)
            f.write(f"{key.capitalize()}: {value:.4f} ± {std_val:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
    
    logger.info(f"Enhanced visualizations saved to {save_dir}")
    logger.info(f"Average IoU: {avg_metrics['jaccard']:.4f}")
    logger.info(f"Average Dice: {avg_metrics['dice']:.4f}")
    logger.info(f"Average HD95: {avg_metrics['hausdorff95']:.4f}")
    
    return avg_metrics, all_metrics

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Enhanced Visualizations for Cell Segmentation')
    
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with test images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory with test masks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--save_dir', type=str, default='enhanced_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of sample images to visualize')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Number of base channels in the model')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    test_dataset = CellSegmentationDataset(
        args.test_dir,
        args.mask_dir,
        img_size=args.img_size,
        is_training=False
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
    
    # Generate visualizations
    generate_enhanced_visualizations(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        threshold=args.threshold,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()

