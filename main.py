############################# Updated Training with Gradient Clipping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import time
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import logging
from datetime import datetime
import multiprocessing
import torch.nn.utils as torch_utils

# Set multiprocessing start method
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Import our components
from dataset import CellSegmentationDataset
from model import get_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

def check_cuda():
    """Check CUDA availability and set device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA is available!")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        
        # Set optimized CUDA settings
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Using CPU instead.")
    
    return device

def dice_coef(pred, target, threshold=0.5, smooth=1.0):
    """Dice coefficient for evaluation"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2. * intersection + smooth) / 
            (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()

####################################################################Loss Functions
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-bce)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()

def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5, focal_weight=0.0):
    """Combined loss function with multiple components"""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    
    if focal_weight > 0:
        focal = focal_loss(pred, target)
        return bce_weight * bce + dice_weight * dice + focal_weight * focal
    else:
        return bce_weight * bce + dice_weight * dice

##############################################################################

def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip_value=1.0, 
                   loss_weights=None):
    """Train model for one epoch with mixed precision and gradient clipping"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_grad_norm = 0
    num_batches = len(loader)
    
    # Default loss weights
    if loss_weights is None:
        loss_weights = {'bce_weight': 0.5, 'dice_weight': 0.5, 'focal_weight': 0.0}
    
    progress_bar = tqdm(loader, desc="Training")
    
    for i, (images, masks) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = combined_loss(outputs, masks, **loss_weights)
        
        # Backward pass with scaling
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping BEFORE optimizer step
        if grad_clip_value > 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients
            grad_norm = torch_utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            epoch_grad_norm += grad_norm.item()
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        batch_loss = loss.item()
        batch_dice = dice_coef(outputs.detach(), masks).item()
        
        epoch_loss += batch_loss
        epoch_dice += batch_dice
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{batch_loss:.4f}",
            'dice': f"{batch_dice:.4f}",
            'grad_norm': f"{grad_norm.item():.3f}" if grad_clip_value > 0 else "N/A"
        })
        
        # Free up memory
        del images, masks, outputs, loss
        if i % 5 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = epoch_loss / num_batches
    avg_dice = epoch_dice / num_batches
    avg_grad_norm = epoch_grad_norm / num_batches if grad_clip_value > 0 else 0
    
    return avg_loss, avg_dice, avg_grad_norm

def validate(model, loader, device, loss_weights=None):
    """Validate model"""
    model.eval()
    val_loss = 0
    val_dice = 0
    num_batches = len(loader)
    
    # Default loss weights
    if loss_weights is None:
        loss_weights = {'bce_weight': 0.5, 'dice_weight': 0.5, 'focal_weight': 0.0}
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = combined_loss(outputs, masks, **loss_weights)
            
            val_loss += loss.item()
            val_dice += dice_coef(outputs, masks).item()
            
            # Free memory
            del images, masks, outputs, loss
    
    avg_loss = val_loss / num_batches
    avg_dice = val_dice / num_batches
    
    return avg_loss, avg_dice

def plot_metrics(train_losses, val_losses, train_dice, val_dice, grad_norms, save_path):
    """Plot training metrics including gradient norms"""
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_dice, label='Train Dice')
    plt.plot(val_dice, label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if grad_norms:
        plt.plot(grad_norms, label='Gradient Norm', color='red')
        plt.title('Gradient Norm')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Gradient Clipping Disabled', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Gradient Norm (Disabled)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    torch.save(state, filename)
    if is_best:
        best_filename = 'best_model.pth.tar'
        import shutil
        shutil.copyfile(filename, best_filename)
        logger.info(f"Saved new best model to {best_filename}")

def create_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                num_epochs=50, checkpoint_dir='checkpoints', grad_clip_value=1.0,
                loss_weights=None, warmup_epochs=0):
    """Full training loop with checkpoints, logging, and gradient clipping"""
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Initialize TensorBoard writer
    log_dir = Path(checkpoint_dir) / 'tensorboard_logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    grad_norms = [] if grad_clip_value > 0 else None
    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Setup warmup scheduler if needed
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = create_warmup_scheduler(optimizer, warmup_epochs, num_epochs)
        logger.info(f"Using warmup scheduler for {warmup_epochs} epochs")
    
    # Default loss weights
    if loss_weights is None:
        loss_weights = {'bce_weight': 0.5, 'dice_weight': 0.5, 'focal_weight': 0.0}
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Gradient clipping: {'Enabled' if grad_clip_value > 0 else 'Disabled'}")
    if grad_clip_value > 0:
        logger.info(f"Gradient clip value: {grad_clip_value}")
    logger.info(f"Loss weights: {loss_weights}")
    
    # Log model graph
    sample_input = next(iter(train_loader))[0].to(device)
    writer.add_graph(model, sample_input)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        train_loss, train_dice, avg_grad_norm = train_one_epoch(
            model, train_loader, optimizer, device, scaler, 
            grad_clip_value, loss_weights
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, device, loss_weights)
        
        # Update learning rate
        if warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)
        if grad_norms is not None:
            grad_norms.append(avg_grad_norm)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if grad_clip_value > 0:
            writer.add_scalar('Gradient_norm', avg_grad_norm, epoch)
        
        # Log some sample predictions every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_masks = sample_masks.to(device)
                sample_preds = torch.sigmoid(model(sample_images))
                
                # Log images
                writer.add_images('Images/Original', sample_images, epoch)
                writer.add_images('Images/Masks', sample_masks, epoch)
                writer.add_images('Images/Predictions', sample_preds, epoch)
            model.train()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        if grad_clip_value > 0:
            logger.info(f"Avg Gradient Norm: {avg_grad_norm:.4f}")
        
        # Check if best model
        is_best = val_dice > best_val_dice
        best_val_dice = max(val_dice, best_val_dice)
        best_val_loss = min(val_loss, best_val_loss)
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'grad_clip_value': grad_clip_value,
            'loss_weights': loss_weights
        }, is_best, filename=f"{checkpoint_dir}/checkpoint_epoch{epoch+1}.pth.tar")
        
        # Plot and save metrics
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_metrics(
                train_losses, val_losses, 
                train_dice_scores, val_dice_scores, grad_norms,
                f"{checkpoint_dir}/metrics_epoch{epoch+1}.png"
            )
    
    # Final plot
    plot_metrics(
        train_losses, val_losses, 
        train_dice_scores, val_dice_scores, grad_norms,
        f"{checkpoint_dir}/final_metrics.png"
    )
    
    # Close TensorBoard writer
    writer.close()
    
    return train_losses, val_losses, train_dice_scores, val_dice_scores

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cell Segmentation Training')
    
    # Data paths
    parser.add_argument('--original_dir', type=str, default='data/original',
                        help='Directory with original images')
    parser.add_argument('--mask_dir', type=str, default='data/mask',
                        help='Directory with mask images')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for dataloader')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Number of base channels in the model')
    
    # New gradient clipping and training parameters
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='reduce', 
                        choices=['reduce', 'cosine'],
                        help='Learning rate scheduler type')
    
    # Loss function weights
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Dice loss weight')
    parser.add_argument('--focal_weight', type=float, default=0.0,
                        help='Focal loss weight')
    
    # Add TensorBoard port argument
    parser.add_argument('--tensorboard_port', type=int, default=6006,
                        help='Port for TensorBoard server')
    
    return parser.parse_args()

def main():
    # Get arguments
    args = get_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Check CUDA
    device = check_cuda()
    
    # Create dataset
    logger.info("Creating datasets...")
    dataset = CellSegmentationDataset(
        args.original_dir, 
        args.mask_dir,
        img_size=args.img_size
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(args.workers, multiprocessing.cpu_count()),
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(args.workers, multiprocessing.cpu_count()),
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    # Create model
    logger.info("Creating model...")
    model = get_model(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:  # reduce
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
    
    # Loss weights
    loss_weights = {
        'bce_weight': args.bce_weight,
        'dice_weight': args.dice_weight,
        'focal_weight': args.focal_weight
    }
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler = GradScaler()
                scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logger.info(f"No checkpoint found at '{args.resume}'")

    # Print TensorBoard instructions
    logger.info("\nTo view TensorBoard:")
    logger.info(f"1. Run: tensorboard --logdir={args.checkpoint_dir}/tensorboard_logs --port={args.tensorboard_port}")
    logger.info(f"2. Open your browser and go to: http://localhost:{args.tensorboard_port}")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        grad_clip_value=args.grad_clip,
        loss_weights=loss_weights,
        warmup_epochs=args.warmup_epochs
    )

if __name__ == '__main__':
    main()

# ############################# normal conv atent
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# import os
# from tqdm import tqdm
# import numpy as np
# import time
# import random
# import argparse
# import matplotlib.pyplot as plt
# from pathlib import Path
# import warnings
# import logging
# from datetime import datetime
# import multiprocessing
# import torch.nn.functional as F


# # Set multiprocessing start method
# if __name__ == '__main__':
#     multiprocessing.set_start_method('spawn', force=True)

# # Import our components
# from dataset import CellSegmentationDataset
# from model import get_model

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def seed_everything(seed=42):
#     """Set seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     logger.info(f"Set random seed to {seed}")

# def check_cuda():
#     """Check CUDA availability and set device"""
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         logger.info(f"CUDA is available!")
#         logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
#         logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
#         logger.info(f"Current GPU: {torch.cuda.current_device()}")
        
#         # Set optimized CUDA settings
#         torch.backends.cudnn.benchmark = True
#     else:
#         device = torch.device("cpu")
#         logger.info("CUDA is not available. Using CPU instead.")
    
#     return device

# # def dice_loss(pred, target, smooth=1.0):
# #     """Dice loss for segmentation"""
# #     pred = pred.contiguous()
# #     target = target.contiguous()    
    
# #     intersection = (pred * target).sum(dim=2).sum(dim=2)
# #     loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
# #     return loss.mean()

# def dice_coef(pred, target, threshold=0.5, smooth=1.0):
#     """Dice coefficient for evaluation"""
#     pred = (pred > threshold).float()
#     intersection = (pred * target).sum(dim=(2, 3))
#     return ((2. * intersection + smooth) / 
#             (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()


# ####################################################################Losss
# def dice_loss(pred, target, smooth=1.0):
#     pred = torch.sigmoid(pred)
#     pred = pred.contiguous().view(pred.size(0), -1)
#     target = target.contiguous().view(target.size(0), -1)
#     intersection = (pred * target).sum(dim=1)
#     return 1 - ((2. * intersection + smooth) /
#                 (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()

# def combined_loss(pred, target):
#     bce = F.binary_cross_entropy_with_logits(pred, target)
#     dice = dice_loss(pred, target)
#     return bce + dice  

# ##############################################################################

# def train_one_epoch(model, loader, optimizer, device, scaler):
#     """Train model for one epoch with mixed precision"""
#     model.train()
#     epoch_loss = 0
#     epoch_dice = 0
#     progress_bar = tqdm(loader, desc="Training")
    
#     for i, (images, masks) in enumerate(progress_bar):
#         images = images.to(device, non_blocking=True)
#         masks = masks.to(device, non_blocking=True)
        
#         # Mixed precision training
#         with autocast():
#             outputs = model(images)
#             loss = combined_loss(outputs, masks)
        
#         optimizer.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         # Calculate metrics
#         batch_loss = loss.item()
#         batch_dice = dice_coef(outputs.detach(), masks).item()
        
#         epoch_loss += batch_loss
#         epoch_dice += batch_dice
        
#         # Update progress bar
#         progress_bar.set_postfix({
#             'loss': f"{batch_loss:.4f}",
#             'dice': f"{batch_dice:.4f}"
#         })
        
#         # Free up memory
#         del images, masks, outputs, loss
#         if i % 5 == 0:
#             torch.cuda.empty_cache()
    
#     avg_loss = epoch_loss / len(loader)
#     avg_dice = epoch_dice / len(loader)
    
#     return avg_loss, avg_dice

# def validate(model, loader, device):
#     """Validate model"""
#     model.eval()
#     val_loss = 0
#     val_dice = 0
    
#     with torch.no_grad():
#         for images, masks in tqdm(loader, desc="Validation"):
#             images = images.to(device, non_blocking=True)
#             masks = masks.to(device, non_blocking=True)
            
#             outputs = model(images)
#             loss = combined_loss(outputs, masks)
            
#             val_loss += loss.item()
#             val_dice += dice_coef(outputs, masks).item()
            
#             # Free memory
#             del images, masks, outputs, loss
    
#     avg_loss = val_loss / len(loader)
#     avg_dice = val_dice / len(loader)
    
#     return avg_loss, avg_dice

# def plot_metrics(train_losses, val_losses, train_dice, val_dice, save_path):
#     """Plot training metrics"""
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Val Loss')
#     plt.title('Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(train_dice, label='Train Dice')
#     plt.plot(val_dice, label='Val Dice')
#     plt.title('Dice Coefficient')
#     plt.xlabel('Epoch')
#     plt.ylabel('Dice')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     """Save checkpoint"""
#     torch.save(state, filename)
#     if is_best:
#         best_filename = 'best_model.pth.tar'
#         import shutil
#         shutil.copyfile(filename, best_filename)
#         logger.info(f"Saved new best model to {best_filename}")

# def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
#                 num_epochs=50, checkpoint_dir='checkpoints'):
#     """Full training loop with checkpoints and logging"""
#     # Create checkpoint directory
#     Path(checkpoint_dir).mkdir(exist_ok=True)
    
#     # Initialize TensorBoard writer
#     log_dir = Path(checkpoint_dir) / 'tensorboard_logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
#     writer = SummaryWriter(log_dir=str(log_dir))
#     logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
#     # Initialize metrics tracking
#     train_losses = []
#     val_losses = []
#     train_dice_scores = []
#     val_dice_scores = []
#     best_val_loss = float('inf')
#     best_val_dice = 0.0
    
#     # Initialize mixed precision scaler
#     scaler = GradScaler()
    
#     logger.info(f"Starting training for {num_epochs} epochs")
#     logger.info(f"Training samples: {len(train_loader.dataset)}")
#     logger.info(f"Validation samples: {len(val_loader.dataset)}")
#     logger.info(f"Batch size: {train_loader.batch_size}")
    
#     # Log model graph
#     sample_input = next(iter(train_loader))[0].to(device)
#     writer.add_graph(model, sample_input)
    
#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
        
#         # Train one epoch
#         train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, device, scaler)
        
#         # Validate
#         val_loss, val_dice = validate(model, val_loader, device)
        
#         # Update learning rate
#         scheduler.step(val_loss)
        
#         # Record metrics
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_dice_scores.append(train_dice)
#         val_dice_scores.append(val_dice)
        
#         # Log metrics to TensorBoard
#         writer.add_scalar('Loss/train', train_loss, epoch)
#         writer.add_scalar('Loss/val', val_loss, epoch)
#         writer.add_scalar('Dice/train', train_dice, epoch)
#         writer.add_scalar('Dice/val', val_dice, epoch)
#         writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
#         # Log some sample predictions every 5 epochs
#         if epoch % 5 == 0:
#             model.eval()
#             with torch.no_grad():
#                 sample_images, sample_masks = next(iter(val_loader))
#                 sample_images = sample_images.to(device)
#                 sample_masks = sample_masks.to(device)
#                 sample_preds = torch.sigmoid(model(sample_images))
                
#                 # Log images
#                 writer.add_images('Images/Original', sample_images, epoch)
#                 writer.add_images('Images/Masks', sample_masks, epoch)
#                 writer.add_images('Images/Predictions', sample_preds, epoch)
#             model.train()
        
#         # Print epoch summary
#         epoch_time = time.time() - epoch_start_time
#         logger.info(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
#         logger.info(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
#         logger.info(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
#         logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Check if best model
#         is_best = val_dice > best_val_dice
#         best_val_dice = max(val_dice, best_val_dice)
#         best_val_loss = min(val_loss, best_val_loss)
        
#         # Save checkpoint
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'scheduler': scheduler.state_dict(),
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'train_dice': train_dice,
#             'val_dice': val_dice,
#             'best_val_loss': best_val_loss,
#             'best_val_dice': best_val_dice
#         }, is_best, filename=f"{checkpoint_dir}/checkpoint_epoch{epoch+1}.pth.tar")
        
#         # Plot and save metrics
#         if epoch % 5 == 0 or epoch == num_epochs - 1:
#             plot_metrics(
#                 train_losses, val_losses, 
#                 train_dice_scores, val_dice_scores,
#                 f"{checkpoint_dir}/metrics_epoch{epoch+1}.png"
#             )
    
#     # Final plot
#     plot_metrics(
#         train_losses, val_losses, 
#         train_dice_scores, val_dice_scores,
#         f"{checkpoint_dir}/final_metrics.png"
#     )
    
#     # Close TensorBoard writer
#     writer.close()
    
#     return train_losses, val_losses, train_dice_scores, val_dice_scores

# def get_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Cell Segmentation Training')
    
#     # Data paths
#     parser.add_argument('--original_dir', type=str, default='data/original',
#                         help='Directory with original images')
#     parser.add_argument('--mask_dir', type=str, default='data/mask',
#                         help='Directory with mask images')
    
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=4,
#                         help='Batch size for training')
#     parser.add_argument('--epochs', type=int, default=100,
#                         help='Number of epochs to train')
#     parser.add_argument('--lr', type=float, default=0.0001,
#                         help='Learning rate')
#     parser.add_argument('--img_size', type=int, default=256,
#                         help='Image size')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed')
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
#                         help='Directory to save checkpoints')
#     parser.add_argument('--workers', type=int, default=4,
#                         help='Number of worker threads for dataloader')
#     parser.add_argument('--resume', type=str, default=None,
#                         help='Path to checkpoint to resume from')
#     parser.add_argument('--base_channels', type=int, default=64,
#                         help='Number of base channels in the model')
    
#     # Add TensorBoard port argument
#     parser.add_argument('--tensorboard_port', type=int, default=6006,
#                         help='Port for TensorBoard server')
    
#     return parser.parse_args()

# def main():
#     # Get arguments
#     args = get_args()
    
#     # Set random seed
#     seed_everything(args.seed)
    
#     # Check CUDA
#     device = check_cuda()
    
#     # Create dataset
#     logger.info("Creating datasets...")
#     dataset = CellSegmentationDataset(
#         args.original_dir, 
#         args.mask_dir,
#         img_size=args.img_size
#     )
    
#     # Split dataset
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size], 
#         generator=torch.Generator().manual_seed(args.seed)
#     )
    
#     logger.info(f"Dataset split: {train_size} training, {val_size} validation")
    
#     # Create data loaders with proper multiprocessing settings
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=min(args.workers, multiprocessing.cpu_count()),
#         pin_memory=True,
#         persistent_workers=True if args.workers > 0 else False,
#         prefetch_factor=2 if args.workers > 0 else None,
#         drop_last=True  # Drop last incomplete batch
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=False, 
#         num_workers=min(args.workers, multiprocessing.cpu_count()),
#         pin_memory=True,
#         persistent_workers=True if args.workers > 0 else False,
#         prefetch_factor=2 if args.workers > 0 else None
#     )
    
#     # Create model
#     logger.info("Creating model...")
#     model = get_model(
#         in_channels=1,
#         out_channels=1,
#         base_channels=args.base_channels,
        
#     ).to(device)
    
#     # Log model parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
#     # Create optimizer
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
#     # Learning rate scheduler
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, 
#          min_lr=1e-6
#     )
    
#     # Resume from checkpoint if specified
#     start_epoch = 0
#     if args.resume:
#         if os.path.isfile(args.resume):
#             logger.info(f"Loading checkpoint '{args.resume}'")
#             checkpoint = torch.load(args.resume)
#             start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             scheduler.load_state_dict(checkpoint['scheduler'])
#             logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
#         else:
#             logger.info(f"No checkpoint found at '{args.resume}'")

#     # Print TensorBoard instructions
#     logger.info("\nTo view TensorBoard:")
#     logger.info(f"1. Run: tensorboard --logdir={args.checkpoint_dir}/tensorboard_logs --port={args.tensorboard_port}")
#     logger.info(f"2. Open your browser and go to: http://localhost:{args.tensorboard_port}")
    
#     # Train model
#     train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         num_epochs=args.epochs,
#         checkpoint_dir=args.checkpoint_dir
#     )

# if __name__ == '__main__':
#     main() 


