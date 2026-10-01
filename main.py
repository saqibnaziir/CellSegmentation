import os
import random
import logging
from pathlib import Path
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import CellSegmentationDataset, ProgressiveSize
# from model import get_model
# from model import EfficientB6AttentionUnet
from model import AttentionUnet

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- reproducibility ----------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed}")

# ---------- loss ----------
def dice_ce_loss(pred, target, smooth=1.0):
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2, 3))
    dice = 1 - ((2. * intersection + smooth) /
                (pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()
    return bce + dice

# ---------- simple boundary loss ----------
class BoundaryLoss(nn.Module):
    """Approximate boundary loss (distance-weighted BCE)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: logits (B,1,H,W)   target: binary mask (B,1,H,W)
        prob = torch.sigmoid(pred)
        # simple 3×3 Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32, device=target.device).view(1,1,3,3)

        gx_t = F.conv2d(target, sobel_x, padding=1).abs()
        gy_t = F.conv2d(target, sobel_y, padding=1).abs()
        edge = (gx_t + gy_t).clamp(0, 1)            # boundary mask

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # weight pixels near the boundary
        w = 1 + 2 * edge
        return (bce * w).mean()

boundary = BoundaryLoss()

def tversky_loss(p, t, alpha=0.3, beta=0.7, smooth=1):
    p = torch.sigmoid(p).flatten(1)
    t = t.flatten(1)
    tp = (p * t).sum(1)
    fp = (p * (1-t)).sum(1)
    fn = ((1-p) * t).sum(1)
    score = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
    return 1 - score.mean()

def combined_loss(pred, target):
    loss = 0.7 * dice_ce_loss(pred, target)
    loss += 0.3 * boundary(pred, target)
    # loss += 0.3 * tversky_loss(pred, target)

    ###### hard-negative mining
    # with torch.no_grad():
    #     prob = torch.sigmoid(pred)
    #     weights = 1 + 3 * torch.abs(prob - target)
    # return (loss * weights).mean()
    return loss

# ---------- metrics ----------
def dice_coef(pred, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2. * intersection + smooth) /
            (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()

# ---------- training ----------
def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    epoch_loss = epoch_dice = 0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss = combined_loss(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_dice += dice_coef(outputs.detach(), masks).item()
    return epoch_loss / len(loader), epoch_dice / len(loader)

def validate(model, loader, device):
    model.eval()
    val_loss = val_dice = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_coef(outputs, masks).item()
    return val_loss / len(loader), val_dice / len(loader)

# ---------- utils ----------
def plot_and_save(train_losses, val_losses, train_dice, val_dice, path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss'); plt.legend(); plt.xlabel('epoch')
    plt.subplot(1, 2, 2)
    plt.plot(train_dice, label='Train')
    plt.plot(val_dice, label='Val')
    plt.title('Dice'); plt.legend(); plt.xlabel('epoch')
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_checkpoint(state, is_best, path):
    torch.save(state, path)
    if is_best:
        torch.save(state, Path(path).parent / "best_model.pth")

def print_model_info(model, device, input_size=(1, 256, 256)):
    """Print concise model information"""
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    
    # Get encoder information
    encoder_name = getattr(model.encoder, 'encoder_name', f"{model.encoder_type.capitalize()} Encoder")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"🔹 ARCHITECTURE:")
    print(f"   Encoder: {encoder_name}")
    print(f"   Decoder: Attention U-Net Decoder")
    print(f"   Input: {input_size[0]} channel, {input_size[1]}×{input_size[2]} images")
    print(f"   Output: 1 channel binary segmentation masks")
    
    print(f"\n🔹 PARAMETERS:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"   Decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    
    print(f"\n🔹 ENCODER DETAILS:")
    print(f"   Output channels per stage: {model.encoder.out_channels}")
    print(f"   Upsampling stages: {len(model.decoder.up_layers)}")
    print(f"   Attention gates: {sum(1 for gate in model.decoder.att_gates if gate is not None)}")
    
    print("="*80)

# ---------- main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str, default='data/original')
    parser.add_argument('--mask_dir', type=str, default='data/mask')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--workers', type=int, default=4)
    
    parser.add_argument('--encoder_type', type=str, default='efficientnet', choices=['efficientnet', 'custom'],
                    help='Choose encoder type: efficientnet (default) or custom')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained weights for efficientnet')
    parser.add_argument('--detailed_summary', action='store_true', help='Print detailed model summary (can be very long)')

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # dataset & progressive size
    base_dataset = CellSegmentationDataset(args.original_dir, args.mask_dir, img_size=256)
    train_size = int(0.8 * len(base_dataset))
    val_size   = len(base_dataset) - train_size
    train_subset, val_subset = random_split(
        base_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # ps = ProgressiveSize(sizes=(256, 512, 768, 1024), epochs_per_size=20)
    ps = ProgressiveSize(sizes=(256,256,256, 256), epochs_per_size=60)

    # Create model with the specified encoder type
    model = AttentionUnet(
        encoder_type=args.encoder_type,
        pretrained=not args.no_pretrained
    ).to(device)

    # Print model information - choose detailed or concise summary
    if args.detailed_summary:
        print("\n" + "="*80)
        print("INITIALIZING MODEL AND PRINTING DETAILED SUMMARY")
        print("="*80)
        from model import print_model_summary
        print_model_summary(model, input_size=(1, 256, 256), device=device)
    else:
        # Print concise model info
        print_model_info(model, device, input_size=(1, 256, 256))
    
    # Log model configuration to file
    logger.info(f"Model Configuration:")
    logger.info(f"  - Encoder: {args.encoder_type}")
    logger.info(f"  - Pretrained: {not args.no_pretrained if args.encoder_type == 'efficientnet' else 'N/A'}")
    logger.info(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - Encoder channels: {model.encoder.out_channels}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # total_steps = args.epochs * (train_size // args.batch_size) ####Changed it for cell seg data
    # With this:
    steps_per_epoch = max(1, train_size // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/runs")

    train_losses, val_losses, train_dice, val_dice = [], [], [], []
    best_dice = 0.0

    for epoch in range(args.epochs):
        # resize dataset on-the-fly
        current_size = ps(epoch)
        base_dataset.img_size = current_size

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va_loss, va_dice = validate(model, val_loader, device)

        scheduler.step()   # OneCycle step per batch

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_dice.append(tr_dice);  val_dice.append(va_dice)

        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', va_loss, epoch)
        writer.add_scalar('Dice/train', tr_dice, epoch)
        writer.add_scalar('Dice/val', va_dice, epoch)

        # Get encoder name for logging
        encoder_name = getattr(model.encoder, 'encoder_name', f"{model.encoder_type.capitalize()}")
        
        logger.info(f"Epoch {epoch+1:02d}/{args.epochs} | "
                    f"{encoder_name} | train {tr_dice:.4f} | val {va_dice:.4f} | size {current_size}")

        is_best = va_dice > best_dice
        best_dice = max(best_dice, va_dice)
        
        # Save checkpoint with encoder type information
        save_checkpoint({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'train_loss': tr_loss, 
            'val_loss': va_loss,
            'train_dice': tr_dice, 
            'val_dice': va_dice,
            'encoder_type': args.encoder_type,
            'pretrained': not args.no_pretrained if args.encoder_type == 'efficientnet' else False,
            'model_config': {
                'encoder_channels': model.encoder.out_channels,
                'total_params': sum(p.numel() for p in model.parameters()),
                'encoder_params': sum(p.numel() for p in model.encoder.parameters()),
                'decoder_params': sum(p.numel() for p in model.decoder.parameters()),
            }
        }, is_best, f"{args.checkpoint_dir}/ckpt_epoch{epoch+1}.pth")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            plot_and_save(train_losses, val_losses, train_dice, val_dice,
                          f"{args.checkpoint_dir}/metrics.png")
    
    writer.close()
    
    # Final summary
    print(f"\n🔹 TRAINING COMPLETED!")
    print(f"   Best validation Dice: {best_dice:.4f}")
    print(f"   Encoder used: {getattr(model.encoder, 'encoder_name', args.encoder_type)}")
    print(f"   Total epochs: {args.epochs}")
    print(f"   Final model saved to: {args.checkpoint_dir}/best_model.pth")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
