"""
Operational Efficiency Metrics Measurement Script
Measures inference time, memory usage, and FLOPs for cell segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import logging
from pathlib import Path
import psutil
import os
from tqdm import tqdm
import pandas as pd

from dataset import CellSegmentationDataset
from model import get_model

# Try to import FLOPs calculation libraries
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: ptflops not available. Install with: pip install ptflops")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_time(model, test_loader, device, num_warmup=10, num_runs=100):
    """
    Measure inference time with proper warmup and statistics
    
    Returns:
        dict with inference time statistics
    """
    model.eval()
    
    # Warmup runs
    logger.info(f"Warming up with {num_warmup} runs...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_warmup:
                break
            images = images.to(device)
            _ = model(images)
    
    # Synchronize GPU if available
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing runs
    logger.info(f"Measuring inference time over {num_runs} runs...")
    inference_times = []
    batch_times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Timing inference")):
            if i >= num_runs:
                break
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Synchronize before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time batch inference
            start_time = time.perf_counter()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            
            # Time per image
            per_image_time = batch_time / batch_size
            inference_times.append(per_image_time)
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    batch_times = np.array(batch_times)
    
    stats = {
        'mean_per_image_ms': np.mean(inference_times) * 1000,
        'std_per_image_ms': np.std(inference_times) * 1000,
        'min_per_image_ms': np.min(inference_times) * 1000,
        'max_per_image_ms': np.max(inference_times) * 1000,
        'median_per_image_ms': np.median(inference_times) * 1000,
        'p95_per_image_ms': np.percentile(inference_times, 95) * 1000,
        'p99_per_image_ms': np.percentile(inference_times, 99) * 1000,
        'mean_batch_ms': np.mean(batch_times) * 1000,
        'std_batch_ms': np.std(batch_times) * 1000,
        'throughput_fps': 1.0 / np.mean(inference_times),  # Frames per second
        'num_runs': len(inference_times)
    }
    
    return stats


def measure_memory_usage(model, test_loader, device):
    """
    Measure memory usage during inference
    
    Returns:
        dict with memory statistics
    """
    model.eval()
    
    # Get initial memory state
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        initial_gpu_memory = 0
    
    process = psutil.Process(os.getpid())
    initial_cpu_memory = process.memory_info().rss / 1024**2  # MB
    
    # Model size
    model_size_mb = get_model_size(model)
    total_params, trainable_params = count_parameters(model)
    
    # Run inference to measure peak memory
    logger.info("Measuring memory usage during inference...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Measuring memory")):
            if i >= 5:  # Run a few batches
                break
            
            images = images.to(device)
            _ = model(images)
    
    # Get peak memory
    if device.type == 'cuda':
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        gpu_memory_used = peak_gpu_memory - initial_gpu_memory
    else:
        peak_gpu_memory = 0
        gpu_memory_used = 0
    
    peak_cpu_memory = process.memory_info().rss / 1024**2  # MB
    cpu_memory_used = peak_cpu_memory - initial_cpu_memory
    
    memory_stats = {
        'model_size_mb': model_size_mb,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'initial_gpu_memory_mb': initial_gpu_memory,
        'peak_gpu_memory_mb': peak_gpu_memory,
        'gpu_memory_used_mb': gpu_memory_used,
        'initial_cpu_memory_mb': initial_cpu_memory,
        'peak_cpu_memory_mb': peak_cpu_memory,
        'cpu_memory_used_mb': cpu_memory_used
    }
    
    return memory_stats


def calculate_flops(model, input_size, device):
    """
    Calculate FLOPs (Floating Point Operations) for the model
    
    Returns:
        dict with FLOPs statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    flops_stats = {}
    
    # Try thop first
    if THOP_AVAILABLE:
        try:
            logger.info("Calculating FLOPs using thop...")
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_stats['flops'] = flops
            flops_stats['flops_giga'] = flops / 1e9
            flops_stats['flops_formatted'] = clever_format([flops], "%.3f")[0]
            flops_stats['method'] = 'thop'
            logger.info(f"FLOPs (thop): {flops_stats['flops_formatted']}")
        except Exception as e:
            logger.warning(f"thop calculation failed: {e}")
    
    # Try ptflops as backup
    if PTFLOPS_AVAILABLE and 'flops' not in flops_stats:
        try:
            logger.info("Calculating FLOPs using ptflops...")
            flops, params = get_model_complexity_info(
                model, 
                tuple(input_size),
                as_strings=False,
                print_per_layer_stat=False
            )
            flops_stats['flops'] = flops
            flops_stats['flops_giga'] = flops / 1e9
            flops_stats['flops_formatted'] = f"{flops / 1e9:.3f} GFLOPs"
            flops_stats['method'] = 'ptflops'
            logger.info(f"FLOPs (ptflops): {flops_stats['flops_formatted']}")
        except Exception as e:
            logger.warning(f"ptflops calculation failed: {e}")
    
    if 'flops' not in flops_stats:
        logger.error("Could not calculate FLOPs. Please install thop or ptflops.")
        flops_stats['flops'] = None
        flops_stats['flops_giga'] = None
        flops_stats['flops_formatted'] = 'N/A'
        flops_stats['method'] = 'none'
    
    return flops_stats


def generate_metrics_table(inference_stats, memory_stats, flops_stats, save_path=None):
    """
    Generate a comprehensive table with all operational efficiency metrics
    """
    metrics_data = []
    
    # Inference Time Metrics
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Mean Inference Time (per image)',
        'Value': f"{inference_stats['mean_per_image_ms']:.2f} ms",
        'Description': 'Average time to process a single image'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Std Inference Time (per image)',
        'Value': f"{inference_stats['std_per_image_ms']:.2f} ms",
        'Description': 'Standard deviation of inference time per image'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Min Inference Time (per image)',
        'Value': f"{inference_stats['min_per_image_ms']:.2f} ms",
        'Description': 'Minimum inference time per image'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Max Inference Time (per image)',
        'Value': f"{inference_stats['max_per_image_ms']:.2f} ms",
        'Description': 'Maximum inference time per image'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Median Inference Time (per image)',
        'Value': f"{inference_stats['median_per_image_ms']:.2f} ms",
        'Description': 'Median inference time per image'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': '95th Percentile Inference Time',
        'Value': f"{inference_stats['p95_per_image_ms']:.2f} ms",
        'Description': '95th percentile of inference time (95% of inferences are faster)'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': '99th Percentile Inference Time',
        'Value': f"{inference_stats['p99_per_image_ms']:.2f} ms",
        'Description': '99th percentile of inference time (99% of inferences are faster)'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Mean Batch Inference Time',
        'Value': f"{inference_stats['mean_batch_ms']:.2f} ms",
        'Description': 'Average time to process a batch of images'
    })
    
    metrics_data.append({
        'Category': 'Inference Time',
        'Metric': 'Throughput (FPS)',
        'Value': f"{inference_stats['throughput_fps']:.2f} fps",
        'Description': 'Frames per second - number of images processed per second'
    })
    
    # Memory Metrics
    metrics_data.append({
        'Category': 'Memory Usage',
        'Metric': 'Model Size',
        'Value': f"{memory_stats['model_size_mb']:.2f} MB",
        'Description': 'Total size of model parameters and buffers in memory'
    })
    
    metrics_data.append({
        'Category': 'Memory Usage',
        'Metric': 'Total Parameters',
        'Value': f"{memory_stats['total_parameters']:,}",
        'Description': 'Total number of model parameters'
    })
    
    metrics_data.append({
        'Category': 'Memory Usage',
        'Metric': 'Trainable Parameters',
        'Value': f"{memory_stats['trainable_parameters']:,}",
        'Description': 'Number of trainable parameters'
    })
    
    if memory_stats['peak_gpu_memory_mb'] > 0:
        metrics_data.append({
            'Category': 'Memory Usage',
            'Metric': 'Peak GPU Memory',
            'Value': f"{memory_stats['peak_gpu_memory_mb']:.2f} MB",
            'Description': 'Peak GPU memory usage during inference'
        })
        
        metrics_data.append({
            'Category': 'Memory Usage',
            'Metric': 'GPU Memory Used',
            'Value': f"{memory_stats['gpu_memory_used_mb']:.2f} MB",
            'Description': 'Additional GPU memory used during inference (excluding initial)'
        })
    
    metrics_data.append({
        'Category': 'Memory Usage',
        'Metric': 'Peak CPU Memory',
        'Value': f"{memory_stats['peak_cpu_memory_mb']:.2f} MB",
        'Description': 'Peak CPU memory (RAM) usage during inference'
    })
    
    metrics_data.append({
        'Category': 'Memory Usage',
        'Metric': 'CPU Memory Used',
        'Value': f"{memory_stats['cpu_memory_used_mb']:.2f} MB",
        'Description': 'Additional CPU memory used during inference (excluding initial)'
    })
    
    # FLOPs Metrics
    if flops_stats.get('flops') is not None:
        metrics_data.append({
            'Category': 'Computational Complexity',
            'Metric': 'FLOPs (Total)',
            'Value': f"{flops_stats['flops']:,}",
            'Description': 'Total Floating Point Operations for one forward pass'
        })
        
        metrics_data.append({
            'Category': 'Computational Complexity',
            'Metric': 'GFLOPs',
            'Value': f"{flops_stats['flops_giga']:.3f} GFLOPs",
            'Description': 'Giga Floating Point Operations (billions of operations)'
        })
        
        metrics_data.append({
            'Category': 'Computational Complexity',
            'Metric': 'FLOPs Calculation Method',
            'Value': flops_stats.get('method', 'N/A'),
            'Description': 'Library used for FLOPs calculation'
        })
    else:
        metrics_data.append({
            'Category': 'Computational Complexity',
            'Metric': 'FLOPs',
            'Value': 'N/A',
            'Description': 'FLOPs calculation not available. Install thop or ptflops.'
        })
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Print table
    print("\n" + "="*100)
    print("OPERATIONAL EFFICIENCY METRICS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save to file
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save as CSV
        csv_path = save_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")
        
        # Save as formatted text
        txt_path = save_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("OPERATIONAL EFFICIENCY METRICS\n")
            f.write("="*100 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*100 + "\n")
            f.write("\nDESCRIPTIONS:\n")
            f.write("-"*100 + "\n")
            for _, row in df.iterrows():
                f.write(f"\n{row['Metric']}:\n")
                f.write(f"  {row['Description']}\n")
        logger.info(f"Formatted metrics saved to {txt_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Measure Operational Efficiency Metrics')
    
    # Data paths
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with test images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory with test masks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Number of base channels in the model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads')
    
    # Measurement parameters
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='Number of warmup runs before timing')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of runs for timing measurements')
    parser.add_argument('--save_dir', type=str, default='efficiency_metrics',
                        help='Directory to save metrics')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
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
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    logger.info("Creating model...")
    model = get_model(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Measure metrics
    logger.info("\n" + "="*80)
    logger.info("MEASURING OPERATIONAL EFFICIENCY METRICS")
    logger.info("="*80 + "\n")
    
    # 1. Measure inference time
    logger.info("1. Measuring Inference Time...")
    inference_stats = measure_inference_time(
        model, test_loader, device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs
    )
    
    # 2. Measure memory usage
    logger.info("\n2. Measuring Memory Usage...")
    memory_stats = measure_memory_usage(model, test_loader, device)
    
    # 3. Calculate FLOPs
    logger.info("\n3. Calculating FLOPs...")
    input_size = (1, args.img_size, args.img_size)
    flops_stats = calculate_flops(model, input_size, device)
    
    # Generate and save metrics table
    logger.info("\n4. Generating Metrics Table...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    df = generate_metrics_table(
        inference_stats,
        memory_stats,
        flops_stats,
        save_path=save_dir / 'operational_efficiency_metrics'
    )
    
    logger.info("\n" + "="*80)
    logger.info("MEASUREMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {save_dir}")
    logger.info("\nKey Metrics Summary:")
    logger.info(f"  Mean Inference Time: {inference_stats['mean_per_image_ms']:.2f} ms")
    logger.info(f"  Throughput: {inference_stats['throughput_fps']:.2f} fps")
    logger.info(f"  Model Size: {memory_stats['model_size_mb']:.2f} MB")
    logger.info(f"  Total Parameters: {memory_stats['total_parameters']:,}")
    if flops_stats.get('flops_giga'):
        logger.info(f"  GFLOPs: {flops_stats['flops_giga']:.3f} GFLOPs")


if __name__ == '__main__':
    main()

