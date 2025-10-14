"""
Detail Enhancement for 8mm Film Footage
Enhances fine details using smart unsharp masking with edge detection
Avoids over-sharpening flat areas and creating artifacts
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import argparse
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def create_edge_mask(frame, blur_radius=5):
    """
    Create an edge mask to focus enhancement on detailed areas
    
    Args:
        frame: Input BGR frame
        blur_radius: Gaussian blur radius for edge detection
    
    Returns:
        Edge mask (0-1 float), higher values = more edges/detail
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients using Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-1
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # Apply slight blur to smooth the mask
    edge_mask = cv2.GaussianBlur(gradient_magnitude, (blur_radius, blur_radius), 0)
    
    return edge_mask.astype(np.float32)


def unsharp_mask(frame, radius=2.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to enhance details
    
    Args:
        frame: Input BGR frame
        radius: Gaussian blur radius (sigma)
        amount: Enhancement strength (1.0 = normal, 2.0 = strong)
        threshold: Minimum difference threshold (0 = all pixels)
    
    Returns:
        Sharpened frame
    """
    # Calculate kernel size from radius (must be odd)
    kernel_size = int(2 * np.ceil(3 * radius) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create blurred version
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), radius)
    
    # Calculate detail layer (difference between original and blurred)
    detail = cv2.subtract(frame, blurred)
    
    # Apply threshold if specified
    if threshold > 0:
        detail = np.where(np.abs(detail) < threshold, 0, detail)
    
    # Add enhanced details back to original
    sharpened = cv2.add(frame, (amount * detail).astype(np.uint8))
    
    return sharpened


def adaptive_unsharp_mask(frame, radius=2.0, amount=1.0, edge_weight=0.7):
    """
    Apply adaptive unsharp masking - stronger on edges, gentler on flat areas
    
    Args:
        frame: Input BGR frame
        radius: Gaussian blur radius
        amount: Maximum enhancement strength
        edge_weight: How much to favor edges (0-1, higher = more edge-focused)
    
    Returns:
        Enhanced frame
    """
    # Create edge mask
    edge_mask = create_edge_mask(frame)
    
    # Apply unsharp mask
    sharpened = unsharp_mask(frame, radius=radius, amount=amount)
    
    # Blend based on edge mask
    # edge_weight controls the contrast: higher = more difference between edges and flat areas
    mask_strength = edge_mask * edge_weight + (1 - edge_weight)
    mask_strength = np.clip(mask_strength, 0, 1)
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_strength] * 3, axis=-1)
    
    # Blend sharpened and original based on mask
    result = (sharpened * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)
    
    return result


def bilateral_sharpen(frame, d=9, sigma_color=75, sigma_space=75, amount=1.0):
    """
    Edge-preserving sharpening using bilateral filter
    
    Args:
        frame: Input BGR frame
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        amount: Sharpening strength
    
    Returns:
        Sharpened frame
    """
    # Apply bilateral filter (smooths while preserving edges)
    smoothed = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    # Calculate detail layer
    detail = cv2.subtract(frame, smoothed)
    
    # Add enhanced details back
    sharpened = cv2.add(frame, (amount * detail).astype(np.uint8))
    
    return sharpened


def enhance_details(input_video, output_video=None, method='adaptive', 
                   radius=2.0, amount=1.5, strength=1.0):
    """
    Enhance details in entire video
    
    Args:
        input_video: Path to input video
        output_video: Path to output video (optional)
        method: Enhancement method ('adaptive', 'unsharp', or 'bilateral')
        radius: Blur radius for unsharp mask (1.0-5.0, default: 2.0)
        amount: Sharpening strength (0.5-3.0, default: 1.5)
        strength: Overall effect strength (0-1, 1 = full effect)
    
    Returns:
        Path to output video
    """
    input_path = Path(input_video)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Set output path
    if output_video is None:
        output_path = input_path.parent / f"{input_path.stem}_detailed.mp4"
    else:
        output_path = Path(output_video)
    
    logger.info(f"Starting detail enhancement: {input_path.name}")
    logger.info(f"Method: {method}, Radius: {radius}, Amount: {amount}, Strength: {strength * 100:.0f}%")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
    
    # Create temporary AVI file (uncompressed)
    temp_avi = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless codec
    writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise ValueError(f"Failed to create video writer: {temp_avi}")
    
    # Process frames
    frame_idx = 0
    start_time = time.time()
    
    logger.info("Processing frames (pass 1/2 - enhancing details)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply selected enhancement method
        if method == 'adaptive':
            enhanced = adaptive_unsharp_mask(frame, radius=radius, amount=amount, edge_weight=0.7)
        elif method == 'bilateral':
            enhanced = bilateral_sharpen(frame, d=9, sigma_color=75, sigma_space=75, amount=amount)
        else:  # 'unsharp'
            enhanced = unsharp_mask(frame, radius=radius, amount=amount)
        
        # Blend with original based on strength
        if strength < 1.0:
            enhanced = cv2.addWeighted(enhanced, strength, frame, 1 - strength, 0)
        
        # Write frame
        writer.write(enhanced)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_idx / elapsed if elapsed > 0 else 0
            progress = (frame_idx / frame_count) * 100
            eta = (frame_count - frame_idx) / fps_current if fps_current > 0 else 0
            logger.info(f"Progress: {frame_idx}/{frame_count} ({progress:.1f}%) - "
                       f"{fps_current:.1f} fps - ETA: {eta:.0f}s")
    
    cap.release()
    writer.release()
    
    elapsed_pass1 = time.time() - start_time
    logger.info(f"Pass 1 complete in {elapsed_pass1:.1f}s ({frame_count/elapsed_pass1:.1f} fps)")
    
    # Convert to MP4 using FFmpeg
    logger.info("Converting to MP4 (pass 2/2)...")
    import subprocess
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', str(temp_avi),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise RuntimeError("FFmpeg conversion failed")
    
    # Clean up temp file
    temp_avi.unlink()
    
    elapsed_total = time.time() - start_time
    logger.info(f"✓ Detail enhancement complete: {output_path}")
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"Average speed: {frame_count/elapsed_total:.1f} fps")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Detail Enhancement for 8mm Film Footage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic usage with adaptive sharpening (recommended)
  python enhance_details.py input_video.mp4
  
  # Adjust sharpening strength
  python enhance_details.py input_video.mp4 --amount 2.0 --radius 2.5
  
  # Reduce overall effect (blend with original)
  python enhance_details.py input_video.mp4 --strength 0.7
  
  # Try bilateral filter method (edge-preserving)
  python enhance_details.py input_video.mp4 --method bilateral --amount 1.8
  
  # Gentle enhancement for already-sharp footage
  python enhance_details.py input_video.mp4 --amount 1.0 --radius 1.5 --strength 0.5

METHODS:
  adaptive:   Smart unsharp mask with edge detection (default, best for most cases)
  unsharp:    Classic unsharp mask (faster, uniform sharpening)
  bilateral:  Edge-preserving bilateral filter (slower, very smooth)

PARAMETERS:
  radius:   Blur radius for detail extraction (1.0-5.0, default: 2.0)
            - Smaller = finer details, larger = broader details
  amount:   Sharpening intensity (0.5-3.0, default: 1.5)
            - 1.0 = subtle, 2.0 = strong, 3.0 = very strong
  strength: Overall effect blend (0-1, default: 1.0)
            - 0.5 = 50% blend with original, 1.0 = full effect
        """
    )
    
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("--method", choices=['adaptive', 'unsharp', 'bilateral'], default='adaptive',
                       help="Enhancement method (default: adaptive)")
    parser.add_argument("--radius", type=float, default=2.0,
                       help="Blur radius for detail extraction (default: 2.0)")
    parser.add_argument("--amount", type=float, default=1.5,
                       help="Sharpening strength (default: 1.5)")
    parser.add_argument("--strength", type=float, default=1.0,
                       help="Overall effect strength 0-1 (default: 1.0)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.radius < 0.5 or args.radius > 10:
        parser.error("Radius must be between 0.5 and 10")
    if args.amount < 0 or args.amount > 5:
        parser.error("Amount must be between 0 and 5")
    if args.strength < 0 or args.strength > 1:
        parser.error("Strength must be between 0 and 1")
    
    try:
        output_path = enhance_details(
            args.input,
            args.output,
            method=args.method,
            radius=args.radius,
            amount=args.amount,
            strength=args.strength
        )
        logger.info(f"✓ Success! Output saved to: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
