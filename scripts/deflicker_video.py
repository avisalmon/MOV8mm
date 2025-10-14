"""
Video deflicker module
Removes brightness flickering common in 8mm projector scans
Uses temporal median filtering to stabilize brightness across frames
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import subprocess

sys.path.append(str(Path(__file__).parent.parent))
import config

def calculate_brightness(frame):
    """
    Calculate average brightness of a frame
    
    Args:
        frame: Input frame (BGR)
    
    Returns:
        Average brightness value (0-255)
    """
    # Convert to LAB color space and get L channel (lightness)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return np.mean(l_channel)

def smooth_brightness_curve(brightness_values, window_size=15):
    """
    Smooth brightness curve using temporal median filtering
    
    Args:
        brightness_values: List of brightness values per frame
        window_size: Size of temporal window for smoothing (odd number)
    
    Returns:
        Smoothed brightness values
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    smoothed = []
    pad = window_size // 2
    
    # Pad the brightness values at edges
    padded = np.pad(brightness_values, pad, mode='edge')
    
    for i in range(len(brightness_values)):
        # Get temporal window
        window = padded[i:i + window_size]
        # Use median for robustness against outliers
        smoothed.append(np.median(window))
    
    return np.array(smoothed)

def apply_brightness_correction(frame, target_brightness, current_brightness):
    """
    Adjust frame brightness to match target
    
    Args:
        frame: Input frame (BGR)
        target_brightness: Target brightness value
        current_brightness: Current frame brightness
    
    Returns:
        Brightness-corrected frame
    """
    if current_brightness == 0:
        return frame
    
    # Calculate correction factor
    correction = target_brightness / current_brightness
    
    # Clamp correction to reasonable range (prevent over-correction)
    correction = np.clip(correction, 0.5, 2.0)
    
    # Convert to LAB for brightness-only adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Apply correction to L channel only (preserves color)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * correction, 0, 255)
    
    # Convert back to BGR
    corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return corrected

def deflicker(input_path, output_path=None, window_size=15, strength=0.8):
    """
    Remove brightness flickering from video
    
    Args:
        input_path: Path to input video
        output_path: Path to save deflickered video (optional)
        window_size: Temporal smoothing window size (default: 15 frames)
        strength: Deflicker strength 0.0-1.0 (default: 0.8, higher = more correction)
    
    Returns:
        Path to deflickered video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_deflickered.mp4"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Deflickering {input_path.name}...")
    logger.info(f"Window size: {window_size} frames, Strength: {strength}")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
    
    # First pass: Calculate brightness for all frames
    logger.info("Pass 1/2: Analyzing brightness fluctuations...")
    brightness_values = []
    all_frames = []
    
    pbar = tqdm(total=frame_count, desc="Analyzing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        brightness = calculate_brightness(frame)
        brightness_values.append(brightness)
        all_frames.append(frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Calculate brightness statistics
    brightness_values = np.array(brightness_values)
    mean_brightness = np.mean(brightness_values)
    std_brightness = np.std(brightness_values)
    flicker_amount = (std_brightness / mean_brightness) * 100
    
    logger.info(f"Original brightness: mean={mean_brightness:.1f}, std={std_brightness:.1f}, flicker={flicker_amount:.1f}%")
    
    # Smooth brightness curve
    logger.info(f"Smoothing brightness curve with {window_size}-frame window...")
    target_brightness = smooth_brightness_curve(brightness_values, window_size)
    
    # Blend original and smoothed based on strength
    target_brightness = brightness_values * (1 - strength) + target_brightness * strength
    
    # Second pass: Apply brightness correction
    logger.info("Pass 2/2: Applying deflicker correction...")
    
    # Create temporary AVI output
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Failed to open video writer for {temp_output}")
        return None
    
    pbar = tqdm(total=frame_count, desc="Deflickering", unit="frame")
    
    for i, frame in enumerate(all_frames):
        # Apply brightness correction
        corrected_frame = apply_brightness_correction(
            frame,
            target_brightness[i],
            brightness_values[i]
        )
        
        out.write(corrected_frame)
        pbar.update(1)
    
    pbar.close()
    out.release()
    
    # Calculate corrected brightness statistics
    corrected_brightness = []
    for i, frame in enumerate(all_frames):
        corrected_frame = apply_brightness_correction(
            frame,
            target_brightness[i],
            brightness_values[i]
        )
        corrected_brightness.append(calculate_brightness(corrected_frame))
    
    corrected_brightness = np.array(corrected_brightness)
    corrected_std = np.std(corrected_brightness)
    corrected_flicker = (corrected_std / mean_brightness) * 100
    improvement = ((flicker_amount - corrected_flicker) / flicker_amount) * 100
    
    logger.info(f"Corrected brightness: std={corrected_std:.1f}, flicker={corrected_flicker:.1f}%")
    logger.info(f"Flicker reduction: {improvement:.1f}%")
    
    # Convert AVI to MP4 using FFmpeg
    logger.info("Converting to H.264...")
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        temp_output.unlink()  # Delete temp AVI
        logger.info(f"✓ Deflicker complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        logger.warning(f"Temp file saved at: {temp_output}")
        return str(temp_output)
    
    return str(output_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove brightness flickering from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deflicker with default settings
  python deflicker_video.py input.mp4
  
  # Custom output path
  python deflicker_video.py input.mp4 -o output.mp4
  
  # Adjust smoothing window (larger = smoother, but may lose intentional brightness changes)
  python deflicker_video.py input.mp4 -w 21
  
  # Adjust strength (0.0 = no correction, 1.0 = full correction)
  python deflicker_video.py input.mp4 -s 0.9
  
  # Test on upscaled file
  python deflicker_video.py "temp\\Movie0030_cut_172-182_step04_faces_restored.mp4"
        """
    )
    
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-w", "--window", type=int, default=15,
                       help="Temporal smoothing window size (default: 15 frames)")
    parser.add_argument("-s", "--strength", type=float, default=0.8,
                       help="Deflicker strength 0.0-1.0 (default: 0.8)")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run deflicker
    deflicker(args.input, args.output, args.window, args.strength)

if __name__ == "__main__":
    main()
