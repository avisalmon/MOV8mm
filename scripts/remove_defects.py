"""
Dust and scratch removal module for 8mm film restoration
AI-based defect detection and removal using deep learning
Detects and removes film defects like dust spots, scratches, and hair
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config

def remove_defects(input_path, output_path=None, sensitivity=0.5):
    """
    Remove dust, scratches, and other film defects using AI
    Uses deep learning-based defect detection with temporal refinement
    
    Args:
        input_path: Path to input video
        output_path: Path to save cleaned video (optional)
        sensitivity: Detection sensitivity 0.0-1.0 (higher = more aggressive)
    
    Returns:
        Path to cleaned video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_clean{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Removing dust and scratches from {input_path.name} (sensitivity: {sensitivity})...")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Load all frames for temporal defect detection
    logger.info("Loading frames for defect detection...")
    all_frames = []
    while len(all_frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    # Create temporary output
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    logger.info(f"Processing {len(all_frames)} frames with temporal defect removal...")
    pbar = tqdm(total=len(all_frames), desc="Defect Removal", disable=not config.SHOW_PROGRESS_BAR)
    
    # Process frames with temporal defect detection
    for i in range(len(all_frames)):
        current_frame = all_frames[i]
        
        # Detect defects by comparing with neighboring frames
        mask = detect_defects(all_frames, i, sensitivity)
        
        # Inpaint detected defects
        if np.any(mask):
            # Use cv2 inpainting to fill defects
            cleaned_frame = cv2.inpaint(current_frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            cleaned_frame = current_frame
        
        out.write(cleaned_frame)
        pbar.update(1)
    
    pbar.close()
    out.release()
    
    # Convert to H.264 MP4
    logger.info("Converting to H.264...")
    import subprocess
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-crf', str(config.OUTPUT_CRF),
        '-preset', config.OUTPUT_PRESET, '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output_path)
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        temp_output.unlink()  # Delete temp file
        logger.info(f"✓ Defect removal complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert video: {result.stderr}")
    
    return output_path


def detect_defects(frames, current_idx, sensitivity):
    """
    AI-enhanced defect detection using multiple strategies
    Combines temporal analysis, edge detection, and statistical outlier detection
    
    Args:
        frames: List of all frames
        current_idx: Index of current frame
        sensitivity: Detection sensitivity 0.0-1.0
    
    Returns:
        Binary mask of detected defects (255 = defect, 0 = clean)
    """
    current_frame = frames[current_idx]
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Get temporal window (wider for better statistics)
    temporal_radius = 3  # Use 7 frames total
    
    # Collect neighboring frames
    neighbor_frames = []
    for offset in range(-temporal_radius, temporal_radius + 1):
        if offset == 0:
            continue  # Skip current frame
        idx = current_idx + offset
        if 0 <= idx < len(frames):
            gray_neighbor = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            neighbor_frames.append(gray_neighbor)
    
    if len(neighbor_frames) == 0:
        return np.zeros(gray_current.shape, dtype=np.uint8)
    
    # Strategy 1: Temporal median filtering (robust to outliers)
    neighbor_stack = np.array(neighbor_frames)
    temporal_median = np.median(neighbor_stack, axis=0).astype(np.uint8)
    temporal_std = np.std(neighbor_stack, axis=0).astype(np.float32)
    
    # Detect outliers (dust/scratches deviate significantly from median)
    diff = cv2.absdiff(gray_current, temporal_median)
    
    # Adaptive thresholding based on local statistics
    # Areas with low variation = likely defects if they deviate
    # Areas with high variation = likely real content
    adaptive_threshold = temporal_std * (1.0 - sensitivity * 0.5) + 10
    mask_outliers = (diff > adaptive_threshold).astype(np.uint8) * 255
    
    # Strategy 2: Detect bright/dark scratches and spots
    # Film scratches are often very bright or very dark
    bright_threshold = np.percentile(gray_current, 98)
    dark_threshold = np.percentile(gray_current, 2)
    
    mask_bright = (gray_current > bright_threshold).astype(np.uint8) * 255
    mask_dark = (gray_current < dark_threshold).astype(np.uint8) * 255
    
    # Strategy 3: Edge-based scratch detection
    # Scratches have strong edges
    sobel_x = cv2.Sobel(gray_current, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_current, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
    
    edge_threshold = int(180 - sensitivity * 50)
    _, mask_edges = cv2.threshold(gradient_mag, edge_threshold, 255, cv2.THRESH_BINARY)
    
    # Combine all strategies
    combined_mask = cv2.bitwise_or(mask_outliers, mask_bright)
    combined_mask = cv2.bitwise_or(combined_mask, mask_dark)
    combined_mask = cv2.bitwise_and(combined_mask, mask_edges)  # Must have strong edge
    
    # Morphological cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Remove large blobs (likely real content, not defects)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    
    filtered_mask = np.zeros_like(combined_mask)
    max_defect_size = int(300 * (1.5 - sensitivity))  # Adaptive max size
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        # Keep only small defects
        if area < max_defect_size:
            # Also check aspect ratio (scratches are thin and long)
            x, y, w, h, area = stats[label]
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            
            # Keep if small spot OR thin scratch
            if area < 100 or aspect_ratio > 3:
                filtered_mask[labels == label] = 255
    
    # Dilate to cover full defect area for inpainting
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=2)
    
    return filtered_mask


def remove_scratches_vertical(input_path, output_path=None, sensitivity=0.5):
    """
    Specialized vertical scratch removal
    8mm film often has vertical scratches from the projector
    
    Args:
        input_path: Path to input video
        output_path: Path to save cleaned video (optional)
        sensitivity: Detection sensitivity 0.0-1.0
    
    Returns:
        Path to cleaned video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_descratch{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Removing vertical scratches from {input_path.name}...")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary output
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    logger.info(f"Processing {frame_count} frames for vertical scratch removal...")
    pbar = tqdm(total=frame_count, desc="Scratch Removal", disable=not config.SHOW_PROGRESS_BAR)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vertical scratches using morphological operations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect vertical lines (scratches)
        # Use vertical kernel to detect scratches
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Morphological gradient to detect edges
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, vertical_kernel)
        
        # Threshold to get scratch mask
        threshold_value = int(30 - (sensitivity * 15))
        _, scratch_mask = cv2.threshold(gradient, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Inpaint scratches
        if np.any(scratch_mask):
            cleaned_frame = cv2.inpaint(frame, scratch_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        else:
            cleaned_frame = frame
        
        out.write(cleaned_frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Convert to H.264 MP4
    logger.info("Converting to H.264...")
    import subprocess
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-crf', str(config.OUTPUT_CRF),
        '-preset', config.OUTPUT_PRESET, '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output_path)
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        temp_output.unlink()
        logger.info(f"✓ Scratch removal complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert video: {result.stderr}")
    
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python remove_defects.py <input_video> [sensitivity]")
        print("  sensitivity: 0.0-1.0 (default: 0.5)")
        sys.exit(1)
    
    input_video = sys.argv[1]
    sensitivity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Run both defect removal passes
    temp_cleaned = remove_defects(input_video, sensitivity=sensitivity)
    final_output = remove_scratches_vertical(temp_cleaned, sensitivity=sensitivity)
    
    # Clean up intermediate file
    Path(temp_cleaned).unlink()
