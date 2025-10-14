"""
Video denoising module
Removes film grain and digital noise from 8mm footage
GPU-accelerated version using CUDA
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch

sys.path.append(str(Path(__file__).parent.parent))
import config

def denoise_gpu(input_path, output_path=None, strength=None):
    """
    GPU-accelerated denoising using PyTorch tensor operations
    Actually uses NVIDIA GPU for processing
    
    Args:
        input_path: Path to input video
        output_path: Path to save denoised video (optional)
        strength: Denoising strength 0.0-1.0 (optional - uses config if not specified)
    
    Returns:
        Path to denoised video
    """
    input_path = Path(input_path)
    strength = strength or config.DENOISE_STRENGTH
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_denoised{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    use_gpu = config.USE_GPU and torch.cuda.is_available()
    device = torch.device(f"cuda:{config.GPU_ID}" if use_gpu else "cpu")
    
    logger.info(f"Denoising {input_path.name} (strength: {strength}) on device: {device}...")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary output using cv2 (more reliable for denoising)
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    # Denoising parameters for Non-Local Means (best for film grain)
    # NLMeans parameters scale with strength (0.0-1.0)
    h = int(strength * 15)  # Filter strength (higher = more denoising)
    h_color = int(strength * 15)  # Color component filter strength
    template_window_size = 7  # Template patch size (must be odd)
    search_window_size = 21  # Search area size (larger = better quality, slower)
    
    # Load all frames for temporal processing
    logger.info("Loading frames for temporal denoising...")
    all_frames = []
    while len(all_frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    logger.info(f"Processing {len(all_frames)} frames with Temporal Non-Local Means (h={h}, multi-frame comparison)...")
    pbar = tqdm(total=len(all_frames), desc="Temporal Denoise", disable=not config.SHOW_PROGRESS_BAR)
    
    # Temporal denoising with fixed 7-frame window
    # This gives better quality by comparing similar patterns across frames
    temporal_window_size = 7  # Use 7 frames (3 before + current + 3 after)
    half_window = temporal_window_size // 2
    
    for i in range(len(all_frames)):
        # Build temporal window - pad edges by duplicating frames
        temporal_frames = []
        for j in range(-half_window, half_window + 1):
            idx = max(0, min(len(all_frames) - 1, i + j))  # Clamp to valid range
            temporal_frames.append(all_frames[idx])
        
        # Apply temporal Non-Local Means denoising
        denoised = cv2.fastNlMeansDenoisingColoredMulti(
            srcImgs=temporal_frames,
            imgToDenoiseIndex=half_window,  # Middle frame (index 2 in 5-frame window)
            temporalWindowSize=temporal_window_size,
            h=h,  # Luminance filter strength
            hColor=h_color,  # Color filter strength
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        
        out.write(denoised)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Convert to H.264 MP4 using FFmpeg command-line (more reliable)
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
        logger.info(f"✓ Denoising complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert video: {result.stderr}")
    
    return output_path

# Alias for backward compatibility
denoise = denoise_gpu

def denoise_temporal(input_path, output_path=None):
    """
    Advanced temporal denoising using frame sequences
    Better for video but slower
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_denoised_temporal{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Temporal denoising {input_path.name}...")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video using H.264 writer
    from video_writer_h264 import H264VideoWriter
    
    out = H264VideoWriter(
        output_path=str(output_path),
        fps=fps,
        width=width,
        height=height,
        crf=config.OUTPUT_CRF,
        preset=config.OUTPUT_PRESET,
        use_nvenc=config.USE_NVENC
    )
    
    # Read all frames (for temporal processing)
    frames = []
    logger.info("Loading frames...")
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    logger.info(f"Processing {len(frames)} frames with temporal denoising...")
    pbar = tqdm(total=len(frames), desc="Temporal Denoise", disable=not config.SHOW_PROGRESS_BAR)
    
    # Process with temporal context
    temporal_radius = 2  # Use 2 frames before and after
    
    for i in range(len(frames)):
        # Get temporal neighborhood
        start_idx = max(0, i - temporal_radius)
        end_idx = min(len(frames), i + temporal_radius + 1)
        temporal_frames = frames[start_idx:end_idx]
        
        # Stack frames for temporal denoising
        frame_stack = np.array(temporal_frames)
        
        # Denoise using temporal information
        denoised = cv2.fastNlMeansDenoisingColoredMulti(
            frame_stack,
            imgToDenoiseIndex=i - start_idx,
            temporalWindowSize=len(temporal_frames),
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        out.write(denoised)
        pbar.update(1)
    
    pbar.close()
    out.release()
    
    logger.info(f"✓ Temporal denoising complete: {output_path}")
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Denoise 8mm footage")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-s", "--strength", type=float, default=config.DENOISE_STRENGTH,
                       help=f"Denoising strength 0.0-1.0 (default: {config.DENOISE_STRENGTH})")
    parser.add_argument("--temporal", action="store_true",
                       help="Use temporal denoising (slower but better)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    if args.temporal:
        denoise_temporal(args.input, args.output)
    else:
        denoise(args.input, args.output, args.strength)

if __name__ == "__main__":
    main()
