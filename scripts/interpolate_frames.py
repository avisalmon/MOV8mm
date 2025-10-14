"""
Frame interpolation module using RIFE
Increases frame rate from 16-18fps to 24/30/60fps
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.nn import functional as F

sys.path.append(str(Path(__file__).parent.parent))

# Add RIFE to path
rife_path = Path(__file__).parent.parent / "temp" / "Practical-RIFE"
if rife_path.exists():
    sys.path.insert(0, str(rife_path))
    os.environ['RIFE_PATH'] = str(rife_path)

import config

def load_rife_model(model_path="models/flownet.pkl"):
    """Load RIFE model for frame interpolation"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # Try to import RIFE model
        try:
            from train_log.RIFE_HDv3 import Model
        except ImportError:
            logger.warning("RIFE model code not found in train_log/. Using fallback interpolation.")
            return None
        
        model = Model()
        model.load_model("temp/Practical-RIFE/train_log", -1)
        model.eval()
        model.device()
        logger.info(f"✓ RIFE model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Could not load RIFE model: {e}. Using fallback interpolation.")
        return None

def interpolate_with_rife_model(model, img0, img1, scale=1.0, device="cuda"):
    """
    Interpolate between two frames using RIFE model
    
    Args:
        model: RIFE model instance
        img0: First frame (numpy array, HWC, BGR)
        img1: Second frame (numpy array, HWC, BGR)
        scale: Downscale factor (use 0.5 for 4K)
        device: cuda or cpu
    
    Returns:
        interpolated_frame: Middle frame (numpy array, HWC, BGR)
    """
    h, w = img0.shape[:2]
    
    # Prepare images for RIFE
    img0_tensor = torch.from_numpy(img0).permute(2, 0, 1).float() / 255.0
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    
    img0_tensor = img0_tensor.unsqueeze(0).to(device)
    img1_tensor = img1_tensor.unsqueeze(0).to(device)
    
    # Pad to multiple of 64 if needed
    n, c, h, w = img0_tensor.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    
    img0_tensor = F.pad(img0_tensor, padding)
    img1_tensor = F.pad(img1_tensor, padding)
    
    # Apply scale if needed
    if scale != 1.0:
        img0_tensor = F.interpolate(img0_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
        img1_tensor = F.interpolate(img1_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
    
    # Interpolate
    mid = model.inference(img0_tensor, img1_tensor, scale)
    
    # Unscale if needed
    if scale != 1.0:
        mid = F.interpolate(mid, scale_factor=1/scale, mode='bilinear', align_corners=False)
    
    # Remove padding and convert back to numpy
    mid = mid[:, :, :h, :w]
    mid_frame = (mid[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    
    return mid_frame

def interpolate(input_path, output_path=None, target_fps=None):
    """
    Interpolate frames to increase frame rate
    
    Args:
        input_path: Path to input video
        output_path: Path to save interpolated video (optional)
        target_fps: Target frame rate (optional - uses config if not specified)
    
    Returns:
        Path to interpolated video
    """
    input_path = Path(input_path)
    target_fps = target_fps or config.TARGET_FPS
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_interpolated_{target_fps}fps{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Interpolating {input_path.name} to {target_fps}fps...")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Source FPS: {fps:.2f}, Target FPS: {target_fps}")
    
    if fps >= target_fps:
        logger.warning(f"Source FPS ({fps:.2f}) >= target FPS ({target_fps}). No interpolation needed.")
        cap.release()
        return input_path
    
    # Calculate interpolation ratio
    frame_multiplier = target_fps / fps
    logger.info(f"Frame multiplier: {frame_multiplier:.2f}x")
    
    # Try to load RIFE model
    use_rife = False
    rife_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(config.RIFE_MODEL_PATH if hasattr(config, 'RIFE_MODEL_PATH') else "models/flownet.pkl")
    if model_path.exists():
        logger.info("Attempting to load RIFE model for AI interpolation...")
        rife_model = load_rife_model(str(model_path))
        if rife_model is not None:
            use_rife = True
            logger.info("✓ Using RIFE AI interpolation")
        else:
            logger.warning("RIFE model failed to load, using simple interpolation")
    else:
        logger.warning(f"RIFE model not found at {model_path}. Using simple interpolation.")
    
    # Create AVI output first (more reliable for opencv)
    output_avi = output_path.with_suffix('.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_avi), fourcc, target_fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Failed to open video writer for {output_avi}")
        cap.release()
        return None
    
    pbar = tqdm(total=frame_count, desc="Interpolating", unit="frame")
    
    prev_frame = None
    for i in range(frame_count):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        if prev_frame is not None:
            # Calculate how many interpolated frames to insert
            num_interpolated = int(frame_multiplier) - 1
            
            if use_rife and rife_model is not None:
                # AI interpolation using RIFE
                for j in range(num_interpolated):
                    try:
                        # RIFE creates frames at fractional timestamps
                        interpolated_frame = interpolate_with_rife_model(
                            rife_model, prev_frame, curr_frame, 
                            scale=1.0, device=str(device)
                        )
                        out.write(interpolated_frame)
                    except Exception as e:
                        logger.warning(f"RIFE interpolation failed: {e}, falling back to linear")
                        alpha = (j + 1) / (num_interpolated + 1)
                        interpolated_frame = cv2.addWeighted(prev_frame, 1-alpha, curr_frame, alpha, 0)
                        out.write(interpolated_frame)
            else:
                # Simple linear interpolation (fallback)
                for j in range(num_interpolated):
                    alpha = (j + 1) / (num_interpolated + 1)
                    interpolated_frame = cv2.addWeighted(prev_frame, 1-alpha, curr_frame, alpha, 0)
                    out.write(interpolated_frame)
        
        out.write(curr_frame)
        prev_frame = curr_frame.copy()
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Convert AVI to MP4 using FFmpeg
    logger.info("Converting to H.264...")
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', str(output_avi),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        output_avi.unlink()  # Delete AVI
        logger.info(f"✓ Interpolation complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        return str(output_avi)  # Return AVI if MP4 failed
    
    return str(output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interpolate frames to increase FPS")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-f", "--fps", type=int, default=config.TARGET_FPS,
                       help=f"Target frame rate (default: {config.TARGET_FPS})")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    interpolate(args.input, args.output, args.fps)

if __name__ == "__main__":
    main()
