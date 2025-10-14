"""
High-quality AI-based video denoising
Uses temporal information to remove film grain while preserving details
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config


class TemporalDenoiser(nn.Module):
    """
    Temporal denoising network using multiple frames
    Similar to BasicVSR++ but simplified for efficiency
    """
    def __init__(self, num_frames=5):
        super().__init__()
        self.num_frames = num_frames
        
        # Spatial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3 * num_frames, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
    
    def forward(self, frames):
        """
        Args:
            frames: Tensor of shape [B, T, C, H, W] where T is temporal window
        Returns:
            Denoised center frame [B, C, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # Concatenate temporal frames
        x = frames.view(B, T * C, H, W)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Temporal fusion
        fused = self.temporal_fusion(features)
        
        # Reconstruct
        out = self.reconstruction(fused)
        
        # Residual connection with center frame
        center_idx = T // 2
        residual = frames[:, center_idx, :, :, :]
        
        return out + residual


def denoise_ai(input_path, output_path=None, model_path=None, temporal_window=5):
    """
    AI-based temporal denoising for film grain removal
    
    Args:
        input_path: Path to input video
        output_path: Path to save denoised video (optional)
        model_path: Path to pretrained model (optional - will use simple model if None)
        temporal_window: Number of frames to use for temporal denoising (odd number)
    
    Returns:
        Path to denoised video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_denoised_ai{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    use_gpu = config.USE_GPU and torch.cuda.is_available()
    device = torch.device(f"cuda:{config.GPU_ID}" if use_gpu else "cpu")
    
    logger.info(f"AI Denoising {input_path.name} on {device}...")
    logger.info(f"Temporal window: {temporal_window} frames")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Load or create model
    model = TemporalDenoiser(num_frames=temporal_window).to(device)
    
    if model_path and Path(model_path).exists():
        logger.info(f"Loading pretrained model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.warning("No pretrained model - using untrained network (for demonstration)")
        logger.info("For best results, use pretrained denoising models")
    
    model.eval()
    
    # Read all frames into memory (for temporal processing)
    logger.info("Loading video frames...")
    all_frames = []
    while len(all_frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    cap.release()
    total_frames = len(all_frames)
    logger.info(f"Loaded {total_frames} frames")
    
    # Process with temporal window
    half_window = temporal_window // 2
    
    logger.info(f"Processing {total_frames} frames with AI denoising...")
    pbar = tqdm(total=total_frames, desc="AI Denoising", disable=not config.SHOW_PROGRESS_BAR)
    
    with torch.no_grad():
        for i in range(total_frames):
            # Get temporal neighborhood
            start_idx = max(0, i - half_window)
            end_idx = min(total_frames, i + half_window + 1)
            
            # Collect frames in temporal window
            temporal_frames = []
            for idx in range(start_idx, end_idx):
                frame = all_frames[idx]
                frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                temporal_frames.append(frame_tensor)
            
            # Pad if necessary (at video boundaries)
            while len(temporal_frames) < temporal_window:
                if i < half_window:
                    temporal_frames.insert(0, temporal_frames[0])  # Pad start
                else:
                    temporal_frames.append(temporal_frames[-1])  # Pad end
            
            # Stack and move to GPU
            temporal_batch = torch.stack(temporal_frames).unsqueeze(0).to(device)  # [1, T, C, H, W]
            
            # Denoise
            denoised_tensor = model(temporal_batch)
            
            # Convert back to numpy
            denoised = (denoised_tensor.squeeze(0).permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
            denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
            
            out.write(denoised_bgr)
            pbar.update(1)
    
    pbar.close()
    out.release()
    
    logger.info(f"✓ AI Denoising complete: {output_path}")
    return output_path


def denoise_fastdvdnet(input_path, output_path=None):
    """
    Use FastDVDnet for temporal denoising (if available)
    FastDVDnet is specifically designed for video denoising
    """
    try:
        # Try to import FastDVDnet if installed
        from models.fastdvdnet import FastDVDnet
        
        logger.info("Using FastDVDnet for temporal denoising...")
        # Implementation would go here
        # This is a placeholder for future implementation
        
    except ImportError:
        logger.warning("FastDVDnet not available, using basic temporal denoising")
        return denoise_ai(input_path, output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-based video denoising for film grain removal")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-m", "--model", help="Path to pretrained model (optional)")
    parser.add_argument("-w", "--window", type=int, default=5,
                       help="Temporal window size (odd number, default: 5)")
    parser.add_argument("--method", choices=["ai", "fastdvdnet"], default="ai",
                       help="Denoising method to use")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    if args.method == "fastdvdnet":
        denoise_fastdvdnet(args.input, args.output)
    else:
        denoise_ai(args.input, args.output, args.model, args.window)


if __name__ == "__main__":
    main()
