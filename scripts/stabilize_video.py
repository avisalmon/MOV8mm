"""
Video stabilization module
Reduces camera shake in 8mm footage
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config

def stabilize(input_path, output_path=None):
    """
    Stabilize video using OpenCV
    
    Args:
        input_path: Path to input video
        output_path: Path to save stabilized video (optional)
    
    Returns:
        Path to stabilized video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_stabilized{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Stabilizing {input_path.name}...")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame")
        return input_path
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Pre-define transformation-store array
    transforms = np.zeros((frame_count-1, 3), np.float32)
    
    logger.info("Analyzing camera motion...")
    pbar = tqdm(total=frame_count-1, desc="Analyzing", disable=not config.SHOW_PROGRESS_BAR)
    
    # Calculate transformations between frames
    for i in range(frame_count-1):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                           minDistance=30, blockSize=3)
        
        if prev_pts is not None and len(prev_pts) > 0:
            # Calculate optical flow (motion vectors)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            # Find transformation matrix
            if len(prev_pts) >= 3:
                m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
                
                if m is not None:
                    # Extract translation
                    dx = m[0,2]
                    dy = m[1,2]
                    
                    # Extract rotation angle
                    da = np.arctan2(m[1,0], m[0,0])
                    
                    # Store transformation
                    transforms[i] = [dx, dy, da]
        
        prev_gray = curr_gray
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Calculate smooth trajectory using cumulative sum
    trajectory = np.cumsum(transforms, axis=0)
    
    # Calculate smooth trajectory using moving average
    smoothing_radius = config.STABILIZE_SMOOTHING
    smoothed_trajectory = np.copy(trajectory)
    
    for i in range(3):
        smoothed_trajectory[:, i] = smooth(trajectory[:, i], smoothing_radius)
    
    # Calculate smooth transforms
    smooth_transforms = transforms + (smoothed_trajectory - trajectory)
    
    # Limit transformations to prevent excessive cropping
    max_shift = config.STABILIZE_MAX_SHIFT
    max_angle = np.radians(config.STABILIZE_MAX_ANGLE)
    
    smooth_transforms[:, 0] = np.clip(smooth_transforms[:, 0], -max_shift, max_shift)  # dx
    smooth_transforms[:, 1] = np.clip(smooth_transforms[:, 1], -max_shift, max_shift)  # dy
    smooth_transforms[:, 2] = np.clip(smooth_transforms[:, 2], -max_angle, max_angle)  # da
    
    # Apply stabilization
    logger.info("Applying stabilization...")
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    pbar = tqdm(total=frame_count, desc="Stabilizing", disable=not config.SHOW_PROGRESS_BAR)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i < len(smooth_transforms):
            dx = smooth_transforms[i, 0]
            dy = smooth_transforms[i, 1]
            da = smooth_transforms[i, 2]
            
            # Reconstruction transformation matrix
            m = np.array([[np.cos(da), -np.sin(da), dx],
                         [np.sin(da), np.cos(da), dy]])
            
            # Apply affine transformation
            frame_stabilized = cv2.warpAffine(frame, m, (width, height), 
                                             borderMode=cv2.BORDER_REPLICATE)
        else:
            frame_stabilized = frame
        
        out.write(frame_stabilized)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    logger.info(f"✓ Stabilization complete: {output_path}")
    return output_path

def smooth(trajectory, radius):
    """Apply moving average filter"""
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    trajectory_padded = np.pad(trajectory, (radius, radius), 'edge')
    trajectory_smoothed = np.convolve(trajectory_padded, f, mode='same')
    return trajectory_smoothed[radius:-radius]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stabilize shaky 8mm footage")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    stabilize(args.input, args.output)

if __name__ == "__main__":
    main()
