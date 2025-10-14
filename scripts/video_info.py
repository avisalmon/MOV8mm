"""
Quick video information and preview script
Shows video properties and extracts sample frames
"""

import sys
from pathlib import Path
import cv2
import argparse
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config

def get_video_info(video_path):
    """Extract and display video information"""
    video_path = Path(video_path)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        'filename': video_path.name,
        'path': str(video_path),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'file_size_mb': video_path.stat().st_size / (1024 * 1024)
    }
    
    info['duration_seconds'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    info['duration_minutes'] = info['duration_seconds'] / 60
    
    # Get codec name
    fourcc = info['codec']
    codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    info['codec_name'] = codec_name
    
    cap.release()
    
    return info

def extract_sample_frames(video_path, output_dir=None, num_frames=5):
    """Extract sample frames from video for preview"""
    video_path = Path(video_path)
    
    if output_dir is None:
        output_dir = Path("preview_frames") / video_path.stem
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames at regular intervals
    frame_indices = [int(i * frame_count / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    extracted = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = output_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted.append(output_path)
            logger.info(f"Saved frame {idx}: {output_path}")
    
    cap.release()
    return extracted

def display_video_info(info):
    """Pretty print video information"""
    print("\n" + "="*60)
    print("VIDEO INFORMATION")
    print("="*60)
    print(f"Filename:     {info['filename']}")
    print(f"Path:         {info['path']}")
    print(f"Resolution:   {info['width']}x{info['height']}")
    print(f"Frame Rate:   {info['fps']:.2f} fps")
    print(f"Total Frames: {info['frame_count']:,}")
    print(f"Duration:     {info['duration_minutes']:.2f} minutes ({info['duration_seconds']:.1f} seconds)")
    print(f"Codec:        {info['codec_name']}")
    print(f"File Size:    {info['file_size_mb']:.2f} MB")
    print("="*60)
    
    # Enhancement estimates
    print("\nENHANCEMENT ESTIMATES (RTX 4060)")
    print("="*60)
    
    # Estimate processing time for 2x upscale
    est_time_2x = (info['duration_minutes']) * 20  # ~20 min processing per min of video
    print(f"2x Upscale:   ~{est_time_2x:.1f} minutes")
    print(f"  Output:     {info['width']*2}x{info['height']*2}")
    
    # Estimate for 4x upscale
    est_time_4x = (info['duration_minutes']) * 60  # ~60 min processing per min of video
    print(f"4x Upscale:   ~{est_time_4x:.1f} minutes")
    print(f"  Output:     {info['width']*4}x{info['height']*4}")
    
    # Frame interpolation
    if info['fps'] < 24:
        fps_increase = 24 / info['fps']
        print(f"\nFrame Interpolation: {info['fps']:.0f}fps → 24fps ({fps_increase:.2f}x frames)")
        print(f"  Additional time: +50-100% of base processing time")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Get video information and preview frames")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("-f", "--frames", type=int, default=5,
                       help="Number of sample frames to extract (default: 5)")
    parser.add_argument("--no-extract", action="store_true",
                       help="Don't extract sample frames, just show info")
    
    args = parser.parse_args()
    
    # Get video info
    info = get_video_info(args.video)
    
    if info:
        display_video_info(info)
        
        if not args.no_extract:
            print(f"Extracting {args.frames} sample frames...")
            frames = extract_sample_frames(args.video, num_frames=args.frames)
            print(f"\n✓ Extracted {len(frames)} frames to: preview_frames/{Path(args.video).stem}/")

if __name__ == "__main__":
    main()
