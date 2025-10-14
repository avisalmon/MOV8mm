"""
Batch processing script for enhancing multiple 8mm videos
"""

import sys
from pathlib import Path
import argparse
from loguru import logger
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
import config
from scripts.enhance_video import enhance_video

def batch_enhance(input_dir=None, output_dir=None, pattern="*.mp4", steps=None):
    """
    Batch process all videos in a directory
    
    Args:
        input_dir: Directory containing input videos (default: config.INPUT_DIR)
        output_dir: Directory to save enhanced videos (default: config.OUTPUT_DIR)
        pattern: File pattern to match (default: *.mp4)
        steps: Processing steps to perform (default: all from config)
    """
    input_dir = Path(input_dir or config.INPUT_DIR)
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Find all matching videos
    video_files = list(input_dir.glob(pattern))
    
    if not video_files:
        logger.warning(f"No videos found matching pattern '{pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process each video
    results = []
    start_time = datetime.now()
    
    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing video {idx}/{len(video_files)}: {video_path.name}")
        logger.info(f"{'='*70}")
        
        output_path = output_dir / f"{video_path.stem}{config.OUTPUT_SUFFIX}{video_path.suffix}"
        
        try:
            success = enhance_video(video_path, output_path, steps)
            results.append({
                'file': video_path.name,
                'success': success,
                'output': output_path if success else None
            })
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            results.append({
                'file': video_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    total_time = datetime.now() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"\n{'='*70}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total videos: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time}")
    
    if failed > 0:
        logger.warning("\nFailed videos:")
        for result in results:
            if not result['success']:
                error = result.get('error', 'Unknown error')
                logger.warning(f"  - {result['file']}: {error}")
    
    logger.info(f"\nEnhanced videos saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch enhance multiple 8mm videos")
    parser.add_argument("-i", "--input", help=f"Input directory (default: {config.INPUT_DIR})")
    parser.add_argument("-o", "--output", help=f"Output directory (default: {config.OUTPUT_DIR})")
    parser.add_argument("-p", "--pattern", default="*.mp4",
                       help="File pattern to match (default: *.mp4)")
    parser.add_argument("-s", "--steps", nargs="+",
                       choices=["stabilization", "upscaling", "denoising", "frame_interpolation"],
                       help="Processing steps to perform (default: all from config)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.TEMP_DIR).mkdir(exist_ok=True)
    
    # Run batch processing
    batch_enhance(args.input, args.output, args.pattern, args.steps)

if __name__ == "__main__":
    main()
