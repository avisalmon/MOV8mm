"""
Video output utility - Write videos with proper H.264 codec
Replaces OpenCV's VideoWriter which outputs incompatible mp4v codec
"""

import subprocess
import numpy as np
from pathlib import Path
import tempfile
import cv2
from loguru import logger


class H264VideoWriter:
    """
    Video writer that outputs H.264 codec using FFmpeg
    Compatible with all video players (VLC, Windows Media Player, etc.)
    """
    
    def __init__(self, output_path, fps, width, height, codec='libx264', crf=18, preset='medium', use_nvenc=False):
        """
        Args:
            output_path: Output video file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (libx264, libx265, or h264_nvenc)
            crf: Quality (0-51, lower=better, 18-23 recommended) - ignored for nvenc
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            use_nvenc: Use NVIDIA hardware encoding (overrides codec to h264_nvenc)
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.use_nvenc = use_nvenc
        self.codec = 'h264_nvenc' if use_nvenc else codec
        self.crf = crf
        self.preset = preset
        self.process = None
        self.frame_count = 0
        
        # Start FFmpeg process
        self._start_ffmpeg()
    
    def _start_ffmpeg(self):
        """Start FFmpeg subprocess to write video"""
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin
            '-an',  # No audio
            '-vcodec', self.codec,
        ]
        
        # Add quality settings based on codec
        if self.use_nvenc:
            # NVENC quality settings
            command.extend([
                '-preset', 'p7',  # p7 = highest quality preset for NVENC
                '-tune', 'hq',    # High quality tuning
                '-rc', 'vbr',     # Variable bitrate
                '-cq', str(self.crf),  # Quality level (lower = better)
                '-b:v', '0',      # Let CQ control bitrate
            ])
        else:
            # Software encoding quality settings
            command.extend([
                '-crf', str(self.crf),
                '-preset', self.preset,
            ])
        
        # Common settings
        command.extend([
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.output_path
        ])
        
        try:
            # Convert path to use forward slashes (works on Windows too)
            output_path_normalized = Path(self.output_path).as_posix()
            command[-1] = output_path_normalized
            
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            logger.debug(f"Started FFmpeg process for {output_path_normalized}")
        except FileNotFoundError:
            logger.error("FFmpeg not found! Please install FFmpeg.")
            raise
    
    def write(self, frame):
        """
        Write a frame to the video
        
        Args:
            frame: NumPy array in BGR format (OpenCV standard)
        """
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg process not started")
        
        # Check if FFmpeg process is still alive
        if self.process.poll() is not None:
            # Process died, get error output
            stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else "No error output"
            logger.error(f"FFmpeg process died. Error: {stderr}")
            raise RuntimeError(f"FFmpeg process terminated unexpectedly: {stderr}")
        
        try:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
        except BrokenPipeError:
            stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else "No error output"
            logger.error(f"FFmpeg broken pipe. Error: {stderr}")
            raise
    
    def release(self):
        """Close the video writer"""
        if self.process and self.process.stdin:
            try:
                # Flush any buffered frames
                logger.debug(f"Flushing {self.frame_count} frames to FFmpeg...")
                self.process.stdin.flush()
                self.process.stdin.close()
                
                # Wait longer for FFmpeg to finish encoding (up to 2 minutes)
                logger.debug("Waiting for FFmpeg to finalize encoding...")
                self.process.wait(timeout=120)
                
                # Check return code
                if self.process.returncode != 0:
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else "No error"
                    logger.error(f"FFmpeg failed with code {self.process.returncode}: {stderr}")
                else:
                    logger.info(f"✓ Successfully wrote {self.frame_count} frames to {self.output_path}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg encoding timeout after 2 minutes! This shouldn't happen.")
                logger.error(f"File may be corrupted. Terminating FFmpeg...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error(f"FFmpeg won't terminate, force killing...")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def convert_video_to_h264(input_path, output_path=None, crf=18, preset='medium', fps=None):
    """
    Convert any video to H.264 codec using FFmpeg
    
    Args:
        input_path: Input video file
        output_path: Output video file (optional)
        crf: Quality (18-23 recommended, lower=better)
        preset: Encoding speed (ultrafast, fast, medium, slow)
        fps: Target FPS (optional, keeps original if None)
    
    Returns:
        Path to output video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_h264.mp4"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Converting {input_path.name} to H.264...")
    
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
    ]
    
    if fps:
        command.extend(['-r', str(fps)])
    
    command.extend(['-y', str(output_path)])
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✓ Converted to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise


def main():
    """Test the H264VideoWriter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert video to H.264 codec")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("--crf", type=int, default=18, help="Quality (18-23 recommended)")
    parser.add_argument("--preset", default="medium",
                       choices=['ultrafast', 'fast', 'medium', 'slow', 'veryslow'],
                       help="Encoding preset")
    parser.add_argument("--fps", type=float, help="Target FPS (optional)")
    
    args = parser.parse_args()
    
    convert_video_to_h264(args.input, args.output, args.crf, args.preset, args.fps)


if __name__ == "__main__":
    main()
