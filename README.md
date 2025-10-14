# 🎬 8mm Film Restoration Pipeline

AI-powered restoration pipeline for digitized 8mm home movies using Real-ESRGAN, GFPGAN, and advanced video processing techniques.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### **Working AI Enhancement Pipeline:**
- 🎯 **Stabilization** - Remove camera shake (OpenCV motion estimation)
- ✨ **Deflicker** - Remove brightness flickering from projector scans (temporal median filtering)
- 🧹 **Denoising** - 7-frame temporal Non-Local Means for aggressive grain removal
- 🚀 **AI Upscaling** - Real-ESRGAN 2x/4x (98% GPU utilization on RTX 4060)
- 👤 **Face Restoration** - GFPGAN v1.4 for enhanced facial details
- 🔍 **Detail Enhancement** - Smart unsharp masking with edge detection
- 🎨 **Scratch Removal** - AI-based temporal defect detection + inpainting (optional - slow)

### **Performance:**
- **RTX 4060 Laptop GPU** - Optimized for maximum GPU utilization (98%)
- **~6 hours** for 60 seconds of video (full classic pipeline)
- **~1 hour** for 10 seconds (quick test recommended)
- **Tile size 2048** for optimal GPU memory usage (8GB VRAM)
- **Two-step encoding** (AVI → MP4) to prevent FFmpeg hangs

## 📋 Requirements

- **Python 3.12+**
- **CUDA 12.1+** (NVIDIA GPU required)
- **8GB+ VRAM** (tested on RTX 4060)
- **FFmpeg** (for video encoding)

## 🚀 Installation

### **1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/8mm-film-restoration.git
cd 8mm-film-restoration
```

### **2. Create virtual environment:**
```bash
python -m venv env
```

### **3. Activate virtual environment:**
- **Windows:** `env\Scripts\activate`
- **Linux/Mac:** `source env/bin/activate`

### **4. Install dependencies:**
```bash
pip install -r requirements.txt
```

### **5. Run setup script (IMPORTANT - First time only):**
```bash
python setup.py
```
This will:
- Create necessary directories (models, temp, output_videos, etc.)
- Download AI models automatically
- Verify your GPU and CUDA installation
- Apply necessary patches (BasicSR fix)

**Alternative:** Manual model download if setup.py fails:
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) → `models/`
- [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth) → `models/`
- [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) → `models/`

See [MANUAL_MODEL_DOWNLOAD.md](MANUAL_MODEL_DOWNLOAD.md) for detailed manual download instructions.

## 📖 Quick Start

### **Complete Example - Classic Pipeline:**

**Process 60 seconds of your 8mm film with all enhancements:**
```bash
# Windows PowerShell:
python scripts\enhance_video.py input_videos\MyFilm.mp4 --cut 0 60 --steps stabilization deflicker denoising upscaling face_restoration detail_enhancement --upscale 2 -o output_videos\MyFilm_restored.mp4

# Linux/Mac:
python scripts/enhance_video.py input_videos/MyFilm.mp4 --cut 0 60 --steps stabilization deflicker denoising upscaling face_restoration detail_enhancement --upscale 2 -o output_videos/MyFilm_restored.mp4
```

**Explanation:**
- `--cut 0 60` = Process first 60 seconds (test before full video!)
- `--steps ...` = All enhancement steps in recommended order
- `--upscale 2` = 2x AI upscaling (864x648 → 1728x1296)
- `-o` = Output file path

### **Quick Test (10 seconds, basic enhancement):**
```bash
python scripts\enhance_video.py input_videos\MyFilm.mp4 --cut 0 10 --steps stabilization denoising upscaling --upscale 2
```

### **Full Video (no time limit):**
```bash
python scripts\enhance_video.py input_videos\MyFilm.mp4 --steps stabilization deflicker denoising upscaling face_restoration detail_enhancement --upscale 2
```

## 🎛️ Command-Line Options

```bash
python scripts/enhance_video.py INPUT_VIDEO [OPTIONS]

Options:
  -o, --output PATH          Output video file path
  --cut START END            Cut video from START to END seconds
  --steps STEP [STEP ...]    Processing steps to perform
  --skip-steps STEP [...]    Steps to skip
  --upscale {2,4}            Upscale factor (default: 2)
  --fps FPS                  Target frame rate
  --nvenc                    Use NVIDIA hardware encoding (if available)
  --model {realesrgan,basicvsr}  AI upscaling model
  --resume-from PATH         Resume from intermediate file
```

### **Available Steps:**
- `stabilization` - Remove camera shake
- `deflicker` - Fix brightness flickering
- `scratch_removal` - AI scratch/dust removal (very slow)
- `denoising` - Temporal grain removal
- `upscaling` - AI upscaling (2x or 4x)
- `face_restoration` - Face enhancement with GFPGAN
- `detail_enhancement` - Adaptive sharpening
- `frame_interpolation` - Increase frame rate (experimental)
- `color_grading` - Manual step placeholder

## ⏱️ Processing Time Estimates (RTX 4060 Laptop GPU)

**Based on actual 60-second video processing (Gesher_Aziv test):**

| Step | Time | Speed | Notes |
|------|------|-------|-------|
| **Stabilization** | ~1.3 min | ~1.3 sec/frame | Fast motion estimation |
| **Deflicker** | ~1.9 min | ~1.9 sec/frame | Temporal brightness smoothing |
| **Denoising** | ~4h 20min | ~4.3 sec/frame | 7-frame temporal NLMeans (most intensive) |
| **Upscaling 2x** | ~1h 19min | ~1.3 sec/frame | Real-ESRGAN with tile size 2048 |
| **Face Restoration** | ~20 min | ~0.3 sec/frame | GFPGAN v1.4 |
| **Detail Enhancement** | *~10 min* | *~0.2 sec/frame* | Adaptive unsharp masking (estimated) |
| **TOTAL (Full Pipeline)** | **~6h 12min** | | For 60 seconds of video |

**Extrapolated estimates:**
- **10 seconds:** ~62 minutes (1 hour)
- **60 seconds:** ~6 hours 12 minutes
- **5 minutes (300 sec):** ~31 hours
- **Full 3-minute reel:** ~18-19 hours

**Note:** Denoising is the bottleneck (70% of total time). You can:
- Skip denoising for faster processing (not recommended - grain will be upscaled)
- Reduce denoising strength in `config.py` (DENOISE_STRENGTH = 0.5 instead of 0.7)
- Process overnight for best quality results

**With Scratch Removal (not recommended for routine use):**
- Add ~32 sec/frame = **~32 hours extra** for 60 seconds!

## 📁 Project Structure

```
8mm-film-restoration/
├── scripts/
│   ├── enhance_video.py          # Main pipeline orchestrator
│   ├── stabilize_video.py        # Camera shake removal
│   ├── deflicker_video.py        # Brightness flickering correction
│   ├── denoise_video.py          # Temporal grain removal
│   ├── upscale_video.py          # Real-ESRGAN upscaling
│   ├── restore_faces.py          # GFPGAN face restoration
│   ├── enhance_details.py        # Adaptive detail enhancement
│   ├── film_restoration.py       # Scratch/dust removal
│   └── interpolate_frames.py     # Frame rate increase
├── models/                        # AI model files (download separately)
├── input_videos/                  # Place your source videos here
├── output_videos/                 # Enhanced videos output here
├── temp/                          # Intermediate processing steps
├── logs/                          # Processing logs
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- **GPU settings** - Device selection, tile size
- **Denoising strength** - Grain removal intensity
- **Upscale factor** - 2x or 4x default
- **Output codec** - H.264, HEVC, etc.
- **Processing steps** - Default pipeline order

## 🎨 Standalone Scripts

Each enhancement can be run independently:

```bash
# Stabilization only
python scripts/stabilize_video.py input.mp4

# Deflicker only
python scripts/deflicker_video.py input.mp4 --window 15 --strength 0.8

# Denoising only
python scripts/denoise_video.py input.mp4

# Upscaling only
python scripts/upscale_video.py input.mp4 --scale 2

# Face restoration only
python scripts/restore_faces.py input.mp4

# Detail enhancement only
python scripts/enhance_details.py input.mp4 --amount 1.5 --radius 2.0

# Scratch removal only (very slow)
python scripts/film_restoration.py input.mp4
```

## 🐛 Troubleshooting

### **GPU not detected:**
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Update NVIDIA drivers
- Reinstall PyTorch with CUDA support

### **Out of memory errors:**
- Reduce `UPSCALE_TILE_SIZE` in config.py (try 1024 or 512)
- Process shorter clips with `--cut`
- Use 2x upscaling instead of 4x

### **FFmpeg hangs:**
- Two-step encoding is already implemented (AVI → MP4)
- Check FFmpeg installation: `ffmpeg -version`

### **Slow processing:**
- Verify GPU utilization (should be ~98%)
- Skip `scratch_removal` step (very slow)
- Use `frame_interpolation` only when needed

## 📝 Known Issues

- ❌ **NVENC encoding** - Requires driver API 13.0+ (tested system has 12.2)
- ⚠️ **Scratch removal** - Very slow (~32 sec/frame), use sparingly
- ⚠️ **Frame interpolation** - RIFE model version mismatch, experimental

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI upscaling
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face restoration
- [OpenCV](https://opencv.org/) - Video processing
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - Super-resolution framework

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Made with ❤️ for preserving family memories**
