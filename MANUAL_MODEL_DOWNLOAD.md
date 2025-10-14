# Manual Model Download Instructions

If `python setup.py` fails to download models, follow these steps:

## Download Real-ESRGAN Models Manually

### Option 1: Direct Download Links (Recommended)

**RealESRGAN_x4plus.pth** (for 4x upscaling):
```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```
- Size: ~67 MB
- Copy to: `c:\Projects\MOV8MM-new\models\RealESRGAN_x4plus.pth`

**RealESRGAN_x2plus.pth** (for 2x upscaling):
```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```
- Size: ~67 MB
- Copy to: `c:\Projects\MOV8MM-new\models\RealESRGAN_x2plus.pth`

### Option 2: Download via PowerShell

```powershell
# Make sure you're in the project directory
cd c:\Projects\MOV8MM-new

# Download x4plus model (if needed)
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "models\RealESRGAN_x4plus.pth"

# Download x2plus model
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" -OutFile "models\RealESRGAN_x2plus.pth"
```

### Option 3: Download via Browser

1. Open browser and go to: https://github.com/xinntao/Real-ESRGAN/releases
2. Find these releases:
   - **v0.1.0** - Download `RealESRGAN_x4plus.pth`
   - **v0.2.1** - Download `RealESRGAN_x2plus.pth`
3. Move the downloaded files to: `c:\Projects\MOV8MM-new\models\`

### Verify Downloads

```powershell
# Check if files exist and show their sizes
Get-ChildItem models\*.pth | Select-Object Name, @{Name="Size MB";Expression={[math]::Round($_.Length/1MB, 1)}}
```

**Expected output:**
```
Name                      Size MB
----                      -------
RealESRGAN_x2plus.pth        67.0
RealESRGAN_x4plus.pth        67.1
```

### Test After Manual Download

```powershell
python test_setup.py
```

You should see:
```
✓ RealESRGAN_x4plus.pth (67.1 MB)
✓ RealESRGAN_x2plus.pth (67.0 MB)
AI models ready: Yes
```

---

## Alternative Models (If Above Don't Work)

If the above links don't work, you can use these alternative models:

### RealESRGAN (Original)
- Repository: https://github.com/xinntao/Real-ESRGAN
- Go to "Releases" tab
- Download latest .pth files
- Place in `models/` folder

### Model Compatibility

The scripts are designed to work with:
- `RealESRGAN_x4plus.pth` - Best for general 4x upscaling
- `RealESRGAN_x2plus.pth` - Best for 2x upscaling

**Note:** Other Real-ESRGAN models exist but may require code modifications.

---

## Troubleshooting

**Problem:** "Access Denied" when downloading
- Run PowerShell as Administrator
- Or download via browser method

**Problem:** Download is slow
- Large files (67MB each) may take time
- Be patient or try different internet connection

**Problem:** Files won't move to models folder
- Check if `models/` folder exists
- Create manually if needed: `mkdir models`

**Problem:** Model downloaded but not detected
- Check filename exactly matches (case-sensitive)
- Ensure file is in `models/` folder directly (not subfolder)
- Check file size is ~67MB (not 0 KB or corrupted)

---

## After Manual Download

Once both models are downloaded:

1. ✅ Run test: `python test_setup.py`
2. ✅ Verify GPU: Should show CUDA available
3. ✅ Try processing: `python scripts/upscale_video.py input_videos/test.mp4 -s 2`

You're ready to enhance your 8mm footage! 🎬