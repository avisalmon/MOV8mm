# GitHub Repository Setup Guide

## 📋 Pre-Upload Checklist

### 1. **Verify .gitignore is properly configured**
```bash
# Check if .gitignore exists and contains necessary exclusions
cat .gitignore
```

**Should include:**
- `*.mp4`, `*.avi`, `*.mov` (video files)
- `models/*.pth` (AI model files - too large for git)
- `temp/`, `output_videos/`, `input_videos/` (processing directories)
- `env/`, `venv/` (virtual environment)
- `__pycache__/`, `*.pyc` (Python cache)
- `logs/` (log files)

### 2. **Clean up large files from repo**
```bash
# Remove any accidentally committed large files
git rm --cached models/*.pth
git rm --cached temp/*.mp4
git rm --cached output_videos/*.mp4
git rm --cached input_videos/*.mp4
```

### 3. **Verify requirements.txt**
```bash
# Make sure all dependencies are listed
pip freeze > requirements-frozen.txt
# Compare with requirements.txt
```

## 🚀 GitHub Repository Setup Steps

### **Step 1: Initialize Git Repository**
```bash
cd C:\Projects\MOV8MM-new
git init
```

### **Step 2: Configure Git**
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### **Step 3: Create Repository on GitHub**
1. Go to https://github.com/new
2. Repository name: `8mm-film-restoration` (or your preferred name)
3. Description: "AI-powered restoration pipeline for digitized 8mm home movies"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### **Step 4: Stage Files**
```bash
# Add all files (respecting .gitignore)
git add .

# Verify what will be committed (should NOT include videos or models)
git status
```

### **Step 5: Initial Commit**
```bash
git commit -m "Initial commit: 8mm film restoration pipeline

Features:
- AI upscaling with Real-ESRGAN
- Face restoration with GFPGAN
- Temporal denoising
- Video stabilization
- Deflickering
- Detail enhancement"
```

### **Step 6: Add Remote Repository**
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/8mm-film-restoration.git
```

### **Step 7: Push to GitHub**
```bash
# Push main branch
git branch -M main
git push -u origin main
```

## 📦 Handling Large Model Files

### **Option 1: Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
git push
```

**Note:** GitHub LFS has storage limits (1GB free, then paid).

### **Option 2: External Hosting (Recommended)**
Keep model files OUT of repository and provide download instructions in README:
- Users download models from official sources (Real-ESRGAN, GFPGAN repos)
- Create `MANUAL_MODEL_DOWNLOAD.md` with direct links
- Keeps repository lightweight

## 📝 Post-Upload Tasks

### **1. Create GitHub Releases (Optional)**
```bash
# Tag a version
git tag -a v1.0.0 -m "Initial release: Full pipeline with 6 working steps"
git push origin v1.0.0
```

### **2. Add Topics/Tags on GitHub**
Navigate to your repository → About (gear icon) → Add topics:
- `video-processing`
- `ai-upscaling`
- `8mm-film`
- `real-esrgan`
- `gfpgan`
- `restoration`
- `pytorch`
- `cuda`

### **3. Create Issues/Project Board (Optional)**
Create issues for:
- [ ] Improve frame interpolation (RIFE compatibility)
- [ ] Add NVENC support for newer drivers
- [ ] Optimize scratch removal speed
- [ ] Add white balance correction
- [ ] Create web UI

### **4. Add License File**
```bash
# Create LICENSE file (MIT example)
# Copy MIT license text to LICENSE file
git add LICENSE
git commit -m "Add MIT license"
git push
```

## 🔄 Daily Development Workflow

### **Making Changes:**
```bash
# 1. Make your code changes
# 2. Check what changed
git status
git diff

# 3. Stage changes
git add scripts/enhance_video.py

# 4. Commit with descriptive message
git commit -m "Fix: Corrected detail_enhancement parameter name"

# 5. Push to GitHub
git push
```

### **Updating README:**
```bash
git add README.md
git commit -m "Update README with new features"
git push
```

## ⚠️ Important Notes

### **DO NOT commit:**
- ❌ Video files (*.mp4, *.avi, *.mov)
- ❌ AI model files (*.pth) - too large, use download instructions
- ❌ Virtual environment (env/, venv/)
- ❌ Test output videos
- ❌ API keys or credentials

### **DO commit:**
- ✅ Source code (*.py)
- ✅ Configuration files (config.py)
- ✅ Documentation (*.md)
- ✅ Requirements (requirements.txt)
- ✅ Scripts and utilities

## 🎯 Repository Best Practices

1. **Write clear commit messages**
   - Use present tense: "Add feature" not "Added feature"
   - First line < 50 chars
   - Detailed description if needed

2. **Create branches for features**
   ```bash
   git checkout -b feature/white-balance
   # Make changes
   git commit -m "Add white balance correction"
   git push -u origin feature/white-balance
   # Create Pull Request on GitHub
   ```

3. **Keep README up-to-date**
   - Update features list
   - Add new examples
   - Document configuration changes

4. **Version your releases**
   - Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
   - Tag major milestones

## 📊 Repository Maintenance

### **Check repository size:**
```bash
git count-objects -vH
```

### **Clean up history (if needed):**
```bash
# Remove large files from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch models/*.pth' \
  --prune-empty --tag-name-filter cat -- --all
```

## 🤝 Making Repository Public-Ready

Before making repository public:
- [ ] Remove any personal paths or credentials
- [ ] Add comprehensive README
- [ ] Include license
- [ ] Add usage examples
- [ ] Document installation clearly
- [ ] Test setup on fresh system
- [ ] Add contributing guidelines
- [ ] Create issue templates

---

**You're ready to share your 8mm film restoration pipeline with the world! 🎬✨**
