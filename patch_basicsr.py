"""
Patch for BasicSR to work with newer torchvision versions
This fixes the import error by using the correct torchvision function
"""

import sys
from pathlib import Path

# Find BasicSR installation
basicsr_path = None
for path in sys.path:
    check_path = Path(path) / 'basicsr' / 'data' / 'degradations.py'
    if check_path.exists():
        basicsr_path = check_path
        break

if not basicsr_path:
    print("✗ Could not find BasicSR installation")
    sys.exit(1)

print(f"Found BasicSR at: {basicsr_path}")

# Read the file
with open(basicsr_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'rgb_to_grayscale' in content and 'functional_tensor' in content:
    print("Applying patch...")
    
    # Replace the problematic import
    old_import = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
    new_import = 'from torchvision.transforms.functional import rgb_to_grayscale'
    
    content = content.replace(old_import, new_import)
    
    # Write back
    with open(basicsr_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Patch applied successfully!")
    print("\nNow run: python test_setup.py")
else:
    print("✓ File is already patched or doesn't need patching")
