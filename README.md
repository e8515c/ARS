# Aspect Ratio Changer (Python)
# GUI: Tkinter
# Processing: OpenCV (opencv-contrib-python), Pillow
# Features:
Stretch image to target aspect ratio: 1:1, 16:9, 9:16, 4:3 (non-uniform scaling).
Batch processing: multiple files or a whole folder (optional recursion).
# Upscale options:
None
2× (ML) / 4× (ML) via OpenCV dnn_superres (EDSR/ESPCN .pb models)
2× (fast) / 4× (fast) via standard cv2.resize (no models)
Optimize output size without changing resolution: Simple / Medium / Max.
Remembers paths in %APPDATA%/ARS_e8515c/settings.json.
Unicode-safe read/write on Windows (np.fromfile/imdecode and imencode/tofile).
# Formats: 
jpg/jpeg/png/webp input/output; gif reads first frame and saves as GIF via Pillow.
Every saved file gets a random name like: e8515c_<6-12 letters/digits>. Originals are never overwritten.
# Author:
e8515c
![loli-xdxd](https://github.com/user-attachments/assets/e6ee1e7e-35e6-41cd-89ea-8d05b5c630dc)
