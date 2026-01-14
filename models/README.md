# Models Directory

This directory stores deep learning models for SentinelAI.

## Auto-Downloaded Models

On first run, the following models will automatically download:

1. **YOLOv8n** (~6MB)
   - Path: `yolov8n.pt` (or custom path from .env)
   - Purpose: Weapon and threat object detection
   - Source: Ultralytics
   - License: AGPL-3.0

2. **Facenet512** (~100MB)
   - Cached in: `./cache/`
   - Purpose: Face recognition and matching
   - Source: DeepFace library
   - Pretrained on VGGFace2

3. **VGG-Face weights** (~500MB)
   - Cached in: `./cache/`
   - Backend for face detection
   - Source: Oxford VGG Group

## Directory Structure

```
models/
├── yolov8n.pt              # YOLOv8 nano model (auto-downloaded)
├── yolov8s.pt              # YOLOv8 small (optional, better accuracy)
├── yolov8-weapon.pt        # Custom weapon model (if trained)
└── cache/                  # DeepFace model cache
    ├── facenet512_weights.h5
    ├── vgg_face_weights.h5
    └── ...other model files
```

## Storage Requirements

- **Minimum:** ~650MB (YOLOv8n + Facenet512)
- **Recommended:** 2GB (includes larger YOLO variants)

## Custom Models

Place custom trained models here and update `.env`:

```dotenv
YOLO_MODEL_PATH=./models/yolov8-weapon-custom.pt
```

## .gitignore

All `.pt` and `.h5` files are excluded from git to prevent large file commits.
Models will auto-download on deployment.
