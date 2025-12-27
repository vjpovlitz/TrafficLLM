# Image Library System Guide

Complete guide to the organized image storage and management system for traffic camera captures.

---

## Overview

The Image Library System provides hierarchical, date-based organization for traffic camera images with automatic validation, metadata tracking, and powerful search capabilities.

### Key Features

- **Hierarchical organization**: `YYYY/MM/DD/camera_id/category/`
- **Automatic validation**: Quality scoring and validated image storage
- **Metadata tracking**: JSON metadata for every image
- **Search & query**: Find images by date, camera, quality
- **Storage management**: Cleanup old images, integrity validation
- **Statistics**: Per-camera and date-based analytics

---

## Directory Structure

```
data/image_library/
├── 2025/
│   ├── 01/
│   │   ├── 26/
│   │   │   ├── us_50_sandy_point/
│   │   │   │   ├── raw/                    # Original captures
│   │   │   │   │   ├── US_50_AT_SANDY_POINT_20250126_143052_123.png
│   │   │   │   │   └── ...
│   │   │   │   ├── validated/              # Quality-validated images
│   │   │   │   │   └── US_50_AT_SANDY_POINT_20250126_143052_123.png
│   │   │   │   └── metadata/               # JSON metadata files
│   │   │   │       └── US_50_AT_SANDY_POINT_20250126_143052_123.json
│   │   │   └── us_50_ex23_md2/
│   │   │       ├── raw/
│   │   │       ├── validated/
│   │   │       └── metadata/
│   │   └── 27/
│   │       └── ...
│   └── 02/
│       └── ...
└── _index/                                  # Daily index files
    ├── 2025-01-26.jsonl
    ├── 2025-01-27.jsonl
    └── ...
```

### Categories

- **raw**: Original screenshots as captured
- **validated**: Images passing quality thresholds (score ≥ 0.4)
- **metadata**: JSON files with capture info, quality metrics, weather
- **thumbnails**: (Optional) Smaller preview images

---

## Quick Start

### 1. Test Fullscreen Capture

```bash
cd TrafficLLM

# Test single camera with comparison
python scripts/test_fullscreen_capture.py --camera us_50_sandy_point

# Output: test_captures/
#   - us_50_sandy_point_standard.png
#   - us_50_sandy_point_fullscreen.png
```

**Expected Results:**
- File size increase: +180-200%
- Pixel count increase: Similar to file size
- Fullscreen images are video-only (no webpage elements)

### 2. Capture to Library

```bash
# Single capture with library storage
python scripts/capture_to_library.py --mode single --fullscreen

# Scheduled captures every 15 minutes
python scripts/capture_to_library.py --mode scheduled --interval 15 --fullscreen
```

**What Happens:**
1. Captures screenshot (fullscreen mode)
2. Stores in `library/YYYY/MM/DD/camera_id/raw/`
3. Validates quality (Laplacian blur, brightness, contrast, edges)
4. If valid (score ≥ 0.4), stores copy in `validated/`
5. Saves metadata JSON
6. Updates daily index

### 3. View Library Statistics

```bash
# Overall stats (last 7 days)
python scripts/manage_image_library.py stats

# Detailed per-camera breakdown
python scripts/manage_image_library.py stats --days 30 --detailed
```

**Sample Output:**
```
Overall:
  Total images: 1,248
  Total size: 3,142.5 MB
  Validated: 1,102
  Fullscreen: 1,248
  Avg quality: 0.782

Per-camera breakdown:
  us_50_sandy_point: 624 images
  us_50_ex23_md2: 624 images

Daily breakdown:
  2025-01-20: 192 images
  2025-01-21: 192 images
  ...
```

---

## Usage Examples

### Capture Operations

**Single capture (all cameras):**
```bash
python scripts/capture_to_library.py --mode single --fullscreen
```

**Single capture (specific camera):**
```bash
python scripts/capture_to_library.py \
    --mode single \
    --camera us_50_sandy_point \
    --fullscreen
```

**Scheduled captures (every 10 minutes):**
```bash
python scripts/capture_to_library.py \
    --mode scheduled \
    --interval 10 \
    --fullscreen
```

**Without validation (faster, stores all images):**
```bash
python scripts/capture_to_library.py \
    --mode single \
    --fullscreen \
    --no-validate
```

### Search & Query

**Find all images from camera in last 7 days:**
```bash
python scripts/manage_image_library.py search \
    --camera us_50_sandy_point \
    --days 7
```

**Find high-quality validated images:**
```bash
python scripts/manage_image_library.py search \
    --category validated \
    --min-quality 0.7 \
    --days 30
```

**Search specific date range:**
```bash
python scripts/manage_image_library.py search \
    --start-date 2025-01-20 \
    --end-date 2025-01-26 \
    --output search_results.json
```

### Library Management

**Export full catalog:**
```bash
python scripts/manage_image_library.py export --output library_catalog.json
```

**Validate integrity:**
```bash
python scripts/manage_image_library.py validate
```

**Cleanup old raw images (dry run):**
```bash
python scripts/manage_image_library.py cleanup \
    --days 30 \
    --category raw \
    --dry-run
```

**Actually delete (after reviewing dry run):**
```bash
python scripts/manage_image_library.py cleanup \
    --days 30 \
    --category raw
```

---

## Python API

### Basic Usage

```python
from pathlib import Path
from trafficllm.data_collection.image_library import ImageLibrary

# Initialize library
library = ImageLibrary("data/image_library")

# Store an image
record = library.store_image(
    camera_id="us_50_sandy_point",
    camera_name="US_50_AT_SANDY_POINT",
    image_path=Path("screenshot.png"),
    category="raw",
    metadata={
        'location': 'Sandy Point',
        'region': 'Maryland',
        'quality_score': 0.85,
        'fullscreen_mode': True
    }
)

print(f"Stored: {record.file_path}")
print(f"Size: {record.file_size / 1024:.1f} KB")
```

### Search Images

```python
from datetime import datetime, timedelta

# Search last 7 days
start_date = datetime.now() - timedelta(days=7)
results = library.search_images(
    camera_id="us_50_sandy_point",
    start_date=start_date,
    category="validated",
    min_quality=0.7
)

for record in results:
    print(f"{record.capture_time}: {record.file_path}")
    print(f"  Quality: {record.quality_score:.3f}")
```

### Get Statistics

```python
# Overall stats
stats = library.get_camera_stats(days=30)
print(f"Total images: {stats['total_images']}")
print(f"Total size: {stats['total_size_mb']:.1f} MB")
print(f"Cameras: {stats['cameras']}")

# Per-camera stats
cam_stats = library.get_camera_stats(
    camera_id="us_50_sandy_point",
    days=7
)
print(f"Images: {cam_stats['total_images']}")
print(f"Avg quality: {cam_stats['avg_quality']:.3f}")
```

### Integration with CameraManager

```python
from trafficllm.data_collection.camera_manager import CameraManager
from trafficllm.data_collection.image_library import ImageLibrary

# Initialize
manager = CameraManager(config_path="config/camera_config.json")
library = ImageLibrary("data/image_library")

# Capture and store
camera = manager.cameras["us_50_sandy_point"]
success, image_path, metadata = manager.capture_screenshot(
    "us_50_sandy_point",
    camera,
    fullscreen_video=True
)

if success:
    # Store in library
    record = library.store_image(
        camera_id="us_50_sandy_point",
        camera_name=camera.name,
        image_path=Path(image_path),
        category="raw",
        metadata=metadata,
        move=True  # Move instead of copy
    )
    print(f"Stored: {record.file_path}")
```

---

## Metadata Format

Each image has a corresponding JSON metadata file:

```json
{
  "camera_id": "us_50_sandy_point",
  "camera_name": "US_50_AT_SANDY_POINT",
  "file_path": "data/image_library/2025/01/26/us_50_sandy_point/raw/US_50_AT_SANDY_POINT_20250126_143052_123.png",
  "capture_time": "2025-01-26T14:30:52.123456",
  "file_size": 2458624,
  "quality_score": 0.823,
  "validated": true,
  "fullscreen_mode": true,
  "location": "Sandy Point",
  "region": "Maryland",
  "weather": {
    "temperature": 42.5,
    "conditions": "Clear",
    "humidity": 65
  }
}
```

---

## Integration with SAM 3

After collecting images in the library, annotate with SAM 3:

```bash
# Annotate validated images from last 7 days
python scripts/manage_image_library.py search \
    --category validated \
    --days 7 \
    --output recent_validated.json

# Extract paths and annotate
python scripts/annotate_with_sam3.py \
    --input data/image_library/2025/01/26/*/validated/ \
    --output data/annotations/ \
    --prompts "car" "truck" "bus" "motorcycle"
```

Or annotate directly from library structure:

```bash
# Annotate all validated images
find data/image_library/ -path "*/validated/*.png" | \
    xargs python scripts/annotate_with_sam3.py --input
```

---

## Performance Considerations

### Storage

**Typical storage usage:**
- Raw image (fullscreen): 1.5-2.5 MB
- Validated copy: +1.5-2.5 MB (duplicate)
- Metadata JSON: ~1 KB
- **Total per image**: ~3-5 MB

**For 5000 images:**
- Raw: 7.5-12.5 GB
- Validated: 7.5-12.5 GB (if all pass validation)
- Total: 15-25 GB

**Recommendation:**
- Keep validated images indefinitely (for training)
- Clean up raw images after 30 days (once validated)
- Use compression for long-term storage

### Cleanup Strategy

```bash
# Weekly: Clean raw images older than 30 days
python scripts/manage_image_library.py cleanup \
    --days 30 \
    --category raw

# Monthly: Archive validated images older than 90 days
# (Move to external storage or compress)

# Never delete: Recent validated images (last 90 days)
```

---

## Troubleshooting

### Issue: "No images found in library"

**Check:**
1. Library path: `data/image_library/` exists?
2. Images captured: Run `python scripts/capture_to_library.py --mode single`
3. Index files: Check `data/image_library/_index/`

### Issue: "Validation failing for all images"

**Possible causes:**
- Low image quality (blurry, dark, low resolution)
- Incorrect camera URL (not actually showing video)
- Fullscreen expansion failing (check logs)

**Solution:**
```bash
# Capture without validation to see raw quality
python scripts/capture_to_library.py --mode single --no-validate

# Check image manually
open data/image_library/2025/*/*/*/raw/*.png

# Adjust validation thresholds in ImageValidator if needed
```

### Issue: "Fullscreen mode not working"

**Diagnosis:**
```bash
# Test fullscreen vs standard comparison
python scripts/test_fullscreen_capture.py --camera us_50_sandy_point

# Check if fullscreen images are larger
# - Standard: ~682 KB
# - Fullscreen: ~1.2-2.2 MB
```

**If fullscreen same size as standard:**
- Camera URL may not have video element
- JavaScript fullscreen expansion failed (check logs)
- Try different camera

### Issue: "Library index corrupt"

**Fix:**
```bash
# Validate integrity
python scripts/manage_image_library.py validate

# Rebuild index (if needed)
# TODO: Implement rebuild_index() function
```

---

## Best Practices

### 1. Capture Strategy

**For continuous monitoring:**
```bash
# Scheduled captures every 15 minutes, fullscreen, validated
nohup python scripts/capture_to_library.py \
    --mode scheduled \
    --interval 15 \
    --fullscreen \
    > capture.log 2>&1 &
```

**For dataset collection:**
```bash
# High-frequency captures during rush hour (7-9 AM, 4-7 PM)
# Lower frequency at night (hourly)
# Use cron jobs or systemd timers
```

### 2. Storage Management

- **Daily**: Check library stats
- **Weekly**: Clean raw images >30 days old
- **Monthly**: Archive validated images >90 days
- **Quarterly**: Validate integrity, rebuild index if needed

### 3. Quality Control

```bash
# Weekly quality check: View validated images
python scripts/manage_image_library.py search \
    --category validated \
    --days 7 \
    --min-quality 0.8

# Identify cameras with low quality scores
python scripts/manage_image_library.py stats --detailed
# Look for cameras with avg_quality < 0.5
```

### 4. Pre-SAM 3 Annotation

```bash
# 1. Collect diverse dataset (different times, weather, traffic levels)
python scripts/capture_to_library.py --mode scheduled --interval 10

# 2. Validate library has good coverage
python scripts/manage_image_library.py stats --detailed

# 3. Export validated images for annotation
python scripts/manage_image_library.py search \
    --category validated \
    --days 30 \
    --output dataset_manifest.json

# 4. Annotate with SAM 3
python scripts/annotate_with_sam3.py \
    --input data/image_library/2025/*/*/*/validated/ \
    --output data/annotations/
```

---

## Next Steps

1. **Test fullscreen capture**: Verify quality improvement
   ```bash
   python scripts/test_fullscreen_capture.py --camera us_50_sandy_point
   ```

2. **Start collecting to library**: Build initial dataset
   ```bash
   python scripts/capture_to_library.py --mode scheduled --interval 15 --fullscreen
   ```

3. **Monitor library growth**: Check stats daily
   ```bash
   python scripts/manage_image_library.py stats
   ```

4. **Annotate with SAM 3**: Once you have 500-1000 images
   ```bash
   python scripts/annotate_with_sam3.py --input data/image_library/*/validated/
   ```

5. **Train custom YOLO**: After annotation
   ```bash
   python scripts/train_yolo_with_sam3.py --data data/annotations/yolo_format/dataset.yaml
   ```

---

## Resources

- [CameraManager Documentation](../src/trafficllm/data_collection/camera_manager.py)
- [ImageLibrary API](../src/trafficllm/data_collection/image_library.py)
- [ImageValidator](../src/trafficllm/data_collection/image_validator.py)
- [SAM 3 Integration Guide](SAM3_INTEGRATION_ANALYSIS.md)
- [SAM 3 Quick Start](SAM3_QUICKSTART.md)
