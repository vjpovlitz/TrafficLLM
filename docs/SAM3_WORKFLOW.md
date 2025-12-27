# SAM 3 Integration Workflow

## Overview
This document outlines the workflow for integrating **SAM 3 (Segment Anything Model 3)** into the TrafficCamLLM pipeline. The goal is to leverage SAM 3's promptable concept segmentation to automate the generation of high-quality segmentation masks for traffic analysis.

## Workflow Stages

### 1. Environment Setup
- **Dependencies**: Ensure the environment supports SAM 3 requirements (PyTorch with CUDA support recommended).
- **Model Acquisition**: Download SAM 3 checkpoints to `models/`.

### 2. Data Ingestion & Preprocessing
- **Input**: Raw images from `screenshots/` or video frames.
- **Preprocessing**: Resize or normalize if required by the specific SAM 3 model variant (though SAM handles most resolutions inherently).

### 3. Promptable Concept Segmentation (PCS)
We will use the **Text Prompt** capability of SAM 3 to identify vehicles without manual bounding boxes.
- **Prompts**: `["car", "truck", "bus", "motorcycle", "vehicle"]`
- **Execution**:
    1. Load image.
    2. Pass text prompts to the SAM 3 image encoder.
    3. Model returns segmentation masks for all instances matching the concepts.

### 4. Mask Refinement & Filtering
- **Confidence Filtering**: Discard masks with low predicted IoU scores.
- **NMS (Non-Maximum Suppression)**: If overlapping masks are generated for the same object, apply NMS to select the best mask.
- **Label Assignment**: Map simple text prompts (e.g., "car") to specific class IDs used by the YOLO model.

### 5. Output Generation
- **Training Data**: Save masks in standard formats (e.g., COCO JSON or YOLO polygon format) to `traffic_data/annotations`.
- **Visualization**: Overlay masks on original images for manual verification (saved to `outputs/sam_visualizations`).

## Integration with YOLO
SAM 3 serves as a **"Label Assist"** tool:
1. Run SAM 3 on unlabeled data.
2. Generate initial masks/boxes.
3. Human verifier confirms or corrects annotations.
4. Train YOLOv8 on the high-quality, semi-automatically labeled dataset.

## Future Video Tracking
Future iterations will utilize SAM 3's video memory capabilities to track specific vehicles across frames for flow analysis, handling occlusion and re-appearance.
