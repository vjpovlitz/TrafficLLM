"""
SAM 3 Annotator Module

Provides high-level interface for annotating traffic images using Meta's SAM 3.
Generates segmentation masks and bounding boxes with text prompts.

Supports multiple backends:
    - HuggingFace transformers (recommended)
    - Ultralytics SAM wrapper
    - Native Meta SAM 3

Usage:
    # HuggingFace backend (recommended)
    annotator = SAM3Annotator(backend="huggingface")

    # Ultralytics backend (local weights)
    annotator = SAM3Annotator(model_path="sam3.pt", backend="ultralytics")

    # Run annotation
    results = annotator.annotate_image(
        image_path="traffic.jpg",
        prompts=["car", "truck", "bus"],
        conf_threshold=0.25
    )
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """
    Single annotation result from SAM 3.

    Attributes:
        class_name: Text label (e.g., "car", "truck")
        class_id: Numeric class ID for training
        mask: Binary segmentation mask (H, W) numpy array
        box: Bounding box [x1, y1, x2, y2] in pixel coordinates
        score: Confidence score [0, 1]
        polygon: Optional polygon coordinates for COCO format
    """
    class_name: str
    class_id: int
    mask: np.ndarray
    box: np.ndarray  # [x1, y1, x2, y2]
    score: float
    polygon: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'class_name': self.class_name,
            'class_id': self.class_id,
            'box': self.box.tolist(),
            'score': float(self.score),
            'mask_shape': self.mask.shape,
            'has_polygon': self.polygon is not None
        }

    def get_yolo_format(self, image_width: int, image_height: int) -> str:
        """
        Convert to YOLO format: class_id x_center y_center width height (normalized)

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            YOLO format string: "class_id x_center y_center width height"
        """
        x1, y1, x2, y2 = self.box

        # Convert to YOLO format (normalized center coordinates + size)
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


class SAM3Annotator:
    """
    High-level wrapper for SAM 3 annotation.

    Handles model loading, inference, and result formatting.
    Thread-safe for batch processing.

    Example:
        >>> annotator = SAM3Annotator()
        >>> results = annotator.annotate_image(
        ...     "traffic.jpg",
        ...     prompts=["car", "truck", "bus", "motorcycle"],
        ...     conf_threshold=0.25
        ... )
        >>> print(f"Detected {len(results)} vehicles")
    """

    # Class mapping for common traffic objects
    DEFAULT_CLASS_MAPPING = {
        'car': 0,
        'truck': 1,
        'bus': 2,
        'motorcycle': 3,
        'bicycle': 4,
        'person': 5,
        'traffic light': 6,
        'stop sign': 7
    }

    # Default prompts for traffic scenes
    DEFAULT_VEHICLE_PROMPTS = ["car", "truck", "bus", "motorcycle"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        backend: Literal["huggingface", "ultralytics", "native"] = "huggingface",
        model_id: str = "facebook/sam2.1-hiera-large",
        use_ultralytics: bool = None,  # Deprecated, use backend instead
    ):
        """
        Initialize SAM 3 annotator.

        Args:
            model_path: Path to SAM 3 checkpoint (for ultralytics/native backends).
            device: Device for inference ('cuda', 'mps', 'cpu'). Auto-detects if None.
            class_mapping: Custom class name → ID mapping. Uses DEFAULT_CLASS_MAPPING if None.
            backend: Backend to use:
                - "huggingface": Load from HuggingFace Hub (recommended)
                - "ultralytics": Use Ultralytics SAM wrapper with local weights
                - "native": Use native Meta SAM 3 implementation
            model_id: HuggingFace model ID (for huggingface backend).
                Options: "facebook/sam2.1-hiera-large", "facebook/sam3", etc.
            use_ultralytics: Deprecated. Use backend="ultralytics" instead.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.class_mapping = class_mapping or self.DEFAULT_CLASS_MAPPING
        self.model_id = model_id

        # Handle deprecated use_ultralytics parameter
        if use_ultralytics is not None:
            logger.warning("use_ultralytics is deprecated. Use backend='ultralytics' instead.")
            backend = "ultralytics" if use_ultralytics else "native"

        self.backend = backend

        logger.info(f"Initializing SAM 3 Annotator")
        logger.info(f"  Backend: {self.backend}")
        logger.info(f"  Device: {self.device}")
        if self.backend == "huggingface":
            logger.info(f"  Model: {self.model_id}")

        # Load model based on backend
        self._load_model(model_path)

        logger.info("SAM 3 Annotator ready")

    def _load_model(self, model_path: Optional[str] = None):
        """
        Load SAM 3 model.

        Supports three backends:
        1. HuggingFace (recommended): Clean API, auto-downloads from Hub
        2. Ultralytics: Easier API, integrated with YOLO
        3. Native SAM 3: More control, video tracking support
        """
        try:
            if self.backend == "huggingface":
                self._load_huggingface_model()
            elif self.backend == "ultralytics":
                self._load_ultralytics_model(model_path)
            else:
                self._load_native_sam3_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load SAM 3 model: {e}")
            raise

    def _load_huggingface_model(self):
        """
        Load SAM using HuggingFace transformers.

        Requires:
        - pip install transformers>=4.36.0 huggingface-hub accelerate
        - huggingface-cli login (for gated models like SAM 3)
        """
        from .sam_hf_pipeline import SAM3HFPipeline, SAM3Config

        config = SAM3Config(
            model_id=self.model_id,
            device=self.device,
        )

        self.hf_pipeline = SAM3HFPipeline(config)
        logger.info(f"Loaded HuggingFace SAM: {self.model_id}")

    def _load_ultralytics_model(self, model_path: Optional[str] = None):
        """
        Load SAM 3 via Ultralytics package.

        Requires:
        - pip install -U ultralytics (version 8.3.237+)
        - sam3.pt weights from Hugging Face (requires access request)

        To get SAM 3 weights:
        1. Request access: https://huggingface.co/facebook/sam3
        2. Download: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
        3. Place sam3.pt in working directory or specify path
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            raise ImportError(
                "Ultralytics SAM 3 not found. Install with:\n"
                "pip install -U ultralytics\n"
                "Requires version 8.3.237 or later."
            )

        # SAM 3 weights are NOT auto-downloaded - must get from Hugging Face
        checkpoint = model_path or "sam3.pt"

        # Check if weights exist
        from pathlib import Path
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM 3 weights not found at: {checkpoint}\n\n"
                "SAM 3 requires manual download:\n"
                "1. Request access: https://huggingface.co/facebook/sam3\n"
                "2. Download sam3.pt (3.4GB)\n"
                "3. Place in working directory or specify path\n\n"
                "Alternatively, use SAM 2 (no access required):\n"
                "  annotator = SAM3Annotator(use_sam2=True)"
            )

        self.predictor = SAM3SemanticPredictor(
            overrides=dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=checkpoint,
                half=True,  # FP16 for speed
                device=self.device,
                verbose=False,
            )
        )

        logger.info(f"Loaded Ultralytics SAM 3 from {checkpoint}")

    def _load_native_sam3_model(self, model_path: Optional[str] = None):
        """
        Load SAM 3 using native Meta implementation.

        Requires:
        - pip install git+https://github.com/facebookresearch/sam3.git
        - HuggingFace authentication: huggingface-cli login
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            raise ImportError(
                "SAM 3 native package not found. Install with:\n"
                "pip install git+https://github.com/facebookresearch/sam3.git\n"
                "Then authenticate: huggingface-cli login"
            )

        # Build model (auto-downloads from HuggingFace if authenticated)
        self.model = build_sam3_image_model(checkpoint=model_path)
        self.processor = Sam3Processor(self.model)

        logger.info("Loaded native SAM 3 model")

    def annotate_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5
    ) -> List[AnnotationResult]:
        """
        Annotate a single image with SAM 3.

        Args:
            image: Image path, numpy array (H,W,3), or PIL Image
            prompts: List of text prompts (e.g., ["car", "truck"]). Uses DEFAULT_VEHICLE_PROMPTS if None.
            conf_threshold: Minimum confidence score [0, 1]
            nms_threshold: NMS IoU threshold for removing duplicate detections

        Returns:
            List of AnnotationResult objects

        Example:
            >>> results = annotator.annotate_image(
            ...     "traffic.jpg",
            ...     prompts=["red car", "white truck", "school bus"],
            ...     conf_threshold=0.3
            ... )
        """
        # Use default vehicle prompts if none provided
        prompts = prompts or self.DEFAULT_VEHICLE_PROMPTS

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            image_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            image_array = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            image_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Run inference based on backend
        if self.backend == "huggingface":
            results = self._annotate_huggingface(pil_image, prompts, conf_threshold)
        elif self.backend == "ultralytics":
            results = self._annotate_ultralytics(pil_image, prompts, conf_threshold)
        else:
            results = self._annotate_native(image_array, prompts, conf_threshold)

        # Apply NMS to remove duplicates
        results = self._apply_nms(results, nms_threshold)

        logger.info(f"Annotated image: {len(results)} detections from prompts {prompts}")
        return results

    def _annotate_ultralytics(
        self,
        image: Image.Image,
        prompts: List[str],
        conf_threshold: float
    ) -> List[AnnotationResult]:
        """Annotate using Ultralytics SAM 3"""
        # Set image
        self.predictor.set_image(image)

        # Run prediction with text prompts
        prediction = self.predictor(text=prompts, conf=conf_threshold)

        # Parse results
        results = []
        if prediction and len(prediction) > 0:
            pred = prediction[0]  # First image (batch size 1)

            # Extract detections
            if hasattr(pred, 'boxes') and pred.boxes is not None:
                boxes = pred.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = pred.boxes.conf.cpu().numpy()
                class_ids = pred.boxes.cls.cpu().numpy().astype(int)

                # Extract masks if available
                masks = None
                if hasattr(pred, 'masks') and pred.masks is not None:
                    masks = pred.masks.data.cpu().numpy()  # (N, H, W)

                # Create AnnotationResult for each detection
                for i in range(len(boxes)):
                    # Map class ID to name
                    class_id = int(class_ids[i])
                    if class_id < len(prompts):
                        class_name = prompts[class_id]
                    else:
                        class_name = f"unknown_{class_id}"

                    # Get mapped class ID (for YOLO training)
                    mapped_class_id = self.class_mapping.get(class_name, class_id)

                    # Extract mask
                    mask = masks[i] if masks is not None else None

                    result = AnnotationResult(
                        class_name=class_name,
                        class_id=mapped_class_id,
                        mask=mask,
                        box=boxes[i],
                        score=float(scores[i])
                    )

                    results.append(result)

        return results

    def _annotate_native(
        self,
        image: np.ndarray,
        prompts: List[str],
        conf_threshold: float
    ) -> List[AnnotationResult]:
        """Annotate using native SAM 3"""
        # Set image
        inference_state = self.processor.set_image(image)

        # Run prediction for each prompt
        all_results = []

        for prompt in prompts:
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            masks = output["masks"]  # (N, H, W)
            boxes = output["boxes"]  # (N, 4) [x1, y1, x2, y2]
            scores = output["scores"]  # (N,)

            # Filter by confidence
            valid_indices = scores >= conf_threshold

            # Create results
            for i in np.where(valid_indices)[0]:
                # Get class ID from mapping
                class_id = self.class_mapping.get(prompt, 0)

                result = AnnotationResult(
                    class_name=prompt,
                    class_id=class_id,
                    mask=masks[i],
                    box=boxes[i],
                    score=float(scores[i])
                )

                all_results.append(result)

        return all_results

    def _apply_nms(
        self,
        results: List[AnnotationResult],
        iou_threshold: float = 0.5
    ) -> List[AnnotationResult]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            results: List of annotation results
            iou_threshold: IoU threshold for considering boxes as duplicates

        Returns:
            Filtered list of annotation results
        """
        if len(results) <= 1:
            return results

        # Extract boxes and scores
        boxes = np.array([r.box for r in results])
        scores = np.array([r.score for r in results])

        # Compute areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU below threshold
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return [results[i] for i in keep]

    def annotate_batch(
        self,
        image_paths: List[Union[str, Path]],
        prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        save_dir: Optional[Path] = None
    ) -> Dict[str, List[AnnotationResult]]:
        """
        Annotate multiple images in batch.

        Args:
            image_paths: List of image file paths
            prompts: Text prompts for segmentation
            conf_threshold: Confidence threshold
            save_dir: Optional directory to save annotations (YOLO format)

        Returns:
            Dictionary mapping image path → list of AnnotationResult
        """
        logger.info(f"Batch annotating {len(image_paths)} images...")

        results_dict = {}

        for image_path in image_paths:
            try:
                results = self.annotate_image(
                    image_path,
                    prompts=prompts,
                    conf_threshold=conf_threshold
                )

                results_dict[str(image_path)] = results

                # Optionally save to disk
                if save_dir:
                    self._save_yolo_annotation(
                        image_path,
                        results,
                        save_dir
                    )

            except Exception as e:
                logger.error(f"Failed to annotate {image_path}: {e}")
                results_dict[str(image_path)] = []

        logger.info(f"Batch annotation complete: {len(results_dict)} images processed")
        return results_dict

    def _save_yolo_annotation(
        self,
        image_path: Union[str, Path],
        results: List[AnnotationResult],
        save_dir: Path
    ):
        """
        Save annotations in YOLO format.

        Creates a .txt file with same name as image, containing:
        class_id x_center y_center width height (normalized)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get image dimensions
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Create annotation file
        image_name = Path(image_path).stem
        annotation_path = save_dir / f"{image_name}.txt"

        with open(annotation_path, 'w') as f:
            for result in results:
                yolo_line = result.get_yolo_format(img_width, img_height)
                f.write(yolo_line + '\n')

        logger.debug(f"Saved YOLO annotation: {annotation_path}")


# Singleton pattern for efficient model loading
_global_annotator = None

def get_annotator(**kwargs) -> SAM3Annotator:
    """
    Get or create global SAM3Annotator instance (singleton pattern).

    Useful for avoiding repeated model loading in scripts.

    Args:
        **kwargs: Arguments passed to SAM3Annotator constructor (only on first call)

    Returns:
        Shared SAM3Annotator instance
    """
    global _global_annotator
    if _global_annotator is None:
        _global_annotator = SAM3Annotator(**kwargs)
    return _global_annotator
