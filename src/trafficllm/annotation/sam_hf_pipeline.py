"""
SAM 3 Hugging Face Pipeline

Direct integration with Hugging Face Hub for loading and running SAM 3 models.
Provides a cleaner API than downloading weights manually.

Prerequisites:
    pip install transformers huggingface-hub accelerate

    # Authenticate with HuggingFace (for gated models like SAM 3)
    huggingface-cli login

Usage:
    from trafficllm.annotation.sam_hf_pipeline import SAM3HFPipeline

    pipeline = SAM3HFPipeline()
    results = pipeline.segment(image, prompts=["car", "truck"])
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class SAM3Config:
    """Configuration for SAM 3 HuggingFace pipeline"""

    # Model selection
    model_id: str = "facebook/sam2.1-hiera-large"  # Default to SAM 2.1 (widely available)

    # Alternative model options:
    # - "facebook/sam2.1-hiera-large" (SAM 2.1 - best available)
    # - "facebook/sam2.1-hiera-base-plus"
    # - "facebook/sam2.1-hiera-small"
    # - "facebook/sam2.1-hiera-tiny"
    # - "facebook/sam3" (SAM 3 - requires access approval)

    # Device configuration
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # Inference settings
    points_per_side: int = 32  # For automatic mask generation
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95

    # Text prompting (SAM 3 specific)
    use_text_prompts: bool = True

    # Performance
    use_flash_attention: bool = True
    compile_model: bool = False  # torch.compile for faster inference

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)

    def get_device(self) -> str:
        """Auto-detect best available device"""
        if self.device != "auto":
            return self.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


@dataclass
class SegmentationResult:
    """Result from SAM 3 segmentation"""
    masks: np.ndarray  # (N, H, W) binary masks
    boxes: np.ndarray  # (N, 4) bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # (N,) confidence scores
    class_names: List[str] = field(default_factory=list)  # Text labels
    class_ids: np.ndarray = field(default_factory=lambda: np.array([]))


class SAM3HFPipeline:
    """
    SAM 3 Hugging Face Pipeline for traffic segmentation.

    Provides a clean interface for:
    - Loading SAM models from HuggingFace Hub
    - Text-prompted segmentation (SAM 3)
    - Point/box-prompted segmentation (SAM 2/3)
    - Automatic mask generation

    Example:
        >>> pipeline = SAM3HFPipeline()
        >>> result = pipeline.segment("traffic.jpg", prompts=["car", "truck"])
        >>> print(f"Found {len(result.masks)} segments")
    """

    # Vehicle class mapping for traffic analysis
    VEHICLE_CLASSES = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "motorcycle": 3,
        "bicycle": 4,
        "person": 5,
        "traffic light": 6,
        "stop sign": 7,
    }

    def __init__(self, config: Optional[SAM3Config] = None):
        """
        Initialize the SAM 3 HuggingFace pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or SAM3Config()
        self.device = self.config.get_device()
        self.dtype = self.config.get_torch_dtype()

        logger.info(f"Initializing SAM 3 HF Pipeline")
        logger.info(f"  Model: {self.config.model_id}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {self.dtype}")

        # Load model and processor
        self._load_model()

        logger.info("SAM 3 HF Pipeline ready")

    def _load_model(self):
        """Load SAM model from HuggingFace Hub"""
        try:
            # Try transformers SAM implementation first
            self._load_transformers_sam()
        except ImportError:
            logger.warning("transformers SAM not available, trying ultralytics...")
            self._load_ultralytics_sam()

    def _load_transformers_sam(self):
        """Load SAM using HuggingFace transformers"""
        try:
            from transformers import SamModel, SamProcessor
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with:\n"
                "pip install transformers>=4.36.0 accelerate"
            )

        logger.info(f"Loading SAM from HuggingFace: {self.config.model_id}")

        # Check if model requires authentication
        model_id = self.config.model_id

        try:
            # Load processor
            self.processor = SamProcessor.from_pretrained(model_id)

            # Load model with optimizations
            model_kwargs = {
                "torch_dtype": self.dtype,
            }

            # Enable flash attention if available
            if self.config.use_flash_attention and self.device == "cuda":
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                except Exception:
                    logger.info("Flash Attention 2 not available, using default")

            self.model = SamModel.from_pretrained(model_id, **model_kwargs)
            self.model.to(self.device)
            self.model.eval()

            # Optional: Compile model for faster inference
            if self.config.compile_model and hasattr(torch, "compile"):
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")

            self.backend = "transformers"

        except Exception as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                raise PermissionError(
                    f"Model {model_id} requires access approval.\n"
                    f"1. Visit: https://huggingface.co/{model_id}\n"
                    f"2. Request access and wait for approval\n"
                    f"3. Run: huggingface-cli login\n"
                    f"4. Try again"
                )
            raise

    def _load_ultralytics_sam(self):
        """Load SAM using Ultralytics (fallback)"""
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError(
                "Neither transformers nor ultralytics available.\n"
                "Install one of:\n"
                "  pip install transformers>=4.36.0 accelerate\n"
                "  pip install ultralytics>=8.3.0"
            )

        # Map HF model ID to ultralytics model name
        model_map = {
            "facebook/sam2.1-hiera-large": "sam2.1_l.pt",
            "facebook/sam2.1-hiera-base-plus": "sam2.1_b.pt",
            "facebook/sam2.1-hiera-small": "sam2.1_s.pt",
            "facebook/sam2.1-hiera-tiny": "sam2.1_t.pt",
            "facebook/sam3": "sam3.pt",
        }

        model_name = model_map.get(self.config.model_id, "sam2.1_l.pt")

        logger.info(f"Loading SAM via Ultralytics: {model_name}")
        self.model = SAM(model_name)
        self.processor = None
        self.backend = "ultralytics"

    def segment(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        prompts: Optional[List[str]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None,
        conf_threshold: float = 0.5,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        Segment image using SAM.

        Args:
            image: Input image (path, array, or PIL Image)
            prompts: Text prompts for semantic segmentation (SAM 3 only)
            points: Point prompts [(x, y), ...]
            boxes: Box prompts [[x1, y1, x2, y2], ...]
            point_labels: Labels for points (1=foreground, 0=background)
            conf_threshold: Minimum confidence score
            multimask_output: Return multiple masks per prompt

        Returns:
            SegmentationResult with masks, boxes, and scores
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Route to appropriate segmentation method
        if self.backend == "transformers":
            return self._segment_transformers(
                pil_image, prompts, points, boxes, point_labels,
                conf_threshold, multimask_output
            )
        else:
            return self._segment_ultralytics(
                pil_image, prompts, points, boxes,
                conf_threshold
            )

    def _segment_transformers(
        self,
        image: Image.Image,
        prompts: Optional[List[str]],
        points: Optional[List[Tuple[int, int]]],
        boxes: Optional[List[List[int]]],
        point_labels: Optional[List[int]],
        conf_threshold: float,
        multimask_output: bool,
    ) -> SegmentationResult:
        """Segment using transformers SAM"""

        # Prepare inputs based on prompt type
        input_points = None
        input_boxes = None
        input_labels = None

        if points:
            input_points = [[[p[0], p[1]] for p in points]]
            input_labels = [[1] * len(points)] if point_labels is None else [point_labels]

        if boxes:
            input_boxes = [boxes]

        # If no prompts provided, use automatic mask generation
        if not points and not boxes and not prompts:
            return self._auto_segment_transformers(image, conf_threshold)

        # Process inputs
        inputs = self.processor(
            image,
            input_points=input_points,
            input_boxes=input_boxes,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)

        # Post-process
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        scores = outputs.iou_scores.cpu().numpy()[0]

        # Convert masks to numpy
        masks_np = masks.squeeze(0).numpy()

        # Filter by confidence
        if len(masks_np.shape) == 3:
            # Multiple masks per prompt - take best one
            best_idx = scores.argmax(axis=-1)
            masks_np = np.array([masks_np[i, best_idx[i]] for i in range(len(best_idx))])
            scores = np.array([scores[i, best_idx[i]] for i in range(len(best_idx))])

        valid = scores >= conf_threshold
        masks_np = masks_np[valid]
        scores = scores[valid]

        # Extract bounding boxes from masks
        boxes_np = self._masks_to_boxes(masks_np)

        # Assign class names/IDs based on prompts
        class_names = []
        class_ids = []

        if prompts:
            for i in range(len(masks_np)):
                prompt_idx = min(i, len(prompts) - 1)
                class_names.append(prompts[prompt_idx])
                class_ids.append(self.VEHICLE_CLASSES.get(prompts[prompt_idx], 0))

        return SegmentationResult(
            masks=masks_np,
            boxes=boxes_np,
            scores=scores,
            class_names=class_names,
            class_ids=np.array(class_ids) if class_ids else np.array([])
        )

    def _auto_segment_transformers(
        self,
        image: Image.Image,
        conf_threshold: float
    ) -> SegmentationResult:
        """Automatic mask generation using grid points"""
        from transformers import SamAutomaticMaskGenerator

        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
        )

        # Generate masks
        image_np = np.array(image)
        masks_data = mask_generator.generate(image_np)

        # Filter and convert
        masks = []
        boxes = []
        scores = []

        for m in masks_data:
            if m["predicted_iou"] >= conf_threshold:
                masks.append(m["segmentation"])
                boxes.append(m["bbox"])  # [x, y, w, h]
                scores.append(m["predicted_iou"])

        # Convert bbox format [x, y, w, h] to [x1, y1, x2, y2]
        boxes_np = np.array(boxes) if boxes else np.zeros((0, 4))
        if len(boxes_np) > 0:
            boxes_np[:, 2] = boxes_np[:, 0] + boxes_np[:, 2]
            boxes_np[:, 3] = boxes_np[:, 1] + boxes_np[:, 3]

        return SegmentationResult(
            masks=np.array(masks) if masks else np.zeros((0, *image.size[::-1])),
            boxes=boxes_np,
            scores=np.array(scores) if scores else np.array([])
        )

    def _segment_ultralytics(
        self,
        image: Image.Image,
        prompts: Optional[List[str]],
        points: Optional[List[Tuple[int, int]]],
        boxes: Optional[List[List[int]]],
        conf_threshold: float,
    ) -> SegmentationResult:
        """Segment using Ultralytics SAM"""

        results = self.model(
            image,
            points=points,
            bboxes=boxes,
            texts=prompts,  # SAM 3 text prompts
            conf=conf_threshold,
            verbose=False
        )[0]

        # Extract results
        masks_np = np.array([])
        boxes_np = np.array([])
        scores_np = np.array([])
        class_names = []
        class_ids = []

        if results.masks is not None:
            masks_np = results.masks.data.cpu().numpy()

        if results.boxes is not None:
            boxes_np = results.boxes.xyxy.cpu().numpy()
            scores_np = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Map class IDs to names
            if prompts:
                for cls_id in cls_ids:
                    if cls_id < len(prompts):
                        class_names.append(prompts[cls_id])
                        class_ids.append(self.VEHICLE_CLASSES.get(prompts[cls_id], cls_id))
                    else:
                        class_names.append(f"class_{cls_id}")
                        class_ids.append(cls_id)

        return SegmentationResult(
            masks=masks_np,
            boxes=boxes_np,
            scores=scores_np,
            class_names=class_names,
            class_ids=np.array(class_ids) if class_ids else np.array([])
        )

    def _masks_to_boxes(self, masks: np.ndarray) -> np.ndarray:
        """Extract bounding boxes from binary masks"""
        if len(masks) == 0:
            return np.zeros((0, 4))

        boxes = []
        for mask in masks:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any() or not cols.any():
                boxes.append([0, 0, 0, 0])
                continue

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            boxes.append([x1, y1, x2, y2])

        return np.array(boxes)

    def segment_traffic(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        vehicle_types: Optional[List[str]] = None,
        conf_threshold: float = 0.5,
    ) -> SegmentationResult:
        """
        Convenience method for traffic scene segmentation.

        Args:
            image: Input image
            vehicle_types: Vehicle types to detect. Defaults to common vehicles.
            conf_threshold: Minimum confidence

        Returns:
            SegmentationResult with detected vehicles
        """
        if vehicle_types is None:
            vehicle_types = ["car", "truck", "bus", "motorcycle"]

        return self.segment(
            image,
            prompts=vehicle_types,
            conf_threshold=conf_threshold
        )

    def visualize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        result: SegmentationResult,
        alpha: float = 0.5,
        show_boxes: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Visualize segmentation results on image.

        Args:
            image: Original image
            result: Segmentation result to visualize
            alpha: Mask transparency (0-1)
            show_boxes: Draw bounding boxes
            show_labels: Draw class labels

        Returns:
            Annotated image as numpy array
        """
        import cv2

        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        # Generate random colors for each detection
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(result.masks), 3)).tolist()

        # Apply masks
        for i, mask in enumerate(result.masks):
            color = colors[i]

            # Resize mask if needed
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (img.shape[1], img.shape[0])
                ) > 0.5

            # Apply colored mask
            img[mask] = img[mask] * (1 - alpha) + np.array(color) * alpha

        # Draw boxes and labels
        if show_boxes or show_labels:
            for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
                x1, y1, x2, y2 = map(int, box)
                color = colors[i]

                if show_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                if show_labels:
                    label = result.class_names[i] if i < len(result.class_names) else "object"
                    label = f"{label} {score:.2f}"
                    cv2.putText(
                        img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        return img

    def to_annotation_results(self, result: SegmentationResult) -> List:
        """
        Convert SegmentationResult to list of AnnotationResult objects.

        For integration with the existing annotation system.
        """
        from .sam3_annotator import AnnotationResult

        annotations = []
        for i in range(len(result.masks)):
            ann = AnnotationResult(
                class_name=result.class_names[i] if i < len(result.class_names) else "object",
                class_id=int(result.class_ids[i]) if i < len(result.class_ids) else 0,
                mask=result.masks[i],
                box=result.boxes[i],
                score=float(result.scores[i])
            )
            annotations.append(ann)

        return annotations


def create_pipeline(
    model_id: str = "facebook/sam2.1-hiera-large",
    device: str = "auto",
    **kwargs
) -> SAM3HFPipeline:
    """
    Factory function to create SAM 3 HuggingFace pipeline.

    Args:
        model_id: HuggingFace model ID
        device: Device for inference
        **kwargs: Additional config options

    Returns:
        Configured SAM3HFPipeline instance

    Example:
        >>> pipeline = create_pipeline("facebook/sam2.1-hiera-large")
        >>> result = pipeline.segment_traffic("traffic.jpg")
    """
    config = SAM3Config(model_id=model_id, device=device, **kwargs)
    return SAM3HFPipeline(config)
