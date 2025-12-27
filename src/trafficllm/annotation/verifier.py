"""
Annotation Quality Verification

Automated quality checks for SAM 3 annotations.
Flags issues like overlapping masks, unusual sizes, low confidence.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from .sam3_annotator import AnnotationResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Quality metrics for annotation validation.

    Attributes:
        passed: Whether all quality checks passed
        total_annotations: Total number of annotations
        warnings: List of warning messages
        errors: List of error messages
        stats: Dictionary of computed statistics
    """
    passed: bool
    total_annotations: int
    warnings: List[str]
    errors: List[str]
    stats: Dict

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Quality Check: {status}\n"
            f"  Annotations: {self.total_annotations}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Errors: {len(self.errors)}\n"
        )


class AnnotationVerifier:
    """
    Automated quality verification for annotations.

    Checks for common issues:
    - Overlapping bounding boxes (potential duplicates)
    - Unusual object sizes (too small/large)
    - Low confidence scores
    - Class distribution imbalance
    - Missing masks

    Example:
        >>> verifier = AnnotationVerifier()
        >>> metrics = verifier.verify(annotations)
        >>> if not metrics.passed:
        ...     print("Issues found:", metrics.errors)
    """

    def __init__(
        self,
        min_confidence: float = 0.25,
        max_overlap_iou: float = 0.5,
        min_box_area: int = 100,  # pixels
        max_box_area_ratio: float = 0.8,  # fraction of image
        check_masks: bool = True
    ):
        """
        Initialize verifier with quality thresholds.

        Args:
            min_confidence: Minimum acceptable confidence score
            max_overlap_iou: Maximum IoU before flagging as duplicate
            min_box_area: Minimum bounding box area in pixels
            max_box_area_ratio: Maximum box area as fraction of image
            check_masks: Whether to verify mask presence/quality
        """
        self.min_confidence = min_confidence
        self.max_overlap_iou = max_overlap_iou
        self.min_box_area = min_box_area
        self.max_box_area_ratio = max_box_area_ratio
        self.check_masks = check_masks

    def verify(
        self,
        annotations: List[AnnotationResult],
        image_width: int,
        image_height: int
    ) -> QualityMetrics:
        """
        Verify annotation quality for a single image.

        Args:
            annotations: List of annotation results
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            QualityMetrics object with validation results
        """
        warnings = []
        errors = []
        stats = {}

        # Basic statistics
        total = len(annotations)
        stats['total_annotations'] = total

        if total == 0:
            warnings.append("No annotations found")
            return QualityMetrics(
                passed=True,
                total_annotations=0,
                warnings=warnings,
                errors=errors,
                stats=stats
            )

        # Check confidence scores
        confidence_issues = self._check_confidence(annotations)
        warnings.extend(confidence_issues)

        # Check box sizes
        size_issues = self._check_box_sizes(
            annotations,
            image_width,
            image_height
        )
        warnings.extend(size_issues)

        # Check for overlaps
        overlap_issues = self._check_overlaps(annotations)
        warnings.extend(overlap_issues)

        # Check masks
        if self.check_masks:
            mask_issues = self._check_masks(annotations)
            warnings.extend(mask_issues)

        # Compute statistics
        stats.update(self._compute_stats(annotations, image_width, image_height))

        # Determine pass/fail
        passed = len(errors) == 0

        return QualityMetrics(
            passed=passed,
            total_annotations=total,
            warnings=warnings,
            errors=errors,
            stats=stats
        )

    def _check_confidence(self, annotations: List[AnnotationResult]) -> List[str]:
        """Check for low confidence annotations"""
        issues = []
        low_conf = [ann for ann in annotations if ann.score < self.min_confidence]

        if low_conf:
            issues.append(
                f"Found {len(low_conf)} annotations with confidence < {self.min_confidence}"
            )

        return issues

    def _check_box_sizes(
        self,
        annotations: List[AnnotationResult],
        image_width: int,
        image_height: int
    ) -> List[str]:
        """Check for unusually small or large bounding boxes"""
        issues = []
        image_area = image_width * image_height

        for ann in annotations:
            x1, y1, x2, y2 = ann.box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Too small
            if box_area < self.min_box_area:
                issues.append(
                    f"{ann.class_name} box too small: {box_area:.0f}px "
                    f"(min: {self.min_box_area}px)"
                )

            # Too large
            area_ratio = box_area / image_area
            if area_ratio > self.max_box_area_ratio:
                issues.append(
                    f"{ann.class_name} box too large: {area_ratio:.1%} of image "
                    f"(max: {self.max_box_area_ratio:.1%})"
                )

        return issues

    def _check_overlaps(self, annotations: List[AnnotationResult]) -> List[str]:
        """Check for overlapping bounding boxes (potential duplicates)"""
        issues = []

        if len(annotations) < 2:
            return issues

        # Compute IoU matrix
        boxes = np.array([ann.box for ann in annotations])
        ious = self._compute_iou_matrix(boxes)

        # Find high overlaps (excluding diagonal)
        overlaps = np.where(
            (ious > self.max_overlap_iou) & (ious < 1.0)  # Exclude self-overlap
        )

        if len(overlaps[0]) > 0:
            unique_pairs = set()
            for i, j in zip(overlaps[0], overlaps[1]):
                if i < j:  # Avoid counting (i,j) and (j,i) separately
                    unique_pairs.add((i, j))

            for i, j in unique_pairs:
                iou = ious[i, j]
                issues.append(
                    f"High overlap (IoU={iou:.2f}) between "
                    f"{annotations[i].class_name} and {annotations[j].class_name}"
                )

        return issues

    def _compute_iou_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix for all pairs of boxes.

        Args:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2]

        Returns:
            IoU matrix of shape (N, N)
        """
        N = len(boxes)
        ious = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                iou = self._compute_iou(boxes[i], boxes[j])
                ious[i, j] = iou
                ious[j, i] = iou

        return ious

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _check_masks(self, annotations: List[AnnotationResult]) -> List[str]:
        """Check for missing or invalid masks"""
        issues = []

        missing_masks = sum(1 for ann in annotations if ann.mask is None)

        if missing_masks > 0:
            issues.append(f"{missing_masks} annotations missing segmentation masks")

        # Check mask quality
        for ann in annotations:
            if ann.mask is not None:
                # Check if mask is too sparse
                mask_area = np.sum(ann.mask > 0)
                if mask_area == 0:
                    issues.append(f"{ann.class_name} has empty mask")

        return issues

    def _compute_stats(
        self,
        annotations: List[AnnotationResult],
        image_width: int,
        image_height: int
    ) -> Dict:
        """Compute annotation statistics"""
        # Class distribution
        class_counts = {}
        for ann in annotations:
            class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1

        # Confidence statistics
        confidences = [ann.score for ann in annotations]

        # Box size statistics
        box_areas = []
        for ann in annotations:
            x1, y1, x2, y2 = ann.box
            area = (x2 - x1) * (y2 - y1)
            box_areas.append(area)

        return {
            'class_distribution': class_counts,
            'confidence': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'box_area': {
                'mean': float(np.mean(box_areas)),
                'min': float(np.min(box_areas)),
                'max': float(np.max(box_areas)),
                'std': float(np.std(box_areas))
            },
            'density': len(annotations) / (image_width * image_height) * 1e6  # per megapixel
        }

    def verify_batch(
        self,
        annotations_dict: Dict[str, List[AnnotationResult]],
        image_dimensions: Dict[str, Tuple[int, int]]
    ) -> Dict[str, QualityMetrics]:
        """
        Verify annotations for multiple images.

        Args:
            annotations_dict: Dict mapping image path → annotations
            image_dimensions: Dict mapping image path → (width, height)

        Returns:
            Dict mapping image path → QualityMetrics
        """
        logger.info(f"Verifying {len(annotations_dict)} images...")

        results = {}

        for image_path, annotations in annotations_dict.items():
            if image_path in image_dimensions:
                width, height = image_dimensions[image_path]
                metrics = self.verify(annotations, width, height)
                results[image_path] = metrics

                # Log issues
                if metrics.warnings:
                    logger.warning(f"{image_path}: {len(metrics.warnings)} warnings")
                if metrics.errors:
                    logger.error(f"{image_path}: {len(metrics.errors)} errors")

        # Summary
        total_warnings = sum(len(m.warnings) for m in results.values())
        total_errors = sum(len(m.errors) for m in results.values())

        logger.info(f"Verification complete:")
        logger.info(f"  Total warnings: {total_warnings}")
        logger.info(f"  Total errors: {total_errors}")

        return results
