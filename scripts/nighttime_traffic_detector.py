#!/usr/bin/env python3
"""
Nighttime Traffic Detector

Detects vehicles at night by identifying headlight patterns:
1. Finds bright spots (white/yellow lights) in the image
2. Pairs nearby lights into potential vehicles
3. Filters by expected vehicle headlight spacing
4. Uses SAM 2 to segment illuminated road areas for traffic density

Optimized for Apple Silicon Macs using MPS backend.

Usage:
    python scripts/nighttime_traffic_detector.py --image path/to/night_image.jpg
    python scripts/nighttime_traffic_detector.py --input data/image_library
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


@dataclass
class HeadlightPair:
    """Detected headlight pair representing a vehicle."""
    left_pos: Tuple[int, int]   # (x, y) of left headlight
    right_pos: Tuple[int, int]  # (x, y) of right headlight
    center: Tuple[int, int]     # Center point
    spacing: float              # Distance between headlights
    brightness: float           # Average brightness
    bbox: List[int]             # [x1, y1, x2, y2] estimated vehicle bbox
    confidence: float           # Detection confidence


@dataclass
class TrafficAnalysis:
    """Overall traffic analysis for an image."""
    headlight_pairs: List[HeadlightPair]
    vehicle_count: int
    traffic_density: str        # "none", "light", "moderate", "heavy"
    illuminated_road_area: float  # Percentage of road lit by headlights
    raw_bright_spots: int       # Total bright spots detected
    image_brightness: float     # Overall image brightness (0-255)


class NighttimeTrafficDetector:
    """
    Detect vehicles at night using headlight patterns.

    Strategy:
    1. Convert to grayscale and find bright regions
    2. Use blob detection to find individual lights
    3. Pair lights that are horizontally aligned and properly spaced
    4. Estimate vehicle bounding boxes from headlight pairs
    5. Use SAM 2 (optional) to segment illuminated road for density
    """

    # Headlight detection parameters
    BRIGHTNESS_THRESHOLD = 200      # Min brightness for headlight (0-255)
    MIN_HEADLIGHT_AREA = 10         # Min pixels for a headlight blob
    MAX_HEADLIGHT_AREA = 2000       # Max pixels (filter out large bright areas)

    # Headlight pairing parameters
    MIN_HEADLIGHT_SPACING = 15      # Min pixels between paired headlights
    MAX_HEADLIGHT_SPACING = 150     # Max pixels (depends on distance)
    MAX_VERTICAL_OFFSET = 20        # Max vertical difference for a pair

    # Region of Interest - exclude UI overlays
    ROI_TOP_MARGIN = 0.18           # Exclude top 18% (text overlays like "US 50 EAST")
    ROI_RIGHT_MARGIN = 0.15         # Exclude right 15% (logos)
    ROI_BOTTOM_MARGIN = 0.05        # Exclude bottom 5% (timestamps)

    # Vehicle size estimation (pixels, will vary by distance)
    VEHICLE_WIDTH_MULTIPLIER = 2.5  # bbox width = spacing * multiplier
    VEHICLE_HEIGHT_RATIO = 0.6      # bbox height = width * ratio

    def __init__(self, use_sam: bool = True, device: str = None):
        """
        Initialize detector.

        Args:
            use_sam: Whether to use SAM 2 for road segmentation
            device: Device for SAM ('mps', 'cuda', 'cpu')
        """
        self.use_sam = use_sam
        self.sam_model = None

        if use_sam:
            self._load_sam(device)

    def _load_sam(self, device: str = None):
        """Load SAM 2 for road segmentation."""
        try:
            import torch
            from ultralytics import SAM

            if device is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self.device = device
            self.sam_model = SAM("sam2_t.pt")
            print(f"SAM 2 loaded on {device}")

        except Exception as e:
            print(f"SAM 2 not available: {e}")
            self.use_sam = False

    def detect_bright_spots(self, image: np.ndarray) -> List[dict]:
        """
        Find bright spots (potential headlights) in image.

        Args:
            image: BGR image array

        Returns:
            List of bright spot info dicts
        """
        h, w = image.shape[:2]

        # Calculate ROI boundaries (exclude UI overlays)
        roi_top = int(h * self.ROI_TOP_MARGIN)
        roi_bottom = int(h * (1 - self.ROI_BOTTOM_MARGIN))
        roi_right = int(w * (1 - self.ROI_RIGHT_MARGIN))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to find bright regions
        _, thresh = cv2.threshold(
            blurred,
            self.BRIGHTNESS_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        # Find connected components (blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        bright_spots = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by area
            if self.MIN_HEADLIGHT_AREA <= area <= self.MAX_HEADLIGHT_AREA:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                # IMPORTANT: Skip spots outside the road ROI (UI elements)
                if cy < roi_top or cy > roi_bottom or cx > roi_right:
                    continue

                # Calculate average brightness in this region
                mask = (labels == i)
                avg_brightness = np.mean(gray[mask])

                # Check if it's roughly circular (headlight-like)
                aspect_ratio = bw / bh if bh > 0 else 0
                is_circular = 0.3 < aspect_ratio < 3.0

                if is_circular:
                    bright_spots.append({
                        'center': (int(cx), int(cy)),
                        'bbox': [x, y, x + bw, y + bh],
                        'area': area,
                        'brightness': avg_brightness,
                        'aspect_ratio': aspect_ratio
                    })

        return bright_spots

    def pair_headlights(
        self,
        bright_spots: List[dict],
        image_width: int
    ) -> List[HeadlightPair]:
        """
        Pair bright spots into headlight pairs.

        Looks for horizontally aligned pairs with appropriate spacing.

        Args:
            bright_spots: List of detected bright spots
            image_width: Image width for scaling

        Returns:
            List of HeadlightPair objects
        """
        if len(bright_spots) < 2:
            return []

        # Extract centers
        centers = np.array([s['center'] for s in bright_spots])

        pairs = []
        used = set()

        # Sort by x-coordinate (left to right)
        sorted_indices = np.argsort(centers[:, 0])

        for i in sorted_indices:
            if i in used:
                continue

            spot1 = bright_spots[i]
            x1, y1 = spot1['center']

            # Look for a matching right headlight
            best_match = None
            best_score = float('inf')

            for j in sorted_indices:
                if j <= i or j in used:
                    continue

                spot2 = bright_spots[j]
                x2, y2 = spot2['center']

                # Check horizontal spacing
                h_spacing = x2 - x1
                if not (self.MIN_HEADLIGHT_SPACING <= h_spacing <= self.MAX_HEADLIGHT_SPACING):
                    continue

                # Check vertical alignment
                v_offset = abs(y2 - y1)
                if v_offset > self.MAX_VERTICAL_OFFSET:
                    continue

                # Check brightness similarity
                brightness_diff = abs(spot1['brightness'] - spot2['brightness'])
                if brightness_diff > 50:  # Should be similar brightness
                    continue

                # Score this pairing (lower is better)
                score = v_offset + brightness_diff * 0.1

                if score < best_score:
                    best_score = score
                    best_match = j

            if best_match is not None:
                spot2 = bright_spots[best_match]
                x2, y2 = spot2['center']

                # Calculate pair properties
                spacing = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                avg_brightness = (spot1['brightness'] + spot2['brightness']) / 2

                # Estimate vehicle bounding box
                bbox_width = int(spacing * self.VEHICLE_WIDTH_MULTIPLIER)
                bbox_height = int(bbox_width * self.VEHICLE_HEIGHT_RATIO)

                bbox = [
                    center[0] - bbox_width // 2,
                    center[1] - bbox_height // 2,
                    center[0] + bbox_width // 2,
                    center[1] + bbox_height // 2
                ]

                # Confidence based on how "ideal" the pair is
                confidence = self._calculate_pair_confidence(
                    spacing, y1, y2, avg_brightness, image_width
                )

                pair = HeadlightPair(
                    left_pos=(x1, y1),
                    right_pos=(x2, y2),
                    center=center,
                    spacing=spacing,
                    brightness=avg_brightness,
                    bbox=bbox,
                    confidence=confidence
                )

                pairs.append(pair)
                used.add(i)
                used.add(best_match)

        return pairs

    def _calculate_pair_confidence(
        self,
        spacing: float,
        y1: int,
        y2: int,
        brightness: float,
        image_width: int
    ) -> float:
        """Calculate confidence score for a headlight pair."""
        # Ideal spacing is proportional to image width (perspective)
        ideal_spacing = image_width * 0.03  # ~3% of image width
        spacing_score = 1.0 - min(abs(spacing - ideal_spacing) / ideal_spacing, 1.0)

        # Vertical alignment score
        v_offset = abs(y1 - y2)
        alignment_score = 1.0 - min(v_offset / self.MAX_VERTICAL_OFFSET, 1.0)

        # Brightness score (brighter = more confident)
        brightness_score = min(brightness / 255.0, 1.0)

        # Combined confidence
        confidence = (spacing_score * 0.4 + alignment_score * 0.3 + brightness_score * 0.3)

        return round(confidence, 3)

    def detect_single_headlights(
        self,
        bright_spots: List[dict],
        paired_indices: set,
        min_brightness: float = 220
    ) -> List[dict]:
        """
        Detect bright single lights that might be motorcycles or distant vehicles.

        Args:
            bright_spots: All detected bright spots
            paired_indices: Indices already used in pairs
            min_brightness: Minimum brightness for single lights

        Returns:
            List of single headlight detections
        """
        singles = []

        for i, spot in enumerate(bright_spots):
            if i in paired_indices:
                continue

            if spot['brightness'] >= min_brightness:
                # Estimate bbox for single light (smaller vehicle/motorcycle)
                cx, cy = spot['center']
                size = int(np.sqrt(spot['area']) * 3)

                singles.append({
                    'center': spot['center'],
                    'bbox': [cx - size, cy - size//2, cx + size, cy + size//2],
                    'brightness': spot['brightness'],
                    'type': 'single_light'
                })

        return singles

    def estimate_traffic_density(
        self,
        image: np.ndarray,
        headlight_pairs: List[HeadlightPair]
    ) -> Tuple[str, float]:
        """
        Estimate overall traffic density.

        Args:
            image: BGR image
            headlight_pairs: Detected headlight pairs

        Returns:
            (density_label, illuminated_percentage)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate illuminated road area (bright pixels in lower half of image)
        h, w = gray.shape
        road_region = gray[h//2:, :]  # Lower half (road)

        # Count bright pixels in road region
        bright_mask = road_region > 150
        illuminated_pct = np.sum(bright_mask) / bright_mask.size * 100

        # Determine density based on vehicle count and illumination
        vehicle_count = len(headlight_pairs)

        if vehicle_count == 0 and illuminated_pct < 5:
            density = "none"
        elif vehicle_count <= 2 or illuminated_pct < 10:
            density = "light"
        elif vehicle_count <= 5 or illuminated_pct < 20:
            density = "moderate"
        else:
            density = "heavy"

        return density, illuminated_pct

    def segment_road_with_sam(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Use SAM 2 to segment the road area.

        Args:
            image_path: Path to image

        Returns:
            Binary mask of road area, or None if SAM unavailable
        """
        if not self.use_sam or self.sam_model is None:
            return None

        try:
            # Use bottom-center point prompt (road area)
            img = Image.open(image_path)
            w, h = img.size

            # Point prompts for road (bottom center area)
            road_points = [
                [w * 0.5, h * 0.8],  # Bottom center
                [w * 0.3, h * 0.7],  # Left road
                [w * 0.7, h * 0.7],  # Right road
            ]

            results = self.sam_model(
                str(image_path),
                points=road_points,
                labels=[1, 1, 1],  # All foreground
                device=self.device,
                verbose=False
            )

            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                # Combine all masks
                combined = np.any(masks, axis=0).astype(np.uint8)
                return combined

        except Exception as e:
            print(f"SAM road segmentation failed: {e}")

        return None

    def analyze_image(
        self,
        image_path: Path,
        visualize: bool = True
    ) -> TrafficAnalysis:
        """
        Analyze a nighttime traffic image.

        Args:
            image_path: Path to image file
            visualize: Whether to save visualization

        Returns:
            TrafficAnalysis with all detection results
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect bright spots
        bright_spots = self.detect_bright_spots(image)
        print(f"  Found {len(bright_spots)} bright spots")

        # Step 2: Pair into headlights
        headlight_pairs = self.pair_headlights(bright_spots, w)
        print(f"  Paired into {len(headlight_pairs)} vehicle headlights")

        # Step 3: Estimate traffic density
        density, illuminated_pct = self.estimate_traffic_density(image, headlight_pairs)

        # Step 4: Optional SAM road segmentation
        road_mask = None
        if self.use_sam:
            road_mask = self.segment_road_with_sam(image_path)

        # Create analysis result
        analysis = TrafficAnalysis(
            headlight_pairs=headlight_pairs,
            vehicle_count=len(headlight_pairs),
            traffic_density=density,
            illuminated_road_area=illuminated_pct,
            raw_bright_spots=len(bright_spots),
            image_brightness=np.mean(gray)
        )

        # Visualize if requested
        if visualize:
            self._save_visualization(image_path, image, analysis, road_mask)

        return analysis

    def _save_visualization(
        self,
        image_path: Path,
        image: np.ndarray,
        analysis: TrafficAnalysis,
        road_mask: Optional[np.ndarray] = None
    ):
        """Save annotated visualization."""
        vis = image.copy()
        h, w = vis.shape[:2]

        # Draw ROI boundary (show excluded regions)
        roi_top = int(h * self.ROI_TOP_MARGIN)
        roi_bottom = int(h * (1 - self.ROI_BOTTOM_MARGIN))
        roi_right = int(w * (1 - self.ROI_RIGHT_MARGIN))

        # Dim the excluded regions
        vis[:roi_top, :] = (vis[:roi_top, :] * 0.3).astype(np.uint8)
        vis[:, roi_right:] = (vis[:, roi_right:] * 0.3).astype(np.uint8)
        vis[roi_bottom:, :] = (vis[roi_bottom:, :] * 0.3).astype(np.uint8)

        # Draw ROI boundary lines
        cv2.line(vis, (0, roi_top), (w, roi_top), (100, 100, 100), 1)
        cv2.line(vis, (roi_right, 0), (roi_right, h), (100, 100, 100), 1)

        # Draw road mask if available
        if road_mask is not None:
            # Resize mask to image size
            mask_resized = cv2.resize(
                road_mask.astype(np.float32),
                (vis.shape[1], vis.shape[0])
            ) > 0.5

            # Apply blue tint to road area
            overlay = vis.copy()
            overlay[mask_resized] = overlay[mask_resized] * 0.7 + np.array([255, 100, 0]) * 0.3
            vis = overlay.astype(np.uint8)

        # Draw headlight pairs
        for pair in analysis.headlight_pairs:
            # Draw headlight circles
            cv2.circle(vis, pair.left_pos, 8, (0, 255, 255), 2)  # Yellow
            cv2.circle(vis, pair.right_pos, 8, (0, 255, 255), 2)

            # Draw connection line
            cv2.line(vis, pair.left_pos, pair.right_pos, (0, 255, 255), 1)

            # Draw estimated vehicle bbox
            x1, y1, x2, y2 = [int(v) for v in pair.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence
            label = f"car {pair.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add summary text
        summary = f"Vehicles: {analysis.vehicle_count} | Density: {analysis.traffic_density}"
        cv2.rectangle(vis, (10, 10), (400, 40), (0, 0, 0), -1)
        cv2.putText(vis, summary, (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save
        output_path = image_path.parent / f"{image_path.stem}_nighttime_detected.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles at night using headlight patterns"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Single image to analyze"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/image_library",
        help="Directory of images to analyze"
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Disable SAM 2 road segmentation"
    )
    parser.add_argument(
        "--brightness-threshold",
        type=int,
        default=200,
        help="Minimum brightness for headlight detection (0-255)"
    )
    args = parser.parse_args()

    # Initialize detector
    detector = NighttimeTrafficDetector(use_sam=not args.no_sam)
    detector.BRIGHTNESS_THRESHOLD = args.brightness_threshold

    # Find images to process
    if args.image:
        images = [Path(args.image)]
    else:
        input_dir = Path(args.input)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(input_dir.rglob(ext))
        # Filter out already-processed images
        images = [
            img for img in images
            if '_detected' not in img.stem
            and '_annotated' not in img.stem
            and '_sam2' not in img.stem
        ]

    if not images:
        print("No images found!")
        return

    print(f"Analyzing {len(images)} images for nighttime traffic...\n")

    results = []
    for image_path in images:
        print(f"Processing: {image_path.name}")
        try:
            analysis = detector.analyze_image(image_path)
            results.append({
                'image': image_path.name,
                'vehicles': analysis.vehicle_count,
                'density': analysis.traffic_density,
                'bright_spots': analysis.raw_bright_spots
            })
            print(f"  → {analysis.vehicle_count} vehicles, {analysis.traffic_density} traffic\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_vehicles = sum(r['vehicles'] for r in results)
    print(f"Images processed: {len(results)}")
    print(f"Total vehicles detected: {total_vehicles}")

    density_counts = {}
    for r in results:
        density_counts[r['density']] = density_counts.get(r['density'], 0) + 1
    print(f"Traffic density distribution: {density_counts}")


if __name__ == "__main__":
    main()
