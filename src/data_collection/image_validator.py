"""
Image Validation and Quality Assessment System
"""
import cv2
import numpy as np
from PIL import Image, ImageStat
import os
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import json

class ImageValidator:
    """Comprehensive image validation and quality assessment"""
    
    def __init__(self, min_resolution: Tuple[int, int] = (640, 480), 
                 min_file_size: int = 10000):  # 10KB minimum
        self.min_resolution = min_resolution
        self.min_file_size = min_file_size
        self.logger = logging.getLogger(__name__)
        
    def validate_image(self, image_path: str) -> Dict:
        """
        Comprehensive image validation
        Returns validation results with quality metrics
        """
        validation_result = {
            "file_path": image_path,
            "is_valid": False,
            "file_exists": False,
            "file_size": 0,
            "resolution": None,
            "format": None,
            "is_corrupted": False,
            "quality_score": 0.0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "errors": []
        }
        
        try:
            # Check file existence
            if not os.path.exists(image_path):
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            validation_result["file_exists"] = True
            validation_result["file_size"] = os.path.getsize(image_path)
            
            # Check minimum file size
            if validation_result["file_size"] < self.min_file_size:
                validation_result["errors"].append(f"File size {validation_result['file_size']} below minimum {self.min_file_size}")
            
            # Try to open and validate image
            try:
                with Image.open(image_path) as img:
                    validation_result["resolution"] = img.size
                    validation_result["format"] = img.format
                    
                    # Check minimum resolution
                    if (img.size[0] < self.min_resolution[0] or 
                        img.size[1] < self.min_resolution[1]):
                        validation_result["errors"].append(
                            f"Resolution {img.size} below minimum {self.min_resolution}"
                        )
                    
                    # Convert to RGB for analysis
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculate quality metrics
                    validation_result.update(self._calculate_quality_metrics(img))
                    
            except Exception as e:
                validation_result["is_corrupted"] = True
                validation_result["errors"].append(f"Image corruption: {str(e)}")
                return validation_result
            
            # Additional OpenCV-based validation
            try:
                cv_metrics = self._opencv_quality_analysis(image_path)
                validation_result.update(cv_metrics)
            except Exception as e:
                validation_result["errors"].append(f"OpenCV analysis failed: {str(e)}")
            
            # Calculate overall quality score
            validation_result["quality_score"] = self._calculate_overall_quality(validation_result)
            
            # Determine if image is valid
            validation_result["is_valid"] = (
                validation_result["file_exists"] and
                not validation_result["is_corrupted"] and
                len(validation_result["errors"]) == 0 and
                validation_result["quality_score"] >= 0.4  # Minimum quality threshold
            )
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Error validating {image_path}: {e}")
        
        return validation_result
    
    def _calculate_quality_metrics(self, img: Image.Image) -> Dict:
        """Calculate image quality metrics using PIL"""
        metrics = {}
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Brightness analysis
            brightness = np.mean(img_array)
            metrics["brightness_score"] = self._normalize_brightness(brightness)
            
            # Contrast analysis using standard deviation
            contrast = np.std(img_array)
            metrics["contrast_score"] = min(contrast / 64.0, 1.0)  # Normalize to 0-1
            
            # Color distribution analysis
            stat = ImageStat.Stat(img)
            metrics["color_variance"] = np.var(stat.mean)
            
        except Exception as e:
            self.logger.warning(f"PIL quality analysis failed: {e}")
            metrics.update({
                "brightness_score": 0.0,
                "contrast_score": 0.0,
                "color_variance": 0.0
            })
        
        return metrics
    
    def _opencv_quality_analysis(self, image_path: str) -> Dict:
        """Advanced quality analysis using OpenCV"""
        metrics = {}
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"blur_score": 0.0}
            
            # Convert to grayscale for blur detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics["blur_score"] = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Edge density (good for detecting meaningful content)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            metrics["edge_density"] = edge_density
            
            # Noise analysis using standard deviation in small patches
            h, w = gray.shape
            patches = []
            patch_size = 20
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = gray[i:i+patch_size, j:j+patch_size]
                    patches.append(np.std(patch))
            
            if patches:
                noise_score = 1.0 - min(np.var(patches) / 100.0, 1.0)
                metrics["noise_score"] = max(noise_score, 0.0)
            else:
                metrics["noise_score"] = 0.5
                
        except Exception as e:
            self.logger.warning(f"OpenCV analysis failed: {e}")
            metrics.update({
                "blur_score": 0.0,
                "edge_density": 0.0,
                "noise_score": 0.0
            })
        
        return metrics
    
    def _normalize_brightness(self, brightness: float) -> float:
        """Normalize brightness to a quality score (0-1)"""
        # Optimal brightness is around 128 for 8-bit images
        # Score is highest at optimal brightness, decreases as it gets too dark or bright
        optimal = 128.0
        max_deviation = 128.0
        
        deviation = abs(brightness - optimal)
        score = max(0.0, 1.0 - (deviation / max_deviation))
        return score
    
    def _calculate_overall_quality(self, metrics: Dict) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "brightness_score": 0.2,
            "contrast_score": 0.3,
            "blur_score": 0.3,
            "edge_density": 0.1,
            "noise_score": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def batch_validate(self, image_directory: str, output_report: str = None) -> Dict:
        """Validate all images in a directory"""
        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory {image_directory} does not exist")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        self.logger.info(f"Validating {len(image_files)} images in {image_directory}")
        
        results = {
            "total_images": len(image_files),
            "valid_images": 0,
            "invalid_images": 0,
            "average_quality": 0.0,
            "validation_results": []
        }
        
        quality_scores = []
        
        for image_file in image_files:
            validation_result = self.validate_image(str(image_file))
            results["validation_results"].append(validation_result)
            
            if validation_result["is_valid"]:
                results["valid_images"] += 1
                quality_scores.append(validation_result["quality_score"])
            else:
                results["invalid_images"] += 1
        
        if quality_scores:
            results["average_quality"] = np.mean(quality_scores)
        
        # Save report if requested
        if output_report:
            with open(output_report, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Validation report saved to {output_report}")
        
        return results
    
    def filter_valid_images(self, image_directory: str, output_directory: str = None, 
                          min_quality: float = 0.4) -> List[str]:
        """Filter and optionally copy valid images to output directory"""
        validation_results = self.batch_validate(image_directory)
        
        valid_images = [
            result["file_path"] for result in validation_results["validation_results"]
            if result["is_valid"] and result["quality_score"] >= min_quality
        ]
        
        if output_directory:
            output_dir = Path(output_directory)
            output_dir.mkdir(exist_ok=True)
            
            for image_path in valid_images:
                src = Path(image_path)
                dst = output_dir / src.name
                
                try:
                    # Copy file
                    import shutil
                    shutil.copy2(src, dst)
                except Exception as e:
                    self.logger.warning(f"Failed to copy {src} to {dst}: {e}")
        
        self.logger.info(f"Found {len(valid_images)} valid images out of {validation_results['total_images']}")
        return valid_images

class ROIDetector:
    """Dynamic Region of Interest detection for traffic cameras"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_traffic_regions(self, image_path: str) -> List[Dict]:
        """
        Detect potential traffic regions in the image
        Returns list of ROI coordinates with confidence scores
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Use edge detection to find structured regions
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            roi_candidates = []
            
            for contour in contours:
                # Calculate bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Filter by size (should be reasonable portion of image)
                min_area = (w * h) * 0.05  # At least 5% of image
                max_area = (w * h) * 0.8   # At most 80% of image
                area = w_box * h_box
                
                if min_area <= area <= max_area:
                    # Calculate confidence based on edge density in region
                    roi_edges = edges[y:y+h_box, x:x+w_box]
                    edge_density = np.sum(roi_edges > 0) / roi_edges.size
                    
                    roi_candidates.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w_box),
                        "height": int(h_box),
                        "confidence": float(edge_density),
                        "area": int(area)
                    })
            
            # Sort by confidence and return top candidates
            roi_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            return roi_candidates[:3]  # Return top 3 candidates
            
        except Exception as e:
            self.logger.error(f"ROI detection failed for {image_path}: {e}")
            return []
    
    def suggest_optimal_roi(self, image_path: str) -> Optional[Dict]:
        """Suggest the most optimal ROI for traffic analysis"""
        roi_candidates = self.detect_traffic_regions(image_path)
        
        if not roi_candidates:
            return None
        
        # Return the highest confidence ROI
        return roi_candidates[0]

def main():
    """Test the image validation system"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = ImageValidator()
    
    # Test with screenshots directory if it exists
    screenshots_dir = "screenshots"
    if os.path.exists(screenshots_dir):
        print(f"Validating images in {screenshots_dir}...")
        results = validator.batch_validate(screenshots_dir, "validation_report.json")
        
        print(f"Validation Results:")
        print(f"  Total images: {results['total_images']}")
        print(f"  Valid images: {results['valid_images']}")
        print(f"  Invalid images: {results['invalid_images']}")
        print(f"  Average quality: {results['average_quality']:.2f}")
        
        # Test ROI detection
        roi_detector = ROIDetector()
        if results["validation_results"]:
            test_image = results["validation_results"][0]["file_path"]
            if os.path.exists(test_image):
                optimal_roi = roi_detector.suggest_optimal_roi(test_image)
                if optimal_roi:
                    print(f"  Suggested ROI: {optimal_roi}")
    else:
        print("No screenshots directory found. Run camera capture first.")

if __name__ == "__main__":
    main()