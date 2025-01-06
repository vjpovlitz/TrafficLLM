import cv2
import numpy as np
import logging

class TrafficImageAnalyzer:
    def __init__(self):
        self.density_weight = 1.0
        self.congestion_weight = 1.0
        self.edge_threshold = 50
        self.traffic_level_thresholds = [0.1, 0.3, 0.5, 0.7]
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image):
        """Convert image to proper format for analysis"""
        try:
            # Convert list to numpy array if needed
            if isinstance(image, list):
                image = np.array(image)
                
            # Handle tensor format (if from PyTorch)
            if hasattr(image, 'numpy'):
                image = image.numpy()
                
            # Ensure correct shape and format
            if len(image.shape) == 3:
                if image.shape[0] == 3:  # If channels first (C, H, W)
                    image = np.transpose(image, (1, 2, 0))
                    
            # Normalize values to [0, 255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
                
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def analyze_traffic(self, image):
        """Analyze traffic in image"""
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            if processed is None:
                return self._get_default_response("Failed to process image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate density and congestion
            density = min(1.0, edge_density * 2.0)
            std_dev = np.std(gray) / 50.0
            congestion = min(1.0, std_dev)
            
            # Calculate traffic score
            traffic_score = (
                self.density_weight * density +
                self.congestion_weight * congestion
            )
            
            # Determine traffic level
            level = 0
            for i, threshold in enumerate(self.traffic_level_thresholds):
                if traffic_score > threshold:
                    level = i + 1
            
            return {
                'valid': True,
                'density': float(density),
                'congestion': float(congestion),
                'traffic_level': int(level),
                'edge_density': float(edge_density)
            }
            
        except Exception as e:
            self.logger.error(f"Error in traffic analysis: {str(e)}")
            return self._get_default_response(str(e))

    def _get_default_response(self, reason="Unknown error"):
        return {
            'valid': False,
            'reason': reason,
            'density': 0.0,
            'congestion': 0.0,
            'traffic_level': 0,
            'edge_density': 0.0
        }

    def update_parameters(self, density_weight=None, congestion_weight=None, 
                        edge_threshold=None, traffic_level_thresholds=None):
        """Update analyzer parameters based on feedback"""
        if density_weight is not None:
            self.density_weight = density_weight
            self.logger.info(f"Updated density weight to: {density_weight}")
            
        if congestion_weight is not None:
            self.congestion_weight = congestion_weight
            self.logger.info(f"Updated congestion weight to: {congestion_weight}")
            
        if edge_threshold is not None:
            self.edge_threshold = edge_threshold
            self.logger.info(f"Updated edge threshold to: {edge_threshold}")
            
        if traffic_level_thresholds is not None:
            self.traffic_level_thresholds = traffic_level_thresholds
            self.logger.info(f"Updated traffic level thresholds to: {traffic_level_thresholds}")
            
        return {
            'density_weight': self.density_weight,
            'congestion_weight': self.congestion_weight,
            'edge_threshold': self.edge_threshold,
            'traffic_level_thresholds': self.traffic_level_thresholds
        }

    def get_parameters(self):
        """Get current analyzer parameters"""
        return {
            'density_weight': self.density_weight,
            'congestion_weight': self.congestion_weight,
            'edge_threshold': self.edge_threshold,
            'traffic_level_thresholds': self.traffic_level_thresholds
        } 