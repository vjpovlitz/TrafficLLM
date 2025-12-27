import torch
from ultralytics import YOLO
from pathlib import Path
import logging

class TrafficYOLODetector:
    def __init__(self, model_size='s', model_path=None):
        """
        Initialize TrafficYOLODetector.
        
        Args:
            model_size (str): Size of YOLO model ('n', 's', 'm', 'l', 'x')
            model_path (str, optional): Path to custom model weights.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(f'yolov8{model_size}.pt')
            
        print(f"YOLO Detector initialized on {self.device}")

    def detect_vehicles(self, image_path, conf_threshold=0.25):
        """
        Detect vehicles in an image.
        
        Args:
            image_path (str): Path to image.
            conf_threshold (float): Confidence threshold.
            
        Returns:
            list: List of detections (class, confidence, bbox)
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Filter for vehicle classes (COCO indices)
                # 2: car, 3: motorcycle, 5: bus, 7: truck
                if cls_id in [2, 3, 5, 7]:
                    detections.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': box.xyxy[0].tolist()
                    })
                    
        return detections

    def count_vehicles(self, image_path, conf_threshold=0.25):
        """
        Count vehicles by type.
        
        Args:
            image_path (str): Path to image.
            
        Returns:
            dict: Counts by class.
        """
        detections = self.detect_vehicles(image_path, conf_threshold)
        counts = {'total': len(detections), 'by_type': {}}
        
        for det in detections:
            cls = det['class']
            counts['by_type'][cls] = counts['by_type'].get(cls, 0) + 1
            
        return counts
