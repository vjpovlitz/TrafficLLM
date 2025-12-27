"""
Unified Traffic Pipeline
Integrates Data Collection, Object Detection (YOLO), and Classification (ResNet).
Removes silos by providing a single entry point for processing traffic data.
"""
import logging
import cv2
import numpy as np
import torch
import supervision as sv
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO

# Import existing components
from trafficllm.data_collection.enhanced_pipeline import EnhancedDataCollectionPipeline
from trafficllm.models.resnet_classifier import EnhancedTrafficNet

class UnifiedTrafficPipeline(EnhancedDataCollectionPipeline):
    """
    Unified pipeline that extends data collection with:
    1. Real-time Object Detection (YOLO)
    2. Traffic Classification (ResNet)
    3. Visualization (Supervision)
    """

    def __init__(self, 
                 config_path: str = "camera_config.json",
                 output_dir: str = "traffic_data",
                 yolo_model_size: str = 's',
                 classifier_path: Optional[str] = None,
                 device: str = None,
                 **kwargs):
        
        # Initialize parent (Data Collection)
        super().__init__(config_path, output_dir, **kwargs)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initializing Unified Pipeline on {self.device}")

        # 1. Initialize YOLO
        self.logger.info(f"Loading YOLOv8{yolo_model_size}...")
        self.yolo_model = YOLO(f'yolov8{yolo_model_size}.pt')

        # 2. Initialize Classifier (if path provided)
        self.classifier = None
        if classifier_path and Path(classifier_path).exists():
            self.logger.info(f"Loading Classifier from {classifier_path}...")
            self.classifier = EnhancedTrafficNet(num_classes=5).to(self.device)
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            self.classifier.eval()
        else:
            self.logger.warning("No classifier checkpoint provided. Classification will be skipped.")

        # 3. Initialize Annotators (for visualization)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Process a single frame through the full pipeline:
        Detection -> Classification -> Visualization
        """
        results = {}

        # --- 1. Object Detection (YOLO) ---
        yolo_results = self.yolo_model(frame, conf=conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        
        # Filter for vehicles
        vehicle_class_ids = [2, 3, 5, 7] # car, motorcycle, bus, truck
        detections = detections[np.isin(detections.class_id, vehicle_class_ids)]
        
        results['detections'] = detections
        results['vehicle_count'] = len(detections)

        # --- 2. Classification (ResNet) ---
        if self.classifier:
            # Preprocess for ResNet (resize, normalize, etc.)
            # Assuming model expects standard ImageNet normalization
            # For now, just resizing to 640x640 or whatever the model expects
            # This is a placeholder for the actual preprocessing logic
            input_tensor = self._preprocess_for_classifier(frame)
            with torch.no_grad():
                cls_outputs = self.classifier(input_tensor)
                # Get traffic level (argmax)
                traffic_level_idx = torch.argmax(cls_outputs['traffic_level'], dim=1).item()
                results['traffic_level'] = traffic_level_idx
                results['density_score'] = cls_outputs['density'].item()

        # --- 3. Visualization ---
        annotated_frame = frame.copy()
        
        labels = [
            f"{self.yolo_model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        results['annotated_frame'] = annotated_frame
        
        return results

    def _preprocess_for_classifier(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for ResNet classifier"""
        # Resize to expected input size (e.g., 224x224 or 640x640 depending on training)
        # Using 640x640 as default for now
        resized = cv2.resize(frame, (640, 640))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and transpose to (C, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device) # Add batch dim

    def run_monitor(self, source: str, conf_threshold: float = 0.25):
        """Run the pipeline in monitor mode (visualize loop)"""
        self.logger.info(f"Starting monitor on source: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error("Could not open video source")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.process_frame(frame, conf_threshold)
                
                # Display
                cv2.imshow("Unified Traffic Monitor", results['annotated_frame'])
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

