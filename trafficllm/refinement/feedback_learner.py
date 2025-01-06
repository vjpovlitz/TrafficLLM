from .path_config import PathConfig
import json
import os
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict

class FeedbackLearner:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        self.paths = PathConfig()
        self.parameter_adjustments = {
            'density_weight': 1.0,
            'congestion_weight': 1.0,
            'edge_threshold': 50,
            'traffic_level_thresholds': [0.2, 0.4, 0.6, 0.8]
        }
        self.learning_rate = 0.1
    
    def load_feedback(self):
        """Load all feedback files using path config"""
        all_feedback = []
        feedback_files = self.paths.get_feedback_files()
        
        for file_path in feedback_files:
            try:
                with open(file_path, 'r') as f:
                    feedback_data = json.load(f)
                    all_feedback.extend(feedback_data['feedback'])
                    self.logger.info(f"Loaded feedback from {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading feedback file {file_path}: {str(e)}")
        
        return all_feedback
    
    def analyze_feedback_patterns(self, feedback):
        """Analyze patterns in model errors"""
        error_patterns = defaultdict(list)
        
        for entry in feedback:
            if not entry['was_correct']:
                error_patterns['density_vs_level'].append({
                    'density': entry['density'],
                    'predicted': entry['model_prediction'],
                    'actual': entry['user_correction']
                })
                error_patterns['congestion_vs_level'].append({
                    'congestion': entry['congestion'],
                    'predicted': entry['model_prediction'],
                    'actual': entry['user_correction']
                })
        
        return error_patterns
    
    def adjust_parameters(self, error_patterns):
        """Adjust model parameters based on feedback patterns"""
        if not error_patterns['density_vs_level']:
            return
        
        # Calculate average error direction for density
        density_errors = [(p['actual'] - p['predicted']) * p['density'] 
                         for p in error_patterns['density_vs_level']]
        avg_density_error = np.mean(density_errors)
        
        # Calculate average error direction for congestion
        congestion_errors = [(p['actual'] - p['predicted']) * p['congestion'] 
                           for p in error_patterns['congestion_vs_level']]
        avg_congestion_error = np.mean(congestion_errors)
        
        # Adjust weights
        self.parameter_adjustments['density_weight'] += self.learning_rate * avg_density_error
        self.parameter_adjustments['congestion_weight'] += self.learning_rate * avg_congestion_error
        
        # Ensure weights stay positive
        self.parameter_adjustments['density_weight'] = max(0.1, self.parameter_adjustments['density_weight'])
        self.parameter_adjustments['congestion_weight'] = max(0.1, self.parameter_adjustments['congestion_weight'])
        
        # Adjust traffic level thresholds
        level_errors = defaultdict(list)
        for entry in error_patterns['density_vs_level']:
            level_errors[int(entry['actual'])].append(entry['density'])
        
        # Update thresholds based on average density per level
        new_thresholds = []
        for i in range(1, 4):  # For levels 1-3
            if i in level_errors and level_errors[i]:
                threshold = np.mean(level_errors[i])
                new_thresholds.append(threshold)
            else:
                new_thresholds.append(self.parameter_adjustments['traffic_level_thresholds'][i-1])
        
        self.parameter_adjustments['traffic_level_thresholds'] = new_thresholds
        
        self.logger.info("Updated parameters:")
        self.logger.info(f"Density weight: {self.parameter_adjustments['density_weight']:.3f}")
        self.logger.info(f"Congestion weight: {self.parameter_adjustments['congestion_weight']:.3f}")
        self.logger.info(f"Traffic level thresholds: {self.parameter_adjustments['traffic_level_thresholds']}")
    
    def apply_learned_parameters(self):
        """Apply learned parameters to the analyzer"""
        self.analyzer.update_parameters(
            density_weight=self.parameter_adjustments['density_weight'],
            congestion_weight=self.parameter_adjustments['congestion_weight'],
            traffic_level_thresholds=self.parameter_adjustments['traffic_level_thresholds']
        ) 