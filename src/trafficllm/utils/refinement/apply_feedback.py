import sys
import os
from pathlib import Path

# Get project root from current file location
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change relative imports to absolute imports
from refinement.feedback_learner import FeedbackLearner
from data_processing.traffic_analysis import TrafficImageAnalyzer
from refinement.path_config import PathConfig
import logging

def apply_feedback_learning():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    paths = PathConfig()
    analyzer = TrafficImageAnalyzer()
    learner = FeedbackLearner(analyzer)
    
    # Load feedback
    feedback_data = learner.load_feedback()
    
    if not feedback_data:
        logger.warning("No feedback data found!")
        return analyzer
    
    # Analyze patterns and adjust parameters
    error_patterns = learner.analyze_feedback_patterns(feedback_data)
    learner.adjust_parameters(error_patterns)
    
    # Apply learned parameters
    learner.apply_learned_parameters()
    
    return analyzer

if __name__ == "__main__":
    updated_analyzer = apply_feedback_learning()