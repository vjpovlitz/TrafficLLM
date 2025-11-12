import os

class PathConfig:
    def __init__(self):
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define all relevant paths
        self.paths = {
            'data_processing': os.path.join(self.project_root, 'trafficllm', 'data_processing'),
            'data_visualization': os.path.join(self.project_root, 'trafficllm', 'data_visualization'),
            'refinement': os.path.join(self.project_root, 'trafficllm', 'refinement'),
            'outputs': os.path.join(self.project_root, 'trafficllm', 'data_visualization', 'outputs'),
            'model_preparation': os.path.join(self.project_root, 'trafficllm', 'model_preparation'),
            'training': os.path.join(self.project_root, 'trafficllm', 'training'),
            'data': os.path.join(self.project_root, 'data'),
            'processed_data': os.path.join(self.project_root, 'data', 'processed'),
            'raw_data': os.path.join(self.project_root, 'data', 'raw')
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def get_path(self, key):
        """Get path by key"""
        return self.paths.get(key, '')
    
    def get_feedback_files(self):
        """Get all feedback files from outputs directory"""
        feedback_files = []
        outputs_dir = self.paths['outputs']
        for folder in os.listdir(outputs_dir):
            folder_path = os.path.join(outputs_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.startswith('feedback_') and file.endswith('.json'):
                        feedback_files.append(os.path.join(folder_path, file))
        return feedback_files 