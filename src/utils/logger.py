"""
Logger utility for tracking experiments and training progress.
"""

import logging
import os
from datetime import datetime
import json


class Logger:
    """
    Logger for experiment tracking and console output.
    """
    
    def __init__(self, log_dir='logs', experiment_name=None):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory for log files
            experiment_name (str): Name of experiment
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Setup file logger
        self.log_file = os.path.join(log_dir, f"{experiment_name}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
        self.metrics = []
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, step, metrics_dict):
        """
        Log metrics for a step.
        
        Args:
            step (int): Step number
            metrics_dict (dict): Dictionary of metrics
        """
        metrics_entry = {'step': step, **metrics_dict}
        self.metrics.append(metrics_entry)
        
        # Log to console
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics_dict.items()])
        self.info(f"Step {step} - {metrics_str}")
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.info(f"Metrics saved to {metrics_file}")
    
    def log_config(self, config_dict):
        """
        Log experiment configuration.
        
        Args:
            config_dict (dict): Configuration parameters
        """
        config_file = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.info(f"Configuration saved to {config_file}")
