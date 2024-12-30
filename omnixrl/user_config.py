from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import json

@dataclass
class Config:
    """Base configuration class for RL algorithms
    
    Attributes:
        learning_rate (float): Learning rate for optimization
        batch_size (int): Batch size for training
        hidden_sizes (List[int]): Neural network architecture
        max_steps (int): Maximum number of training steps
        device (str): Device to run on ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducibility
    """
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    hidden_sizes: List[int] = (256, 256)
    max_steps: int = 1_000_000
    device: str = "cpu"
    seed: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate configuration values"""
        try:
            assert self.learning_rate > 0, "Learning rate must be positive"
            assert self.batch_size > 0, "Batch size must be positive"
            assert all(h > 0 for h in self.hidden_sizes), "Hidden sizes must be positive"
            assert self.max_steps > 0, "Max steps must be positive"
            assert self.device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def from_dict(cls, config: Dict) -> "Config":
        """Create a Config instance from a dictionary"""
        # Filter only valid keys
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {
            k: v for k, v in config.items() 
            if k in valid_keys
        }
        
        # Create instance
        instance = cls(**filtered_config)
        instance.validate()
        return instance
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from a YAML or JSON file"""
        path = Path(path)
        
        # Load file
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)




# Example usage:
if __name__ == "__main__":
    # From dictionary
    config_dict = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_sizes": [64, 64],
        "device": "cuda"
    }
    config1 = Config.from_dict(config_dict)
    print("Config from dict:", config1)
    
    # Create example YAML file
    yaml_config = """
    learning_rate: 0.0003
    batch_size: 64
    hidden_sizes: [256, 256]
    device: cpu
    seed: 42
    """
    
    with open("example_config.yaml", "w") as f:
        f.write(yaml_config)
    
    # Load from YAML
    config2 = Config.from_file("example_config.yaml")
    print("\nConfig from YAML:", config2)
    
    # Validation example
    invalid_config = {
        "learning_rate": -0.001,  # Invalid negative learning rate
        "batch_size": 64
    }
    print("\nTrying invalid config:")
    config3 = Config.from_dict(invalid_config)  # Will print validation error