from dataclasses import dataclass
from typing import Union, Dict, Any
from pathlib import Path
import yaml
import json

from .configs.base_config import EnvConfig, NetworkConfig
from .configs.algo_configs import PPOConfig, SACConfig

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    max_episode_steps: int = 1000
    action_space_type: str = "discrete"
    state_space_type: str = "continuous"
    reward_scale: float = 1.0

@dataclass
class ModelConfig:
    actor_hidden_sizes: List[int] = (256, 256)
    critic_hidden_sizes: List[int] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = False


@dataclass
class PPOConfig:
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

@dataclass
class SACConfig:
    buffer_size: int = 1_000_000
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
@dataclass
class Config:
    """Main configuration class that combines all configs"""
    env: EnvConfig
    model: ModelConfig
    algorithm: Union[PPOConfig, SACConfig]
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Config":
        # Create component configs
        env_config = EnvConfig(**config.get("env", {}))
        model_config =  ModelConfig(**config.get("network", {}))
        
        # Determine algorithm type and create config
        algo_type = config.get("algorithm_type", "ppo")
        if algo_type == "ppo":
            algo_config = PPOConfig(**config.get("algorithm", {}))
        elif algo_type == "sac":
            algo_config = SACConfig(**config.get("algorithm", {}))
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
        
        return cls(
            env=env_config,
            model=model_config,
            algorithm=algo_config
        )
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        path = Path(path)
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """Validate all configurations"""
        try:
            # Validate environment config
            assert self.env.max_episode_steps > 0
            assert self.env.action_space_type in ["discrete", "continuous"]
            
            # Validate network config
            assert all(h > 0 for h in self.network.actor_hidden_sizes)
            assert self.network.activation in ["relu", "tanh"]
            
            # Validate algorithm config
            if isinstance(self.algorithm, PPOConfig):
                assert 0 < self.algorithm.clip_range < 1
                assert 0 < self.algorithm.gamma <= 1
            elif isinstance(self.algorithm, SACConfig):
                assert self.algorithm.buffer_size > 0
                assert 0 < self.algorithm.tau <= 1
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False