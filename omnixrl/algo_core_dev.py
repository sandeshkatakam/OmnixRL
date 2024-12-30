from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp

class AlgorithmConfig(NamedTuple):
    """Base configuration for all RL algorithms"""
    learning_rate: float
    gamma: float
    batch_size: int
    max_steps: int
    seed: int
    # ... other common parameters

class Algorithm(ABC):
    """Base class for all RL algorithms"""
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
    
    @abstractmethod
    def init_params(self):
        """Initialize algorithm parameters"""
        pass
    
    @abstractmethod
    def update(self, batch):
        """Update algorithm parameters"""
        pass
    
    @abstractmethod
    def get_action(self, state):
        """Get action from policy"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save algorithm state"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load algorithm state"""
        pass

class OnPolicyAlgorithm(Algorithm):
    """Base class for on-policy algorithms"""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.trajectory_buffer = None
    
    @abstractmethod
    def collect_trajectories(self, env):
        """Collect trajectories using current policy"""
        pass
    
    @abstractmethod
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages for policy update"""
        pass
    
    def train_epoch(self, env):
        """Single training epoch for on-policy algorithms"""
        # 1. Collect fresh trajectories
        trajectories = self.collect_trajectories(env)
        
        # 2. Compute advantages
        advantages = self.compute_advantages(
            trajectories.rewards,
            trajectories.values,
            trajectories.dones
        )
        
        # 3. Update policy (limited number of times)
        metrics = []
        for _ in range(self.config.n_epochs):
            batch = self.get_minibatch(trajectories, advantages)
            update_info = self.update(batch)
            metrics.append(update_info)
            
            # Check for early stopping (e.g., KL divergence)
            if self.should_early_stop(update_info):
                break
        
        return metrics

class OffPolicyAlgorithm(Algorithm):
    """Base class for off-policy algorithms"""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.replay_buffer = self.init_replay_buffer()
    
    @abstractmethod
    def init_replay_buffer(self):
        """Initialize replay buffer"""
        pass
    
    @abstractmethod
    def sample_batch(self):
        """Sample batch from replay buffer"""
        pass
    
    def add_experience(self, experience):
        """Add experience to replay buffer"""
        self.replay_buffer.add(experience)
    
    def train_step(self):
        """Single training step for off-policy algorithms"""
        # 1. Sample batch from replay buffer
        batch = self.sample_batch()
        
        # 2. Update policy (can do multiple updates)
        metrics = []
        for _ in range(self.config.n_updates):
            update_info = self.update(batch)
            metrics.append(update_info)
        
        return metrics
    

class ReplayBuffer:
    def __init__(self, size, observation_shape, action_shape):
        self.size = size
        self.current_size = 0
        self.pointer = 0
        
        # Initialize buffers
        self.observations = jnp.zeros((size,) + observation_shape)
        self.actions = jnp.zeros((size,) + action_shape)
        self.rewards = jnp.zeros(size)
        self.next_observations = jnp.zeros((size,) + observation_shape)
        self.dones = jnp.zeros(size)
    
    def add(self, experience):
        idx = self.pointer
        
        self.observations = self.observations.at[idx].set(experience.observation)
        self.actions = self.actions.at[idx].set(experience.action)
        self.rewards = self.rewards.at[idx].set(experience.reward)
        self.next_observations = self.next_observations.at[idx].set(
            experience.next_observation)
        self.dones = self.dones.at[idx].set(experience.done)
        
        self.pointer = (self.pointer + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)
    
    def sample(self, batch_size):
        indices = jax.random.randint(
            key=jax.random.PRNGKey(0),
            shape=(batch_size,),
            minval=0,
            maxval=self.current_size
        )
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }