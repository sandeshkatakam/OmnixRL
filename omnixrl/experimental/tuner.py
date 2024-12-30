from typing import Dict, Any, List, Union, Optional, Callable
import optuna
import numpy as np
from pathlib import Path
import json
import wandb
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

from ..env.base import BaseEnvLoader
from ..algorithms.base import BaseAlgorithm
from .config import TuningConfig

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import yaml

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    experiment_name: str
    output_dir: str
    method: str = "optuna"  # "grid", "random", or "optuna"
    n_trials: int = 50
    n_timesteps: int = 1_000_000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    n_jobs: int = -1
    seed: int = 42
    metric: str = "eval/mean_return"
    direction: str = "maximize"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TuningConfig':
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))


class ParameterSpace:
    """Define parameter search space."""
    
    def __init__(self):
        self.params = {}
    
    def add_continuous(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False
    ):
        """Add continuous parameter."""
        self.params[name] = {
            'type': 'continuous',
            'low': low,
            'high': high,
            'log': log
        }
    
    def add_discrete(
        self,
        name: str,
        choices: List[Any]
    ):
        """Add discrete parameter."""
        self.params[name] = {
            'type': 'discrete',
            'choices': choices
        }
    
    def add_categorical(
        self,
        name: str,
        choices: List[Any]
    ):
        """Add categorical parameter."""
        self.params[name] = {
            'type': 'categorical',
            'choices': choices
        }
    
    def sample(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        """Sample parameters."""
        if trial is None:
            # Random sampling
            params = {}
            for name, spec in self.params.items():
                if spec['type'] == 'continuous':
                    if spec['log']:
                        params[name] = np.exp(
                            np.random.uniform(
                                np.log(spec['low']),
                                np.log(spec['high'])
                            )
                        )
                    else:
                        params[name] = np.random.uniform(
                            spec['low'],
                            spec['high']
                        )
                elif spec['type'] in ['discrete', 'categorical']:
                    params[name] = np.random.choice(spec['choices'])
            return params
        
        # Optuna-based sampling
        params = {}
        for name, spec in self.params.items():
            if spec['type'] == 'continuous':
                if spec['log']:
                    params[name] = trial.suggest_float(
                        name,
                        spec['low'],
                        spec['high'],
                        log=True
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        spec['low'],
                        spec['high']
                    )
            elif spec['type'] == 'discrete':
                params[name] = trial.suggest_int(
                    name,
                    min(spec['choices']),
                    max(spec['choices'])
                )
            elif spec['type'] == 'categorical':
                params[name] = trial.suggest_categorical(
                    name,
                    spec['choices']
                )
        return params

class HyperparameterTuner:
    """Hyperparameter tuner for RL algorithms."""
    
    def __init__(
        self,
        config: TuningConfig,
        env_loader: BaseEnvLoader,
        algorithm_class: type,
        parameter_space: ParameterSpace
    ):
        """
        Initialize tuner.
        
        Args:
            config: Tuning configuration
            env_loader: Environment loader instance
            algorithm_class: RL algorithm class
            parameter_space: Parameter search space
        """
        self.config = config
        self.env_loader = env_loader
        self.algorithm_class = algorithm_class
        self.parameter_space = parameter_space
        
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_params = None
        self.best_score = float('-inf') if config.direction == 'maximize' else float('inf')
        self.results = []
        
        # Initialize W&B
        self.wandb_run = wandb.init(
            project=f"{config.experiment_name}_tuning",
            config=vars(config)
        )
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        seed: int
    ) -> float:
        """Evaluate a set of parameters."""
        # Create algorithm instance
        algorithm = self.algorithm_class(
            env_info=self.env_loader.get_env_info(),
            **params,
            seed=seed
        )
        
        # Training loop
        obs = self.env_loader.reset().observation
        returns = []
        
        for timestep in range(self.config.n_timesteps):
            action = algorithm.select_action(obs)
            step_result = self.env_loader.step(action)
            algorithm.update(obs, action, step_result)
            obs = step_result.observation
            
            # Evaluation
            if timestep % self.config.eval_freq == 0:
                eval_returns = []
                for _ in range(self.config.n_eval_episodes):
                    eval_obs = self.env_loader.reset().observation
                    eval_return = 0
                    done = False
                    while not done:
                        eval_action = algorithm.select_action(eval_obs)
                        eval_step = self.env_loader.step(eval_action)
                        eval_return += eval_step.reward
                        eval_obs = eval_step.observation
                        done = eval_step.done
                    eval_returns.append(eval_return)
                
                mean_return = np.mean(eval_returns)
                returns.append(mean_return)
                
                # Log to W&B
                wandb.log({
                    'eval/mean_return': mean_return,
                    'eval/std_return': np.std(eval_returns),
                    'timestep': timestep,
                    **params
                })
        
        return np.mean(returns[-10:])  # Average of last 10 evaluations
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        params = self.parameter_space.sample(trial)
        score = self._evaluate_params(params, self.config.seed)
        
        self.results.append({
            'params': params,
            'score': score
        })
        
        return score
    
    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        if self.config.method == "optuna":
            study = optuna.create_study(
                direction=self.config.direction,
                study_name=self.config.experiment_name
            )
            study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                n_jobs=self.config.n_jobs
            )
            self.best_params = study.best_params
            self.best_score = study.best_value
            
        elif self.config.method == "random":
            for _ in tqdm(range(self.config.n_trials)):
                params = self.parameter_space.sample()
                score = self._evaluate_params(params, self.config.seed)
                
                if (self.config.direction == "maximize" and score > self.best_score) or \
                   (self.config.direction == "minimize" and score < self.best_score):
                    self.best_params = params
                    self.best_score = score
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / "tuning_results.csv")
        
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump({
                'params': self.best_params,
                'score': float(self.best_score)
            }, f)
        
        # Log best results to W&B
        wandb.log({
            'best_score': self.best_score,
            'best_params': self.best_params
        })
        
        return self.best_params
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot tuning results."""
        results_df = pd.DataFrame(self.results)
        
        # Parameter importance plot
        if self.config.method == "optuna":
            optuna.visualization.plot_param_importances(study)
        
        # Learning curves for different parameters
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=results_df,
            x=results_df.index,
            y='score',
            alpha=0.6
        )
        plt.axhline(
            y=self.best_score,
            color='r',
            linestyle='--',
            label='Best score'
        )
        plt.xlabel('Trial')
        plt.ylabel(self.config.metric)
        plt.title('Hyperparameter Tuning Results')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()



####################################### Example Usage #######################################################
from src.tuning.tuner import HyperparameterTuner, ParameterSpace
from src.tuning.config import TuningConfig
from src.algorithms import PPO
from src.env import GymEnvLoader

def tune_ppo():
    # Load config
    config = TuningConfig.from_yaml('configs/tuning_config.yaml')
    
    # Create environment
    env_loader = GymEnvLoader("HalfCheetah-v4")
    
    # Define parameter space
    param_space = ParameterSpace()
    param_space.add_continuous(
        "learning_rate",
        low=1e-5,
        high=1e-3,
        log=True
    )
    param_space.add_discrete(
        "n_steps",
        choices=[128, 256, 512, 1024, 2048]
    )
    param_space.add_continuous(
        "gamma",
        low=0.9,
        high=0.9999,
        log=True
    )
    param_space.add_continuous(
        "clip_range",
        low=0.1,
        high=0.4
    )
    
    # Create tuner
    tuner = HyperparameterTuner(
        config=config,
        env_loader=env_loader,
        algorithm_class=PPO,
        parameter_space=param_space
    )
    
    # Run tuning
    best_params = tuner.tune()
    print(f"Best parameters: {best_params}")
    
    # Plot results
    tuner.plot_results("tuning_results.png")

if __name__ == "__main__":
    tune_ppo()