<div align="center">

# OmnixRL
## **NOTE: [Development in Progress]**  
Inspired by OpenAI Spinning Up RL Algorithms Educational Resource implemented in JAX

[![Tests](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/tests.yml/badge.svg)](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/tests.yml)
[![Release](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/release.yml/badge.svg)](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/release.yml)
[![Docs](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/docs.yml/badge.svg)](https://github.com/sandeshkatakam/SpinningUp-RL-JAX/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/sandeshkatakam/SpinningUp-RL-JAX/branch/main/graph/badge.svg)](https://codecov.io/gh/sandeshkatakam/SpinningUp-RL-JAX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center" style="text-align: center; width: 100%;">
  <img src="./assets/imgs/OmnixRL_logo.png" alt="OmnixRL Logo" style="max-width: 600px; width: 100%; height: auto; display: block; margin: 0 auto;">
</p>

</div>
A comprehensive reinforcement learning library implemented in JAX, inspired by OpenAI's Spinning Up. This library provides a clean, modular implementation of popular RL algorithms with a focus on research experimentation and serves as a research framework for developing novel RL algorithms.

## Core Features

- 🚀 **High Performance**: Implemented in JAX for efficient training on both CPU and GPU
- 📊 **Comprehensive Logging**: Built-in support for Weights & Biases and CSV logging
- 🔧 **Modular Design**: Easy to extend and modify for research purposes
- 🎯 **Hyperparameter Tuning**: Integrated Optuna-based tuning with parallel execution
- 📈 **Experiment Analysis**: Tools for ablation studies and result visualization
- 🧪 **Benchmarking**: Automated benchmark suite with baseline comparisons
- 📝 **Documentation**: Detailed API documentation and educational tutorials


## Implemented Algorithms

| Algorithm | Paper | Description | Key Features | Status |
|-----------|-------|-------------|--------------|--------|
| VPG | [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | Basic policy gradient algorithm with value function baseline | - Simple implementation<br>- Value function baseline<br>- GAE support<br>- Continuous/Discrete actions | 🚧 |
| PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | On-policy algorithm with clipped objective | - Clipped surrogate objective<br>- Adaptive KL penalty<br>- Value function clipping<br>- Mini-batch updates | 🚧 |
| SAC | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor](https://arxiv.org/abs/1801.01290) | Off-policy maximum entropy algorithm | - Automatic entropy tuning<br>- Twin Q-functions<br>- Reparameterization trick<br>- Experience replay | 🚧 |
| DQN | [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) | Value-based algorithm with experience replay | - Double Q-learning<br>- Priority replay<br>- Dueling networks<br>- N-step returns | 🚧 |
| DDPG | [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) | Off-policy algorithm for continuous control | - Deterministic policy<br>- Target networks<br>- Action noise<br>- Batch normalization | 🚧 |
| TD3 | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) | Enhanced version of DDPG | - Twin Q-functions<br>- Delayed policy updates<br>- Target policy smoothing<br>- Clipped double Q-learning | 🚧 |
| TRPO | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | On-policy algorithm with trust region constraint | - KL constraint<br>- Conjugate gradient<br>- Line search<br>- Natural policy gradient | 🚧 |

**Legend:**

- ✅ Fully Supported: Thoroughly tested and documented
- 🚧 In Development: Basic implementation available, under testing
- ⭕ Planned: On the roadmap
- ❌ Not Supported: No current plans for implementation

**Implementation Details:**

- All algorithms support both continuous and discrete action spaces (except DQN: discrete only)
- JAX-based implementations with automatic differentiation
- Configurable network architectures
- Comprehensive logging and visualization
- Built-in hyperparameter tuning

## Performance Benchmarks
<!-- 
### Training Speed (Steps/Second)

| Algorithm | CPU (AMD 5900X) | GPU (RTX 3080) | TPU v3-8 | Notes |
|-----------|----------------|----------------|-----------|--------| -->
<!-- | VPG | 12,450 ± 320 | 45,800 ± 520 | 124,500 ± 1,200 | Single environment |
| PPO | 8,900 ± 250 | 38,600 ± 480 | 98,400 ± 950 | 8 parallel environments |
| SAC | 6,800 ± 180 | 32,400 ± 420 | 84,600 ± 880 | With replay buffer |
| DQN | 9,200 ± 220 | 41,200 ± 460 | 102,800 ± 1,100 | Priority replay enabled |
| DDPG | 7,400 ± 200 | 35,600 ± 440 | 89,200 ± 920 | With target networks |
| TD3 | 7,100 ± 190 | 34,200 ± 430 | 86,400 ± 900 | Twin Q-networks |

### Final Performance (Average Returns)

#### Continuous Control (1M steps)

| Environment | VPG | PPO | SAC | DDPG | TD3 | Published Baseline¹ |
|-------------|-----|-----|-----|------|-----|-------------------|
| HalfCheetah-v4 | 4,142 ± 512 | 5,684 ± 425 | 9,150 ± 392 | 6,243 ± 448 | 9,543 ± 376 | 9,636 ± 412 |
| Hopper-v4 | 2,345 ± 321 | 2,965 ± 284 | 3,254 ± 245 | 2,876 ± 312 | 3,412 ± 268 | 3,528 ± 285 |
| Walker2d-v4 | 3,156 ± 428 | 4,235 ± 386 | 4,892 ± 342 | 3,945 ± 398 | 4,978 ± 356 | 5,012 ± 384 |
| Ant-v4 | 3,845 ± 486 | 4,892 ± 442 | 5,648 ± 412 | 4,234 ± 468 | 5,786 ± 428 | 5,864 ± 446 |
| Humanoid-v4 | 4,234 ± 645 | 5,234 ± 586 | 6,124 ± 524 | 4,856 ± 612 | 6,234 ± 542 | 6,456 ± 568 |

#### Discrete Control (10M steps)

| Environment | VPG | PPO | DQN | Published Baseline² |
|-------------|-----|-----|-----|-------------------|
| Pong | 19.2 ± 1.2 | 20.4 ± 0.8 | 20.8 ± 0.6 | 20.9 ± 0.7 |
| Breakout | 354 ± 42 | 425 ± 38 | 442 ± 35 | 448 ± 40 |
| Qbert | 14,235 ± 1,245 | 16,485 ± 1,124 | 17,256 ± 1,084 | 17,452 ± 1,186 |
| Seaquest | 1,824 ± 284 | 2,245 ± 246 | 2,456 ± 228 | 2,512 ± 242 |

### Memory Usage (Peak MB)

| Algorithm | CPU Mode | GPU Mode | TPU Mode |
|-----------|----------|-----------|----------|
| VPG | 245 | 486 | 524 |
| PPO | 312 | 645 | 686 |
| SAC | 486 | 824 | 886 |
| DQN | 524 | 886 | 945 |
| DDPG | 386 | 724 | 768 |
| TD3 | 412 | 768 | 812 | -->

<!-- ### Training Time to Performance Threshold³

| Environment | Algorithm | Steps to Threshold | Wall Time (GPU) | Wall Time (TPU) |
|-------------|-----------|-------------------|-----------------|-----------------|
| HalfCheetah-v4 | PPO | 425K ± 45K | 12m 24s | 4m 45s |
| HalfCheetah-v4 | SAC | 285K ± 32K | 9m 12s | 3m 36s |
| Hopper-v4 | PPO | 225K ± 28K | 6m 48s | 2m 42s |
| Hopper-v4 | SAC | 184K ± 24K | 5m 36s | 2m 12s |

### Notes:
¹ Baselines from "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)  
² Baselines from "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2018)  
³ Performance threshold: 90% of published baseline performance   -->

<!-- ### Hardware Specifications:

- **CPU**: AMD Ryzen 9 5900X (12 cores, 24 threads)
- **GPU**: NVIDIA RTX 3080 (10GB VRAM)
- **TPU**: Google Cloud TPU v3-8
- **RAM**: 32GB DDR4-3600
- **Storage**: NVMe SSD

### Software Environment:

- JAX 0.4.20
- CUDA 11.8
- Python 3.9
- Ubuntu 22.04 LTS

### Methodology:

- All results averaged over 5 runs with different random seeds
- 95% confidence intervals reported
- Training performed with default hyperparameters
- GPU results using mixed precision (float16/float32)
- TPU results using bfloat16/float32 -->

## Installation

```bash
Clone the repository
--------------------
git clone https://github.com/yourusername/omnixRL.git
cd omnixRL

Install Dependencies
---------------------
pip install -e .


```

## Quick Start

```python
from omnixrl import PPO
from omnixrl.env import GymEnvLoader
# Create environment
env = GymEnvLoader("HalfCheetah-v4", normalize_obs=True)
# Initialize algorithm
ppo = PPO(
env_info=env.get_env_info(),
learning_rate=3e-4,
n_steps=2048,
batch_size=64
)
# Train
ppo.train(total_timesteps=1_000_000)
```

<!-- ## Advanced Features

### Hyperparameter Tuning

```python
from omnixrl.tuning import HyperparameterTuner, ParameterSpace
# Define parameter space
param_space = ParameterSpace()
param_space.add_continuous("learning_rate", 1e-5, 1e-3, log=True)
param_space.add_discrete("n_steps", [128, 256, 512, 1024, 2048])
# Run  Hyperparameter tuning
tuner = HyperparameterTuner(config, env, PPO, param_space)
best_params = tuner.tune()
```
### Benchmarking
```python
from omnixrl.benchmarks import BenchmarkRunner
runner = BenchmarkRunner(config)
results = runner.run_benchmark(
algo_names=["PPO", "SAC"],
env_ids=["HalfCheetah-v4", "Hopper-v4"]
)
```

### Ablation Studies

```python

from omnixrl.analysis import AblationStudy
study = AblationStudy(config, env, PPO, base_config)
study.add_component_ablation("value_function", variants)
study.add_parameter_ablation("clip_range", values=[0.1, 0.2, 0.3])
study.run()
``` -->
<!-- 
## Documentation

Detailed documentation is available at [readthedocs link]. This includes:

- Algorithm implementations and theory
- API reference
- Tutorials and examples
- Experiment reproduction guides
- Contributing guidelines -->

## Citing

If you use this library in your research, please cite:

```bibtex
@software{omnixRL,
author = {Sandesh Katakam},
title = {OmnixRL: A JAX Implementation of Deep RL Algorithms},
year = {2024},
publisher = {GitHub},
url = {https://github.com/sandeshkatakam/omnixRL}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest features
- Submit pull requests
- Add new algorithms
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Spinning Up for the original inspiration
- JAX team for the excellent framework