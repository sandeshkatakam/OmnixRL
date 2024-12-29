



import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
from jax.random.distributions.normal import Normal
from jax.random.distributions.categorical import Categorical
import gym.spaces import Box, Discrete
from typing import List, Callable # NOTE: Use JAXTyping library later on
from omnixrl import BaseNN

def combined_shape(length, shape = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



class MLP:
    def __init__(self, layer_sizes: List[int], activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu):
        """
        Initialize an MLP with variable layer sizes.

        Args:
            layer_sizes: List of integers specifying the number of units in each layer, including input and output layers.
            activation: Activation function to use between layers.
        """
        self.layer_sizes = layer_sizes
        self.activation = activation

        # Initialize weights and biases
        self.params = self.initialize_params()

    def initialize_params(self):
        """Initialize the weights and biases of the MLP."""
        params = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight = jax.random.normal(jax.random.PRNGKey(0), (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
            bias = jnp.zeros(out_size)
            params.append({'weight': weight, 'bias': bias})
        return params

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Perform a forward pass through the MLP."""
        for i, layer in enumerate(self.params):
            x = jnp.dot(x, layer['weight']) + layer['bias']
            if i < len(self.params) - 1:  # Apply activation for all but the last layer
                x = self.activation(x)
        return x

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """Alias for forward."""
        return self.forward(x)




class Actor(BaseNN): # Implement Base Neural Network to inherit methods for Actor Network
    def _distribution(self, obs):
        raise NotImplementedError
    
    def _log_prob_from_distributions(self, pi, act):
        raise NotImplementedError

    def forward(self,obs, act = None):
        pass # Do this Later

class MLPCategoricalActor(Actor):
    def __init__(self, act):
        super().__init__()

class MLPGaussianActor(Actor):
    def __init__(self,act):
        super().__init__()

    def _distribution(self,obs):
        distrax.distributions.normal.Normal(obs)



class MLPCritic(BaseNN):
    def __init__(self,values):
        super().__init__()
