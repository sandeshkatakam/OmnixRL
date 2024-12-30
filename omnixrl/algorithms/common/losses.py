from flax import linen as nn
import jax.numpy as jnp

def huber_loss(
    targets: jnp.ndarray, predictions: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
  """Implementation of the Huber loss with threshold delta.

  Let `x = |targets - predictions|`, the Huber loss is defined as:
  `0.5 * x^2` if `x <= delta`
  `0.5 * delta^2 + delta * (x - delta)` otherwise.

  Args:
    targets: Target values.
    predictions: Prediction values.
    delta: Threshold.

  Returns:
    Huber loss.
  """
  x = jnp.abs(targets - predictions)
  return jnp.where(x <= delta, 0.5 * x**2, 0.5 * delta**2 + delta * (x - delta))


def mse_loss(targets: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
  """Implementation of the mean squared error loss."""
  return jnp.power((targets - predictions), 2)


def softmax_cross_entropy_loss_with_logits(
    labels: jnp.ndarray, logits: jnp.ndarray
) -> jnp.ndarray:
  """Implementation of the softmax cross entropy loss."""
  return -jnp.sum(labels * nn.log_softmax(logits))