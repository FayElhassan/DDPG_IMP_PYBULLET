import jax
import jax.numpy as jnp
import haiku as hk

class Actor(hk.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def __call__(self, state):
        x = hk.Linear(256,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(state)
        x = jax.nn.relu(x)
        x = hk.Linear(256,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.action_dim,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(x)
        return jax.nn.tanh(x)

class Critic(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = hk.Linear(256,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(256,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1,  w_init=hk.initializers.VarianceScaling(scale=1.0, distribution='uniform'))(x)
        return x
