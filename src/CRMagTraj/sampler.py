import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray


class SamplerBase:
    """
    Base class for samplers.
    """

    def sample(self, key: PRNGKeyArray, num_samples: int) -> Array:
        """
        Sample initial conditions.

        Args:
            key: JAX random key.
            num_samples: Number of samples to generate.

        Returns:
            Array of shape (num_samples, dim) where dim is the dimension of the sample space.
        """
        raise NotImplementedError(
            "SamplerBase.sample must be implemented in subclasses."
        )


class Sine:
    """
    Class to sample directions from a sine distribution:
        p(theta) ~ sin(theta)
    """

    def __init__(self, xmin: Float = 0.0, xmax: Float = jnp.pi):
        self.xmin = xmin
        self.xmax = xmax

    def sample(self, key: PRNGKeyArray, n_samples: int) -> Array:
        x = jax.random.uniform(key, shape=(n_samples,), minval=0.0, maxval=1.0)
        return jnp.arccos(
            jnp.cos(self.xmin) + x * (jnp.cos(self.xmax) - jnp.cos(self.xmin))
        )


class Lambertian:
    """
    Class to sample directions from a Lambertian distribution:
        p(theta_nadir) ~ cos(theta_nadir) sin(theta_nadir) ~ sin(2 theta_nadir)
        p(phi_nadir) ~ const
    """

    def __init__(self):
        self.theta_sampler = Sine(xmin=0.0, xmax=jnp.pi)

    def sample(self, key: PRNGKeyArray, n_samples: int) -> Array:
        key, subkey = jax.random.split(key)
        phi = jax.random.uniform(
            subkey, shape=(n_samples,), minval=0.0, maxval=2 * jnp.pi
        )
        key, subkey = jax.random.split(subkey)
        theta = self.theta_sampler.sample(key, n_samples) / 2.0
        return jnp.stack([theta, phi], axis=1)


class UniformSphericalSurface:
    """
    Class to sample points uniformly on a spherical surface.
        p(theta, phi) ~ sin(theta)
    """

    def __init__(self, r: Float):
        self.r = r
        self.theta_sampler = Sine(xmin=0.0, xmax=jnp.pi)

    def sample(self, key: PRNGKeyArray, n_samples: int) -> Array:
        key, subkey = jax.random.split(key)
        phi = jax.random.uniform(
            subkey, shape=(n_samples,), minval=0.0, maxval=2 * jnp.pi
        )
        key, subkey = jax.random.split(subkey)
        theta = self.theta_sampler.sample(subkey, n_samples)
        r = jnp.full((n_samples,), self.r)
        return jnp.stack([r, theta, phi], axis=1)
