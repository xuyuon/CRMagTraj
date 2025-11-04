from jaxtyping import Float, Array, Integer

import jax
import jax.numpy as jnp


def cartesian_to_spherical(x: Float, y: Float, z: Float) -> tuple[Float, Float, Float]:
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(
    r: Float, theta: Float, phi: Float
) -> tuple[Float, Float, Float]:
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return x, y, z


def spherical_vec_to_cartesian_vec(
    theta: Float, phi: Float, v_r: Float, v_theta: Float, v_phi: Float
) -> tuple[Float, Float, Float]:
    """ """
    v_x = (
        jnp.sin(theta) * jnp.cos(phi) * v_r
        + jnp.cos(theta) * jnp.cos(phi) * v_theta
        - jnp.sin(phi) * v_phi
    )
    v_y = (
        jnp.sin(theta) * jnp.sin(phi) * v_r
        + jnp.cos(theta) * jnp.sin(phi) * v_theta
        + jnp.cos(phi) * v_phi
    )
    v_z = jnp.cos(theta) * v_r - jnp.sin(theta) * v_theta
    return v_x, v_y, v_z
