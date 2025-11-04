import jax.numpy as jnp
import jax

from CRMagTraj.utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_vec_to_cartesian_vec,
)


def test_spherical_to_cartesian():
    # test case: r=1, theta=pi/2, phi=0 -> x=1, y=0, z=0
    # test case: r=1, theta=0, phi=0 -> x=0, y=0, z=1
    # test case: r=1, theta=pi/2, phi=pi/2 -> x=0, y=1, z=0
    r = jnp.array([1.0, 1.0, 1.0])
    theta = jnp.array([jnp.pi / 2, 0.0, jnp.pi / 2])
    phi = jnp.array([0.0, 0.0, jnp.pi / 2])
    x_expected = jnp.array([1.0, 0.0, 0.0])
    y_expected = jnp.array([0.0, 0.0, 1.0])
    z_expected = jnp.array([0.0, 1.0, 0.0])
    x, y, z = jax.vmap(spherical_to_cartesian)(r, theta, phi)
    assert jnp.allclose(x, x_expected, atol=1e-6)
    assert jnp.allclose(y, y_expected, atol=1e-6)
    assert jnp.allclose(z, z_expected, atol=1e-6)


def test_cartesian_to_spherical():
    # test case: x=1, y=0, z=0 -> r=1, theta=pi/2, phi=0
    x = 1.0
    y = 0.0
    z = 0.0
    r, theta, phi = cartesian_to_spherical(x, y, z)
    assert jnp.isclose(r, 1.0)
    assert jnp.isclose(theta, jnp.pi / 2)
    assert jnp.isclose(phi, 0.0)
