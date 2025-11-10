import jax.numpy as jnp
import jax

from CRMagTraj.utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_vec_to_cartesian_vec,
    nadir_to_cartesian,
    cartesian_to_nadir,
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
    # test case: x=0, y=1, z=0 -> r=1, theta=pi/2, phi=pi/2
    # test case: x=0, y=0, z=1 -> r=1, theta=0, phi=0
    x = jnp.array([1.0, 0.0, 0.0])
    y = jnp.array([0.0, 1.0, 0.0])
    z = jnp.array([0.0, 0.0, 1.0])
    r_expected = jnp.array([1.0, 1.0, 1.0])
    theta_expected = jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0])
    phi_expected = jnp.array([0.0, jnp.pi / 2, 0.0])
    r, theta, phi = jax.vmap(cartesian_to_spherical)(x, y, z)
    assert jnp.allclose(r, r_expected, atol=1e-6)
    assert jnp.allclose(theta, theta_expected, atol=1e-6)
    assert jnp.allclose(phi, phi_expected, atol=1e-6)


def test_nadir_to_cartesian():
    # test case: pos_theta=pi/2, pos_phi=0, nadir_r=1, nadir_theta=pi/2, nadir_phi=0 -> x=0, y=0, z=1
    # test case: pos_theta=pi/2, pos_phi=pi/2, nadir_r=1, nadir_theta=pi/2, nadir_phi=0 -> x=0, y=0, z=1
    # test case: pos_theta=0, pos_phi=0, nadir_r=1, nadir_theta=0, nadir_phi=0 -> x=0, y=0, z=1
    # test case: pos_theta=pi/2, pos_phi=0, nadir_r=1, nadir_theta=0, nadir_phi=0 -> x=1, y=0, z=0
    # test case: pos_theta=pi/2, pos_phi=pi/2, nadir_r=1, nadir_theta=0, nadir_phi=0 -> x=0, y=1, z=0
    # test case: pos_theta=pi/2, pos_phi=0, nadir_r=1, nadir_theta=pi/2, nadir_phi=pi/2 -> x=0, y=1, z=0
    # test case: pos_theta=pi/2, pos_phi=pi/2, nadir_r=1, nadir_theta=pi/2, nadir_phi=pi/2 -> x=-1, y=0, z=0
    pos_theta = jnp.array(
        [jnp.pi / 2, jnp.pi / 2, 0.0, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
    )
    pos_phi = jnp.array([0.0, jnp.pi / 2, 0.0, 0.0, jnp.pi / 2, 0.0, jnp.pi / 2])
    nadir_r = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nadir_theta = jnp.array(
        [jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, jnp.pi / 2, jnp.pi / 2]
    )
    nadir_phi = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, jnp.pi / 2, jnp.pi / 2])
    x_expected = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    y_expected = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    z_expected = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    x, y, z = jax.vmap(nadir_to_cartesian)(
        pos_theta, pos_phi, nadir_r, nadir_theta, nadir_phi
    )
    print(x, y, z)
    assert jnp.allclose(x, x_expected, atol=1e-6)
    assert jnp.allclose(y, y_expected, atol=1e-6)
    assert jnp.allclose(z, z_expected, atol=1e-6)


def test_cartesian_to_nadir():
    # test case: pos_theta=pi/2, pos_phi=0, x=0, y=0, z=1 -> nadir_r=1, nadir_theta=pi/2, nadir_phi=0
    # test case: pos_theta=pi/2, pos_phi=pi/2, x=0, y=0, z=1 -> nadir_r=1, nadir_theta=pi/2, nadir_phi=0
    # test case: pos_theta=pi/2, pos_phi=0, x=1, y=0, z=0 -> nadir_r=1, nadir_theta=0, nadir_phi=0
    # test case: pos_theta=pi/2, pos_phi=pi/2, x=0, y=sqrt(1/2), z=sqrt(1/2) -> nadir_r=1, nadir_theta=pi/4, nadir_phi=0
    # test case: pos_theta=pi/2, pos_phi=0, x=0, y=1, z=0 -> nadir_r=1, nadir_theta=pi/2, nadir_phi=pi/2
    # test case: pos_theta=pi/2, pos_phi=pi/2, x=-1, y=0, z=0 -> nadir_r=1, nadir_theta=pi/2, nadir_phi=pi/2
    pos_theta = jnp.array(
        [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
    )
    pos_phi = jnp.array([0.0, jnp.pi / 2, 0.0, jnp.pi / 2, 0.0, jnp.pi / 2])
    x = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    y = jnp.array([0.0, 0.0, 0.0, jnp.sqrt(0.5), 1.0, 0.0])
    z = jnp.array([1.0, 1.0, 0.0, jnp.sqrt(0.5), 0.0, 0.0])
    nadir_r_expected = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nadir_theta_expected = jnp.array(
        [jnp.pi / 2, jnp.pi / 2, 0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi / 2]
    )
    nadir_phi_expected = jnp.array([0.0, 0.0, 0.0, 0.0, jnp.pi / 2, jnp.pi / 2])
    nadir_r, nadir_theta, nadir_phi = jax.vmap(cartesian_to_nadir)(
        pos_theta, pos_phi, x, y, z
    )
    print(nadir_r, nadir_theta, nadir_phi)
    print(nadir_r_expected, nadir_theta_expected, nadir_phi_expected)
    assert jnp.allclose(nadir_r, nadir_r_expected, atol=1e-6)
    assert jnp.allclose(nadir_theta, nadir_theta_expected, atol=1e-6)
    assert jnp.allclose(nadir_phi, nadir_phi_expected, atol=1e-6)
