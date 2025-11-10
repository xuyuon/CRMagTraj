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


def cartesian_vec_to_spherical_vec(
    theta: Float, phi: Float, v_x: Float, v_y: Float, v_z: Float
) -> tuple[Float, Float, Float]:
    """ """
    v_r = (
        jnp.sin(theta) * jnp.cos(phi) * v_x
        + jnp.sin(theta) * jnp.sin(phi) * v_y
        + jnp.cos(theta) * v_z
    )
    v_theta = (
        jnp.cos(theta) * jnp.cos(phi) * v_x
        + jnp.cos(theta) * jnp.sin(phi) * v_y
        - jnp.sin(theta) * v_z
    )
    v_phi = -jnp.sin(phi) * v_x + jnp.cos(phi) * v_y
    return v_r, v_theta, v_phi


def nadir_to_cartesian(
    pos_theta: Float,
    pos_phi: Float,
    nadir_mag: Float,
    nadir_theta: Float,
    nadir_phi: Float,
) -> tuple[Float, Float, Float]:
    """_summary_

    Args:
        pos_theta (Float): _description_
        pos_phi (Float): _description_
        nadir_mag (Float): _description_
        nadir_theta (Float): _description_
        nadir_phi (Float): _description_

    Returns:
        tuple[Float, Float, Float]: _description_
    """
    # phi_nadir=0 is northward, phi_nadir=pi/2 is eastward
    v_north, v_east, v_down = spherical_to_cartesian(
        nadir_mag, jnp.pi - nadir_theta, nadir_phi
    )

    vx, vy, vz = spherical_vec_to_cartesian_vec(
        pos_theta, pos_phi, -v_down, -v_north, v_east
    )

    return vx, vy, vz


def cartesian_to_nadir(
    pos_theta: Float, pos_phi: Float, v_x: Float, v_y: Float, v_z: Float
) -> tuple[Float, Float, Float]:
    """_summary_

    Args:
        theta (Float): _description_
        phi (Float): _description_
        v_r (Float): _description_
        v_theta (Float): _description_
        v_phi (Float): _description_

    Returns:
        tuple[Float, Float, Float]: _description_
    """
    v_up, v_south, v_east = cartesian_vec_to_spherical_vec(
        pos_theta, pos_phi, v_x, v_y, v_z
    )
    nadir_mag, nadir_theta, nadir_phi = cartesian_to_spherical(-v_south, v_east, -v_up)
    nadir_theta = jnp.pi - nadir_theta
    return nadir_mag, nadir_theta, nadir_phi
