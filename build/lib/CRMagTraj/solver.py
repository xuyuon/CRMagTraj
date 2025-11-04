from abc import ABC, abstractmethod
from jaxtyping import Float, Array, Integer
import jax
import jax.numpy as jnp

from diffrax import (
    diffeqsolve,
    Dopri5,
    ODETerm,
    SaveAt,
    PIDController,
    Event,
    TextProgressMeter,
    RecursiveCheckpointAdjoint,
)

from .field import FieldBase
from .particle import ParticleBase
from .constants import C_SI
from .utils import spherical_to_cartesian, spherical_vec_to_cartesian_vec


class TrajectorySolver:
    """
    A wrapper for solving particle trajectories in a magnetic field.
    """

    def __init__(
        self, field_model, particle, dD: Float, Dmax: Float, rmin: Float, rmax: Float
    ):
        self.field_model: FieldBase = field_model
        self.particle: ParticleBase = particle
        self.dD: Float = dD
        self.Dmax: Float = Dmax

        self.init_ODETerm()
        self.init_stopping_event(rmin, rmax)

    def init_ODETerm(self):
        def acceleration(d, y, args):
            # position is first three elements, velocity last three
            pos = y[0:3]
            vel = jnp.array(y[3:6])

            const, v = args
            B = self.field_model.B_field(pos[0] / 1000, pos[1] / 1000, pos[2] / 1000)
            dvel = const * jnp.cross(vel, B)
            dx = vel / v
            dy = dx[0], dx[1], dx[2], dvel[0], dvel[1], dvel[2]
            return dy

        self.term = ODETerm(acceleration)

    def init_stopping_event(self, r_min, r_max):
        def cond_fn(d, y, args, **kwargs):
            pos = jnp.array(y[0:3])
            r = jnp.linalg.norm(pos)
            return jnp.where((r < r_min) | (r > r_max), 0.0, 1.0)

        self.event = Event(cond_fn=cond_fn)

    def run(self, x0: Array, v0: Array, coordinate_system: str = "cartesian"):
        solver = Dopri5()
        progress = TextProgressMeter()
        saveat = SaveAt(ts=jnp.linspace(0.0, self.Dmax, 100))

        if coordinate_system == "spherical":
            r, theta, phi = x0
            x, y, z = spherical_to_cartesian(r, theta, phi)
            x0 = jnp.array([x, y, z])

            theta_nadir, phi_nadir = v0
            # phi_nadir=0 is northward, phi_nadir=pi/2 is eastward
            mag = self.particle.get_beta() * C_SI
            v_north, v_east, v_down = spherical_to_cartesian(
                mag, jnp.pi - theta_nadir, phi_nadir
            )

            vx, vy, vz = spherical_vec_to_cartesian_vec(
                theta, phi, -v_down, -v_north, v_east
            )
            v0 = jnp.array([vx, vy, vz])

        sol = diffeqsolve(
            self.term,
            solver,
            t0=0.0,
            t1=self.Dmax,
            dt0=self.dD,
            y0=(x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]),
            args=(
                self.particle.get_charge()
                * C_SI
                / self.particle.get_energy()
                / self.particle.get_beta(),
                C_SI * self.particle.get_beta(),
            ),
            saveat=saveat,
            # stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(),
            event=self.event,
            max_steps=None,
            progress_meter=progress,
        )

        return sol.ys

    def run_batch(self, x0: Array, v0: Array, coordinate_system: str = "cartesian"):
        def single_run(x, v):
            return self.run(x, v, coordinate_system)

        batched_run = jax.vmap(single_run, in_axes=(0, 0))
        return batched_run(x0, v0)

    def run_jacobian(self, x0: Array, v0: Array, coordinate_system: str = "cartesian"):
        solver = Dopri5()
        progress = TextProgressMeter()
        saveat = SaveAt(ts=jnp.linspace(0.0, self.Dmax, 100))

        if coordinate_system == "spherical":
            r, theta, phi = x0
            x, y, z = spherical_to_cartesian(r, theta, phi)
            x0 = jnp.array([x, y, z])

            theta_nadir, phi_nadir = v0
            # phi_nadir=0 is northward, phi_nadir=pi/2 is eastward
            mag = self.particle.get_beta() * C_SI
            v_north, v_east, v_down = spherical_to_cartesian(
                mag, jnp.pi - theta_nadir, phi_nadir
            )

            vx, vy, vz = spherical_vec_to_cartesian_vec(
                theta, phi, -v_down, -v_north, v_east
            )
            v0 = jnp.array([vx, vy, vz])

        y0 = jnp.array([x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]])

        # calculate the jacobian of the final position with respect to the initial position
        def final_position(y0):
            sol = diffeqsolve(
                self.term,
                solver,
                t0=0.0,
                t1=self.Dmax,
                dt0=self.dD,
                y0=(y0[0], y0[1], y0[2], y0[3], y0[4], y0[5]),
                args=(
                    self.particle.get_charge()
                    * C_SI
                    / self.particle.get_energy()
                    / self.particle.get_beta(),
                    C_SI * self.particle.get_beta(),
                ),
                # stepsize_controller=stepsize_controller,
                adjoint=RecursiveCheckpointAdjoint(),
                event=self.event,
                max_steps=int(self.Dmax / self.dD + 1000),
            )
            return jnp.array(sol.ys)

        jacobian_fn = jax.jacrev(final_position)
        jacobian = jacobian_fn(y0)
        return jacobian

        # return jnp.abs(jnp.linalg.det(jacobian))
