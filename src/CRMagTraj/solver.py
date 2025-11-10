from abc import ABC, abstractmethod
import functools as ft
from tqdm import tqdm, trange
import math

from jaxtyping import Float, Array, Integer
import jax
import jax.numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

from diffrax import (
    diffeqsolve,
    Dopri5,
    ODETerm,
    SaveAt,
    SubSaveAt,
    PIDController,
    Event,
    TextProgressMeter,
    NoProgressMeter,
    RecursiveCheckpointAdjoint,
)

from .field import FieldBase
from .particle import ParticleBase
from .constants import C_SI
from .utils import (
    spherical_to_cartesian,
    spherical_vec_to_cartesian_vec,
    cartesian_to_spherical,
    nadir_to_cartesian,
    cartesian_to_nadir,
)


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

    def single_run(
        self,
        x0: Array,
        v0: Array,
        energy: Float = None,
        coordinate_system: str = "cartesian",
        progress: bool = True,
        saveat: bool = True,
    ):
        solver = Dopri5()
        if progress:
            progress = TextProgressMeter()
        else:
            progress = NoProgressMeter()

        if saveat:
            saveat = SaveAt(ts=jnp.linspace(0.0, self.Dmax, 100))
        else:
            saveat = SaveAt(subs=SubSaveAt(t1=True))

        if coordinate_system == "spherical":
            r, theta, phi = x0
            x, y, z = spherical_to_cartesian(r, theta, phi)
            x0 = jnp.array([x, y, z])

            theta_nadir, phi_nadir = v0
            # phi_nadir=0 is northward, phi_nadir=pi/2 is eastward
            mag = self.particle.get_beta(energy) * C_SI
            vx, vy, vz = nadir_to_cartesian(theta, phi, mag, theta_nadir, phi_nadir)
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
                / energy
                / self.particle.get_beta(energy),
                C_SI * self.particle.get_beta(energy),
            ),
            saveat=saveat,
            # stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(),
            event=self.event,
            max_steps=None,
            progress_meter=progress,
        )

        # transform the result back to spherical coordinates
        if coordinate_system == "spherical":
            pos_r, pos_theta, pos_phi = jax.vmap(cartesian_to_spherical)(
                jnp.array(sol.ys[0]), jnp.array(sol.ys[1]), jnp.array(sol.ys[2])
            )
            nadir_r, nadir_theta, nadir_phi = jax.vmap(cartesian_to_nadir)(
                pos_theta,
                pos_phi,
                jnp.array(sol.ys[3]),
                jnp.array(sol.ys[4]),
                jnp.array(sol.ys[5]),
            )
            return jnp.array(
                [
                    pos_r,
                    pos_theta,
                    pos_phi,
                    nadir_r,
                    nadir_theta,
                    nadir_phi,
                ]
            )
        else:
            return jnp.array(sol.ys)

    def batch_run(
        self, x0: Array, v0: Array, energy: Array, coordinate_system: str = "cartesian"
    ):
        n_device = jax.device_count()
        n_run = x0.shape[0]
        n_padding = n_device - n_run % n_device
        n_epochs = math.ceil(n_run / n_device)
        n_saveat = 1

        print(n_device, "devices detected. Running a batch size of ", n_device, ".")
        mesh = Mesh(create_device_mesh((n_device,)), ["i"])
        spec = PartitionSpec("i")

        @jax.jit
        @ft.partial(
            shard_map, mesh=mesh, in_specs=spec, out_specs=spec, check_rep=False
        )
        @jax.vmap
        def single_run_fn(x0_sr, v0_sr, energy_sr):
            return self.single_run(
                x0_sr, v0_sr, energy_sr, coordinate_system, progress=False, saveat=False
            )

        sharding = NamedSharding(mesh, spec)

        results = jnp.zeros((n_run + n_padding, 6, n_saveat))
        # add padding if necessary
        if n_padding > 0:
            x0 = jnp.vstack([x0, jnp.zeros((n_padding, x0.shape[1]))])
            v0 = jnp.vstack([v0, jnp.zeros((n_padding, v0.shape[1]))])
            energy = jnp.hstack([energy, jnp.zeros((n_padding,))])

        pbar = trange(n_epochs, desc="Running solver")
        for epoch in pbar:
            start_idx = epoch * n_device
            end_idx = epoch * n_device + n_device  # stop at end_idx - 1
            x0_sharded = jax.device_put(x0[start_idx:end_idx], sharding)
            v0_sharded = jax.device_put(v0[start_idx:end_idx], sharding)
            energy_sharded = jax.device_put(energy[start_idx:end_idx], sharding)
            results = results.at[start_idx:end_idx, :, :].set(
                single_run_fn(x0_sharded, v0_sharded, energy_sharded)
            )

        # Remove padding
        results = results[0:n_run, :, :]
        return results

    def run_jacobian(
        self, x0: Array, v0: Array, energy: Float, coordinate_system: str = "cartesian"
    ):
        solver = Dopri5()
        progress = TextProgressMeter()
        saveat = SaveAt(ts=jnp.linspace(0.0, self.Dmax, 100))

        if coordinate_system == "spherical":
            r, theta, phi = x0
            x, y, z = spherical_to_cartesian(r, theta, phi)
            x0 = jnp.array([x, y, z])

            theta_nadir, phi_nadir = v0
            # phi_nadir=0 is northward, phi_nadir=pi/2 is eastward
            mag = self.particle.get_beta(energy) * C_SI
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
                    / energy
                    / self.particle.get_beta(energy),
                    C_SI * self.particle.get_beta(energy),
                ),
                # stepsize_controller=stepsize_controller,
                adjoint=RecursiveCheckpointAdjoint(),
                event=self.event,
                max_steps=int(self.Dmax / self.dD + 1000),
            )
            return jnp.array(sol.ys)

        jacobian_fn = jax.jacrev(final_position)
        jacobian = jacobian_fn(y0)
        return jnp.abs(jnp.linalg.det(jacobian))
