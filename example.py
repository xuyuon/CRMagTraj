import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np
import multiprocessing

from src.CRMagTraj.solver import TrajectorySolver
from src.CRMagTraj.field import IGRFField, ConstField
from src.CRMagTraj.particle import Antiproton
from src.CRMagTraj.sampler import Sine, Lambertian, UniformSphericalSurface

jax.config.update("jax_enable_x64", True)

antiproton = Antiproton(energy=3251319351.8354297)  # energy in eV
field = IGRFField("IGRF14.shc", 2011, 4, 28, 5)
solver = TrajectorySolver(
    field_model=field,
    particle=antiproton,
    dD=100.0,
    Dmax=5 * 2 * np.pi * 10 * 6371.2 * 1000.0,
    rmin=7371.2 * 1000,
    rmax=10 * 6371.2 * 1000,
)

n_samples = 10
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x0_list = UniformSphericalSurface(7371.2 * 1000).sample(key, n_samples)
key, subkey = jax.random.split(subkey)
v0_list = Lambertian().sample(subkey, n_samples)

batch_sols = solver.batch_run(x0_list, v0_list, coordinate_system="spherical")

# save input
np.save("input_x0.npy", np.array(x0_list))
np.save("input_v0.npy", np.array(v0_list))
# save output
np.save("output_batch_sols_ex.npy", np.array(batch_sols))
