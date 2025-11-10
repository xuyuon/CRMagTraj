from abc import ABC, abstractmethod
from jaxtyping import Float, Array, Integer

import jax
import jax.numpy as jnp

from .sampler import SamplerBase


class ParticleBase(ABC):
    """Abstract base class for particles."""

    @abstractmethod
    def get_mass(self) -> Float:
        """Get the mass of the particle."""
        pass

    @abstractmethod
    def get_charge(self) -> Float:
        """Get the charge of the particle."""
        pass

    @abstractmethod
    def get_gamma(self) -> Float:
        """Get the Lorentz factor of the particle."""
        pass

    @abstractmethod
    def get_beta(self) -> Float:
        """Get the velocity of the particle in units of c."""
        pass


class Antiproton(ParticleBase):
    """Class to represent an antiproton."""

    def __init__(self):
        self.mass = 938.272089e6  # in eV/c^2
        self.charge = -1.0  # in elementary charge units

    def get_mass(self) -> Float:
        return self.mass

    def get_charge(self) -> Float:
        return self.charge

    def get_gamma(self, energy) -> Float:  # energy in eV
        return energy / self.mass

    def get_beta(self, energy) -> Float:
        gamma = self.get_gamma(energy)
        return jnp.sqrt(1.0 - 1.0 / (gamma * gamma))
