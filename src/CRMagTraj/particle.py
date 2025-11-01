from abc import ABC, abstractmethod
from jaxtyping import Float, Array, Integer

import jax
import jax.numpy as jnp


class ParticleBase(ABC):
    """Abstract base class for particles. """

    @abstractmethod
    def get_mass(self) -> Float:
        """Get the mass of the particle."""
        pass
    
    @abstractmethod
    def get_charge(self) -> Float:
        """Get the charge of the particle."""
        pass

    @abstractmethod
    def get_energy(self) -> Float:
        """Get the energy of the particle."""
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
    
    def __init__(self, energy: Float):
        self.mass = 938.272089e6  # in eV/c^2
        self.charge = -1.0       # in elementary charge units
        self.energy = energy     # in eV
        
    def get_mass(self) -> Float:
        return self.mass
    
    def get_charge(self) -> Float:
        return self.charge

    def get_energy(self) -> Float:
        return self.energy
    
    def get_gamma(self) -> Float:
        return self.energy / self.mass
    
    def get_beta(self) -> Float:
        gamma = self.get_gamma()
        return jnp.sqrt(1.0 - 1.0 / (gamma * gamma))