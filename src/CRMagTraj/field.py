from abc import ABC, abstractmethod
from jaxtyping import Float, Array, Integer

import jax
import jax.numpy as jnp

import io


class FieldBase(ABC):
    """Abstract base class for magnetic field models."""

    @abstractmethod
    def B_field(self, x: Float, y: Float, z: Float) -> Array:
        """Get the magnetic field at position (x, y, z)."""
        pass


class ConstField(FieldBase):
    """Class to represent a constant magnetic field."""

    def __init__(self, Bx: Float, By: Float, Bz: Float):
        self.B = jnp.array([Bx, By, Bz])

    def B_field(self, x: Float, y: Float, z: Float) -> Array:
        """Return the constant magnetic field."""
        return self.B

class IGRFField:
    """Class to represent the IGRF magnetic field model."""

    def __init__(self, datafile, year: Integer, month: Integer, day: Integer, n_max: Integer):
        self.r_earth = 6371.2 # in km
        self.year_frac = self.datetime_to_year_fraction(year, month, day)
        self.load_field_data(datafile)
        
        if (n_max <= self.N_MAX):
            self.n_max = n_max
        else:
            self.n_max = self.N_MAX
        
        self.g_interp = self.g_interp[:self.n_max+1,:self.n_max+1]
        self.h_interp = self.h_interp[:self.n_max+1,:self.n_max+1]
        
        # initialize the normalization constant for associated Legendre Polynomials
        self.init_Scmidt_normalization(self.n_max, self.n_max)
        self.init_alp_mask(self.n_max, self.n_max)
        
        
    @staticmethod
    def datetime_to_year_fraction(year: Integer, month: Integer, day: Integer) -> Float:
        """Convert a date to a fractional year representation."""
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_year = 365
        
        def is_leap_year(year: Integer) -> bool:
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        
        if (is_leap_year(year)):
            days_in_month[1] = 29
            days_in_year = 366
            
        n_days = day
        for m in days_in_month[:month - 1]:
            n_days += m
            
        return float(year) + (float(n_days) - 1.0) / float(days_in_year)
        
        
    def load_field_data(self, datafile):
        """Load the IGRF field data from a file."""
        print("Loading IGRF field data from:", datafile)
        
        lines = []
        
        with io.open(datafile, 'r') as f:
            for line in f:
                # skip comments and empty lines
                if not line.startswith('#') or not line.strip():
                    lines.append(line.split())

        # the first line contains the meta data
        self.N_MIN = int(lines[0][0])
        self.N_MAX = int(lines[0][1])
        self.NTIMES = int(lines[0][2])
        self.SP_ORDER = int(lines[0][3])
        self.N_STEPS = int(lines[0][4])
        
        # initialize the g and h vectors to be of size (N_MAX+1) x (N_MAX+1) x NTIMES
        g = jnp.zeros((self.N_MAX + 1, self.N_MAX + 1, self.NTIMES))
        h = jnp.zeros((self.N_MAX + 1, self.N_MAX + 1, self.NTIMES))
        
        # load the time points from the second line
        times = jnp.array([float(t) for t in lines[1]])
        if (self.year_frac < times[0] or self.year_frac > times[-1]):
            raise ValueError("Date out of range for IGRF data.")

        # load the Gauss coefficients for the Spherical Harmonic
        for line in lines[2:]:
            n = int(line[0])
            m = int(line[1])
            
            if m >= 0:
                for t in range(self.NTIMES):
                    g = g.at[n, m, t].set(float(line[2 + t]))
            else:
                for t in range(self.NTIMES):
                    h = h.at[n, -m, t].set(float(line[2 + t]))
        
        # Find t1 (the time point just before the input year) and
        # t2 (the time point just after the input year)
        t1 = 0
        t2 = 0
        t1_index = 0
        t2_index = 0
        for i in range(self.NTIMES - 1):
            if times[i] <= self.year_frac <= times[i + 1]:
                t1 = times[i]
                t2 = times[i + 1]
                t1_index = i
                t2_index = i + 1
                break
        
        # Interpolate the Gauss coefficients for the input year     
        self.g_interp = g[:, :, t1_index] + (g[:, :, t2_index] - g[:, :, t1_index]) / (t2 - t1) * (self.year_frac - t1)
        self.h_interp = h[:, :, t1_index] + (h[:, :, t2_index] - h[:, :, t1_index]) / (t2 - t1) * (self.year_frac - t1)

    
    def init_Scmidt_normalization(self, n_max, m_max):
        """
        calculating s(n,m)
        """
        s = jnp.zeros((n_max+1, m_max+1))

        a_idx = jnp.arange(1, n_max+1)
        b_idx = jnp.arange(0, n_max+1)

        s = s.at[0,0].set(1.0) # initial value s(0,0)
        s = s.at[1:,0].set(jnp.cumprod((2*a_idx-1)/a_idx)) # setting s(n,0)

        n = jnp.broadcast_to(b_idx, (m_max, b_idx.shape[0])).T
        m = jnp.broadcast_to(a_idx, (n_max+1, a_idx.shape[0]))
        s = s.at[:,1:].set(jnp.sqrt((n - m + 1)/(n+m) * jnp.where(m==1, 2, 1)))
        s = jnp.cumprod(s, axis=1)
        
        self.s = s.at[jnp.isnan(s)].set(0.0)
    
    
    def init_alp_mask(self, n_max, m_max):
        n_mat, m_mat = jnp.meshgrid(
            jnp.arange(n_max+1),
            jnp.arange(n_max+1),
            indexing='ij'
        )
        
        d0 = jnp.ones((n_max+1, n_max+1))
        d1 = ((n_mat-1.0)**2-m_mat**2)/(2*n_mat-1)/(2*n_mat-3)

        d0_mask_indices = jnp.tril_indices(n_max + 1, -1)
        d1_mask_indices = jnp.tril_indices(n_max + 1, -1)
        d_zeros = jnp.zeros((n_max+1, n_max+1))

        d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
        d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

        i, j, k = jnp.ogrid[:n_max+1, :n_max+1, :n_max+1]
        mask = (i-j+k==0)
        
        self.d0_mask_3d = jnp.einsum('jk,ijk->ijk', d0_mask, mask)
        self.d1_mask_3d = jnp.einsum('jk,ijk->ijk', d1_mask, mask)
    
    @staticmethod
    def associated_Legendre_polynomial(theta, n_max, m_max, d0_mask_3d, d1_mask_3d, s):
        sinth = jnp.sin(theta)
        costh = jnp.cos(theta)

        p = jnp.zeros((n_max+1, m_max+1))
        dp = jnp.zeros((n_max+1, m_max+1))

        p = p.at[(0,0)].set(1.0) # initial value p(0,0)
        # initial value dp(0,0) is 0.0

        # Compute the diagonal entries p(n,n) with recurrence
        p_diag = jnp.cumprod(jnp.broadcast_to(sinth, (n_max)), axis=0)
        diag_indices = jnp.diag_indices(n_max + 1)
        p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)
        # Compute the diagonal entries dp(n,n) with recurrence
        def gen_dp_diag(i, dp_val):
            dp_val = dp_val.at[i,i].set(dp_val[i-1,i-1]*sinth+costh*p[i-1,i-1])
            return dp_val
        dp = jax.lax.fori_loop(lower=1, upper=n_max+1, body_fun=gen_dp_diag, init_val=dp)

        # Compute the off-diagonal entries p(n+1,n) with recurrence
        p_offdiag = costh * p[jnp.diag_indices(n_max)]
        offdiag_indicies = (diag_indices[0][:n_max]+1, diag_indices[1][:n_max])
        p = p.at[offdiag_indicies].set(p_offdiag)
        # Compute the off-diagonal entries dp(n+1,n) with recurrence
        def gen_dp_offdiag(i, dp_val):
            dp_val = dp_val.at[i+1,i].set(costh*dp_val[i,i]-sinth*p[i,i])
            return dp_val
        dp = jax.lax.fori_loop(lower=0, upper=n_max+1, body_fun=gen_dp_offdiag, init_val=dp)

        # Compute the remaining entries with recurrence   
        def body_func(i, carry):
            p_val, dp_val = carry
            coeff_0 = d0_mask_3d[i]
            coeff_1 = d1_mask_3d[i]
            h = jnp.roll(p_val, shift=1, axis=0) * costh * coeff_0 - jnp.roll(p_val, shift=2, axis=0) * coeff_1
            dh = jnp.roll(dp_val, shift=1, axis=0) * costh * coeff_0 - jnp.roll(p_val, shift=1, axis=0) * sinth * coeff_0 - jnp.roll(dp_val, shift=2, axis=0) * coeff_1
            p_val = p_val + h
            dp_val = dp_val + dh
            return (p_val, dp_val)
        
        if n_max > 1:
            p, dp = jax.lax.fori_loop(lower=2, upper=n_max+1, body_fun=body_func, init_val=(p,dp))
        
        return p * s, dp * s
    
    
    # def B_field(self, x, y, z):
    #     """
    #     return B-field at position in Tesla
    #     """
    #     return jnp.array([0.0, 1e4, 0.0])

    def B_field(self, x: Float, y: Float, z: Float) -> Array:
        """
        Calculate the magnetic field at a given position (x, y, z) in Tesla.
        
        input in km
        """
        # Convert Cartesian coordinates to spherical coordinates
        r = jnp.sqrt(x**2 + y**2 + z**2) 
        theta = jnp.arccos(z / r)
        phi = jnp.arctan2(y, x)
        
        p, dp = self.associated_Legendre_polynomial(theta, self.n_max, self.n_max, self.d0_mask_3d, self.d1_mask_3d, self.s)
        
        n_mat, m_mat = jnp.meshgrid(
            jnp.arange(self.n_max+1),
            jnp.arange(self.n_max+1),
            indexing='ij'
        )
        
        B_r = jnp.sum((n_mat + 1) * ((self.r_earth/r)**(n_mat + 2)) * (self.g_interp * jnp.cos(m_mat * phi) + self.h_interp * jnp.sin(m_mat * phi)) * p)
        B_theta = -jnp.sum((self.r_earth/r)**(n_mat+2)*(self.g_interp * jnp.cos(m_mat * phi) + self.h_interp * jnp.sin(m_mat * phi)) * dp)
        B_phi = - jnp.sum((self.r_earth/r)**(n_mat+2)*m_mat*(- self.g_interp * jnp.sin(m_mat * phi) + self.h_interp * jnp.cos(m_mat * phi))*p)/jnp.sin(theta)

        # Convert from spherical to Cartesian coordinates
        B_x = jnp.sin(theta) * jnp.cos(phi) * B_r + jnp.cos(theta) * jnp.cos(phi) * B_theta - jnp.sin(phi) * B_phi
        B_y = jnp.sin(theta) * jnp.sin(phi) * B_r + jnp.cos(theta) * jnp.sin(phi) * B_theta + jnp.cos(phi) * B_phi
        B_z = jnp.cos(theta) * B_r - jnp.sin(theta) * B_theta

        return jnp.array([B_x, B_y, B_z]) * 1e-9
        # return jnp.array([B_x, B_y, B_z]) * 1e-9