# -*- coding: utf-8 -*-

import numpy as np
import he3_tools.he3_props as h3p
import he3_tools.he3_bases as h3b
import he3_tools.he3_constants as h3c
import he3_tools.he3_data as h3d
import he3_tools.he3_matrix as h3m
import he3_tools.he3_free_energy as h3f


# Theory functions
def s_scale(p):

    """ Natural entropy density scale, units joule per kelvin per nm3 .
    """
    return h3p.f_scale(p) / (h3p.T_mK(1, p) * 1e-3)


def entropy_density_norm(t, p, phase, squeeze_me=True, diagonal=False):
    """
    Normalised entropy density, in units of $(1/3)N(0)k_B^2T_c$, calculated from 
    approximate volumetric specific heat (Greywall 1986), by integrating from $T_c$.

    Parameters
    ----------
    t : int, float, numpy.ndarray 1D
        Reduced temperature $T/T_c$.
    p : nt, float, numpy.ndarray 1D
        Pressure in bar.
    phase : str
        Superfluid phase: [ A | B | planar | polar ].

    Returns
    -------
    float, numpy.ndarray
        Normalised entropy density. If t and p are both ndarrays of length greater 
        than 1, the returned array has shape (len(t), len(p)). 

    """
    
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)
    # N0 is single-spin density of states
    c_scale = (np.pi**2/3) * (2*h3p.N0(p)) * h3c.kB**2 *(h3p.Tc_mK(p)/1000) 

    C_V_norm_Tc = h3p.C_V(1, p, phase)/(c_scale)
    
    sf_int_fac = (t**3 - 1)/3
    sf_int_fac[sf_int_fac > 0] = 0.0
    # Entropy density at Tc is equal to the volumetric specific heat, as it is linear in t
    if diagonal:
        s_norm = specific_heat_N_norm(t) + sf_int_fac * C_V_norm_Tc
    else:
        s_norm = specific_heat_N_norm(t)[:,None] + np.outer(sf_int_fac,  C_V_norm_Tc)
    
    if squeeze_me:
        s_norm = h3p.squeeze_float(s_norm)
    return s_norm

def specific_heat_N_norm(t):
    """
    Specific heat in the normal phase, in of $(1/3)N(0)k_B^2T_c$.

    Parameters
    ----------
    t : TYPE
        Reduced temperature, $T/T_c$.
    p : TYPE
        Pressure in bar.

    Returns
    -------
    Specific heat, same type as t.

    """
    
    return np.pi**2 * t * 2 # spin states

def enthalpy_density_norm(t, p, phase, squeeze_me=True, diagonal=False):
    
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)
    if diagonal:
        w = t * entropy_density_norm(t, p, phase, squeeze_me=False, diagonal=True)
    else:
        w = t[:,None] * entropy_density_norm(t, p, phase, squeeze_me=False)
    
    if squeeze_me:
        w = h3p.squeeze_float(w)
        
    return w

def latent_heat_norm(p):
    
    tAB = h3p.tAB(p)
    
    LAB = enthalpy_density_norm(tAB, p, 'B', diagonal=True) - enthalpy_density_norm(tAB, p, 'A', diagonal=True)
    
    return LAB

def energy_density_norm(t, p, phase):
    
    return enthalpy_density_norm(t, p, phase) - h3p.f_phase_norm(t, p, phase)
    
def theta_norm(t, p, phase, squeeze_me=True, diagonal=False):
    
    return 0.25*enthalpy_density_norm(t, p, phase, squeeze_me, diagonal) - h3p.f_phase_norm(t, p, phase, squeeze_me, diagonal)

def alpha_pt_AB_norm(t, p):

    delta_theta = theta_norm(t, p, 'A') - theta_norm(t, p, 'B')
    
    thermal_energy_density = 0.75 * enthalpy_density_norm(t, p, 'A')
    
    return delta_theta/thermal_energy_density

