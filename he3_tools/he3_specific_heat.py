#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:03:31 2025

@author: hindmars with assistace of MicrdSoft CoPilot
"""


# import math
import numpy as np
import he3_tools.he3_constants as h3c
import he3_tools.he3_props as h3p

# he3.set_default("DEFAULT_T_SCALE", 'Greywall')

def C_V_normal(t, p, squeeze_me=True, diagonal=False):
    """
    Normal phase specific heat in units J / K / nm$^3$, with strong coupling 
    corrections included.  Uses formula (2.13) from Vollhardt & Woelfle.
    $$
    C_V = \frac{\pi^2}{3} N_F k_B^2 T
    $$
    where $N_F$ is the total density of states, including the spin sum.

    Parameters
    ----------
    t : float, int, numpy.ndarray (1D)
        Reduced temperature, $T/T_c$.
    p : float, int, numpy.ndarray (1D)
        Pressure in bar.


    Returns
    -------
    float or numpy.ndarray
        Normal phase specific heat at temperature $t$ and pressure $p$. 
        If both t and p are 1D arrays of length greater than 1, the return 
        value has shape (len(t), len(p))

    """
    
    # Remember that N0 is the single-spin density of states
    c_scale =  np.pi**2 * (2*h3p.N0(p)) * h3c.kB**2 *(h3p.Tc_K(p)) / 3 
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)
    # c_v =np.outer(t, np.pi**2 * f_scale(p) / (Tc_mK(p)/1000))
    if diagonal:
        c_v = t * c_scale
    else:
        c_v = np.outer(t, c_scale)    
            
    if squeeze_me:
        c_v = np.squeeze(c_v)

    try:
        c_v = float(c_v)
    except:
        pass

    # return  np.pi**2 * f_scale(p) / (Tc_mK(p)/1000) * t
    return c_v

def delta_C_V_Tc(p, phase):
    """
    Jump in specific heat at superfluid phase transition in units J / K / nm$^3$, 
    with strong coupling corrections included.  Uses formula (3.78) from 
    Vollhardt & Woelfle.
    $$
    \Delta C_V = \frac{1} N(0) \frac{\partial}{\partial T} \langle \Delta_{\bf k}^\dagger \Delta_{\bf k} \rangle_{\hat{\bf k}}
    $$

    Parameters
    ----------
    p : float, int, numpy.ndarray
        Pressure in bar.
    phase : str
        Phase string, e.g. "A", "B".
        

    Returns
    -------
    float or numpy.ndarray
        Specific heat jump at pressure p.

    """
    return  0.25 * h3p.f_scale(p) / (h3p.Tc_K(p) * h3p.beta_phase_norm(1, p, phase))

def C_V(t, p, phase):
    """
    Specific heat in given phase, with strong coupling 
    corrections included.  Uses formula (3.77) from Vollhardt & Woelfle.
    $$
    C_V = C_N + \Delta C_V
    $$
    Below $T_c$ (i.e. for t < 1) it uses the emperical observation that the 
    specific heat deecreases apprximately as $t^3$.

    Units:  units J / K / nm$^3$

    Parameters
    ----------
    t : float, int, numpy.ndarray
        Reduced temperature, $T/T_c$.
    p : float, int, numpy.ndarray
        Pressure in bar.

    If both t and p are arrays, then C_V is 2D array.

    Returns
    -------
    float or numpy.ndarray
        Specific heat at temperature $t$ and pressure p.

    """
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)
    
    c_v = np.outer(t**3, C_V_normal(1,p) + delta_C_V_Tc(p, phase) )
        
    c_v[t>1,:] = C_V_normal(t[t>1], p, squeeze_me=False)

    # c_v = np.squeeze(c_v)
    
    # try:
    #     c_v = float(c_v)
    # except:
    #     pass
    
    return h3p.squeeze_float(c_v)

# ===============================
# Coefficients from Table IV
# ===============================

# Eq. (7): C_V = sum_{i=1..4} sum_{j=0..3} a_ij * V^j * T^{-i}, valid below ~0.1 K

T_pow_a = [1, 3, 4, 5]

a = [    [-2.9190414,  5.2893401e2,  -1.8869641e4,  2.6031315e5],   # Tpow=1    
       [-2.4752597e3, 1.8377260e5, -3.4946553e6,  0.0],           # Tpow=3   
       [ 3.8887481e4, -2.8649769e6, 5.2526785e7,  0.0],           # Tpow=4    
       [-1.7505655e5, 1.2809001e7, -2.3037701e8,  0.0]            # Tpow=5
       ]

# Eq. (8): C_V = sum_{i=0..3} sum_{j=0..3} b_ij * V^j * T^{-i}
#             + exp( -T^{-1} * sum_{j=0..3} d_j * V^j ) * sum_{i=1..3} sum_{j=0..3} c_ij * V^j * T^{-i}

b = [[-6.5521193e-2,  1.3502371e-2, 0.0, 0.0],       # j=0    
       [ 4.1359033e-2,  3.8233755e-4, -5.3468396e-5,  0.0 ],       # j=1    
       [ 5.7976786e-3, -6.5611532e-4, 1.2689707e-5,  0.0],       # j=2    
       [-3.8374623e-4,  3.2072581e-5,  -5.3038906e-7,  0.0]        # j=3
       ]

c = [    [-2.5482958e1,  1.6416936e0,  -1.5110378e-2],   # j=0    
      [ 3.7882751e1, -2.8769188e0,   3.5751181e-2],   # j=1    
      [ 2.4412956e1, -2.4244083e0,   6.7775905e-2]   # j=2 
      ]

d = [-7.1613436e0,  6.0525139e-1,  -7.1295855e-3]

# ===============================
# Implementations
# ===============================

def specific_heat_low(V, T):
    """
    Eq. (7): Valid below ~0.1 K
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        Cv (float): specific heat in the same units used by the fit (per mole)
    """
    
    Cv = 0.0
    for i in range(4):          
        Ti = T ** T_pow_a[i]
        Vpow = 1.0
        for j in range(4):      # j=0..3
            Cv += a[i][j] * Vpow * Ti
            Vpow /= V
    return Cv

def specific_heat_high(V, T):
    """
    Eq. (8): Valid at/above ~0.1 K
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        Cv (float): specific heat in the same units used by the fit (per mole)
    """
    # if T <= 0:
    #     raise ValueError("Temperature T must be positive.")
    # First sum over b_ij terms
    Cv = 0.0
    for i in range(4):          # i=0..3
        Ti = T ** (-i)
        Vpow = 1.0
        for j in range(4):
            Cv += b[i][j] * Vpow * Ti
            Vpow *= V

    # Exponential argument: -T^{-1} * sum_j d_j V^j
    # Clip to avoid overflow in exp()
    exp_arg_sum = 0.0
    Vpow = 1.0
    for j in range(3):
        exp_arg_sum += d[j] * Vpow
        Vpow *= V
    exp_arg = -(exp_arg_sum) / T
    # exp_arg = max(min(exp_arg, 700.0), -700.0)
    exp_factor = np.exp(exp_arg)

    # Second sum over c_ij terms (i=1..3)
    second_sum = 0.0
    for i in range(3):          # i=0..2 corresponds to i=1..3
        Ti = T ** (-(i + 1))
        Vpow = 1.0
        for j in range(3):
            second_sum += c[i][j] * Vpow * Ti
            Vpow *= V

    Cv += exp_factor * second_sum
    return Cv

# ===============================
# Combined convenience function
# ===============================

def specific_heat_Greywall(V, T):
    """
    Selects Eq. (7) for T < 0.1 K and Eq. (8) for T >= 0.1 K.
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        Cv (float): specific heat (per mole) from the empirical fit.
    """
    T = np.atleast_1d(T)
    low_T = T < 0.1
    Cv = np.zeros_like(T)
    Cv[low_T] = specific_heat_low(V, T[low_T])  
    Cv[~low_T] = specific_heat_high(V, T[~low_T])
    return Cv

def specific_heat_normal_liquid(T, p, units='default', T_K_lowest=0.007):
    r"""
    Selects Eq. (7) for T < 0.1 K and Eq. (8) for T >= 0.1 K.
    Inputs:
        T (float): temperature (K)
        p : pressure (bar)
    Returns:
        Cv (float): volumetric specific heat from the empirical fit.
        
        Units:
            
        default: J /  K nm^3
        SI: J / K m^3
        R: divided by gas constant R
        dimless: J/K in $k_B$, length unit $\xi_{GL}(0)$

    """
    T = np.atleast_1d(T)
    vlow_T = T < T_K_lowest
    low_T = (0.007 <= T) & (T <= 0.1)
    high_T = 0.1 < T
    Cv = np.zeros_like(T)
    V = h3p.molar_vol_cm3(p)
    V_m3 = V * (1e-2)**3
    if np.any(vlow_T):
        Cv[vlow_T] = C_V_normal(T[vlow_T]/h3p.Tc_K(p), p) * (h3c.N_A/h3p.npart(0))/h3c.R
    if np.any(low_T):
        Cv[low_T] = specific_heat_low(V, T[low_T])  
    if np.any(high_T):
        Cv[high_T] = specific_heat_high(V, T[high_T])
        
    if units == 'R':
        factor = 1.0
    elif units == 'SI':
        factor = h3c.R / V_m3
    elif units == 'dimless':
        factor = (h3c.R/h3c.kB) / V_m3 * (h3p.xi(0, p)*1e-9)**3
    else: # he3_tools uses J / K / nm^3
        factor = h3c.R / V_m3 * (1e-9)**3
            
    return Cv*factor