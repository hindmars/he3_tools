#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:03:31 2025

@author: hindmars with assistace of MicrdSoft CoPilot
"""


# import math
import numpy as np
import he3_tools as he3

he3.set_default("DEFAULT_T_SCALE", 'Greywall')

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

def specific_heat_normal_liquid(T, p, units='default', T_K_lowest= 0.007):
    """
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

    """
    T = np.atleast_1d(T)
    vlow_T = T < T_K_lowest
    low_T = (0.007 <= T) & (T <= 0.1)
    high_T = 0.1 < T
    Cv = np.zeros_like(T)
    V = he3.molar_vol_cm3(p)
    V_m3 = V * (1e-2)**3
    if np.any(vlow_T):
        Cv[vlow_T] = he3.C_V_normal(T[vlow_T]/he3.Tc_K(p), p) * (he3.N_A/he3.npart(0))/he3.R
    if np.any(low_T):
        Cv[low_T] = specific_heat_low(V, T[low_T])  
    if np.any(high_T):
        Cv[high_T] = specific_heat_high(V, T[high_T])
        
    if units == 'R':
        factor = 1.0
    elif units == 'SI':
        factor = he3.c.R / V_m3
    else: # he3_tools uses J / K / nm^3
        factor = he3.c.R / V_m3 * (1e-9)**3
            
    return Cv*factor