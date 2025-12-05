#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:03:31 2025

@author: hindmars with assistace of MicroSoft CoPilot
"""


# import math
import numpy as np
import he3_tools.he3_props as h3p
import he3_tools.he3_data as h3d

h3p.set_default("DEFAULT_T_SCALE", 'Greywall')

# ===============================
# Coefficients from Table III
# ===============================

a = h3d.a_Gre84_table3
b = h3d.b_Gre84_table3

T_pow_a = [0, 1, 2, 3]
T_pow_b = [-2, -1, 0, 1, 2]

T_thresh = 0.05

conv_erg_sec_cm_K_SI = 1e-7 / 1e-2  

# ===============================
# Implementations
# ===============================

def thermal_conductivity_low(V, T):
    """
    Eq. (6): Valid below 50 mK
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        kappa (float): thermal conductivity in erg/sec cm K
    """
    
    kappa = 0.0
    i_sum = 0.0
    for i in range(len(T_pow_a)):          
        Ti = T ** T_pow_a[i]
        Vpow = 1.0
        j_sum = 0.0
        for j in range(3):      # j=0..2
            j_sum += a[i][j] * Vpow 
            Vpow *= V
            
        i_sum += Ti/j_sum 
    
    kappa = 1/(T * i_sum)
    return kappa

def thermal_conductivity_high(V, T):
    """
    Eq. (7): Valid at/above 50 mK
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        kappa (float): thermal conductivity in erg/sec cm K
    """
    # if T <= 0:
    #     raise ValueError("Temperature T must be positive.")
    # First sum over b_ij terms
    kappa = 0.0
    for i in range(len(T_pow_b)):          # i=0..3
        Ti = T ** T_pow_b[i]
        Vpow = 1.0
        for j in range(3):
            kappa += b[i][j] * Vpow * Ti
            Vpow *= V

    return kappa

# ===============================
# Combined convenience function
# ===============================

def thermal_conductivity_Greywall84(V, T):
    """
    Selects Eq. (6) for T < 50 mK and Eq. (7) for T >= 50 mK.
    Inputs:
        V (float): molar volume (e.g., cm^3/mol)
        T (float): temperature (K)
    Returns:
        kappa (float): thermal conductivity in erg/sec cm K
    """
    T = np.atleast_1d(T)
    low_T = T < T_thresh
    kappa = np.zeros_like(T)
    kappa[low_T] = thermal_conductivity_low(V, T[low_T])  
    kappa[~low_T] = thermal_conductivity_high(V, T[~low_T])
    return kappa

def thermal_conductivity_normal_liquid(T, p, units='default', T_K_lowest= 0.007):
    """
    Selects Eq. (6) for T < 0.05 K and Eq. 7) for T >= 0.05 K.
    Inputs:
        T (float): temperature (K)
        p : pressure (bar)
    Returns:
        kappa (float): thermal conductivity in 

        default: J / ns nm K
        SI: J / s m K
        cgs: erg/sec cm K
    """
    T = np.atleast_1d(T)
    vlow_T = T < T_K_lowest
    low_T = (T_K_lowest <= T) & (T <= T_thresh)
    high_T = T_thresh < T
    kappa = np.zeros_like(T)
    V = h3p.molar_vol_cm3(p)
    if np.any(vlow_T):
        kappa[vlow_T] = h3p.kappa(T[vlow_T]/h3p.Tc_K(p), p)*1e18 / conv_erg_sec_cm_K_SI
    if np.any(low_T):
        kappa[low_T] = thermal_conductivity_low(V, T[low_T])  
    if np.any(high_T):
        kappa[high_T] = thermal_conductivity_high(V, T[high_T])
        
    if units == 'SI':
        factor = conv_erg_sec_cm_K_SI
    elif units == 'cgs':
        factor = 1.0
    else:
        factor = conv_erg_sec_cm_K_SI * 1e-18
        
    return kappa*factor