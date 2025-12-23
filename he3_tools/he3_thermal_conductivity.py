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
import he3_tools.he3_constants as h3c
# import he3_tools.he3_specific_heat as h3sh


# ===============================
# mK temperatures
# ===============================


def kappa_0(t, p):
    """
    Gas-kinetic expression for thermal conductivity, V&W eqn 2.40.  
    
    Units: J / ns / nm / K

    Parameters
    ----------
    t : float, int, numpy.ndarray
        Reduced temperature, $T/T_c$.
    p : float, int, numpy.ndarray
        Pressure in bar.

    Only one or other of t and p can be an array.

    Returns
    -------
    float or numpy.ndarray
        Gas-kinetic thermal conductivity at temperature $t$ and pressure p.

    """
    # C_V is in J / K / nm^3, vf is in m/s (also nm/ns), tau_qp is in seconds
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)

    vf2 = (h3p.vf(p))**2 
    k0 = (1e9/3) * h3p.C_V_normal(t, p) * h3p.tau_qp(t, p) * vf2[None, :]

    k0 = np.squeeze(k0)

    try:
        k0 = float(k0)
    except:
        pass
    # return (1/3) * C_V_normal(t, p) * (vf(p))**2 * tau_qp(t, p) * 1e9
    return k0


def kappa(t, p):
    """
    Experimental low temperature thermal conductivity, Greywall 1984  
    
    Units:    J / ns / nm / K

    Parameters
    ----------
    t : float, int, numpy.ndarray
        Reduced temperature, $T/T_c$.
    p : float, int, numpy.ndarray
        Pressure in bar.

    Only one or other of t and p can be an array.

    Returns
    -------
    float or numpy.ndarray
        Extrapolated measured thermal conductivity at temperature $t$ and pressure p.

    """
    t = np.atleast_1d(t)
    p = np.atleast_1d(p)

    p_data = h3d.data_Gre84_therm_cond[:, 0]
    kappaT_data = h3d.data_Gre84_therm_cond[:, 4]
    b_data = h3d.data_Gre84_therm_cond[:, 6]
    
    
    kappaT0_interp = np.interp(p, p_data, kappaT_data) 
    a = 1/kappaT0_interp
    b = np.interp(p, p_data, b_data) 

    if h3p.get_setting('DEFAULT_T_SCALE') != 'Greywall':
        tmp_set = h3p.get_setting('DEFAULT_T_SCALE')
        h3p.set_default('DEFAULT_T_SCALE', 'Greywall')
        T = h3p.Tc_K(p)*t
        h3p.set_default('DEFAULT_T_SCALE', tmp_set)
    else:
        T = h3p.Tc_K(p)*t

    kappaT_interp = 1/(a + b*T)

    k = kappaT_interp[None, :] / T

    k = np.squeeze(k)
    
    try:
        k = float(k)
    except:
        pass
    # print(p, T, kappaT_interp)

    # Greywall is in erg/sec/cm, convert to J/ns/nm 
    return k * 1e-7/(1e9 * 1e9 * 1e-2)


# ===============================
# Coefficients from Table III
# ===============================

a = h3d.a_Gre84_table3
b = h3d.b_Gre84_table3

T_pow_a = [0, 1, 2, 3]
T_pow_b = [-2, -1, 0, 1, 2]
# T in K above which to switch functions from low to high
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
    Thermal conductivity from Greywall 1984.
    Selects Eq. (6) for T < 0.05 K and Eq. 7) for T >= 0.05 K.
    Inputs:
        T (float): temperature (K)
        p : pressure (bar)
    units:
        default: J / ns nm K
        SI: J / s m K
        cgs: erg/sec cm K
        dimless: J / K in k_B, length unit xlGL(0), time unit xiGL(0)/vF
    T_K_lowest: (K) switch to theoretical below this temp

    Returns:
        kappa (float): thermal conductivity in chosen units

    """
    T = np.atleast_1d(T)
    vlow_T = T < T_K_lowest
    low_T = (T_K_lowest <= T) & (T < T_thresh)
    high_T = T_thresh <= T
    kappa_arr = np.zeros_like(T)
    V = h3p.molar_vol_cm3(p)
    if np.any(vlow_T):
        kappa_arr[vlow_T] = kappa(T[vlow_T]/h3p.Tc_K(p), p)*1e18 / conv_erg_sec_cm_K_SI
    if np.any(low_T):
        kappa_arr[low_T] = thermal_conductivity_low(V, T[low_T])  
    if np.any(high_T):
        kappa_arr[high_T] = thermal_conductivity_high(V, T[high_T])
        
    if units == 'SI':
        factor = conv_erg_sec_cm_K_SI
    elif units == 'cgs':
        factor = 1.0
    elif units == 'dimless':
        factor = conv_erg_sec_cm_K_SI/h3c.kB * (h3p.xi(0,p)*1e-9)**2 / h3p.vf(p)
    else:
        factor = conv_erg_sec_cm_K_SI * 1e-18
        
    return kappa_arr*factor