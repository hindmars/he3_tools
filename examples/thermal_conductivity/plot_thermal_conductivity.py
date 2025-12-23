#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:44:28 2025

Test plot of thermal conductivity. Tries to reproduce Figs 6 and 12 of 
Greywall 1984, Phys Rev B, using Table III and Equations 6 and 7.

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

import he3_tools as he3
# import he3_thermal_conductivity as k

#%%

V_arr = np.array([36.68, 32.83, 30.33, 27.65, 26.27])

def p_from_molar_vol_cm3(V_in):
    
    V_in = np.atleast_1d(V_in)
    
    p_out = []
    for V in V_in:
        p_out.append(spo.fsolve(lambda p: he3.molar_vol_cm3(p) - V, 10))
        
    return np.squeeze(np.array(p_out))


p_arr = p_from_molar_vol_cm3(V_arr)

print('Pressure equivalents', p_arr)

#%%

fig, ax = plt.subplots(figsize=(4, 5))

for p, V in zip(p_arr, V_arr):
    T_arr = np.logspace(np.log10(he3.Tc_K(p)), 0.0, 500)
    ax.loglog(T_arr, he3.thermal_conductivity_normal_liquid(T_arr, p, units='cgs', T_K_lowest=1e-3), label = f'{V:.2f}')

ax.set_xlabel(r'$T/$K')
ax.set_ylabel(r'$\kappa/$ erg s$^{-1}$ cm$^{-1}$ K$^{-1}$')

ax.grid()
ax.legend(title=r'$V$ [cm$^3$/mol]')

ax.set_xlim(1e-3, 1.0)
# ax.set_xlim(40/1000, 60/1000)

ax.set_ylim(2e2, 3e4)

# ax.axvline(0.05, ls='--')

#%% 

fig_low, ax_low = plt.subplots(figsize=(4, 5))

T_arr = np.linspace(0.007, 0.06, 50)

for p, V in zip(p_arr, V_arr):
    ax_low.plot(T_arr*1e3, 100/(he3.thermal_conductivity_normal_liquid(T_arr, p, units='cgs') * T_arr), label = f'{V:.2f}')

ax_low.set_xlabel(r'$T/$mK')
ax_low.set_ylabel(r'$1/\kappa T$ cm s erg$^{-1}$')

ax_low.set_xlim(0, 60)
ax_low.set_ylim(0.0, 12.0)

ax_low.grid()
ax_low.legend(title=r'$V$ [cm$^3$/mol]')
