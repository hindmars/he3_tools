#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:44:28 2025

Test plot of specific heat, trying to reproduce Fig 9 of Greywall 1983.

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

import he3_tools as he3
# import he3_specific_heat as c

#%%

# V_arr = np.array([36.68, 32.83, 30.33, 27.65, 26.27])
V_arr = np.array([36.82, 32.59, 30.39, 28.89, 27.70, 26.84, 26.17])

def p_from_molar_vol_cm3(V_in):
    
    V_in = np.atleast_1d(V_in)
    
    p_out = []
    for V in V_in:
        p_out.append(spo.fsolve(lambda p: he3.molar_vol_cm3(p) - V, 10))
        
    return np.squeeze(np.array(p_out))


p_arr = p_from_molar_vol_cm3(V_arr)

print(p_arr)

#%%

fig, ax = plt.subplots(figsize=(5, 5))

for p, V in zip(p_arr, V_arr):
    T_arr = np.linspace(he3.Tc_K(p), 2.5, 100)
    ax.plot(T_arr, he3.specific_heat_normal_liquid(T_arr, p, units='R', T_K_lowest=1e-3), label = f'{V:.2f}')

ax.set_xlabel(r'$T/$K')
ax.set_ylabel(r'$C_V/R$')
# ax.set_ylabel(r'$C_V$')

ax.grid()
ax.legend(title=r'$V$ [cm$^3$/mol]')

ax.set_xlim(0.0, 2.5)
ax.set_ylim(0.2, 0.9)

