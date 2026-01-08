#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 5 11:44:28 2025

Test plot of heat capacity, trying to reproduce Fig 9 of Greywall 1983 and 
Fig 15 of Greywall 1986.

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

import he3_tools as he3

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
    ax.plot(T_arr, he3.heat_capacity_normal_liquid(T_arr, p, units='R', T_K_lowest=1e-3), label = f'{V:.2f}')

ax.set_xlabel(r'$T/$K')
ax.set_ylabel(r'$C_V/R$')
# ax.set_ylabel(r'$C_V$')

ax.grid()
ax.legend(title=r'$V$ [cm$^3$/mol]')

ax.set_xlim(0.0, 2.5)
ax.set_ylim(0.2, 0.9)

#%% Normal phase heat capacity near Tc

fig_lo, ax_lo = plt.subplots(figsize=(4,6))

p_expt = he3.data_Gre86_expt_data_df['$P\ (\mathrm{bar})$']
Cv_RT_expt = he3.data_Gre86_expt_data_df['$C_n / RT$ (K$^{-1}$)']
Tc_expt = he3.data_Gre86_expt_data_df['$T_c$ (mK)']

for p, Cv_RT in zip(p_expt, Cv_RT_expt):
    T_arr = np.linspace(he3.Tc_mK(p), 5.5, 2)
    # Cv_R = he3.heat_capacity_normal_liquid(T_arr*1e-3, p, 'R')
    # Cv_RT = Cv_R/(T_arr*1e-3)
    ax_lo.plot(T_arr, Cv_RT*np.ones_like(T_arr), ls='--', c='k')
    ax_lo.text(T_arr[-1]+0.25, Cv_RT - 0.02, rf'{p:.2f}', fontsize='small')

ax_lo.plot(Tc_expt, Cv_RT_expt, ls='', c='k', marker='.', label='Table II')


p_arr = np.linspace(p_expt.iloc[0], p_expt.iloc[-1], 20)
Tc_arr = he3.Tc_mK(p_arr)
# Cv_Tc = he3.heat_capacity_normal_liquid(he3.Tc_mK(p_arr)*1e-3, p_arr,'R')
ax_lo.plot(Tc_arr, he3.C_RT_poly_Greywall(p_arr), ls='--', c='k', label='Eqs 5, 17')

ax_lo.set_xlim(0, 7)
ax_lo.set_ylim(2.6, 4.8)


ax_lo.set_xlabel(r'$T/$mK')
ax_lo.set_ylabel(r'$C_V/RT$ [K$^{-1}$]')
_ = ax_lo.set_yticks(np.arange(2.6, 4.801, 0.2))

ax_lo.grid(True)
ax_lo.legend(ncols=2)

ax_lo.set_title(r'$^3$He heat capacity, Greywall 1986 Fig. 15')