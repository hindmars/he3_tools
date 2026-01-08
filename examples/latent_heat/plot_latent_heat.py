#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 13:22:06 2025

Latent heat checker

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt

import he3_tools as he3

from he3_tools import he3_thermo as the3


#%%

p_arr = np.linspace(he3.p_pcp_bar, 34, 100)

fig, ax = plt.subplots(figsize=(5,5))

p_expt = he3.data_Gre86_lat_heat_df.iloc[:,0]
LAB_expt = he3.data_Gre86_lat_heat_df.iloc[:,1]
dLAB_expt = np.array((LAB_expt - he3.data_Gre86_lat_heat_df.iloc[:,2], he3.data_Gre86_lat_heat_df.iloc[:,3] - LAB_expt))

p_arr = np.linspace(he3.p_pcp_bar,34)
c_scale =  np.pi**2 * (2*he3.N0(p_arr)) * he3.kB**2 *(he3.Tc_mK(p_arr)*1e-3) / 3 

ax.errorbar(p_expt, LAB_expt, dLAB_expt, ls='', marker='.', label='Greywall 1986 Fig 14')

# Why this factor 0.5?
LAB = 0.5*the3.latent_heat_norm(p_arr) * c_scale * he3.Tc_K(p_arr) * he3.N_A/he3.npart(p_arr)* 1e6
ax.plot(p_arr, LAB, label=r'Greywall 1986 $\frac{T_{AB}}{T_c}\int^{T_c}_{T_{AB}}\frac{dT}{T} (C_B - C_A)$')

ax.set_xlabel(r'$p$/bar')
ax.set_ylabel(r'$L_{AB}$/$\,\mu$J mol$^{-1}$')

ax.set_xlim(20, 36)
ax.set_ylim(0, 1.7)

ax.legend()
ax.grid(True)

#%% Strength of transition, cosmological definition

p_arr = np.linspace(he3.p_pcp_bar, 34, 100)

fig_th, ax_th = plt.subplots()

p_arr = np.linspace(he3.p_pcp_bar,34)
t_fix = he3.tAB(p_arr)
# c_scale =  np.pi**2 * (2*he3.N0(p_arr)) * he3.kB**2 *(he3.Tc_mK(p_arr)*1e-3) / 3 

# Why this factor 0.5?
LAB = 0.5 * the3.latent_heat_norm(p_arr) 
wA = the3.enthalpy_density_norm(t_fix, p_arr, 'A', diagonal=True)
ax_th.plot(p_arr, LAB/(3*wA) )

for t_tAB in np.arange(0.9,0.99,0.02):
    alpha_AB = the3.alpha_pt_AB_norm(t_tAB, p_arr, diagonal=True)
    ax_th.plot(p_arr, alpha_AB, label=f'{t_tAB:.2f}')

ax_th.set_xlabel(r'$p$/bar')
ax_th.set_ylabel(r'$L_{AB}/3 w_A(T_{AB})$')

ax_th.set_xlim(20, 36)
# ax_th.set_ylim(0, 1e-2)

ax_th.grid(True)
ax_th.legend()
ax_th.set_title('Cosmological transition strength of AB transition.')
