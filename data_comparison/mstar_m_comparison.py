#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:39:18 2025

Comparing $m_*/m$ data

- Greywall 1986 Table VI
- Also, computed from density of states and heat capacity
- Regan, Wiman, Sauls 2019 Table 1

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import pandas as pd

import he3_tools as he3

#%% Tableau colorblind 10 palette
good_col_list = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
plt.rc('axes', prop_cycle=cycler('color', good_col_list))

#%%

def N0_from_C_RT_V(p):
    """Density of states per spin calculated from low T normal phase 
    heat capacity and molar volume, as obtained from Greywall 1986 polynomials.
    
    Molar volume: Eq. 2.
    Very low T heat capacity divided by RT: Eq. 17.
    
    Units: 1/J/nm^3
    """
    C_RT = he3.C_RT_poly_Greywall(p) 
    V = he3.molar_volume(p) * 1e7**3 # nm^3 per mol
    denom = (np.pi**2/3) * he3.kB * V / he3.N_A
    # denom = (np.pi**2/3) * he3.molar_volume(p) * 1e-7**3
    return 0.5 * C_RT/denom
    
def pf_from_V(p):
    """Fermi momentum calculated from molar volume, calculated from Greywall 
    1986 polynomial.

    Molar volume: Eq. 2.
    
    Units: kg m /s """
    V = he3.molar_volume(p) * 1e-2**3 # m^3 per mol
    return he3.hbar * (2*np.pi) * (3/(8*np.pi) * he3.N_A/V)**(1/3.)

def mstar_m_from_N0_pf(p):
    """Effective mass ratio from density of states and Fermi momentum, derived 
    from Greywall 1986 polynomials for molar volume and v low T heat capacity. 
    
    Density of states: N0_from_C_RT_V
    Fermi momentum: pf_from_V
    
    Dimensionless, unitless.
    """
    N0_m3_J = N0_from_C_RT_V(p)*(1e9)**3
    pf = pf_from_V(p)
    return (he3.hbar * (2*np.pi))**3/(8*np.pi*he3.mhe3_kg) * (2*N0_m3_J/pf)
    

fig_msm, ax_msm = plt.subplots()

p_arr = he3.p_nodes

# V_gre86 = he3.molar_volume(p_arr)
# V_rws19 = he3.molar_vol_cm3(p_arr)

# d_whe75 = he3.data_Whe75_expt_prop_df
d_gre86 = he3.data_Gre86_heat_cap_df
d_rws19 = he3.data_RWS19_mat_pars_df

# label_whe75_str = d_whe75.columns[1]
label_gre86_str = d_gre86.columns[1]

msm = mstar_m_from_N0_pf(d_gre86.iloc[:,0])

# ax_msm.plot(d_whe75.iloc[:,0], d_whe75.iloc[:,1] - he3.molar_volume(d_whe75.iloc[:,0]), label='Wheatley 1975', ls='', marker='.')
ax_msm.plot(d_gre86.iloc[:,0], d_gre86.iloc[:,3] - msm, label='Greywall 1986, table VI', ls='', marker='.')

# ax_msm.plot(p_arr, V_gre86, label='Greywall 1986, polynomial (Eq. 2)')

label_rws19_str = 'RWS 2019, table 1'
# if he3.DEFAULT_RWS19_PATCH:
#     label_rws19_str += ', patched\n'
#     label_rws19_str += rf'$n(22) = {he3.npart(22):}$, $n(34) = {he3.npart(34):}$'
ax_msm.plot(d_rws19.iloc[:,0], d_rws19.iloc[:,2] - mstar_m_from_N0_pf(d_rws19.iloc[:,0]), label=label_rws19_str, ls='', marker='.')


ax_msm.set_xlabel(r'$p$/bar')
ax_msm.set_ylabel(r'$\Delta m^*/m$')

ax_msm.set_xlim(0, 35)
ax_msm.set_ylim(-0.008, 0.008)
ax_msm.legend()

ax_msm.grid(True)
ax_msm.set_title(r'$m^*/m$: comparison to Greywall 1986 polynomials')

fig_msm.tight_layout()

