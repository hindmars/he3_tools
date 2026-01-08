#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:39:18 2025

Comparing molar volume and number density data.

- Greywall 1986 Equation 2, from Wheatley 1975
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

fig_V, ax_V = plt.subplots()

p_arr = he3.p_nodes

V_gre86 = he3.molar_volume(p_arr)

d_whe75 = he3.data_Whe75_expt_prop_df
d_gre86 = he3.data_Gre86_expt_data_df

label_whe75_str = d_whe75.columns[1]
label_gre86_str = d_gre86.columns[1]

ax_V.plot(d_whe75.iloc[:,0], d_whe75.iloc[:,1] - he3.molar_volume(d_whe75.iloc[:,0]), label='Wheatley 1975', ls='', marker='.')
ax_V.plot(d_gre86.iloc[4:,1], d_gre86.iloc[4:,2] - he3.molar_volume(d_gre86.iloc[4:,1]), label='Greywall 1986, table II', ls='', marker='.')

# ax_V.plot(p_arr, V_gre86, label='Greywall 1986, polynomial (Eq. 2)')


for DEFAULT_RWS19_PATCH in he3.SET_RWS19_PATCH:
    he3.set_default('DEFAULT_RWS19_PATCH', DEFAULT_RWS19_PATCH)
    V_rws19 = he3.molar_vol_cm3(p_arr)
    d_rws19 = he3.data_RWS19_mat_pars_df
    
    label_rws19_str = 'RWS 2019, table I'
    if he3.get_setting('DEFAULT_RWS19_PATCH'):
        label_rws19_str += ', patched'
    label_rws19_str += '\n' + rf'$n(22) = {he3.npart(22):}$, $n(34) = {he3.npart(34):}$'
    
    ax_V.plot(p_arr, V_rws19 - V_gre86, label=label_rws19_str, ls='', marker='.')


ax_V.set_xlabel(r'$p$/bar')
ax_V.set_ylabel(r'$\Delta V$ [cm$^3$/mol]')

ax_V.set_xlim(0, 35)
ax_V.set_ylim(-1.25, 0.20)
ax_V.legend()

ax_V.grid(True)
ax_V.set_title('Molar volume: comparison to Greywall 1986 polynomial Eq. 2')

fig_V.tight_layout()

