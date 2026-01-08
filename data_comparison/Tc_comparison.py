#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:39:18 2025

Comparing $T_c$ data

- Greywall 1986 Table VI
- Also, computed from density of states and heat capacity
- Regan, Wiman, Sauls 2019 Table 1

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import he3_tools as he3

#%% Tableau colorblind 10 palette
good_col_list = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
plt.rc('axes', prop_cycle=cycler('color', good_col_list))

#%%

fig_Tc, ax_Tc = plt.subplots()

d_gre86 = he3.data_Gre86_expt_data_df
d_rws19 = he3.data_RWS19_mat_pars_df

xlabel_gre86_str = d_gre86.columns[4]
ylabel_gre86_str = d_gre86.columns[1]

p_expt = d_gre86.iloc[4:,1]
Tc_expt = d_gre86.iloc[4:,4]

p_arr = np.linspace(0.0, he3.p_A_bar, 50)
Tc_poly = he3.Tc_mK_expt(p_arr)

ax_Tc.plot(p_expt, Tc_expt - he3.Tc_mK_expt(p_expt), label='Greywall 1986, table II', ls='', marker='.')

label_rws19_str = 'RWS 2019, table 1'
ax_Tc.plot(d_rws19.iloc[:,0], d_rws19.iloc[:,4] - he3.Tc_mK_expt(d_rws19.iloc[:,0]), label=label_rws19_str, ls='', marker='.')

ax_Tc.set_xlabel(r'$p$/bar')
ax_Tc.set_ylabel(r'$\Delta T_c$/mK')

ax_Tc.set_ylim(-0.002, 0.002)
ax_Tc.set_xlim(0, 35)
ax_Tc.legend()

ax_Tc.grid(True)
ax_Tc.set_title(r'$T_c$: comparison to Greywall 1986 polynomials')

fig_Tc.tight_layout()

