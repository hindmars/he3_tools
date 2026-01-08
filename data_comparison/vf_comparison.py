#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:39:18 2025

Comparing $v_F$ (Fermi velocity) data

- Greywall 1986, computed from density of states and heat capacity polynomials
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

fig_vf, ax_vf = plt.subplots()

d_rws19 = he3.data_RWS19_mat_pars_df

xlabel_gre86_str = d_rws19.columns[0]
ylabel_gre86_str = d_rws19.columns[5]

p_arr = np.linspace(0.0, he3.p_A_bar, 50)
vf_poly = he3.vf_from_pf_mstar_m(p_arr)

label_rws19_str = 'RWS 2019, table 1'
ax_vf.plot(d_rws19.iloc[:,0], d_rws19.iloc[:,4] - he3.Tc_mK_expt(d_rws19.iloc[:,0]), label=label_rws19_str, ls='', marker='.')

ax_vf.set_xlabel(r'$p$/bar')
ax_vf.set_ylabel(r'$\Delta v_F$/mK')

ax_vf.set_ylim(-0.002, 0.002)
ax_vf.set_xlim(0, 35)
ax_vf.legend()

ax_vf.grid(True)
title_str = r'$v_F$: comparison to Greywall 1986 polynomials ($V$, $C/RT$)'
title_str += '\n'
title_str += r'$v_F = p_F/m^*$, $p_F$ from $V$, $m^*$ from $N(0)/p_F$, $N(0)$ from $V$ and $C/RT$'
ax_vf.set_title(title_str)

fig_vf.tight_layout()

