#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Sun Dec 28 10:39:18 2025

Comparing $\xi_0$ (Cooper pair) data

- Greywall 1986, computed from molar volume and heat capacity polynomials, and $T_c$
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

fig_xi0, ax_xi0 = plt.subplots()

d_rws19 = he3.data_RWS19_mat_pars_df

xlabel_gre86_str = d_rws19.columns[0]
ylabel_gre86_str = d_rws19.columns[6]

p_arr = np.linspace(0.0, he3.p_A_bar, 50)
xi0_poly = he3.xi0_from_vf_Tc(p_arr)

label_rws19_str = 'RWS 2019, table 1'
ax_xi0.plot(d_rws19.iloc[:,0], d_rws19.iloc[:,6] - he3.xi0_from_vf_Tc(d_rws19.iloc[:,0]), label=label_rws19_str, ls='', marker='.')

ax_xi0.set_xlabel(r'$p$/bar')
ax_xi0.set_ylabel(r'$\Delta \xi_0$/nm')

ax_xi0.set_ylim(-0.1, 0.1)
ax_xi0.set_xlim(0, 35)
ax_xi0.legend()

ax_xi0.grid(True)
title_str = r'$\xi_0$: comparison to Greywall 1986 polynomials ($V$, $C/RT$, $T_c$)'
title_str += '\n'
title_str += r'$\xi_0 = \hbar v_F/2\pi T_c$'
ax_xi0.set_title(title_str)

fig_xi0.tight_layout()

