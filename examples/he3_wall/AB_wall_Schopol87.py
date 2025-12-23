#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:54:21 2022

He3 AB boundary - reproducing Schopol 1987.

@author: hindmars
"""


import numpy as np
import matplotlib.pyplot as plt
import he3_tools as h
import he3_tools.he3_wall as hw

import pandas as pd

#%%
al = -1.0

# Schopol values 1 - note beta normalisation  Eq 1a, 2, and text above, 
# where beta_hat = beta/( 3*beta345 + beta12).
beta_hat_arr_1 = np.array([-0.288, 0.513, 0.504, 0.464, -0.643])
# Schopol values 2 
beta_hat_arr_2 = np.array([-0.230, 0.461, 0.434, 0.384, -0.511])

# "times" for relaxation and recording of solution
tau_max = 4.0
tau_eval = np.linspace(0, tau_max, 11)

# arbitrary t and p
p = 28
t = h.tAB(p)

# In our equations, we have 2*beta appearing in quartic terms, 
# so we divide by 2 here.
Asols, pot, gr = hw.relax(tau_eval, al, beta_hat_arr_1/2, gr_pars=(384, 24), dim=1)

#%%
## This would be the ratio, but Schopohl makes an approximation for xi
# xi_ratio =  h.xi(t,p)/h.xi(0,p)

# Length unit as quoted in footnote 9
xi_sch = np.sqrt(3/5) * 0.13 * h.hbar*h.vf(p)/(h.kB*h.Tc_K(p)) * (1 - t)**(-0.5)
xi_ratio = xi_sch/(h.xi(0,p)*1e-9)

sigma_AB_arr = np.zeros_like(tau_eval)
for n, tau in enumerate(tau_eval):
    A = Asols[:,:,:,n]
    # Free energy functional has a factor (3*beta345 + beta12)/2 taken out
    f_B_norm_scho = 2.0*h.f_B_norm(t, p)
    sigma_AB_arr[n] = hw.surface_energy(A, pot, gr)/(xi_ratio * np.abs(f_B_norm_scho))

    print(f'tau={tau:.1f}, Surface energy: {sigma_AB_arr[n]:}')

A = Asols[:,:,:,-1]

#%%

sch_com_list = ['Uxx', 'Uyy', 'Uzz', 'Vzx', 'Vxz']

com2aj_list = [(1,1), (2,2), (3,3), (1,2), (2,1)]
com2RI_list = ['Re']*3 + ['Im']*2
operator_list = [np.real]*3 + [np.imag]*2

com2aj_dict = dict(zip(sch_com_list, com2aj_list))
com2RI_dict = dict(zip(sch_com_list, com2RI_list))
op_dict = dict(zip(sch_com_list, operator_list))

# Read in digitised figure 1
filename_stem = 'Sch87_fig1_'
df_sch_dict = {}

for sch_com in sch_com_list:
    df_sch_dict[sch_com] = pd.read_csv(filename_stem + sch_com + '.csv')
    df_sch_dict[sch_com].columns=['x', sch_com]

# Plot Schopol and he3_tools solution

fig_sch, ax_sch = plt.subplots()

for sch_com in sch_com_list:
    x = df_sch_dict[sch_com]['x']
    y = df_sch_dict[sch_com][sch_com]
    line = ax_sch.plot(x, y, label=sch_com)
    a, i = com2aj_dict[sch_com]
    x_ht = gr.x - np.max(gr.x)/2
    ax_sch.plot(x_ht, op_dict[sch_com](A[:,a-1,i-1]), 
                c=line[0].get_c(), 
                ls='--', 
                label=com2RI_dict[sch_com] + rf' $A_{{{a:}{i:}}}$' )

ax_sch.set_xlabel(r'$x$')
ax_sch.set_ylabel(r'$A_{\alpha i}$')
ax_sch.legend()
ax_sch.grid(True)

title_str = 'Schopohl 1987 (solid), he3_tools (dashed)' 
title_str += '\n'
title_str += rf'cool time {tau_max:}, $\sigma_{{AB}}/\xi_{{\rm GL}} |f_B| = {sigma_AB_arr[n]:.3f}$'
ax_sch.set_title(title_str, fontsize='medium')
