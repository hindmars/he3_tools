#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 18:23:10 2022

He3 phase boundary solutions

@author: hindmars
"""

import numpy as np
import he3_tools as h
import he3_tools.he3_wall as hw
import pickle as pkl

h.set_default("DEFAULT_T_SCALE", "Greywall")

#%%
savefig=False
pickle = True

#%% Set pressure and rduced temperature

# p = 25.5
p = 28
# t = 1.5/h.Tc_mK(p)
t = h.tAB(p)

#%% Lattice size: grid points and physical llength

# N, L_xi = (500, 25)
N, L_xi0 = (512, 32)

#%% Get wall solution

Apg_kry, sigma_bw, sigma_tot = hw.get_wall(t, p, L_xi0 * h.xi(0,p), N=N, right_phase='B', f_tol=1e-6)

# Apg_kry is a tuple
# Apg_kry[0] is order parameter array, Apg_kry[0].shape = (N, 3, 3), complex
# Apg_kry[1] is a quartic potential object
# Apg_kry[2] is a grid obkect

#%% Plot wall

ax2 = hw.plot_wall(*Apg_kry, plot_gap=True, phase_marker=True)

title_str = ax2[0].get_title()
dx = L_xi0/N
title_str += rf' $dx = {dx:.1f}\xi_{{\rm GL}}^0$'
ax2[0].set_title(title_str, fontsize='small')
ax2[2].set_xlabel(r'$x/\xi_{{\rm GL}}^0$')

pot = Apg_kry[1]

sigma_AB = hw.surface_energy(*Apg_kry, zero_point='A')/abs(pot.mat_pars.f_B_norm() * (h.xi(t,p)/h.xi(0,p)))
w_AB, w_AB = hw.wall_width(*Apg_kry)

print(rf'NK: Surface energy: {sigma_AB:}, width: {w_AB:} xiGL0')

#%% Plot R1

x = Apg_kry[2].x
R_vec = h.R_terms(Apg_kry[0])
xiGL0 = h.xi(0,p)
# xiGL = h.xi(t,p)

ax2[2].plot((x - np.max(x)/2)/xiGL0, 4*R_vec[0]*(1 - R_vec[0]), '--', label=r'$4R_1(1-R_1)$')
ax2[2].legend(fontsize='small', bbox_to_anchor=(0.9, 0.5))

#%% Make a figure if wanted

if savefig:
    file_name = 'AB_wall_t={:.2f}_p={:.1f}.pdf'.format(t,p)
    print('print to figure',file_name)
    ax2[0].get_figure().savefig(file_name)

#%% pickle the data

if pickle:
    data_dict = {}
    
    data_dict['A'] = Apg_kry[0]
    data_dict['physics params'] = Apg_kry[1].mat_pars.__dict__
    data_dict['grid'] = Apg_kry[2].__dict__
    
    filename = f'AB_wall_data_p{p:}.pkl'
    
    with open(filename, 'wb') as f:
        pkl.dump(data_dict, f)

