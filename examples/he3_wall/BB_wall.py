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

#%%
figures_dir = '../../../AB_projects_QUEST-DMC/walls/'
savefig=False

#%%

p = 15
t = 0.8

# N, L_xi = (500, 50)
N, L_xi0 = (200, 50)
dx_xi0 = L_xi0/N

w = L_xi0 * h.xi(0,p)

B_phases = ['B'+ str(x) for x in range(1,8)]

Apg_dict = {}
ax_dict = {}

for right_phase in B_phases:
    print(right_phase)
    Apg_dict[right_phase] = hw.krylov_BB(t, p, gr_pars=(N,L_xi0), 
                           left_phase='B', 
                           right_phase=right_phase,
                           bcs = [hw.bc_neu, hw.bc_neu])

#%%

for right_phase in B_phases:
    ax_dict[right_phase] = hw.plot_wall(*Apg_dict[right_phase], 
                      real_comp_list=[(0,0), (1,1), (2,2)],
                      imag_comp_list=[], title_extra= ', B' + right_phase + rf'$dx = {dx_xi0:.1f}$', 
                      plot_gap=True,
                      phase_marker=True)

#%%



if savefig:
    data = []
    for right_phase in B_phases:
        file_name = figures_dir + 'B' + right_phase + '_wall_t={:.2f}_p={:.1f}'.format(t,p)
        print('print to figure',file_name + '.pdf')
        ax_dict[right_phase][0].get_figure().savefig(file_name + '.pdf')
        print('print to figure',file_name + '.png')
        ax_dict[right_phase][0].get_figure().savefig(file_name + '.png')
        sigma_BB = hw.surface_energy(*Apg_dict[right_phase])*h.xi(0,p)/(abs(Apg_dict[right_phase][1].mat_pars.f_B_norm())*h.xi(t,p))
        data.append([t, p, sigma_BB])
        
    np.savetxt(figures_dir + 'BB_wall_data_t={:.2f}_p={:.1f}.txt'.format(t,p), data, 
               fmt=('%.3e, %.3e, %.18e'), header='right_phase, t, p, sigma*xi0/f_B*xi')
