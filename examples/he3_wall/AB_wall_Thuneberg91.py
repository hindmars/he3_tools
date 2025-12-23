#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:43:44 2022

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt
import he3_tools as h
import he3_tools.he3_wall as hw



#%%
h.set_default("DEFAULT_SC_CORRS", "SS81-interp")

surface_energy_thune = []

N, L_xi = (400, 20)
p_array_short = np.linspace(28.5,34,3)

for p in p_array_short[::-1]:
    t = h.tAB(p)
    # print(t, p, L_xi0 * h.xi(0,p))
    # A, pot, gr = hw.krylov_bubble(t, p, gr_pars=(N, L_xi*h.xi(t,p)), dim=1)
    Apg, sigma_AB, sigma_tot  = hw.get_wall(t, p, L_xi * h.xi(t,p), N=N, right_phase='B', f_tol=1e-6)
    # A, pot, gr = hw.krylov_bubble(h.alpha_norm(t), h.beta_norm_asarray(t,p), gr_pars=(500,125), dim=1)
    
    # eden, eden_grad, eden_pot = hw.energy_density(A, pot, gr)
    
    # sigma_AB = hw.surface_energy(A,pot,gr)*h.xi(0,p)/(abs(pot.mat_pars.f_B_norm())*h.xi(t,p))
    print(t, p, sigma_AB)
    surface_energy_thune.append(sigma_AB)
    # surface_energy_thune.append(hw.thuneberg_formula(t, p)) #/(-h.f_B_norm(t, p)*h.xi(t,p)))

#%% 

hw.plot_wall(*Apg)

#%%

h.set_default("DEFAULT_SC_CORRS", "RWS19")

surface_energy_us = []

p_array = np.linspace(22,34,6)

for p in p_array:
    t = h.tAB(p)
    # print(t, p, L_xi0 * h.xi(0,p))
    # A, pot, gr = hw.krylov_bubble(t, p, gr_pars=(N, L_xi*h.xi(t,p)), dim=1)
    Apg, sigma_AB, sigma_tot  = hw.get_wall(t, p, L_xi * h.xi(t,p), N=N, right_phase='B', f_tol=1e-6)
    # A, pot, gr = hw.krylov_bubble(h.alpha_norm(t), h.beta_norm_asarray(t,p), gr_pars=(500,125), dim=1)
    
    pot = Apg[1]
    
    # eden, eden_grad, eden_pot = hw.energy_density(A, pot, gr)
    
    beta_norm = pot.mat_pars.beta_arr/h.beta_const
    # sigma_AB = hw.surface_energy(A,pot,gr)*h.xi(0,p)/(abs(pot.mat_pars.f_B_norm())*h.xi(t,p))
    print(f'{t:.3f}, {p:.1f}, {sigma_AB:.4f}, ' + ', '.join(f'{b:.3f}' for b in beta_norm ) )
    surface_energy_us.append(sigma_AB)
    # surface_energy_thune.append(hw.thuneberg_formula(t, p)) #/(-h.f_B_norm(t, p)*h.xi(t,p)))


#%%

surface_energy_us = np.array(surface_energy_us)
surface_energy_thune = np.array(surface_energy_thune)

plt.figure(figsize=(5,3))

plt.plot(p_array_short, surface_energy_thune, label=r'Sauls-Serene $\beta_a$ (Thuneberg 1991)')
plt.plot(p_array, surface_energy_us, label=r'RWS19 $\beta_a$')
plt.xlabel(r'$p$/bar')
plt.ylabel(r'$\sigma_{AB}/|f_B|\xi_{GL}(T)$')
plt.grid(True)
plt.xlim(22,34)
plt.ylim(0.8,1.0)
title_str = "AB interface surface surface energy"
# title_str +=  '$N={}$, $L/\xi_{{\rm GL}}(T)={}$'.format(N,L_xi)
title_str += '\n'
title_str += r'Osheroff, Cross 1977: $\sigma_{AB}/|f_B|\xi_{GL}(T) \simeq 0.9$'

plt.title(title_str)
plt.legend()
plt.tight_layout()

savefig=True

if savefig:
    plt.savefig('ABsurface_energy.pdf')
