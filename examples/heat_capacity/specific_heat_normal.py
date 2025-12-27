#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 13:22:06 2025

Latent heat checker

@author: hindmars
"""

import numpy as np
import matplotlib.pyplot as plt

import he3_tools as h

from he3_tools import he3_thermo as th

def gamma(p):
    """
    Computes the value of gamma = C/RT based on equation (17) of Greywall 86:
    γ(P) = a0 + a1*P + a2*P^2 + a3*P^3 + a4*P^4.
    Produced by CoPilot.
    """
    # Coefficients from the paper
    a0 = 0.27840464 * 10**1
    a1 = 0.69575243 * 10**-1
    a2 = -0.14738303 * 10**-2
    a3 = 0.46153498 * 10**-4
    a4 = -0.53785385 * 10**-6

    # Polynomial evaluation
    return a0 + a1*p + a2*p**2 + a3*p**3 + a4*p**4


#%% Normal phase specific heat

p_list = [0, 2.18, 5.21, 10.25, 14.95, 20.30, 25.31, 29.08, 33.95]

R_gas = h.N_A * h.kB

fig_cvn, ax_cvn = plt.subplots(figsize=(5,5))

# convert CV to J/K/mol
p = np.array(p_list)
cv_normal_molar = h.C_V_normal(1, p) * (h.N_A / h.npart(p) )
T_K = h.Tc_mK(p)*1e-3

gamma_this = cv_normal_molar/(R_gas*T_K)

gamma_greywall = gamma(p)

ax_cvn.plot(p, gamma_this, marker='x', label='he3_tools' )
ax_cvn.plot(p, gamma_greywall, marker='o', label='Greywall 86 poly' )

ax_cvn.set_xlim(0,35)
ax_cvn.set_ylim(2.6,4.8)
ax_cvn.set_xlabel(r'$p$/bar')
ax_cvn.set_ylabel(r'$C_V/ R T$ (K$^{-1})$')
ax_cvn.legend()
ax_cvn.grid(True)
ax_cvn.set_yticks(np.arange(2.6,4.8,0.2))

ax_cvn.set_title('Normal phase heat capacity at $T_c$')

fig_cvn.tight_layout()


