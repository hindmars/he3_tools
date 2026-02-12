#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:04:41 2025

Tabulate some thermal properties at the superflui critical temperature

N phase specific heat
A phase specific heat
thermal conductivity times T^2
thermal diffusivity (conductivity divided by specific heat) in physical units
thermal diffusivity (conductivity divided by specific heat) in dimensionless units

@author: hindmars
"""

import numpy as np
import he3_tools as he3

p_arr = np.array([5.5, 9.0, 12.0, 22.0])

print(r'\begin{tabular}{c c c c c c}')
header_line = r'\parbox[b][10pt]{1cm}{$p$\\ (bar)} '
header_line += r'& \parbox[b][20pt]{3cm}{$C_V(\Tc) $\\(eV/$\mu$m$^3$/mK)} '
header_line += r'& \parbox[b][20pt]{3cm}{$C_V^A(\Tc) $\\(eV/$\mu$m$^3$/mK)} '
header_line += r'& \parbox[b][20pt]{3cm}{$\ka T $\\ (eV/ns/$\mu$m} '
header_line += r'& \parbox[b][20pt]{2.5cm}{$D_{t,c}$ \\ (10$^2$ $\mu$m$^2$/$\mu$s)} '
header_line += r'& \parbox[b][20pt]{2cm}{$\tilde{D}_{t,c} $ \\ \relax\null} '
print(header_line)
print(r' \\\hline')

for p in p_arr:
    
    table_line = rf'${p:}$'
    C_V = he3.C_V_normal(1, p)/(he3.h3c.c.e) * 1e-3 * (1e3)**3 
    table_line += rf'& ${C_V:.1f}$ '
    CA_V = he3.C_V(1, p, 'A')/(he3.h3c.c.e) * 1e-3 * (1e3)**3 
    table_line += rf'& ${CA_V:.1f}$ '
    # kappa_T2 = he3.kappa(1, p)/(he3.h3c.c.e) 
    kappa_T = (he3.kappa(1, p)/(he3.h3c.c.e) ) * he3.Tc_K(p) * 1e3
    table_line += rf'& ${kappa_T:.2f}$ '
    D = he3.thermal_diffusivity(1, p)*1e-3/100#/(he3.h3c.c.e) 
    table_line += rf'& ${D:.2f}$ '
    Dn = he3.thermal_diffusivity_norm(1, p)
    table_line += rf'& ${Dn:.1f}$ '
    if p != p_arr[-1]:
        table_line += r' \\'
    
    print(table_line)
    
print('\end{tabular}')
