#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:00:42 2026

Utilities for ESTAR range and stopping power for electrons in Helium and Silicon

@author: Mark Hindmarsh
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Returns the absolute path to your package folder
data_sources_dir = BASE_DIR + os.sep + 'data_sources' + os.sep

import numpy as np
import he3_tools.he3_props as h3p

estar_range_data = {}
estar_stop_data = {}
density = {}

estar_range_data['4He'] = np.loadtxt(data_sources_dir + 'estar_range_data_helium.txt', skiprows=8)
estar_range_data['14Si'] = np.loadtxt(data_sources_dir + 'estar_range_data_silicon.txt', skiprows=8)

estar_stop_data['4He'] = np.loadtxt(data_sources_dir + 'estar_stopping_power_data_helium.txt', skiprows=8)
estar_stop_data['14Si'] = np.loadtxt(data_sources_dir + 'estar_stopping_power_data_silicon.txt', skiprows=8)

density['4He']  = lambda p : h3p.density(p) * (4/3) # approximation to density of 4He, g / cm3
density['14Si'] = lambda p : 2.329085 # g / cm3 Wikipedia, at 20 C

def estar_range(Ee_MeV, element='4He'):
    """
    Interpolator for ESTAR electron range data, for 4He and 14Si.
    
    For energies below 0.01 MeV, makes an E^2 approximation matched to the 
    ESTAR value at 0.01 MeV
    
    Return in units of cm2 / g
    """
    scalar = np.isscalar(Ee_MeV)
    Ee_MeV = np.atleast_1d(Ee_MeV)
    
    E = estar_range_data[element][:,0] # g / cm2
    Emin = E[0]
    estar_range_cm2_g = estar_range_data[element][:,1]
    estar_range_cm2_g_minE = estar_range_cm2_g[0]

    range_cm2_g = np.interp(Ee_MeV, E, estar_range_cm2_g, left = 0.0)
    if np.any(Ee_MeV < Emin):
        low_Ee = Ee_MeV < Emin
        range_cm2_g[low_Ee] = estar_range_cm2_g_minE*(Ee_MeV[low_Ee]/Emin)**2

    if scalar:
        range_cm2_g = float(range_cm2_g)
    return range_cm2_g

def estar_stopping_power(Ee_MeV, element='4He'):
    """
    Interpolator for ESTAR electron stopping power data, for 4He and 14Si.
    
    For energies below 0.001 MeV, makes a 1/E^2 approximation matched to the 
    ESTAR value at 0.001 MeV

    Return in units of MeV g / cm2"""

    scalar = np.isscalar(Ee_MeV)
    Ee_MeV = np.atleast_1d(Ee_MeV)
    
    E = estar_stop_data[element][:,0] # g / cm2
    Emin = E[0]
    estar_stop_MeVg_cm2 = estar_stop_data[element][:,1]
    estar_stop_MeVg_cm2_minE = estar_stop_MeVg_cm2[0]

    stop_MeVg_cm2 = np.interp(Ee_MeV, E, estar_stop_MeVg_cm2, left = 0.0)
    if np.any(Ee_MeV < Emin):
        low_Ee = Ee_MeV < Emin
        stop_MeVg_cm2[low_Ee] = estar_stop_MeVg_cm2_minE*(Emin/Ee_MeV[low_Ee])

    if scalar:
        stop_MeVg_cm2 = float(stop_MeVg_cm2)
    return stop_MeVg_cm2


def e_stop_dist(Ee_MeV, element='4He', p = 0.0, units='cgs'):
    """
    Electron stopping distance in 4He and 14Si, using (extrapolated) ESTAR 
    data.
    
    Returns distance in cm, m, or micron
    """
    
    scalar = np.isscalar(Ee_MeV)
    Ee_MeV = np.atleast_1d(Ee_MeV)
    rho = density[element](p) # g/cm3

    dist_cm = estar_range(Ee_MeV, element=element) / rho
    
    if units == 'cm' or units == 'cgs':
        factor = 1.0
    elif units == 'SI' or units == 'm':
        factor = 1e-2
    elif units == 'um' or units == 'mu.m' or units == 'micron':
        factor = 1e4
    else:
        raise ValueError('e_stop_dist: units not recognised')
        
    if scalar:
        dist_cm = float(dist_cm)
        
    return dist_cm * factor


def e_dE_dx(Ee_MeV, element='4He', p = 0.0, units='MeV_cm'):
    """
    Electron energy loss per unit length in 4He and 14Si, using (extrapolated) ESTAR 
    data.
    
    Returns energy in MeV/cm [MeV_cm], or eV/micron [eV_um]
    """
    
    scalar = np.isscalar(Ee_MeV)
    Ee_MeV = np.atleast_1d(Ee_MeV)
    rho = density[element](p) # g/cm3

    dEdx = estar_stopping_power(Ee_MeV, element=element) * rho
    
    if units == 'MeV_cm':
        factor = 1.0
    elif units == 'eV_um':
        factor = 1e6 * 1e-4
    else:
        raise ValueError('e_stop_dist: units not recognised')

    if scalar:
        dEdx = float(dEdx)
        
    return dEdx * factor
    
    
    