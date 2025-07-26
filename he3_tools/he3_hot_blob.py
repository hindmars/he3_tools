#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:05:49 2025

Hot blob model utilities


@author: hindmars
"""

import numpy as np
import he3_tools.he3_props as h3p
import he3_tools.he3_constants as h3c
import he3_tools.he3_magnetic as h3m

def hot_blob_length_scale(p, phase='A'):
    """
    Length scale of a normal region produced by 1 eV energy, injected into 
    superfluid in given phase at zero temperature, defined as 
    $$
    L = (C_V T_c/ 1 {\rm eV})^{-1/3}
    $$
    Units: nm.
    
    Radius of hot blob would contain geometric factors from region shape, and 
    algebraic factors from integration of temperature between $0$ and $T_c$.

    Parameters
    ----------
    p : float, int, numpy.ndarray 
        Pressure in bar.

    Returns
    -------
    type(p)
        Length scale of the heated region, in nm..

    """
    C_V_Tc = h3p.C_V_normal(1, p) + h3p.delta_C_V_Tc(p, phase)
    return (h3p.Tc_mK(p)/1000 * C_V_Tc/h3c.c.e)**(-1/3)

def hot_blob_size(t, p, Q_eV, phase='A'):
    """
    Size of a normal region produced by Q_eV energy, injected into 
    superfluid in given phase at reduced temperature t. 
    
    Units: nm.
    
    Radius of hot blob would contain geometric factors from region shape, and 
    algebraic factors from integration of temperature between $0$ and $T_c$.

    Parameters
    ----------
    t : float, int, numpy.ndarray 
        Reduced temperature (T/Tc).
    p : float, int, numpy.ndarray 
        Pressure in bar.
    Q_eV : float, int, numpy.ndarray 
        Deposited heat energy in eV.
    phase : str
        Thermodynamic superfuid phase ('A' or 'B')

    Returns
    -------
    type(p)
        Length scale of the heated region, in nm..

    """
    return hot_blob_length_scale(p, phase='A') * (Q_eV/(1 - t))**(1/3)

def hot_blob_quench_time(t, p, Q_eV=1.0, phase='A'):
    """
    Quench time of a normal region produced by Q_eV energy, injected into 
    superfluid in given phase at reduced temperature t. 
    
    Units: ns.
    
    Parameters
    ----------
    t : float, int, numpy.ndarray 
        Reduced temperature (T/Tc).
    p : float, int, numpy.ndarray 
        Pressure in bar.
    Q_eV : float, int, numpy.ndarray 
        Deposited heat energy in eV.
    phase : str
        Thermodynamic superfuid phase ('A' or 'B')


    Returns
    -------
    type(p)
        Quench time for hot blob, in ns..

    """
    return hot_blob_size(p, t, Q_eV, phase='A')**2/h3p.thermal_diffusivity(t, p) *1/(1-t)
    
def hot_blob_critical_energy_eV(t, p, H=0, geo_fac=(4*np.pi)**(-1.5)):
    """
    Energy required to make a hot blob of same size as critical bubble.

    Parameters
    ----------
    t : float, int, numpy.ndarray 
        Reduced temperature (T/Tc).
    p : float, int, numpy.ndarray 
        Pressure in bar.
    geo_fac : float, optional
        Geometrical/algebraic factor. The default is (4*np.pi)**(-1.5).

    Returns
    -------
    type(t) or type(p).
        Critical energy (eV) for a hot blob to be as big as the critical bubble

    """
    
    Rc = h3m.critical_radius(t, p, H)
    
    vol = Rc**3 * (np.exp(0)/6)**1.5 * (1/geo_fac) * (1 - t)
    
    return vol * h3p.C_V_normal(1, p)*h3p.Tc_mK(p)/1e3/h3c.c.e