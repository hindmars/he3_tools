#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:05:49 2025

Hot blob model utilities


@author: hindmars
"""

import numpy as np
from scipy.optimize import fsolve

import he3_tools.he3_props as h3p
import he3_tools.he3_constants as h3c
import he3_tools.he3_magnetic as h3m

#%% RHUL cell data

rhul_lake_vol_mm3 = np.zeros((6,))

rhul_lake_vol_mm3[1] = 0.024
rhul_lake_vol_mm3[2] = 0.072
rhul_lake_vol_mm3[3] = 0.056
rhul_lake_vol_mm3[4] = 0.075
rhul_lake_vol_mm3[5] = 0.120

rhul_lake_vol_mm3[0] = np.sum(rhul_lake_vol_mm3[1:])

#%%

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

def hot_blob_size(t, p, Q_eV, phase='A',):
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
    integral_t3_dt = 0.25*(1 - t**4)
    
    return hot_blob_length_scale(p, phase='A') * (Q_eV/integral_t3_dt)**(1/3)

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

def truncated_sphere_volume(Rn, h):
    """
    Average volume of a sphere radius Rn in a cell of height h.

    Parameters
    ----------
    Rn : float, int, numpy.ndarray 
        Curvature radius.
    h : float, int
        Call height.

    Returns
    -------
    vol : type(Rn)
        Volume occupied by truncated sphere.

    """
    Rn = np.atleast_1d(Rn)
    
    sphere_vol = 4*np.pi*Rn**3/3
    cylinder_vol = np.pi*Rn**2 * h
    
    r = Rn/h

    hot_puck = Rn > h
    vol = sphere_vol*(1 - 3*r/8)
    vol[hot_puck] = cylinder_vol[hot_puck] * (1 - 1/(6*r[hot_puck]**2))
        
    return vol

def truncated_sphere_radius(vol, h):
    """
    Radius of a truncated sphere with average volume vol in a cell of height h.

    Parameters
    ----------
    vol : float, int, numpy.ndarray 
        Average volume (nm$^3$)
    h : float, int
        Cell height (nm)

    Returns
    -------
    Rn : type(Rn)
        Radius of truncated sphere

    """
    vol = np.atleast_1d(vol)
    
    Rb = fsolve(lambda Rn: truncated_sphere_volume(Rn, h) - vol, x0 = h )
        
    return Rb

def hot_blob_vol(t, p, Q_eV, phase='A'):

    C_V_Tc = h3p.C_V_normal(1, p) + h3p.delta_C_V_Tc(p, phase)
    integral_t3_dt = 0.25*(1 - t**4)

    heat_eV_per_vol = (h3p.Tc_mK(p)/1000) * integral_t3_dt * C_V_Tc / h3c.c.e

    vol_nm3 = Q_eV/heat_eV_per_vol

    return vol_nm3

def hot_blob_radius(t, p, Q_eV, h, phase='A', model='average'):
    
    vol_nm3 = hot_blob_vol(t, p, Q_eV, phase)
    
    vol_mm3 = np.atleast_1d(vol_nm3)
    
    Rb_arr = (vol_mm3*3/(4*np.pi))**(1/3)
    
    if model == 'average':
            
        for n, v in enumerate(vol_mm3):
            Rb_arr[n] = fsolve(lambda Rn: truncated_sphere_volume(Rn, h) - v, x0=1e9)

    elif model == 'simple':
        Rb2 = (vol_mm3*3/(np.pi * h))**(1/2)
        
        Rb_arr[Rb_arr > h/2] = Rb2[Rb_arr > h/2]

    else:
        
        print('warning: blob radius model not recognised.  Using 3D')

    return Rb_arr

def hot_blob_critical_energy_eV(t, p, H=0, 
                                Rc_fac=1.0, 
                                geo_fac=(4*np.pi)**(-0.5), 
                                cell_height=np.inf):
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
    
    t = np.atleast_1d(t)
    
    Rc3 = h3m.critical_radius(t, p, H, dim=3)*Rc_fac
    Rc2 = h3m.critical_radius(t, p, H, dim=2)*Rc_fac

    Rc = Rc3
    dim=3*np.ones_like(t) 
    
    hot_puck = Rc3 > cell_height
    intermediate = (Rc3 < cell_height) & (Rc3 > cell_height/2)
    dim[hot_puck] = 2
    Rc[hot_puck] = Rc2[hot_puck]

    dim[intermediate] = 3 - 2*(Rc3[intermediate] - cell_height/2)/cell_height
    Rc[intermediate] = Rc3[intermediate]*(dim[intermediate]-2) + Rc2[intermediate]*(3 - dim[intermediate]) 
    
    vol = hot_blob_volume(Rc, cell_height) * (np.exp(0)/6)**(dim/2) * (1/geo_fac)**dim * (1 - t)
    
    return vol * h3p.C_V_normal(1, p)*h3p.Tc_mK(p)/1e3/h3c.c.e, dim

def critical_radius_confined_eV(t, p, H=0, 
                                sigma_wall=None, 
                                geo_fac=(4*np.pi)**(-0.5), 
                                cell_height=np.inf, 
                                z0=0):
    """
    Critical bubble in confined geometry.

    Parameters
    ----------
    t : float, int, numpy.ndarray 
        Reduced temperature (T/Tc).
    p : float, int, numpy.ndarray 
        Pressure in bar.
    H : float, optional
        Magnetic field in tesla.
    cell_height: float, optional
        Length of confined dimension in nm
    z0 : float, optional        
        Position of bubble centre relative to half way height in cell, 
        Must be between -cell_height/2 and cell_height/2.
        
    Returns
    -------
    type(t) or type(p).
        Critical radius (eV) of critical bubble

    """
    # sigma_AB = h.f
   
    # if sigma_wall is None:
    
    t = np.atleast_1d(t)

    Rc = np.ones_like(t)*np.nan
    dim = np.ones_like(t)*3
    
    Rc3 = h3m.critical_radius(t, p, H, dim=3)
    Rc2 = h3m.critical_radius(t, p, H, dim=2)

    z0abs = np.abs(z0)
    case0 = Rc3 < (0.5 - z0abs) # free bulk 3D bubble, no walls touched
    case2 = Rc2 > (0.5 + z0abs) # effectively 2D bubble filling cell height, touching two walls
    case1 = (Rc3 < (0.5 - z0abs)) & (Rc2 < (0.5 + z0abs)) # touching one wall only ?

    Rc[case0] = Rc3

    
    hot_puck = Rc3 > cell_height
    intermediate = (Rc3 < cell_height) & (Rc3 > cell_height/2)
    dim[hot_puck] = 2
    Rc[hot_puck] = Rc2[hot_puck]

    dim[intermediate] = 3 - 2*(Rc3[intermediate] - cell_height/2)/cell_height
    Rc[intermediate] = Rc3[intermediate]*(dim[intermediate]-2) + Rc2[intermediate]*(3 - dim[intermediate]) 
    
    vol = hot_blob_volume(Rc, cell_height) * (np.exp(0)/6)**(dim/2) * (1/geo_fac)**dim * (1 - t)
    
    return vol * h3p.C_V_normal(1, p)*h3p.Tc_mK(p)/1e3/h3c.c.e, dim, 

def rhul_cell_rate_cdf(E_eV):
    """
    
    Cumulative distribution function of particle collision events in RHUL 3He 
    experiment, from Geant4 simulation (Elisabeth Leason, Rob)

    Parameters
    ----------
    E_eV : float, in, numpy.ndarray
        Particle energy.

    Returns
    -------
    type(E_eV)
        Cumulative distribution function.

    """
    
    A = 1.67268555e+03
    B = 1.12383036e+01
    C = 2.57511862e+00
    D = 1.45850000e+00
    F = -6.84359815e-01

    return (A / (B + E_eV / C))**D + E_eV**F

def rhul_lifetime_to_eV(life, efficiency=1.0, vol=rhul_lake_vol_mm3[1]):
    """
    Converts a lifetime (hours scaled to Lake 1) in the RHUL experiment to a minimum 
    ebergy, using the event rate CDF derived from 

    Parameters
    ----------
    r : float, int, numpy.ndarray
        Event rate in 1/hours.
    efficiency : float,int, optional
        Efficiency of conversion of energy into heat. The default is 1.0.
    vol : float, int, optional
        Fiducial volume in mm$^3$. The default is rhul_lake_vol_mm3[1].

    Returns
    -------
    Qi_eV : type(r)
        Minimum energy of particle causing events.

    """
    life_arr = np.atleast_1d(life)
    rate_per_day = 24/life_arr
    Qi_eV = np.zeros_like(life_arr)
    
    for n, rx in enumerate(rate_per_day):
        Qi_eV[n] = fsolve(lambda E_eV: rhul_cell_rate_cdf(E_eV/efficiency)*vol/rhul_lake_vol_mm3[0] - rx, x0=0.03)
    
    return Qi_eV
    
