# he3_tools
Tools for computation of static properties of superfluid Helium 3

Put the top-level he3_tools directory on your PYTHONPATH.

See examples folder for some more examples of how the module can be used.

## Example functions

```
import he3_tools as h

# critical temperature
print('critical temperature at 5.5 bar', h.Tc_mK(5.5))

# specific heat
print('volumetric specific heat in the A phase at critical temperature at 5.5 bar', h.C_V(1, 5.5, 'A'))

# thermal diffusivity (in units of zero-temperature GL coherence length and Fermi velocity)
print('thermal diffusivity at the critical temperature and at p=5.5 bar', h.thermal_diffusivity_norm(1, 5.5))

```
