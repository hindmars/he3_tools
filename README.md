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
## Changing default settings

There are some default settings which are reported when the module is loaded.
```
>>> import he3_tools as h
he3_tools: <class 'bool'> variable DEFAULT_SC_ADJUST set to False
he3_tools: <class 'str'> variable DEFAULT_SC_CORRS set to RWS19
he3_tools: <class 'str'> variable DEFAULT_T_SCALE set to PLTS
he3_tools: <class 'str'> variable DEFAULT_ALPHA_TYPE set to GL
```
Settings can be changed with e.g.
```
h.set_default('DEFAULT_T_SCALE', 'Greywall')
```
As an example:
```
>>> print(h.Tc_mK(5.5))
1.499831031273556
>>> h.set_default('DEFAULT_T_SCALE', 'Greywall')
he3_tools: <class 'str'> variable DEFAULT_T_SCALE set to Greywall
>>> print(h.Tc_mK(5.5))
1.5202018671958002
```

A list of possible settings:

`DEFAULT_T_SCALE: {'PLTS' , 'Greywall'}`

`DEFAULT_SC_CORRS: {'Choi-interp', 'Choi-poly', 'RWS19', 'RWS19-interp', 'RWS19-poly', 'WS15', 'WS15-poly', 'Wiman-thesis'}`

`DEFAULT_SC_ADJUST: {True, False}`

`DEFAULT_ALPHA_TYPE: {'GL', 'BCS'}`









