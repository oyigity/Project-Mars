#Yigit Yazgan
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
import matplotlib.pyplot as plt
import os

# Compile the C library if not already compiled
if not os.path.exists("libode.so"):
    os.system("gcc -shared -O2 -fPIC ode.c -o libode.so")

# Load our C library
lib = ctypes.CDLL("./libode.so")
solve_ode_c = lib.solve_ode
solve_ode_c.restype = None
solve_ode_c.argtypes = [
    ndpointer(ctypes.c_double), 
    ndpointer(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_double),
    ctypes.CFUNCTYPE(None, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double))
]

# Python wrapper for the ODE solver
def solve_ode(fun, t_span, nsteps, y0, method="RK4", args=None):
    t_span = np.asarray(t_span, dtype=np.double)
    t = np.linspace(t_span[0], t_span[1], nsteps + 1, dtype=np.double)
    nvar = len(y0)
    y = np.zeros([nsteps + 1, nvar], dtype=np.double, order='C')
    y[0, :] = np.asarray(y0, dtype=np.double)

    if "ctypes" in dir(fun):
        fun_c = fun.ctypes
    else:
        FUNCTYPE = CFUNCTYPE(None, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double))
        fun_c = FUNCTYPE(fun)

    dt = (t_span[1] - t_span[0]) / nsteps
    if args is not None: 
        args = np.asarray(args, dtype=np.double)
    if method in ["RK2", "RKO2"]:
        order = 2
    elif method in ["Euler"]:
        order = 1
    elif method in ["Euler-Cromer"]:
        order = -1
    else:
        order = 4

    solve_ode_c(t, y, dt, nsteps, nvar, order, args, fun_c)
    return t, y


from numba import cfunc, types

c_sig = types.void(types.double,
                   types.CPointer(types.double),
                   types.CPointer(types.double),
                   types.CPointer(types.double))



@cfunc(c_sig)
def func(t, vars, params, dfdt):
    x = vars[0]
    dx_dt = vars[1]
    y = vars[2]
    dy_dt = vars[3]
    #soln for x
    dfdt[0] = dx_dt
    dfdt[1] = - (params[0] * x)/ ((x**2 + y**2) * np.sqrt((x**2 + y**2)))
    #soln  for y
    dfdt[2] = dy_dt
    dfdt[3] = - (params[0] * y)/ ((x**2 + y**2) * np.sqrt((x**2 + y**2)))