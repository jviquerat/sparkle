# Generic imports
import math
import pytest

# Custom imports
from sparkle.tst.tst         import *
from sparkle.src.utils.lsrk4 import *

###############################################
### Test lsrk4 integrator
def test_lsrk4():

    # integrate the following system:
    # du/dt(x,t) = cos((x+1)*t)
    # u(x,0)     = 0
    nt = 1000         # nb of timesteps
    nx = 11           # nb of spatial points
    u  = np.zeros(nx) # field
    uk = np.zeros(nx) # lsrk storage
    fu = np.zeros(nx) # rhs
    dt = 0.1          # timestep
    dx = 0.1          # spatial step
    t  = 0.0          # time

    # space
    x = np.linspace(0.0, (nx-1)*dx, nx) + 1.0

    # instanciate lsrk4 class
    integrator = lsrk4()

    # loop on timesteps
    for i in range(nt):

        # update intermediate storage
        uk[:] = u[:]

        # loop on integration steps
        for j in range(integrator.steps()):

            # retrieve source time
            src_time = integrator.source_time(j, t, dt)

            # compute rhs
            fu       = np.cos(x*src_time)

            # update
            integrator.update(u, uk, fu, j, dt)

        # update main storage
        u[:] = uk[:]
        t   += dt

    # exact solution
    u_ex = np.sin(x*t)/x

    # compute error
    assert(np.linalg.norm(u-u_ex) < 1.0e-8)
