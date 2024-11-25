# Generic imports
import os
import math
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

from   matplotlib.patches import Rectangle

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for rayleigh control
class rayleigh(base_env):

    # Initialize instance
    def __init__(self, cpu, path, pms=None):

        # Main parameters
        self.name        = "rayleigh"
        self.base_path   = path
        self.cpu         = cpu
        self.dim         = 10
        self.C           = 0.75
        self.x0          = np.zeros(self.dim)
        self.xmin        =-np.ones(self.dim)
        self.xmax        = np.ones(self.dim)

        self.L           = 1.0              # length of the domain
        self.H           = 1.0              # height of the domain
        self.ra          = 3.0e3            # rayleigh number
        self.nx          = 50*int(self.L)   # nb of pts in x direction
        self.ny          = 50*int(self.H)   # nb of pts in y direction
        self.pr          = 0.71             # prandtl number
        self.Tc          =-0.5              # top plate temperature
        self.Th          = 0.5              # bottom plate reference temperature
        self.dt          = 0.005            # timestep
        self.t_max       = 200.0            # action time after warmup
        self.n_sgts      = self.dim         # nb of temperature segments
        self.init_file   = "init_field.dat" # initialization file

        # Deduced parameters
        self.dx             = float(self.L/self.nx)             # x spatial step
        self.dy             = float(self.H/self.ny)             # y spatial step
        self.n_steps        = math.floor(self.t_max/self.dt)    # total nb of steps
        self.nx_sgts        = self.nx//self.n_sgts              # nb of pts in each segment

        ### Declare arrays
        self.u       = np.zeros((self.nx+2, self.ny+2)) # u field
        self.v       = np.zeros((self.nx+2, self.ny+2)) # v field
        self.p       = np.zeros((self.nx+2, self.ny+2)) # p field
        self.T       = np.zeros((self.nx+2, self.ny+2)) # T field
        self.us      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.vs      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.phi     = np.zeros((self.nx+2, self.ny+2)) # projection field

        self.u_init  = np.zeros((self.nx+2, self.ny+2)) # u initialization
        self.v_init  = np.zeros((self.nx+2, self.ny+2)) # v initialization
        self.p_init  = np.zeros((self.nx+2, self.ny+2)) # p initialization
        self.T_init  = np.zeros((self.nx+2, self.ny+2)) # T initializatino

        # Load initialization file
        self.load(self.init_file)

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        self.temperature_path = self.path+"/temperature"
        self.field_path       = self.path+"/field"
        self.action_path      = self.path+"/action"
        os.makedirs(self.temperature_path, exist_ok=True)
        os.makedirs(self.field_path,       exist_ok=True)
        os.makedirs(self.action_path,      exist_ok=True)

        self.reset_fields()

        return True

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.u[:,:] = 0.0
        self.v[:,:] = 0.0
        self.p[:,:] = 0.0
        self.T[:,:] = 0.0

        # Other fields
        self.us[:,:]  = 0.0
        self.vs[:,:]  = 0.0
        self.phi[:,:] = 0.0

        # Nusselt
        self.nu = np.empty((0,2))

    # Cost function
    def cost(self, x):

        # Set initial fields
        self.u[:] = self.u_init[:]
        self.v[:] = self.v_init[:]
        self.p[:] = self.p_init[:]
        self.T[:] = self.T_init[:]

        # Zero-mean the actions
        xx = x.copy()
        xx[:] = xx[:] - np.mean(xx)
        m    = max(1.0, np.amax(np.abs(xx)/self.C))
        for i in range(self.n_sgts):
            xx[i] = xx[i]/m

        i = 0
        while (i < self.n_steps):

            # Set boundary conditions
            # Left wall
            self.u[1,1:-1]  = 0.0
            self.v[0,2:-1]  =-self.v[1,2:-1]
            self.T[0,1:-1]  = self.T[1,1:-1]

            # Right wall
            self.u[-1,1:-1] = 0.0
            self.v[-1,2:-1] =-self.v[-2,2:-1]
            self.T[-1,1:-1] = self.T[-2,1:-1]

            # Top wall
            self.u[1:,-1]   =-self.u[1:,-2]
            self.v[1:-1,-1] = 0.0
            self.T[1:-1,-1] = 2.0*self.Tc - self.T[1:-1,-2]

            # Bottom wall
            self.u[1:,0]    =-self.u[1:,1]
            self.v[1:-1,1]  = 0.0

            for j in range(self.n_sgts):
                s = 1 + j*self.nx_sgts
                e = 1 + (j+1)*self.nx_sgts
                self.T[s:e,0]  = 2.0*(self.Th + xx[j]) - self.T[s:e,1]

            # Predictor step
            predictor(self.u, self.v, self.us, self.vs, self.p, self.T,
                      self.nx, self.ny, self.dt, self.dx, self.dy, self.pr, self.ra)

            # Poisson step
            itp, ovf = poisson(self.us, self.vs, self.u, self.phi,
                               self.nx, self.ny, self.dx, self.dy, self.dt)
            self.p[:,:] += self.phi[:,:]

            if (ovf):
                print("\n")
                print("Exceeded max number of iterations in solver")
                exit(1)

            # Corrector step
            corrector(self.u, self.v, self.us, self.vs, self.phi,
                      self.nx, self.ny, self.dx, self.dy, self.dt)

            # Transport step
            transport(self.u, self.v, self.T,
                      self.nx, self.ny, self.dx, self.dy, self.dt, self.pr, self.ra)

            # Update loop index
            i += 1

            # Update cost
            nu = 0.0
            for j in range(1,self.nx+1):
                dT  = (self.T[j,2] - self.T[j,1])/self.dy
                nu -= dT
            nu /= self.nx

            # Store nusselt
            self.nu = np.append(self.nu, np.array([[i, nu]]), axis=0)

        return nu

    # Render environment
    def render(self, x):

        # Zero-mean the actions
        xx    = x[0].copy()
        xx[:] = xx[:] - np.mean(xx)
        m    = max(1.0, np.amax(np.abs(xx)/self.C))
        for i in range(self.n_sgts):
            xx[i] = xx[i]/m

        # Set field
        margin = 0.2
        rny    = self.ny+int(margin*self.ny)
        pT     = np.zeros((self.nx, rny))
        pT[0:self.nx,rny-self.ny:rny] = self.T[1:-1,1:-1]

        # Rotate field
        pT = np.rot90(pT)

        # Plot temperature
        fig = plt.figure(figsize=(5,5))
        ax  = plt.gca()
        plt.axis('off')
        plt.imshow(pT,
                   cmap = 'RdBu_r',
                   vmin = self.Tc,
                   vmax = self.Th,
                   extent=[0.0, self.L, -margin*self.H, self.H])

        # Plot control
        scale = self.L/self.n_sgts
        ax.add_patch(Rectangle((0.0,-0.99*margin), 0.998*self.L, margin*self.H,
                               color='k', fill=False, lw=0.5))
        ax.add_patch(Rectangle((0.0,-0.51*margin), 0.998*self.L, 0.001,
                               color='k', fill=False, lw=0.3))
        for i in range(self.n_sgts):
            px = (0.5 + i)*scale - 0.5*self.dx
            py = -0.5*margin
            color = 'r' if xx[i] > 0.0 else 'b'
            ax.add_patch(Rectangle((px, py),
                                   0.25*scale, 0.5*margin*xx[i],
                                   color=color, fill=True, lw=1))

        # Save figure
        filename = self.temperature_path+"/"+str(self.it_plt)+".png"
        fig.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')

        # Dump nusselt time trace
        np.savetxt(self.path+"/nu.dat", self.nu, fmt='%.5e')

        self.it_plt += 1

    # Dump fields
    def dump(self, filename):

        array = self.u.copy()
        array = np.vstack((array, self.v))
        array = np.vstack((array, self.p))
        array = np.vstack((array, self.T))

        np.savetxt(filename, array, fmt='%.5e')

    # Load (h,q)
    def load(self, filename):

        f = np.loadtxt(filename)
        self.u_init[:,:] = f[0*(self.nx+2):1*(self.nx+2),:]
        self.v_init[:,:] = f[1*(self.nx+2):2*(self.nx+2),:]
        self.p_init[:,:] = f[2*(self.nx+2):3*(self.nx+2),:]
        self.T_init[:,:] = f[3*(self.nx+2):4*(self.nx+2),:]

    # Closing
    def close(self):
        pass

###############################################
# Predictor step
@nb.njit(cache=False)
def predictor(u, v, us, vs, p, T, nx, ny, dt, dx, dy, pr, ra):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            uE = 0.5*(u[i+1,j] + u[i,j])
            uW = 0.5*(u[i,j]   + u[i-1,j])
            uN = 0.5*(u[i,j+1] + u[i,j])
            uS = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j]   + v[i-1,j])
            conv = (uE*uE-uW*uW)/dx + (uN*vN-uS*vS)/dy

            diff  = ((u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/(dx**2) +
                     (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/(dy**2))
            diff *= math.sqrt(pr/ra)

            pres = (p[i,j] - p[i-1,j])/dx

            us[i,j] = u[i,j] + dt*(diff - conv - pres)

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            vE = 0.5*(v[i+1,j] + v[i,j])
            vW = 0.5*(v[i,j]   + v[i-1,j])
            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i,j])
            vS = 0.5*(v[i,j]   + v[i,j-1])
            conv = (uE*vE-uW*vW)/dx + (vN*vN-vS*vS)/dy

            diff  = ((v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/(dx**2) +
                     (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/(dy**2))
            diff *= math.sqrt(pr/ra)

            pres  = (p[i,j] - p[i,j-1])/dy

            vs[i,j] = v[i,j] + dt*(diff - conv - pres + T[i,j])

###############################################
# Poisson step
@nb.njit(cache=False)
def poisson(us, vs, u, phi, nx, ny, dx, dy, dt):

    tol      = 1.0e-8
    err      = 1.0e10
    itp      = 0
    itmax    = 300000
    ovf      = False
    phi[:,:] = 0.0
    phin     = np.zeros((nx+2,ny+2))
    while(err > tol):

        phin[:,:] = phi[:,:]

        for i in range(1,nx+1):
            for j in range(1,ny+1):

                b = ((us[i+1,j] - us[i,j])/dx +
                     (vs[i,j+1] - vs[i,j])/dy)/dt

                phi[i,j] = 0.5*((phin[i+1,j] + phin[i-1,j])*dy*dy +
                                (phin[i,j+1] + phin[i,j-1])*dx*dx -
                                b*dx*dx*dy*dy)/(dx*dx+dy*dy)

        # Domain left (neumann)
        phi[ 0,1:-1] = phi[ 1,1:-1]

        # Domain right (neumann)
        phi[-1,1:-1] = phi[-2,1:-1]

        # Domain top (dirichlet)
        phi[1:-1,-1] = phi[1:-1,-2]

        # Domain bottom (neumann)
        phi[1:-1, 0] = phi[1:-1, 1]

        # Compute error
        dphi = np.reshape(phi - phin, (-1))
        err  = np.dot(dphi,dphi)

        itp += 1
        if (itp > itmax):
            ovf = True
            break

    return itp, ovf

###############################################
# Corrector step
@nb.njit(cache=False)
def corrector(u, v, us, vs, phi, nx, ny, dx, dy, dt):

    u[2:-1,1:-1] = us[2:-1,1:-1] - dt*(phi[2:-1,1:-1] - phi[1:-2,1:-1])/dx
    v[1:-1,2:-1] = vs[1:-1,2:-1] - dt*(phi[1:-1,2:-1] - phi[1:-1,1:-2])/dy

###############################################
# Transport step
@nb.njit(cache=False)
def transport(u, v, T, nx, ny, dx, dy, dt, pr, ra):

    for i in range(1,nx+1):
        for j in range(1,ny+1):
            uE = u[i+1,j]
            uW = u[i,j]
            vN = v[i,j+1]
            vS = v[i,j]
            TE = 0.5*(T[i+1,j] + T[i,j])
            TW = 0.5*(T[i-1,j] + T[i,j])
            TN = 0.5*(T[i,j+1] + T[i,j])
            TS = 0.5*(T[i,j-1] + T[i,j])
            conv = (uE*TE-uW*TW)/dx + (vN*TN-vS*TS)/dy

            diff  = ((T[i+1,j] - 2.0*T[i,j] + T[i-1,j])/(dx**2) +
                     (T[i,j+1] - 2.0*T[i,j] + T[i,j-1])/(dy**2))
            diff /= math.sqrt(pr*ra)

            T[i,j] += dt*(diff - conv)
