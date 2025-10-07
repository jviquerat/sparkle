import os
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default

###############################################
class heat_source(base_env):
    """
    A heat source positionning problem
    The goal is to place several heat sources in a square domain in
    order to obtain a high homogeneous temperature across a target area
    in the center of the domain
    """
    def __init__(self, cpu, path, pms=None):

        self.name = "heat_source"
        self.base_path = path

        # Domain
        self.L = 1.0
        self.nx = set_default("nx", 100, pms)
        self.ny = set_default("ny", 100, pms)
        self.dx = self.L/(self.nx - 1)
        self.dy = self.L/(self.ny - 1)

        # Properties
        self.k = 1.0                                         # thermal conductivity
        self.n_sources = set_default("n_sources", 4, pms)    # nb of sources
        self.q_source  = set_default("q_source", 500.0, pms) # power of each source

        # Number of free parameters
        self.dim = 2*self.n_sources

        # Initial guess for normalized source coordinates [0,1]
        self.x0   = np.random.rand(self.dim)
        self.xmin = np.zeros(self.dim) # min normalized coordinate
        self.xmax = np.ones(self.dim)  # max normalized coordinate

        # Zone for cost computation
        self.xmin_cost = math.floor(0.25/self.dx)
        self.xmax_cost = math.floor(0.75/self.dx)

        # Solver parameters
        # We either use an iterative jacobi, or a over-relaxation method
        self.method = set_default("method", "sor", pms)
        self.max_iter  = 1000 # max iterations for temperature solver
        self.tol       = 1e-5  # convergence tolerance
        self.omega_sor = 2.0/(1.0 + math.sin(math.pi*self.dx/self.L)) # SOR omega

        # Boundary Conditions
        self.T_bc = 0.0 # fixed temperature on all boundaries

        # Plotting data
        self.it_plt = 0

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)

        return True

    def cost(self, x):
        """
        Compute cost function
        """

        # Clip for safety
        x_clipped = np.clip(x, 0.0, 1.0)

        # Solve and check for possible failure
        T = self.solve(x_clipped)
        if np.isnan(T).any(): return 1.0e8

        # Restrict to cost computation area
        T_cost = T[self.xmin_cost:self.xmax_cost, self.xmin_cost:self.xmax_cost]

        # Compute mean and variance of temperature field
        mean_T     = np.mean(T_cost)
        variance_T = np.mean((T_cost - mean_T)**2)

        return -mean_T + 0.01*variance_T

    def map_to_grid(self, x):
        """
        Map continuous x values to grid and compute source term q'''/k
        The FD scheme is:

          (T_i+1,j + T_i-1,j - 2*T_i,j)/dx^2
        + (T_i,j+1 + T_i,j-1 - 2*T_i,j)/dy^2 + q'''_i,j/k = 0
        """

        # Work on copy
        x_norm = np.copy(x)

        # Reshape and denormalize to physical coordinates
        x_norm = x_norm.reshape((self.n_sources, 2))
        x_norm[:,:] *= self.L

        # Populate x_source for plotting
        self.x_source = x_norm

        # Create grid for the source term at grid points
        q3k = np.zeros((self.nx, self.ny))
        for i in range(self.n_sources):
            xs = x_norm[i,0]
            ys = x_norm[i,1]

            # Find nearest grid cell indices
            ix = min(self.nx - 1, int(round(xs/self.dx)))
            iy = min(self.ny - 1, int(round(ys/self.dy)))

            # Ensure sources are not on Dirichlet boundaries
            ix = max(1, min(self.nx - 2, ix))
            iy = max(1, min(self.ny - 2, iy))

            # q''' = q_source/(dx*dy) is the volumic source term
            q3k[ix, iy] += self.q_source/(self.dx*self.dy*self.k)

        return q3k

    def solve(self, x):
        """
        Solve physical problem and return temperature field
        """

        # Map continuous x values to grid and compute source term
        # q3k is the source term q'''/k
        q3k = self.map_to_grid(x)

        # Initiallize T field with BC value
        T = np.full((self.nx, self.ny), self.T_bc)

        # Jacobi update factor for T_ij when summing neighbors
        dx2 = self.dx**2
        dy2 = self.dy**2
        inv_denom = 1.0/(2.0*(1.0/dx2 + 1.0/dy2))

        if (self.method == "jacobi"):
            jacobi(T, q3k, inv_denom,
                   self.max_iter, self.nx, self.ny, dx2, dy2, self.tol)

        if (self.method == "sor"):
            sor(T, q3k, inv_denom, self.omega_sor,
                   self.max_iter, self.nx, self.ny, dx2, dy2, self.tol)

        # Populate self.T for plotting
        self.T = T

        return T

    def render(self, x_pop, c):
        """
        Render the best candidate in the current population
        """

        best_combined_cost = np.min(c)
        best_idx = np.argmin(c)
        best_u = x_pop[best_idx]

        _ = self.cost(best_u)

        self._render()

    def _render(self):
        """
        Actual rendering
        """

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.axis('off')

        x_coords = np.linspace(0, self.L, self.nx)
        y_coords = np.linspace(0, self.L, self.ny)
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)

        vmin = np.min(self.T)
        vmax = np.max(self.T)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 10)

        im = ax.contourf(X_mesh,
                         Y_mesh,
                         self.T,
                         levels=levels,
                         cmap='inferno',
                         norm=norm)

        ax.plot(self.x_source[:, 1],
                self.x_source[:, 0],
                'wo',
                markersize=10,
                markeredgecolor='teal',
                markerfacecolor='none',
                mew=2)

        lgt = (self.xmax_cost - self.xmin_cost)*self.dx
        ax.add_patch(Rectangle((self.xmin_cost*self.dx, self.xmin_cost*self.dx),
                               lgt, lgt,
                               edgecolor='red',
                               fill=False,
                               lw=3,
                               linestyle='--'))

        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()

        plot_filename = os.path.join(self.path, f"{self.it_plt}.png")
        plt.savefig(plot_filename,
                    dpi=90,
                    bbox_inches='tight',
                    pad_inches=0.0)
        plt.close(fig)

        self.it_plt += 1

    def close(self):
        pass

###############################################
# Jacobi update
@nb.njit(cache=False)
def jacobi(T, q3k, inv_denom, max_iter, nx, ny, dx2, dy2, tol):

    for iteration in range(max_iter):
        T_prv = T.copy()

        # Interior points update
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                term_x  = (T_prv[i-1, j  ] + T_prv[i+1, j  ])/dx2
                term_y  = (T_prv[i,   j-1] + T_prv[i,   j+1])/dy2
                T[i, j] = (term_x + term_y + q3k[i,j])*inv_denom

        # Check for convergence (max absolute difference)
        diff = np.abs(T - T_prv).max()
        if diff < tol: break

###############################################
# Successive over-relaxation update
@nb.njit(cache=False)
def sor(T, q3k, inv_denom, omega, max_iter, nx, ny, dx2, dy2, tol):

    for iteration in range(max_iter):
        T_prv = T.copy()

        # Loop over all interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Gauss-Seidel
                term_x = (T[i-1, j] + T_prv[i+1, j])/dx2
                term_y = (T[i, j-1] + T_prv[i, j+1])/dy2

                # This is the value a Gauss-Seidel iteration would produce
                T_gs = (term_x + term_y + q3k[i, j])*inv_denom

                # Apply the SOR formula
                T[i, j] = (1.0 - omega)*T_prv[i, j] + omega*T_gs

        # Check for convergence (max absolute difference)
        diff = np.abs(T - T_prv).max()
        if diff < tol: break
