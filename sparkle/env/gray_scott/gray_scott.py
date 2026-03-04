import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default

###############################################
class gray_scott(base_env):
    """
    A 2D Gray-Scott PDE environment
    The PDE simulates two chemical species U and V interacting over time
    The goal is to find the feed rate (f) and kill rate (k) that produce
    a specific structural pattern in 2D (here, a target coverage area)

    Variables
        x[0]: Feed rate (f) in [0.01, 0.05]
        x[1]: Kill rate (k) in [0.04, 0.07]
    """
    def __init__(self, cpu, path, pms=None):

        self.name = "gray_scott"
        self.base_path = path
        self.cpu = cpu
        self.render_mode = set_default("render_mode", "png", pms)

        # Number of free parameters
        self.dim = 2

        # Standard range for Turing patterns in 2D
        self.x0   = np.array([0.025, 0.055])
        self.xmin = np.array([0.010, 0.040])
        self.xmax = np.array([0.050, 0.070])

        # Plotting data for metamodel renderer
        self.it_plt = 0
        self.vmin   = 0.0
        self.vmax   = 1.0 # Cost is normalized
        self.levels = [0.1, 0.2, 0.4, 0.6, 0.8]

        # PDE solver parameters
        self.nx = 256        # Spatial resolution X
        self.ny = 256        # Spatial resolution Y
        self.dx = 1.0        # Spatial step
        self.dt = 1.0        # Time step
        self.n_steps = 15000 # Number of time steps to reach steady state

        # Diffusion coefficients
        # It is crucial for Turing patterns that Du > Dv
        self.Du = 0.16
        self.Dv = 0.08

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)
        if self.render_mode == "gif":
            os.makedirs(os.path.join(self.path, self.render_mode), exist_ok=True)

        return True

    def cost(self, x):
        """
        Compute cost function: maximize the total curvature
        This directly penalizes large uniform blobs, which have
        zero curvature, and rewards highly intricate, dense stripes
        and spots patterns, which have large curvature at their edges
        """
        # Ensure x is a 1D array to extract f and k safely
        x_flat = np.ravel(x)
        f = x_flat[0]
        k = x_flat[1]

        # Solve the 2D PDE
        U, V = self.solve(f, k)

        # Compute the discrete Laplacian of the final state
        # using a simple 5-point stencil with periodic boundaries
        laplace_V = (np.roll(V,  1, axis=0) +
                     np.roll(V, -1, axis=0) +
                     np.roll(V,  1, axis=1) +
                     np.roll(V, -1, axis=1) - 4.0 * V)

        # We want to maximize the total absolute curvature
        total_curvature = np.sum(np.abs(laplace_V))

        # Return the negative curvature
        return -total_curvature / 1000.0

    def init_fields(self):
        """
        Initializes the concentration fields with a central perturbation
        and breaking noise
        """
        # Initialize uniform concentrations
        U = np.ones((self.nx, self.ny), dtype=np.float64)
        V = np.zeros((self.nx, self.ny), dtype=np.float64)

        # Introduce a square perturbation in the center
        cx = self.nx // 2
        cy = self.ny // 2
        r  = 5
        V[cx-r:cx+r, cy-r:cy+r] = 0.5

        # Add a tiny bit of noise to break perfect symmetry
        # (deterministic for reproducible results)
        np.random.seed(42)
        V += np.random.rand(self.nx, self.ny) * 0.05

        return U, V

    def solve(self, f, k):
        """
        Setup and solve
        """
        # Initialize concentration fields
        U, V = self.init_fields()

        # Call the numba solver
        U, V = solve_gs_2d(U, V, f, k,
                           self.Du, self.Dv,
                           self.dt, self.dx**2,
                           self.n_steps, self.nx, self.ny)

        # Store for rendering
        self.final_U = U
        self.final_V = V

        return U, V

    def render(self, x_pop, c):
        """
        Renders the concentration profiles of the best candidate based
        on self.render_mode
        """
        # Ensure we can handle both 1D and 2D arrays robustly
        c_flat = np.ravel(c)
        x_pop_2d = np.atleast_2d(x_pop)

        best_idx = np.argmin(c_flat)
        best_u = x_pop_2d[best_idx]
        best_c = c_flat[best_idx]
        f = best_u[0]
        k = best_u[1]

        # Re-solve to get the fields for the best candidate
        self.cost(best_u)

        # png mode is used during training
        if self.render_mode == 'png':
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(5, 5))

            im = ax.imshow(self.final_V,
                           cmap='magma',
                           vmin=0.0, vmax=0.4, origin='lower')

            ax.axis('off')
            fig.tight_layout()

            plot_filename = os.path.join(self.path, "png", f"{self.it_plt}.png")
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            plt.savefig(plot_filename, dpi=100,
                        bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

        # gif mode is used for rendering
        elif self.render_mode == 'gif':

            # Initialize concentration fields
            U, V = self.init_fields()

            n_frames = min(150, self.n_steps)
            chunk_size = max(1, self.n_steps // n_frames)

            # Save initial frame
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(V,
                           cmap='magma',
                           vmin=0.0, vmax=0.4, origin='lower')
            ax.axis('off')
            fig.tight_layout()

            filename = os.path.join(self.path, self.render_mode)
            filename = os.path.join(filename, "0.png")
            plt.savefig(filename,
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.0)

            # Run simulation in chunks and save intermediate frames
            for frame in range(1, n_frames):
                U, V = solve_gs_2d(U, V, f, k,
                                   self.Du, self.Dv,
                                   self.dt, self.dx**2,
                                   chunk_size, self.nx, self.ny)

                im.set_data(V)
                filename = os.path.join(self.path, self.render_mode)
                filename = os.path.join(filename, f"{frame}.png")
                plt.savefig(filename, dpi=100,
                            bbox_inches='tight', pad_inches=0.0)

            plt.close(fig)

        self.it_plt += 1

    def close(self):
        pass

# Numba 2D PDE solver for Gray-Scott
@nb.njit(cache=False)
def solve_gs_2d(U, V, f, k, Du, Dv, dt, dx2, n_steps, nx, ny):
    for step in range(n_steps):
        U_new = np.empty_like(U)
        V_new = np.empty_like(V)

        for i in range(nx):
            for j in range(ny):
                # Periodic boundary conditions
                im1 = i - 1 if i > 0 else nx - 1
                ip1 = i + 1 if i < nx - 1 else 0
                jm1 = j - 1 if j > 0 else ny - 1
                jp1 = j + 1 if j < ny - 1 else 0

                # Discrete Laplacian (5-point stencil)
                laplace_U = (U[im1, j] + U[ip1, j] +
                             U[i, jm1] + U[i, jp1] -
                             4.0 * U[i, j]) / dx2
                laplace_V = (V[im1, j] + V[ip1, j] +
                             V[i, jm1] + V[i, jp1] -
                             4.0 * V[i, j]) / dx2

                u_val = U[i, j]
                v_val = V[i, j]
                uvv = u_val * v_val * v_val

                U_new[i, j] = u_val + dt * (Du * laplace_U -
                                            uvv +
                                            f * (1.0 - u_val))
                V_new[i, j] = v_val + dt * (Dv * laplace_V +
                                            uvv -
                                            (f + k) * v_val)

        U[:] = U_new[:]
        V[:] = V_new[:]

    return U, V
