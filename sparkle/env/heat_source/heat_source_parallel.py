import os
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from mpi4py import MPI

from sparkle.env.heat_source.heat_source import heat_source
from sparkle.src.utils.default import set_default

###############################################
class heat_source_parallel(heat_source):
    """
    A heat source positionning problem, parallelized with MPI
    using a naive domain decomposition strategy
    """
    def __init__(self, cpu, path, comm, pms=None):

        # Initialize from mother class
        super().__init__(cpu, path, pms)

        # MPI setup
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Split the grid into horizontal slices (along the i-axis / nx)
        rows_per_proc = self.nx // self.size
        remainder = self.nx % self.size

        # Determine start and end row for this process
        if self.rank < remainder:
            self.start_row = self.rank*(rows_per_proc + 1)
            self.local_nx = rows_per_proc + 1
        else:
            self.start_row = self.rank*rows_per_proc + remainder
            self.local_nx = rows_per_proc

        self.end_row = self.start_row + self.local_nx

        # Overwrite initial guess on slave processes for consistency
        self.comm.Bcast(self.x0, root=0)

        # Store neighbor ranks
        self.rank_up = self.rank-1 if self.rank > 0 else MPI.PROC_NULL
        self.rank_down = self.rank+1 if self.rank < self.size-1 else MPI.PROC_NULL

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        if self.rank == 0:
            os.makedirs(self.path, exist_ok=True)
        self.comm.Barrier()
        return True

    def cost(self, x):
        """
        Compute cost function
        The internal solver is parallel
        """

        # Clip for safety
        x_clipped = np.clip(x, 0.0, 1.0)

        # Solve in parallel. Returns the full T field on rank 0, None elsewhere
        T = self.solve(x_clipped)

        # Cost is computed only on rank 0, where T_global is available
        cost_val = None
        if self.rank == 0:
            if np.isnan(T).any(): return 1.0e8
            else:
                # Restrict to cost computation area
                T_cost = T[self.xmin_cost:self.xmax_cost, self.xmin_cost:self.xmax_cost]

                # Compute mean and variance of temperature field
                mean_T     = np.mean(T_cost)
                variance_T = np.mean((T_cost - mean_T)**2)

                # Total cost
                cost_val   = -mean_T + 0.01*variance_T

        # Broadcast the computed cost from rank 0 to all other processes
        cost_val = self.comm.bcast(cost_val, root=0)

        return cost_val

    def map_to_grid(self, x):
        """
        Maps continuous x values to the grid. Each process computes only the
        source terms that fall within its own subdomain
        """
        # Work on copy
        x_norm = np.copy(x)

        # Reshape and denormalize to physical coordinates
        x_norm = x_norm.reshape((self.n_sources, 2))
        x_norm[:,:] *= self.L

        # Populate x_source for plotting
        if self.rank == 0:
            self.x_source = x_norm

        # Each process initializes its own local source grid to all zeros
        q3k_local = np.zeros((self.local_nx, self.ny))

        # Each process iterates over all sources to check which ones belong to it
        for i in range(self.n_sources):
            xs = x_norm[i,0]
            ys = x_norm[i,1]

            # Find nearest grid cell indices
            ix = min(self.nx - 1, int(round(xs/self.dx)))
            iy = min(self.ny - 1, int(round(ys/self.dy)))

            # Ensure sources are not on Dirichlet boundaries
            ix = max(1, min(self.nx - 2, ix))
            iy = max(1, min(self.ny - 2, iy))

            # Check if the source global row index falls within the range
            # of rows handled by this process
            if self.start_row <= ix < self.end_row:
                # If it does, compute the corresponding local row index
                ix_local = ix - self.start_row

                # Add the source term to the local grid at the correct local index
                q3k_local[ix_local, iy] += self.q_source/(self.dx*self.dy*self.k)

        return q3k_local

    def solve(self, x):
        """
        Solve physical problem in parallel and return temperature field
        """

        # Map continuous x values to grid and compute source term
        # q3k is the source term q'''/k
        q3k_local = self.map_to_grid(x)

        # Initiallize T field with BC value and halo rows (one above, one below)
        T_local = np.full((self.local_nx + 2, self.ny), self.T_bc)

        # Update factors
        dx2 = self.dx**2
        dy2 = self.dy**2
        inv_denom = 1.0/(2.0*(1.0/dx2 + 1.0/dy2))

        # Main parallel iteration loop
        # We use checkerboard algorithm for Gauss-Seidel update in two times
        for it in range(self.max_iter):
            T_prv_local = T_local.copy()

            # Update all red points in the local domain
            gauss_seidel_local(T_local, q3k_local, inv_denom, self.omega_sor,
                               self.local_nx, self.ny, dx2, dy2,
                               self.start_row, self.nx, 0)

            # Halo exchange for red points
            self.comm.Sendrecv(T_local[1, :], self.rank_up, 0,
                               T_local[0, :], self.rank_up, 1)
            self.comm.Sendrecv(T_local[-2, :], self.rank_down, 1,
                               T_local[-1, :], self.rank_down, 0)

            # Update all black points in the local domain
            gauss_seidel_local(T_local, q3k_local, inv_denom, self.omega_sor,
                               self.local_nx, self.ny, dx2, dy2,
                               self.start_row, self.nx, 1)

            # Halo exchange for black points
            self.comm.Sendrecv(T_local[1, :], self.rank_up, 0,
                               T_local[0, :], self.rank_up, 1)
            self.comm.Sendrecv(T_local[-2, :], self.rank_down, 1,
                               T_local[-1, :], self.rank_down, 0)

            # Global convergence check
            local_diff  = np.abs(T_local[1:-1, :] - T_prv_local[1:-1, :]).max()
            global_diff = self.comm.allreduce(local_diff, op=MPI.MAX)
            local_max   = np.abs(T_local[1:-1, :]).max()
            global_max  = self.comm.allreduce(local_max, op=MPI.MAX)
            diff        = global_diff/global_max

            if global_diff < self.tol:
                break

        # Gather all the local T fields back to the root process
        T_local_no_halos = np.ascontiguousarray(T_local[1:-1, :])
        counts           = self.comm.allgather(T_local_no_halos.size)
        displs           = None
        T_global_flat    = None

        if self.rank == 0:
            displs        = [sum(counts[:i]) for i in range(self.size)]
            total_size    = sum(counts)
            T_global_flat = np.zeros(total_size)

        self.comm.Gatherv(T_local_no_halos,
                          [T_global_flat, counts, displs, MPI.DOUBLE],
                          root=0)

        # Return field
        # Root rank returns the global field, other ranks return None by convention
        if self.rank == 0:
            T_global = T_global_flat.reshape((self.nx, self.ny))
            self.T   = T_global # for plotting
            return T_global
        else:
            return None

    def render(self, x_pop, c):
        """
        Render the best candidate from the population. Only the root process does this.
        """

        best_idx = np.argmin(c)
        best_candidate = x_pop[best_idx]
        _ = self.solve(best_candidate)

        # The actual rendering is only performed by the main rank
        if self.rank != 0:
            return

        # Call mother class rendering helper
        super()._render()

    def close(self):
        pass

###############################################
# Gauss-Seidel with relaxation update
# This is a parallel checkerboard implementation
# 0 is for red points (i+j even), 1 is for black (i+j is odd)
@nb.njit(cache=False)
def gauss_seidel_local(T_local, q3k_local, inv_denom, omega,
                       local_nx, ny, dx2, dy2, start_row, nx, color):

    # Loop over assigned rows
    for i in range(local_nx):
        global_i = start_row + i
        if global_i == 0 or global_i == nx-1:
            continue

        # Loop over interior y-columns
        for j in range(1, ny-1):
            # Check if the current point (global_i, j) matches the target color
            if (global_i + j)%2 == color:

                # Gauss-Seidel
                # The i indexing takes into account the halo layers
                # Reminder: there is no halo layer on q3k array
                term_x = (T_local[i, j] + T_local[i+2, j])/dx2  # T[i-1], T[i+1]
                term_y = (T_local[i+1, j-1] + T_local[i+1, j+1])/dy2 # T[j-1], T[j+1]

                T_gs = (term_x + term_y + q3k_local[i, j])*inv_denom

                # Apply the relaxation formula
                T_local[i+1, j] = (1.0 - omega)*T_local[i+1, j] + omega*T_gs
