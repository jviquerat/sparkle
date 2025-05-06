import math
import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default

###############################################
class n_body(base_env):
    """
    N-body problem, with a cost function aiming at finding periodic orbits
    The bodies are initially positionned on the y=0 axis
    The optimized dof are the initial x positions, as well as the initial velocities (so 3n dof)
    Adapted from:

    A guide to hunting periodic three-body orbits
    M. Suvakov, V. Dmitrasinovicz
    Computational Physics, vol. 82, iss. 6 (2014)
    """
    def __init__(self, cpu, path, pms=None):

        self.name        = 'n_body'
        self.base_path   = path
        self.cpu         = cpu
        self.render_mode = set_default("render_mode", "png", pms)

        self.n_bodies = set_default("n_bodies", 3, pms)
        self.dim      = 3*self.n_bodies # n positions + 2n velocities
        self.sim_dim  = 4*self.n_bodies # n*(x, y, vx, vy)

        default_x0_pos = np.linspace(-3.0, 3.0, self.n_bodies)
        default_x0_vel = np.zeros(2*self.n_bodies)
        self.x0        = np.concatenate((default_x0_pos, default_x0_vel))

        default_xmin_pos = -4.0*np.ones(self.n_bodies)
        default_xmin_vel = -2.0*np.ones(2*self.n_bodies)
        self.xmin        = np.concatenate((default_xmin_pos, default_xmin_vel))

        default_xmax_pos = 4.0*np.ones(self.n_bodies)
        default_xmax_vel = 2.0*np.ones(2*self.n_bodies)
        self.xmax        = np.concatenate((default_xmax_pos, default_xmax_vel))

        # Physical and numerical parameters
        self.G       = 1.0
        self.m       = np.ones(self.n_bodies)
        self.eps     = 1e-6
        self.t_max   = set_default("t_max", 60.0, pms)
        self.dt      = set_default("dt", 0.005, pms)
        self.n_steps = math.floor(self.t_max / self.dt)

        # Scope parameters
        self.scope_radius = set_default("scope_radius", 5.0, pms)
        self.beta         = set_default("penalty_weight", 1.0, pms)

        # Rendering parameters
        self.plot_every_n_steps = set_default("plot_every_n_steps", 10, pms)
        self.trail_length       = set_default("trail_length", 800, pms)

        # Update simulation array sizes based on n_bodies
        self.xk  = np.zeros(self.sim_dim)
        self.acc = np.zeros(2*self.n_bodies)
        self.hx  = np.zeros((self.n_steps + 1, self.sim_dim))
        self.t   = np.linspace(0.0, self.t_max, num=self.n_steps + 1)

        # Minimum index to consider for return proximity
        self.min_return_index = int(set_default("min_return_time", 5.0, pms)/self.dt)

        self.it_plt = 0

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = os.path.join(self.base_path, str(run))
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, self.render_mode), exist_ok=True)

        return True

    def cost(self, u):
        """
        Runs the simulation and compute cost function
        """
        # Input u is the 3n-dim vector [x1, ..., xn, vx1, vy1, ..., vxn, vyn]
        u_pos = u[0:self.n_bodies] # extract n initial x-positions
        u_vel = u[self.n_bodies:]  # extract 2n initial velocities

        # Construct the full initial state (4n dimensions)
        x_initial = np.zeros(self.sim_dim)
        for i in range(self.n_bodies):
            x_initial[4*i]     = u_pos[i]     # initial x_i
            x_initial[4*i + 1] = 0.0          # initial y_i (always 0)
            x_initial[4*i + 2] = u_vel[2*i]   # initial vx_i
            x_initial[4*i + 3] = u_vel[2*i+1] # initial vy_i

        # Initialize simulation state
        self.xk[:]   = x_initial
        self.hx[0,:] = self.xk[:]

        # Unroll trajectories
        penalty = 0.0
        for i in range(1, self.n_steps+1): # starting at 1 as 0 is initial step
            self.xk = verlet(self.xk, self.m, self.n_bodies, self.eps, self.G, self.dt)
            self.hx[i,:] = self.xk[:]

            # Scope penalty
            hx_reshaped = self.hx[i,:].reshape((self.n_bodies, 4))
            for k in range(self.n_bodies):
                r_sq = np.sum(hx_reshaped[k, 0:2]**2) # x^2 + y^2 for kth body
                if r_sq > self.scope_radius**2:
                    penalty += (r_sq - self.scope_radius**2)*self.dt

        # Compute return proximity cost
        start_idx    = min(self.min_return_index, self.n_steps)
        diff         = self.hx[start_idx:, :] - x_initial
        sq_distances = np.sum(diff**2, axis=1)
        total_cost   = np.min(sq_distances) + self.beta*penalty

        return total_cost

    def render(self, x_pop, c):
        """
        Renders the trajectory of the best candidate based on self.render_mode
        """
        best_idx = np.argmin(c)
        best_u = x_pop[best_idx] # Get the best 3n-dim vector
        best_initial_pos_x = best_u[0:self.n_bodies] # Extract n best initial x-positions

        _ = self.cost(best_u) # Rerun simulation to populate self.hx

        plot_limit = self.scope_radius * 1.1

        # png mode is used during training
        if self.render_mode == 'png':
            png_path = os.path.join(self.path, 'png')
            plt.clf(); plt.cla()
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_bodies))

            # Plot trajectories
            for i in range(self.n_bodies):
                ax.plot(self.hx[:, 4*i],
                        self.hx[:, 4*i+1],
                        label=f'B{i+1}',
                        alpha=0.7,
                        linewidth=1,
                        color=colors[i])

            # Plot start markers
            for i in range(self.n_bodies):
                label = 'Start' if i == 0 else None
                ax.scatter(best_initial_pos_x[i],
                           0.0,
                           marker='o',
                           s=50,
                           color=colors[i],
                           zorder=5,
                           label=label)

            # Plot end markers
            for i in range(self.n_bodies):
                label = 'End (t=Tmax)' if i == 0 else None
                ax.scatter(self.hx[-1, 4*i],
                           self.hx[-1, 4*i+1],
                           marker='x',
                           s=70,
                           color=colors[i],
                           zorder=5,
                           label=label)

            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-plot_limit, plot_limit)
            ax.set_ylim(-plot_limit, plot_limit)
            scope_circle = plt.Circle((0, 0),
                                      self.scope_radius,
                                      color='red',
                                      fill=False,
                                      linestyle=':',
                                      linewidth=1,
                                      label='scope')
            ax.add_patch(scope_circle)

            # Consolidate legend for n bodies
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            label_order = [] # Maintain order for bodies
            for handle, label in zip(handles, labels):
                 if label not in unique_labels:
                     unique_labels[label] = handle
                     label_order.append(label)
            ax.legend([unique_labels[lbl] for lbl in label_order],
                      label_order,
                      fontsize='small',
                      loc='upper right')

            filename = os.path.join(png_path, f"{self.it_plt}.png")
            plt.savefig(filename, dpi=100)
            plt.close(fig)

        # gif mode is used for rendering
        elif self.render_mode == 'gif':
            frame_base_dir = os.path.join(self.path, 'gif')
            current_iter_frame_dir = os.path.join(frame_base_dir, f'{self.it_plt}')
            os.makedirs(current_iter_frame_dir, exist_ok=True)

            # Compute plotting limits
            all_x = self.hx[:, 0:self.sim_dim:4].flatten()
            all_y = self.hx[:, 1:self.sim_dim:4].flatten()
            xmin = np.min(all_x)
            xmax = np.max(all_x)
            ymin = np.min(all_y)
            ymax = np.max(all_y)

            with mplstyle.context('dark_background'):
                num_frames_generated = 0
                colors = plt.cm.RdBu(np.linspace(0, 1, self.n_bodies))

                for i in range(0, self.n_steps + 1, self.plot_every_n_steps):
                    plt.clf(); plt.cla()
                    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
                    ax.set_facecolor('black')

                    # Plot trails
                    start_trail = max(0, i - self.trail_length)
                    for body_idx in range(self.n_bodies):
                        ax.plot(self.hx[start_trail:i+1, 4*body_idx],
                                self.hx[start_trail:i+1, 4*body_idx+1],
                                color=colors[body_idx],
                                alpha=0.6,
                                linewidth=1.0)

                    # Plot positions
                    for body_idx in range(self.n_bodies):
                        ax.scatter(self.hx[i, 4*body_idx],
                                   self.hx[i, 4*body_idx+1],
                                   marker='o',
                                   s=100,
                                   color=colors[body_idx],
                                   zorder=5,
                                   edgecolors='w',
                                   linewidth=0.5)

                    ax.set_aspect('equal', adjustable='box')
                    ax.set_xlim(1.1*xmin, 1.1*xmax)
                    ax.set_ylim(1.1*ymin, 1.1*ymax)
                    ax.axis('off')
                    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

                    frame_filename = os.path.join(current_iter_frame_dir,
                                                  f"{num_frames_generated}.png")
                    plt.savefig(frame_filename,
                                dpi=90,
                                facecolor='black',
                                bbox_inches='tight',
                                pad_inches=0.0)
                    plt.close(fig)
                    num_frames_generated += 1

        self.it_plt += 1

    def close(self):
        pass

###############################################
@nb.njit(cache=False)
def verlet(xk, m, n_bodies, eps, G, dt):
    """
    Performs a single Verlet integration step for n_bodies
    """
    # Compute accelerations a(t)
    acc = accelerations(xk, m, n_bodies, eps, G)

    # Reshape for easier vectorized operations
    xk_next = np.copy(xk).reshape((n_bodies, 4))
    acc     = acc.reshape((n_bodies, 2))

    # Update velocities halfway v(t + dt/2)
    xk_next[:, 2:4] += acc*0.5*dt # update vx, vy for all bodies

    # Update positions fully r(t + dt)
    xk_next[:, 0:2] += xk_next[:,2:4]*dt # update x, y for all bodies

    # Update acceleration a(t + dt) based on new positions
    acc_next = accelerations(xk_next.flatten(), m, n_bodies, eps, G)
    acc_next = acc_next.reshape((n_bodies, 2))

    # Update velocities fully v(t + dt)
    xk_next[:, 2:4] += acc_next*0.5*dt # update vx, vy

    return xk_next.flatten()

###############################################
@nb.njit(cache=False)
def accelerations(xk, m, n_bodies, eps, G):
    """
    Computes accelerations for n_bodies based on current positions in self.xk
    """
    acc = np.zeros(2*n_bodies) # ax, ay for each body
    pos = xk.reshape((n_bodies, 4))[:, 0:2] # gets [[x1,y1], [x2,y2], ...]

    for i in range(n_bodies):
        acc_i = np.zeros(2) # Accumulator for body i's acceleration
        pos_i = pos[i, :]
        for j in range(n_bodies):
            if i == j:
                continue

            pos_j = pos[j, :]
            m_j = m[j]

            # Vector from i to j
            r_ij = pos_j - pos_i
            dist_sq = np.sum(r_ij**2) + eps**2
            inv_dist_cubed = dist_sq**(-1.5)

            # Acceleration on i due to j: G * mj * r_ij / |r_ij|^3
            acc_i += G*m[j]*r_ij*inv_dist_cubed

        acc[2*i:2*i+2] = acc_i

    return acc
