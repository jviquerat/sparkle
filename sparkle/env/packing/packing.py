import os
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.patches import Circle, Rectangle

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default

###############################################
class packing(base_env):
    """
    Disc packing in square domain
    The goal is to find the tightest packing of discs in a square
    Known solutions up to dimension 24 are available here:
    https://erich-friedman.github.io/packing/cirinsqu/
    """
    def __init__(self, cpu, path, pms=None):

        self.name      = "packing"
        self.base_path = path
        self.cpu       = cpu

        self.n_spheres         = set_default("n_spheres", 6, pms)
        self.radius            = 1.0
        self.max_boundary_side = set_default("max_boundary_side", 7.0, pms)

        self.dim  = 2*self.n_spheres
        self.x0   = np.full(self.dim, 0.5*self.max_boundary_side)
        self.xmin = np.full(self.dim, self.radius)
        self.xmax = np.full(self.dim, self.max_boundary_side - self.radius)

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = os.path.join(self.base_path, str(run))
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)

        return True

    def cost(self, x):
        """
        Runs the simulation and compute cost function
        """
        coords = x.reshape((self.n_spheres, 2))
        overlap, box_side, _, _ = penalties(coords, self.radius)

        return overlap + box_side

    def render(self, x_pop, c):
        """
        Renders the best sphere packing configuration from the provided population

        The plot area is fixed to [0, max_boundary_side]. The visual content
        (spheres and their bounding box) is shifted to appear centered within
        this fixed plot area.
        """
        best_idx = np.argmin(c)
        best_x = x_pop[best_idx]

        best_x = best_x.reshape((self.n_spheres, 2))
        overlap, box_side, box_x_min, box_y_min = penalties(best_x, self.radius)

        # Center of the bounding box
        content_center_x = box_x_min + 0.5*box_side
        content_center_y = box_y_min + 0.5*box_side

        # Center of the fixed plot window
        plot_center_x = 0.5*self.max_boundary_side
        plot_center_y = 0.5*self.max_boundary_side

        # Offset to move content center to plot center
        offset_x = plot_center_x - content_center_x
        offset_y = plot_center_y - content_center_y

        with mplstyle.context('dark_background'):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal', adjustable='box')

            ax.set_xlim([-2, self.max_boundary_side + 2])
            ax.set_ylim([-2, self.max_boundary_side + 2])
            ax.axis('off')

            # Add circles for spheres, applying the offset for visual centering
            for i in range(self.n_spheres):
                circle = Circle((best_x[i, 0] + offset_x, best_x[i, 1] + offset_y),
                                self.radius,
                                facecolor='white',
                                edgecolor='white',
                                linewidth=1.0,
                                alpha=0.6,
                                zorder=5)
                ax.add_patch(circle)

            # Add the minimum bounding square, applying the offset for visual centering
            # The rectangle's origin (bottom-left) needs the offset. Size remains box_s.
            bounding_box = Rectangle((box_x_min + offset_x, box_y_min + offset_y),
                                     box_side, box_side,
                                     facecolor='none',
                                     edgecolor='red',
                                     linewidth=2,
                                     linestyle='--',
                                     zorder=10)
            ax.add_patch(bounding_box)

            filename = os.path.join(self.path, f"{self.it_plt}.png")
            plt.savefig(filename,
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        facecolor='black')
            plt.close(fig)

        self.it_plt += 1

    def close(self) -> None:
        pass

###############################################
@nb.njit(fastmath=True, cache=False)
def penalties(coords, radius):

    n_spheres   = coords.shape[0]
    min_dist_sq = (2.0*radius - 1e-9)**2
    overlap     = 0.0

    # Compute overlap penalties
    for i in range(n_spheres):
        for j in range(i+1, n_spheres):
            x0 = coords[i, 0]
            y0 = coords[i, 1]
            x1 = coords[j, 0]
            y1 = coords[j, 1]

            dist_sq = (x1 - x0)**2 + (y1 - y0)**2
            if dist_sq < min_dist_sq:
                dist             = np.sqrt(dist_sq)
                overlap         += 5.0*max(0.0, 2.0*radius - dist)**2

    # Compute boundary penalties
    center_x_min = np.min(coords[:, 0])
    center_x_max = np.max(coords[:, 0])
    center_y_min = np.min(coords[:, 1])
    center_y_max = np.max(coords[:, 1])

    box_x_min = center_x_min - radius
    box_x_max = center_x_max + radius
    box_y_min = center_y_min - radius
    box_y_max = center_y_max + radius
    box_side  = max(box_x_max - box_x_min, box_y_max - box_y_min)

    return overlap, box_side, box_x_min, box_y_min
