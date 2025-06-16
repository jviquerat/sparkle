import os
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default

###############################################
class emstack(base_env):
    """
    Optimization of a dielectric mirror for maximum reflectivity.
    For a fixed number of layers N, the optimizer chooses the material
    for each layer (discrete) and the thickness of each layer (continuous).
    The cost function is the negative average reflectivity over a given
    wavelength spectrum.
    """
    def __init__(self, cpu, path, pms=None):

        self.name = 'emstack'
        self.base_path = path

        # Materials parameters
        self.n_layers        = set_default("n_layers", 20, pms)
        self.materials       = set_default("materials", {"TiO2": 2.4, "MgF2": 1.38}, pms)
        self.material_names  = list(self.materials.keys())
        self.material_values = np.array(list(self.materials.values()))
        self.n_materials     = self.material_values.size

        self.n_incident  = set_default("n_incident", 1.0, pms)   # Air
        self.n_substrate = set_default("n_substrate", 1.52, pms) # Glass

        self.continuous_dim = self.n_layers
        self.discrete_dim   = self.n_layers
        self.dim            = self.continuous_dim + self.discrete_dim

        material_indices   = list(range(self.n_materials))
        self.discrete_cats = [material_indices for _ in range(self.discrete_dim)]

        # Thicknesses are continuous values in nm
        self.x0   = set_default("x0",   100.0*np.ones(self.continuous_dim), pms)
        self.xmin = set_default("xmin",  50.0*np.ones(self.continuous_dim), pms)
        self.xmax = set_default("xmax", 150.0*np.ones(self.continuous_dim), pms)

        # Wavelength integration parameters in nm
        self.lambda_min      = set_default("lambda_min", 300.0, pms)
        self.lambda_max      = set_default("lambda_max", 500.0, pms)
        self.n_lambda_points = set_default("n_lambda_points", 200, pms)
        self.wavelengths     = np.linspace(self.lambda_min,
                                           self.lambda_max,
                                           self.n_lambda_points)

        self.it_plt = 0

    def reset(self, run):
        """
        Resets the environment for a new run.
        """

        self.path   = os.path.join(self.base_path, str(run))
        self.png_path = os.path.join(self.path, 'png')
        self.dat_path = os.path.join(self.path, 'dat')
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.png_path, exist_ok=True)
        os.makedirs(self.dat_path, exist_ok=True)

        return True

    def cost(self, u):
        """
        Runs the simulation and computes the cost function.
        The cost is the negative average reflectivity.
        """
        # Unpack the optimization vector u
        # Round materials to nearest integer
        thicknesses   = u[:self.n_layers]
        materials_idx = np.round(u[self.n_layers:]).astype(np.int32)

        # Get the refractive indices corresponding to the chosen materials
        n_values_for_layers = self.material_values[materials_idx]

        # Calculate reflectivity for each wavelength in the spectrum
        spectrum = np.zeros_like(self.wavelengths)
        for i, lambda_val in enumerate(self.wavelengths):
            spectrum[i] = transfer_matrix_reflectivity(thicknesses,
                                                       n_values_for_layers,
                                                       self.n_incident,
                                                       self.n_substrate,
                                                       lambda_val)

        dev = np.max(spectrum) - np.min(spectrum)

        return -np.mean(spectrum)

    def render(self, x_pop, c):
        """
        Renders the reflectivity spectrum and layer stack of the best candidate.
        """

        best_idx = np.argmin(c)
        best_u = x_pop[best_idx]

        # Unpack the best solution for plotting
        thicknesses   = best_u[0:self.continuous_dim]
        materials_idx = best_u[self.continuous_dim:].astype(np.int32)
        n_values_for_layers = self.material_values[materials_idx]

        # Recalculate the spectrum for the best individual
        spectrum = np.zeros_like(self.wavelengths)
        for i, lambda_val in enumerate(self.wavelengths):
            spectrum[i] = transfer_matrix_reflectivity(thicknesses,
                                                       n_values_for_layers,
                                                       self.n_incident,
                                                       self.n_substrate,
                                                       lambda_val)

        with plt.style.context('dark_background'):
            font_settings = {
                'axes.titlesize': 10,
                'axes.labelsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 6,
            }
            plt.rcParams.update(font_settings)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3),
                                           gridspec_kw={'height_ratios': [2, 0.5]},
                                           dpi=100)

            # Reflectivity spectrum
            ax1.plot(self.wavelengths, spectrum, color='#00aaff', linewidth=1.5)
            ax1.set_ylabel("Reflectivity")
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylim(0, 1.05)
            ax1.tick_params(direction='in', colors='gray')
            ax1.grid(True, linestyle='--', color='gray', alpha=0.3)

            mean_r = np.mean(spectrum)
            ax1.axhline(mean_r,
                        color='red',
                        linestyle=':',
                        label=f'mean R = {mean_r:.4f}')

            # Create a list to hold the colored rectangles for the legend
            material_patches = []
            denominator = max(1, self.n_materials - 1)
            for i, name in enumerate(self.material_names):
                normalized_value = 0.5*(i+1)/denominator
                color = plt.cm.inferno(normalized_value)
                patch = plt.Rectangle((0, 0), 1, 1, color=color)
                material_patches.append(patch)

            # Combine legends: the one from axhline and the material patches
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles=handles + material_patches,
                       labels=labels + self.material_names,
                       loc='lower right')

            # Layer stack visualization
            colors = plt.cm.inferno(np.linspace(0.5, 1.0, self.n_materials))
            y_pos = 0
            for i in range(self.n_layers):
                mat_idx = materials_idx[i]
                thick = thicknesses[i]
                ax2.barh(0,
                         thick,
                         left=y_pos,
                         height=1.0,
                         color=colors[mat_idx],
                         edgecolor='black',
                         linewidth=0.5)
                y_pos += thick
            ax2.axis('off')

            plt.tight_layout(pad=0.8)
            filename = os.path.join(self.png_path, f"{self.it_plt}.png")
            plt.savefig(filename)
            plt.close(fig)

        # Save reflectance
        data = np.vstack((self.wavelengths, spectrum)).T
        filename = os.path.join(self.dat_path, f"{self.it_plt}.dat")
        np.savetxt(filename, data, fmt='%.6f', delimiter=' ')

        self.it_plt += 1

    def close(self):
        pass

###############################################
# Transfer matrix computation
@nb.njit(cache=False)
def transfer_matrix_reflectivity(thicknesses,
                                 n_layers,
                                 n_incident,
                                 n_substrate,
                                 lambda_val):
    """
    Calculates reflectivity of a thin-film stack for a single wavelength.

    Args:
        thicknesses (np.array): Thickness of each layer (in nm).
        n_layers (np.array): Refractive index of each layer.
        n_incident (float): Refractive index of the incident medium (e.g., air).
        n_substrate (float): Refractive index of the substrate (e.g., glass).
        lambda_val (float): Wavelength of light (in nm).

    Returns:
        float: The reflectivity (a value between 0 and 1).
    """
    m11, m12, m21, m22 = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    n_current = n_substrate

    for i in range(len(thicknesses) - 1, -1, -1):
        n_prev = n_layers[i]
        d_prev = thicknesses[i]

        r = (n_current - n_prev) / (n_current + n_prev)
        t_inv = (n_current + n_prev) / (2 * n_current)

        delta = (2.0 * np.pi * n_prev * d_prev) / lambda_val
        L_11 = np.exp(1j * delta)
        L_22 = np.exp(-1j * delta)

        lm11, lm12 = L_11 * m11, L_11 * m12
        lm21, lm22 = L_22 * m21, L_22 * m22

        m11_new = (lm11 + r * lm21) * t_inv
        m12_new = (lm12 + r * lm22) * t_inv
        m21_new = (r * lm11 + lm21) * t_inv
        m22_new = (r * lm12 + lm22) * t_inv

        m11, m12, m21, m22 = m11_new, m12_new, m21_new, m22_new
        n_current = n_prev

    # The final interface doesn't need a matrix, just the reflection coefficient
    r_total = m21 / m11
    reflectance = np.abs(r_total)**2

    return reflectance
