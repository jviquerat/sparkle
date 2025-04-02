import numpy as np
from numpy import ndarray
from typing import Any, Callable, List, Optional, Tuple, Union

from sparkle.src.utils.error import warning

###############################################
### Simplified L-BFGS-B algorithm
class LBFGSB():
    def __init__(self) -> None:
        pass

    # f0        : function to minimize
    # x0        : initial guess
    # xmin      : minimal bounds
    # xmax      : maximal bounds
    # df        : gradient of f0
    # max_pairs : max nb of correction pairs
    # tol       : tolerance for the norm of the projected gradient
    # max_iter  : max nb of iteration
    def optimize(self,
                 f0: Callable,
                 x0: ndarray,
                 xmin: ndarray,
                 xmax: ndarray,
                 df: Optional[Callable]=None,
                 m: int=10,
                 tol: float=1e-6,
                 max_iter: int=100) -> Tuple[ndarray, float]:

        # Copy input
        x = x0.copy()

        # Extract lower and upper bounds as NumPy arrays.
        l = xmin
        u = xmax

        # Define dx vectors for differenciation
        dx  = 0.0001*(np.max(u) - np.min(l))

        # Wrap f for array outputs if necessary
        is_fx_array = isinstance(f0(x0), (list, tuple, np.ndarray))
        if (is_fx_array): f = lambda x: f0(x)[0]
        else:             f = f0

        # Select gradient function
        if df is not None: self.grad_f = lambda f, x, dx: df(x)
        else:              self.grad_f = lambda f, x, dx: self.grad_fd(f, x, dx)

        # Lists for recent steps and gradient differences
        s_list = [] # (s = x_k+1 - x_k)
        y_list = [] # (y = grad_k+1 - grad_k)

        # Main loop
        for k in range(max_iter):

            # Compute gradient
            g = self.grad_f(f, x, dx)

            # Check convergence using the norm of the projected gradient.
            pg = self.projected_gradient(x, g, l, u)
            if np.linalg.norm(pg) < tol: break

            # Compute a generalized Cauchy point
            x_c = self.generalized_cauchy_point(x, g, l, u)

            # Compute the search direction
            d = self.search_direction(x, g, s_list, y_list, l, u)

            # Perform line search to determine step length.
            alpha = self.strong_wolfe_line_search(f, x, dx, d, l, u)

            # Update the iterate (always project back onto the feasible set)
            x_new = self.project(x + alpha * d, l, u)

            # Update correction pairs
            s     = x_new - x
            g_new = self.grad_f(f, x_new, dx)
            y     = g_new - g

            if np.dot(s, y) > 1e-10:
                s_list.append(s)
                y_list.append(y)

                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)

            # Update current point
            x = x_new

        # Check if not converged
        if (k == max_iter):
            warning("lbfgs", "optimize",
                    "lbfgs did not converge in "+str(max_iter)+" iterations")

        return x, f(x)

    # Naive gradient computation using finite differences
    def grad_fd(self,
                f: Callable,
                x: ndarray,
                dx: float) -> ndarray:

        n = x.shape[0]
        m = f(x).size
        J = np.zeros((m,n))

        x_f = x.copy()
        x_b = x.copy()

        for j in range(n):
            x_f[j] += dx
            x_b[j] -= dx
            J[:,j]  = 0.5*(f(x_f)-f(x_b))/dx
            x_f[j] -= dx
            x_b[j] += dx

        return np.reshape(J, (n))

    # Project x in bounds
    def project(self,
                x: ndarray,
                l: ndarray,
                u: ndarray) -> ndarray:
        return np.clip(x, l, u)

    # Compute the projected gradient
    # pg = x - project(x - g)
    def projected_gradient(self,
                           x: ndarray,
                           g: ndarray,
                           l: ndarray,
                           u: ndarray) -> ndarray:
        return x - self.project(x - g, l, u)

    # Compute genereralized cauchy point
    # This is a simplified version:
    # For each coordinate, we find the step length required to hit a bound
    # when moving along -g, then choose the smallest positive step
    def generalized_cauchy_point(self,
                                 x: ndarray,
                                 g: ndarray,
                                 l: ndarray,
                                 u: ndarray) -> ndarray:
        alpha_candidates = []
        for i in range(len(x)):
            if g[i] > 0:
                alpha_i = (x[i] - l[i]) / g[i]
            elif g[i] < 0:
                alpha_i = (x[i] - u[i]) / g[i]
            else:
                alpha_i = np.inf
            if alpha_i > 0:
                alpha_candidates.append(alpha_i)
        alpha = min(alpha_candidates) if alpha_candidates else 1.0
        x_c = self.project(x - alpha * g, l, u)
        return x_c

    # Two-loop recursion to compute the approximate Hessian-vector product:
    # Given stored correction pairs (s,y), compute H*g
    def two_loop_recursion(self,
                           g: ndarray,
                           s_list: List[ndarray],
                           y_list: List[ndarray]) -> ndarray:
        q = g.copy()
        alpha_list = []
        rho_list = []
        # Loop backward over stored pairs
        for s, y in zip(reversed(s_list), reversed(y_list)):
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            alpha_list.append(alpha)
            q = q - alpha * y
            rho_list.append(rho)
        # Use scaling factor gamma = (s^T y)/(y^T y) from the most recent pair
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
        else:
            gamma = 1.0
        r = gamma * q
        # Loop forward over stored pairs
        for s, y, rho, alpha in zip(s_list, y_list, rho_list[::-1], alpha_list[::-1]):
            beta = rho * np.dot(y, r)
            r = r + s * (alpha - beta)
        return r

    # Compute the search direction using the L-BFGS two-loop recursion
    # Then adjust the direction for variables that are at their bounds
    def search_direction(self,
                         x: ndarray,
                         g: ndarray,
                         s_list: List[Union[ndarray, Any]],
                         y_list: List[Union[ndarray, Any]],
                         l: ndarray,
                         u: ndarray) -> ndarray:
        if len(s_list) > 0:
            H_g = self.two_loop_recursion(g, s_list, y_list)
        else:
            H_g = g.copy()
        # Proposed direction is the negative of the approximate Hessian times gradient.
        d = -H_g
        # For coordinates at a bound, do not allow a step that would violate the bound.
        d_adjusted = d.copy()
        for i in range(len(x)):
            if x[i] <= l[i] and d[i] < 0:
                d_adjusted[i] = 0
            elif x[i] >= u[i] and d[i] > 0:
                d_adjusted[i] = 0
        return d_adjusted

    # Backtracking line search with strong wolfe
    def strong_wolfe_line_search(self,
                                 f: Callable,
                                 x: ndarray,
                                 dx: float,
                                 d: ndarray,
                                 l: ndarray,
                                 u: ndarray,
                                 alpha_init: float=1.0,
                                 c1: float=1e-3,
                                 c2: float=0.9,
                                 tau: float=0.5,
                                 max_iters: int=20) -> float:

        alpha = alpha_init
        f_x = f(x)
        grad_x = self.grad_f(f, x, dx)
        grad_dot_d = np.dot(grad_x, d)  # ∇f(x_k) · d_k

        for _ in range(max_iters):
            # Compute new candidate point with projection onto bounds
            x_new = self.project(x + alpha * d, l, u)
            f_new = f(x_new)
            grad_new = self.grad_f(f, x_new, dx)
            grad_new_dot_d = np.dot(grad_new, d)

            # Armijo Condition (sufficient decrease)
            if f_new > f_x + c1 * alpha * grad_dot_d:
                alpha *= tau  # Reduce step size
                continue

            # Strong Wolfe Curvature Condition
            if abs(grad_new_dot_d) > c2 * abs(grad_dot_d):
                alpha *= tau  # Reduce step size
                continue

            # Both conditions satisfied
            return alpha

        return alpha  # Return final step size if max iterations reached
