from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from sparkle.src.utils.error import warning


###############################################
class LBFGSB():
    """
    A simplified implementation of the Limited-memory Broyden-Fletcher-Goldfarb-Shanno
    bound-constrained (L-BFGS-B) optimization algorithm.

    This class provides a method to optimize a function subject to box constraints
    (lower and upper bounds on the variables). It uses a limited-memory approximation
    of the inverse Hessian matrix to compute search directions and a line search
    procedure to determine the step length.
    """
    def __init__(self) -> None:
        """
        Initializes the LBFGSB optimizer.
        """
        pass

    def optimize(self,
                 f0: Callable,
                 x0: ndarray,
                 xmin: ndarray,
                 xmax: ndarray,
                 df: Optional[Callable]=None,
                 m: int=10,
                 tol: float=1e-6,
                 max_iter: int=100) -> Tuple[ndarray, float]:
        """
        Optimizes a function using the L-BFGS-B algorithm.

        Args:
            f0: The objective function to minimize. It should accept a NumPy array
                as input and return a scalar or a list/tuple/array with the first element being the scalar value.
            x0: The initial guess for the minimum, as a NumPy array.
            xmin: The lower bounds for the variables, as a NumPy array.
            xmax: The upper bounds for the variables, as a NumPy array.
            df: An optional function to compute the gradient of f0. If not provided,
                finite differences will be used.
            m: The maximum number of correction pairs to store (memory size).
            tol: The tolerance for the norm of the projected gradient, used as a
                convergence criterion.
            max_iter: The maximum number of iterations.

        Returns:
            A tuple containing:
                - The optimized point (NumPy array).
                - The value of the objective function at the optimized point (float).

        Raises:
            Warning: If the algorithm does not converge within the maximum number of iterations.
        """

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

    def grad_fd(self,
                f: Callable,
                x: ndarray,
                dx: float) -> ndarray:
        """
        Computes the gradient of a function using finite differences.

        Args:
            f: The function to compute the gradient of.
            x: The point at which to compute the gradient.
            dx: The step size for finite differences.

        Returns:
            The gradient of f at x, as a NumPy array.
        """

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

    def project(self,
                x: ndarray,
                l: ndarray,
                u: ndarray) -> ndarray:
        """
        Projects a point onto the feasible set defined by lower and upper bounds.

        Args:
            x: The point to project.
            l: The lower bounds.
            u: The upper bounds.

        Returns:
            The projected point.
        """
        return np.clip(x, l, u)

    def projected_gradient(self,
                           x: ndarray,
                           g: ndarray,
                           l: ndarray,
                           u: ndarray) -> ndarray:
        """
        Computes the projected gradient.

        Args:
            x: The current point.
            g: The gradient at the current point.
            l: The lower bounds.
            u: The upper bounds.

        Returns:
            The projected gradient.
        """
        return x - self.project(x - g, l, u)

    def generalized_cauchy_point(self,
                                 x: ndarray,
                                 g: ndarray,
                                 l: ndarray,
                                 u: ndarray) -> ndarray:
        """
        Computes a generalized Cauchy point.

        This is a simplified version where, for each coordinate, we find the step
        length required to hit a bound when moving along -g, then choose the
        smallest positive step.

        Args:
            x: The current point.
            g: The gradient at the current point.
            l: The lower bounds.
            u: The upper bounds.

        Returns:
            The generalized Cauchy point.
        """
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

    def two_loop_recursion(self,
                           g: ndarray,
                           s_list: List[ndarray],
                           y_list: List[ndarray]) -> ndarray:
        """
        Performs the two-loop recursion to compute the approximate Hessian-vector product.

        Given stored correction pairs (s,y), this method computes H*g, where H is
        the approximate inverse Hessian.

        Args:
            g: The vector to multiply by the approximate Hessian.
            s_list: A list of stored step differences (s = x_k+1 - x_k).
            y_list: A list of stored gradient differences (y = grad_k+1 - grad_k).

        Returns:
            The approximate Hessian-vector product H*g.
        """
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

    def search_direction(self,
                         x: ndarray,
                         g: ndarray,
                         s_list: List[Union[ndarray, Any]],
                         y_list: List[Union[ndarray, Any]],
                         l: ndarray,
                         u: ndarray) -> ndarray:
        """
        Computes the search direction using the L-BFGS two-loop recursion.

        Then adjusts the direction for variables that are at their bounds.

        Args:
            x: The current point.
            g: The gradient at the current point.
            s_list: A list of stored step differences.
            y_list: A list of stored gradient differences.
            l: The lower bounds.
            u: The upper bounds.

        Returns:
            The search direction.
        """
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
        """
        Performs a backtracking line search with strong Wolfe conditions.

        Args:
            f: The objective function.
            x: The current point.
            dx: The step size for finite differences.
            d: The search direction.
            l: The lower bounds.
            u: The upper bounds.
            alpha_init: The initial step size.
            c1: The Armijo condition constant.
            c2: The strong Wolfe curvature condition constant.
            tau: The step size reduction factor.
            max_iters: The maximum number of iterations.

        Returns:
            The step size that satisfies the strong Wolfe conditions.
        """

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
