import numpy as np

from sparkle.src.agent.lbfgsb import LBFGSB
from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.src.utils.compare import compare
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_lbfgsb():

    # Set seed for reproducible test
    set_seeds(0)

    # Initial space
    print("")

    # Test with function returning a scalar
    def f(x):
        v = (x[0]-3.14)**2 + (x[1]-2.72)**2 + np.sin(3*x[0]+1.41) + np.sin(4*x[1]-1.73)
        return v

    def df(x):
        v0 = 2.0*x[0]*(x[0]-3.14) + 3.0*np.cos(3.0*x[0]+1.41)
        v1 = 2.0*x[1]*(x[1]-3.14) + 4.0*np.cos(4.0*x[1]-1.73)
        return np.array([v0,v1])

    # Starting point and bounds for each variable.
    opt    = LBFGSB()
    x0     = np.array([2.5, 2.5])
    xmin   = np.array([0.0, 0.0])
    xmax   = np.array([5.0, 5.0])

    # Run the optimizer.
    x, c  = opt.optimize(f, x0, xmin, xmax, m=5, tol=1e-3, max_iter=20)
    x_ref = np.array([3.18515304089873, 3.129799326779645])
    c_ref = -1.8083520357842864
    assert np.allclose(x, x_ref)
    assert compare(c, c_ref, 1.0e-15)

    # Test with function returning a 0D array
    def g(x):
        return np.array([f(x)])

    x, c  = opt.optimize(g, x0, xmin, xmax, m=5, tol=1e-3, max_iter=20)
    assert np.allclose(x, x_ref)
    assert compare(c, c_ref, 1.0e-15)

    # Test by providing the gradient
    x, c  = opt.optimize(f, x0, xmin, xmax, df=df, m=5, tol=1e-3, max_iter=20)
    x_ref = np.array([3.17860673, 3.16116911])
    c_ref = -1.7993714113084178
    print(x,c)
    assert np.allclose(x, x_ref)
    assert compare(c, c_ref, 1.0e-15)

###############################################
def test_mslbfgsb():

    # Set seed for reproducible test
    set_seeds(0)

    # Initial space
    print("")

    def f(x):
        v = (x[0]-3.14)**2 + (x[1]-2.72)**2 + np.sin(3*x[0]+1.41) + np.sin(4*x[1]-1.73)
        return v

    def df(x):
        v0 = 2.0*x[0]*(x[0]-3.14) + 3.0*np.cos(3.0*x[0]+1.41)
        v1 = 2.0*x[1]*(x[1]-3.14) + 4.0*np.cos(4.0*x[1]-1.73)
        return np.array([v0,v1])

    # Starting point and bounds for each variable.
    opt    = MSLBFGSB()
    xmin   = np.array([0.0, 0.0])
    xmax   = np.array([5.0, 5.0])

    # Run the optimizer.
    x, c  = opt.optimize(f, xmin, xmax, n_pts=10, m=5, tol=1e-3, max_iter=20)
    x_ref = np.array([3.18515304089873, 3.129799326779645])
    c_ref = -1.8083520358717708
    assert np.allclose(x, x_ref)
    assert compare(c, c_ref, 1.0e-15)

    # Test with gradient
    x, c  = opt.optimize(f, xmin, xmax, df=df, n_pts=10, m=5, tol=1e-3, max_iter=20)
    x_ref = np.array([3.17955849, 3.13560734])
    c_ref = -1.8078819103729469
    assert np.allclose(x, x_ref)
    assert compare(c, c_ref, 1.0e-15)
