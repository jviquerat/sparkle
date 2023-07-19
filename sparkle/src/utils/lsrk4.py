# Generic imports
import numpy as np

###############################################
### Five-stage fourth-order low-storage Runge-Kutta class
class lsrk4():
    def __init__(self):

        # lsrk coefficients
        self.n_lsrk = 5
        self.a = np.array([ 0.000000000000000, -0.417890474499852,
                           -1.192151694642677, -1.697784692471528,
                           -1.514183444257156])
        self.b = np.array([ 0.149659021999229,  0.379210312999627,
                            0.822955029386982,  0.699450455949122,
                            0.153057247968152])
        self.c = np.array([ 0.000000000000000,  0.149659021999229,
                            0.370400957364205,  0.622255763134443,
                            0.958282130674690])

    # return number of integration steps
    def steps(self):

        return self.n_lsrk

    # return source time at jth step
    def source_time(self, j, t, dt):

        return t + self.c[j]*dt

    # lsrk update
    def update(self, u, uk, f, j, dt):

        u[:]   = self.a[j]*u[:] + dt*f[:]
        uk[:] += self.b[j]*u[:]
