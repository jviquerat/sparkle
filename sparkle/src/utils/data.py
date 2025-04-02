import numpy as np

from sparkle.src.utils.ema import EMA
from numpy import ndarray

###############################################
### Data averager class
### Used to compute avg+/-std of drl-related fields
### n_avg : nb of runs to average
class DataAvg():
    def __init__(self, n_fields: int, n_avg: int) -> None:

        self.n_avg    = n_avg
        self.n_fields = n_fields
        self.n_stp    = 0
        self.stp      = None
        self.data     = None

    def store(self, filename: str, run: int) -> None:

        # Load file
        f = np.loadtxt(filename)

        # Deduce file size if this is the first one
        if (self.stp is None and self.data is None):
            self.n_stp = f.shape[0]
            self.stp   = np.zeros((self.n_stp), dtype=int)
            self.data  = np.zeros((self.n_avg, self.n_stp, self.n_fields), dtype=float)

        # Check file size if this is not the first one
        assert(f.shape[0] == self.n_stp)

        self.stp = f[:, 0]
        for field in range(self.n_fields):
            self.data[run,:,field] = f[:,field+1]

    def average(self, filename: str, avg_type: str="linear") -> ndarray:

        array = np.vstack(self.stp)
        smoother = EMA(0.2, int(self.n_stp/10))

        for field in range(self.n_fields):
            avg   = np.mean(self.data[:,:,field], axis=0)
            std   = np.std (self.data[:,:,field], axis=0)
            p     = avg + std
            m     = avg - std

            if (avg_type == "log"):
                log_avg = np.log(avg)
                log_std = 0.434*(p-avg)/avg
                log_p   = log_avg + log_std
                log_m   = log_avg - log_std
                p       = np.exp(log_p)
                m       = np.exp(log_m)

            smooth_avg = smoother.smooth(avg)
            smooth_p   = smoother.smooth(p)
            smooth_m   = smoother.smooth(m)

            array = np.hstack((array,np.vstack(smooth_avg)))
            array = np.hstack((array,np.vstack(smooth_p)))
            array = np.hstack((array,np.vstack(smooth_m)))

        np.savetxt(filename, array, fmt='%.5e')
        
        return array
