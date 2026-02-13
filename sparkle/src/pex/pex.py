from sparkle.src.core.factory import Factory
from sparkle.src.pex.fpd import FPD
from sparkle.src.pex.lhs import LHS
from sparkle.src.pex.mlhs import MLHS
from sparkle.src.pex.mlhs_sa import MLHS_SA
from sparkle.src.pex.random import Random
from sparkle.src.pex.maxrfps import MaxRFPS
from sparkle.src.pex.minrfps import MinRFPS
from sparkle.src.pex.diff_pex import DiffPex

# Declare factory
pex_factory = Factory()

# Register pex
pex_factory.register("random",     Random)
pex_factory.register("lhs",        LHS)
pex_factory.register("mlhs",       MLHS)
pex_factory.register("mlhs_sa",    MLHS_SA)
pex_factory.register("fpd",        FPD)
pex_factory.register("maxrfps",    MaxRFPS)
pex_factory.register("minrfps",    MinRFPS)
pex_factory.register("diffpex",    DiffPex)
