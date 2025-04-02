from sparkle.src.core.factory import Factory
from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.kernel.matern52 import Matern52

# Declare factory
kernel_factory = Factory()

# Register kernels
kernel_factory.register("gaussian", Gaussian)
kernel_factory.register("matern52", Matern52)

