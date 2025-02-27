# Custom imports
from sparkle.src.core.factory    import factory
from sparkle.src.kernel.gaussian import gaussian
from sparkle.src.kernel.matern52 import matern52

# Declare factory
kernel_factory = factory()

# Register kernels
kernel_factory.register("gaussian", gaussian)
kernel_factory.register("matern52", matern52)

