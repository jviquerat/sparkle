from sparkle.src.core.factory import Factory
from sparkle.bench.pex import BenchPex
from sparkle.bench.lbfgsb import BenchLBFGSB

# Declare factory
bench_factory = Factory()

# Register benchmarks
bench_factory.register("pex", BenchPex)
bench_factory.register("lbfgsb", BenchLBFGSB)

