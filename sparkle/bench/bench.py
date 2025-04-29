from sparkle.src.core.factory import Factory
from sparkle.bench.pex import BenchPex

# Declare factory
bench_factory = Factory()

# Register benchmarks
bench_factory.register("pex", BenchPex)

