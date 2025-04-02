# Generic imports

# Custom imports
from sparkle.src.env.parallel import parallel
from numpy import float64, ndarray

###############################################
# Worker class for slave processes
class MpiWorker():
    def __init__(self, env_name: str, args: None, cpu: int, path: str) -> None:

        # Build environment
        module    = __import__(env_name)
        env_build = getattr(module, env_name)
        if args is not None:
            self.env = env_build(cpu, path, args)
        else:
            self.env = env_build(cpu, path)

    # Working function for slaves
    def work(self):
        while True:
            data    = None
            data    = parallel.comm().scatter(data, root=0)
            command = data[0]
            data    = data[1]

            # Execute commands
            if command == 'cost':
                c = self.cost(data)
                parallel.comm().gather((c), root=0)

            if command == 'reset':
                r = self.reset(data)
                parallel.comm().gather((r), root=0)

            if command == 'render':
                rnd = self.render(data)
                parallel.comm().gather((rnd), root=0)

            if command == 'close':
                self.close()
                parallel.finalize()
                break

    # Compute cost
    def cost(self, x: ndarray) -> float64:

        return self.env.cost(x)

    # Resetting
    def reset(self, run: int) -> bool:

        return self.env.reset(run)

    # Rendering
    def render(self, x, c, **kwargs):

        return self.env.render(x, c, **kwargs)

    # Closing
    def close(self) -> None:

        self.env.close()
