# Generic imports
import sys

# Custom imports
from sparkle.src.env.mpi import *

###############################################
# Worker class for slave processes
class worker():
    def __init__(self, env_name, args, cpu, path):

        # Build environment
        module    = __import__(env_name)
        env_build = getattr(module, env_name)
        if args is not None:
            #self.env = env_build(cpu, path, **args.__dict__)
            self.env = env_build(cpu, path, args)
        else:
            self.env = env_build(cpu, path)

    # Working function for slaves
    def work(self):
        while True:
            data    = None
            data    = mpi.comm.scatter(data, root=0)
            command = data[0]
            data    = data[1]

            # Execute commands
            if command == 'cost':
                c = self.cost(data)
                mpi.comm.gather((c), root=0)

            if command == 'reset':
                r = self.reset()
                mpi.comm.gather((r), root=0)

            if command == 'render':
                rnd = self.render(data)
                mpi.comm.gather((rnd), root=0)

            if command == 'close':
                self.close()
                mpi.finalize()
                break

    # Compute cost
    def cost(self, x):

        return self.env.cost(x)

    # Resetting
    def reset(self, run):

        return self.env.reset(run)

    # Rendering
    def render(self, x):

        return self.env.render(x)

    # Closing
    def close(self):

        self.env.close()
