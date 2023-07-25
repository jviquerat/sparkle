# Generic imports
import sys

# Custom imports
from sparkle.src.env.parallel import parallel

###############################################
# Worker function for slave processes
def multiproc_worker(env_name, args, cpu, path, pipe):

    # Build environment
    module    = __import__(env_name)
    env_build = getattr(module, env_name)
    if args is not None:
        env = env_build(cpu, path, args)
    else:
        env = env_build(cpu, path)

    # Execute tasks
    try:
        while True:
            command, data = pipe.recv()

            if command == 'cost':
                c = env.cost(data)
                pipe.send(c)

            if command == 'reset':
                r = env.reset(data)
                pipe.send(r)

            if (command == 'render'):
                rnd = env.render(data)
                pipe.send(rnd)

            if command == 'close':
                pipe.send(None)
                break

            if command == 'dim':
                pipe.send(env.dim)

            if command == 'xmin':
                pipe.send(env.xmin)

            if command == 'xmax':
                pipe.send(env.xmax)
    finally:
        env.close()
