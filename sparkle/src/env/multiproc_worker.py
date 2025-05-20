###############################################
def MultiprocWorker(env_name, args, cpu, path, pipe):
    """
    Worker function for multiprocessing slave processes.

    This function defines the behavior of a worker process in a
    multiprocessing-based parallel environment. It handles communication
    with the main process and executes commands related to environment
    interaction.

    Args:
        env_name: The name of the environment module.
        args: Additional arguments for the environment constructor.
        cpu: The CPU index for the worker.
        path: The base path for storing results.
        pipe: The communication pipe for sending and receiving data.
    """

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

            if command == 'validate':
                v = env.validate(data)
                pipe.send(v)

            elif command == 'reset':
                r = env.reset(data)
                pipe.send(r)

            elif command == 'render':
                rnd = env.render(data[0], data[1])
                pipe.send(rnd)

            elif command == 'close':
                pipe.send(None)
                break

            elif command == 'dim':
                pipe.send(env.dim)

            elif command == 'x0':
                pipe.send(env.x0)

            elif command == 'xmin':
                pipe.send(env.xmin)

            elif command == 'xmax':
                pipe.send(env.xmax)

            elif command == 'vmin':
                pipe.send(env.vmin)

            elif command == 'vmax':
                pipe.send(env.vmax)

            elif command == 'levels':
                pipe.send(env.levels)

            else:
                pipe.send(None)
    finally:
        env.close()
