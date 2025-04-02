

###############################################
# Worker function for slave processes
def MultiprocWorker(env_name, args, cpu, path, pipe):

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
