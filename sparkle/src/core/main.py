import sys

from sparkle.src.core.average import average
from sparkle.src.core.evaluate import evaluate
from sparkle.src.core.model import model
from sparkle.src.core.sample import sample
from sparkle.src.core.train import train
from sparkle.src.env.parallel import parallel
from sparkle.bench.bench import bench_factory
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import bold, disclaimer, err_print, liner, spacer
from sparkle.src.utils.seeds import set_seeds


def helper():
    """
    Displays the command-line usage instructions.
    """
    liner(err_print("Command line error"))
    spacer("Command line functionalities:")
    spacer("   spk --train <json_file>")
    spacer("   spk --evaluate -dat <dat_file> -json <json_file>")
    spacer("   spk --average <dat_file_0> ... <dat_file_n>")
    spacer("   spk --model <json_file>")
    spacer("   spk --pex -type <pex_type> -n_points <n_points> -dim <dim>")
    spacer("   spk --bench <bench_name> -json <json_file>")
    spacer("Optional arguments:")
    spacer("       --set-seeds <seed>")
    sys.exit(0)

def main():
    """
    Main entry point for the Sparkle framework.

    This function parses command-line arguments and dispatches to the
    appropriate functionality (training, evaluation, averaging, model
    generation, or Pex sampling).
    """

    # Check arguments
    args = sys.argv

    # Check for set_seeds option
    if ("--set-seeds" in args):
        seed = args[args.index("--set-seeds")+1]
        set_seeds(int(seed))

    # Training mode
    if ("--train" in args):

        # Initialize json parser and read parameters
        json_file = args[args.index("--train")+1]
        parser    = JsonParser()
        pms       = parser.read(json_file)

        # Set parallel framework
        parallel.set(pms)

        # Printings
        disclaimer()
        liner(bold('Training mode'))

        if (parallel.is_root):
            spacer("Number of parallel environments: "+str(parallel.n_envs))
            spacer("Number of procs per environment: "+str(parallel.n_procs_per_env))

        train(json_file, pms)
        return

    # Evaluation mode
    if ("--evaluate" in args):

        if ("-dat" not in args): helper()
        dat_file  = args[args.index("-dat")+1]

        if ("-json" not in args): helper()
        json_file = args[args.index("-json")+1]

        # Set parallel framework
        parallel.set(None)

        # Printings
        disclaimer()
        liner(bold('Evaluation mode'))

        evaluate(dat_file, json_file)
        return

    # Averaging mode
    if ("--average" in args):

        # Set parallel framework
        parallel.set({})

        # Printings
        disclaimer()
        liner(bold('Average mode'))

        dat_args = args[args.index("--avg")+1:]
        average(dat_args)
        return

    # Model mode
    if ("--model" in args):

        # Initialize json parser and read parameters
        json_file = args[args.index("--model")+1]
        parser    = JsonParser()
        pms       = parser.read(json_file)

        # Set parallel framework
        parallel.set(pms)

        # Printings
        disclaimer()
        liner(bold('Model mode'))

        model(json_file, pms)
        return

    # Pex sampling mode
    if ("--pex" in args):

        # Read parameters
        if ("-type" not in args): helper()
        pex_type = args[args.index("-type")+1]

        if ("-n_points" not in args): helper()
        n_points = args[args.index("-n_points")+1]
        n_points = int(n_points)

        if ("-dim" not in args): helper()
        dim = args[args.index("-dim")+1]
        dim = int(dim)

        # Set parallel framework
        parallel.set({})

        # Printings
        disclaimer()
        liner(bold('Pex sampling mode'))

        sample(pex_type, n_points, dim)
        return

    # Benchmark mode
    if ("--bench" in args):

        bench_name = args[args.index("--bench")+1]
        bench = bench_factory.create(bench_name)

        disclaimer()
        liner(bold('Benchmark mode: '+bench_name))

        bench.run(args)
        return

    # If no keyword was triggered
    helper()

if __name__ == "__main__":
    main()
