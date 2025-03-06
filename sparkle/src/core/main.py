# Generic imports
import sys

# Custom imports
from sparkle.src.core.train    import train
from sparkle.src.core.evaluate import evaluate
from sparkle.src.core.average  import average
from sparkle.src.core.sample   import sample
from sparkle.src.core.model    import model
from sparkle.src.env.parallel  import parallel
from sparkle.src.utils.json    import json_parser
from sparkle.src.utils.prints  import disclaimer, liner, liner_simple, spacer, bold, err_print

def helper():
    liner(err_print("Command line error"))
    spacer("Command line arguments are:")
    spacer("   spk --train <json_file>")
    spacer("   spk --evaluate -dat <dat_file> -json <json_file>")
    spacer("   spk --average <dat_file_0> ... <dat_file_n>")
    spacer("   spk --model <json_file>")
    spacer("   spk --pex -type <pex_type> -n_points <n_points>")
    exit(0)

def main():

    # Check arguments
    args = sys.argv

    # Training mode
    if ("--train" in args):

        # Initialize json parser and read parameters
        json_file = args[args.index("--train")+1]
        parser    = json_parser()
        pms       = parser.read(json_file)

        # Set parallel framework
        parallel.set(pms)

        # Printings
        disclaimer()
        liner(bold('Training mode'))

        if (parallel.is_root()):
            spacer("Parallelism based on "+parallel.type)
            spacer("Number of parallel environments: "+str(parallel.size))

        train(json_file, pms)
        return

    # Evaluation mode
    if ("--evaluate" in args):

        if ("-dat" not in args): helper()
        dat_file  = args[args.index("-dat")+1]

        if ("-json" not in args): helper()
        json_file = args[args.index("-json")+1]

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
        parser    = json_parser()
        pms       = parser.read(json_file)

        # Set parallel framework
        parallel.set(pms)

        # Printings
        disclaimer()
        liner(bold('Model mode'))

        if (parallel.is_root()):
            spacer("Parallelism based on "+parallel.type)
            spacer("Number of parallel environments: "+str(parallel.size))

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

        # Set parallel framework
        parallel.set({})

        # Printings
        disclaimer()
        liner(bold('Pex sampling mode'))

        sample(pex_type, n_points)
        return

    # If no keyword was triggered
    helper()

if __name__ == "__main__":
    main()
