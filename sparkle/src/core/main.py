# Generic imports
import sys

# Custom imports
from sparkle.src.core.train    import train
from sparkle.src.core.evaluate import evaluate
from sparkle.src.core.average  import average
from sparkle.src.env.parallel  import parallel
from sparkle.src.utils.json    import json_parser
from sparkle.src.utils.prints  import new_line, disclaimer, liner_simple, spacer, bold

def error():
    new_line()
    errr("""Command line error. Possible behaviors:
    spk --train <json_file>""")

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
        new_line()
        liner_simple()
        bold('Training mode')

        if (parallel.is_root()):
            spacer()
            print("Parallelism based on "+parallel.type())
            spacer()
            print("Number of parallel environments: "+str(parallel.size()))

        train(json_file, pms)
        return

    # Evaluation mode
    if ("--evaluate" in args):

        if ("-dat" not in args): error()
        dat_file  = args[args.index("-dat")+1]

        if ("-json" not in args): error()
        json_file = args[args.index("-json")+1]

        # Printings
        disclaimer()
        new_line()
        liner_simple()
        bold('Evaluation mode')

        evaluate(dat_file, json_file)
        return

    # Averaging mode
    if ("--average" in args):

        # Set parallel framework
        parallel.set({})

        # Printings
        disclaimer()
        new_line()
        liner_simple()
        bold('Average mode')

        dat_args = args[args.index("--avg")+1:]
        average(dat_args)

if __name__ == "__main__":
    main()
