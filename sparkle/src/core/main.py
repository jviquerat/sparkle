# Generic imports
import sys

# Custom imports
from sparkle.src.core.train   import *
from sparkle.src.utils.json   import *
from sparkle.src.utils.prints import *

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

        train(json_file, pms)
        return

if __name__ == "__main__":
    main()
