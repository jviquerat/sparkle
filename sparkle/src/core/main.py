# Generic imports
import sys

# Custom imports
from sparkle.src.core.train   import *
from sparkle.src.utils.prints import *

def error():
    new_line()
    errr("""Command line error. Possible behaviors:
    spk --train <json_file>""")

def main():

    # Printings
    disclaimer()

    # Check arguments
    args = sys.argv

    # Training mode
    if ("--train" in args):
        new_line()
        liner_simple()
        bold('Training mode')

        json_file = args[args.index("--train")+1]
        train(json_file)
        return

if __name__ == "__main__":
    main()
