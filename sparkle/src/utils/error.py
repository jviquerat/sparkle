import sys

from sparkle.src.env.parallel import parallel
from sparkle.src.utils.prints import err_print, liner, new_line, spacer, warn_print

###############################################
### A set of functions to print errors and warnings

### Error
def error(module, function, text, call_exit=True):
    """
    Prints an error message and optionally exits the program.

    Args:
        module: The name of the module where the error occurred.
        function: The name of the function where the error occurred.
        text: The error message.
        call_exit: If True, exits the program after printing the error.
    """
    liner(err_print("Sparkle error"))
    spacer("Module "+str(module)+", function "+str(function))
    spacer(text)

    if (call_exit):
        parallel.finalize()
        sys.exit(1)

### Warning
def warning(module, function, text):
    """
    Prints a warning message.

    Args:
        module: The name of the module where the warning occurred.
        function: The name of the function where the warning occurred.
        text: The warning message.
    """
    liner(warn_print("Sparkle warning"))
    spacer("Module "+str(module)+", function "+str(function))
    spacer(text)
    new_line()
