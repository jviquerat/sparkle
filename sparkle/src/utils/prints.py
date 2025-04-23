import subprocess
from typing import Union

from numpy import float64

from sparkle.src.env.parallel import parallel

###############################################
### A set of functions to format printings

### Specific colors for printings
wrn_clr = '\033[93m'
err_clr = '\033[91m'
end_clr = '\033[0m'
bld_clr = '\033[1m'

def new_line():
    """
    Prints a new line if the current process is the root process.
    """
    if (parallel.is_root()):
        print("")

def header():
    """
    Prints a header line if the current process is the root process.
    """
    if (parallel.is_root()):
        print("#################################")

def liner(text):
    """
    Prints a line with a header format if the current process is the root process.

    Args:
        text: The text to print.
    """
    if (parallel.is_root()):
        new_line()
        print("### "+text)

def liner_simple(text: str):
    """
    Prints a line with a header format without a preceding newline if the current process is the root process.

    Args:
        text: The text to print.
    """
    if (parallel.is_root()):
        print("### "+text)

def spacer(text: str) -> None:
    """
    Prints a line with a spacer format if the current process is the root process.

    Args:
        text: The text to print.
    """
    if (parallel.is_root()):
        print("# "+text)

def disclaimer():
    """
    Prints the Sparkle disclaimer, including the library name and Git revision, if the current process is the root process.
    """
    if (parallel.is_root()):
        header()
        liner_simple(bold("Sparkle, an optimization library"))
        liner_simple(git_short_hash())
        header()

def warn_print(text):
    """
    Prints text with a warning color if the current process is the root process.

    Args:
        text: The text to print.

    Returns:
        The text with the warning color codes.
    """
    if (parallel.is_root()):
        return wrn_clr + text + end_clr

def err_print(text):
    """
    Prints text with an error color if the current process is the root process.

    Args:
        text: The text to print.

    Returns:
        The text with the error color codes.
    """
    if (parallel.is_root()):
        return err_clr + text + end_clr

def bold(text):
    """
    Prints text in bold if the current process is the root process.

    Args:
        text: The text to print.

    Returns:
        The text with the bold formatting codes.
    """
    if (parallel.is_root()):
        return bld_clr + text + end_clr

def fmt_float(x: float) -> str:
    """
    Formats a float for output, using either scientific notation or fixed-point notation.

    Args:
        x: The float to format.

    Returns:
        The formatted float as a string.
    """
    if (x < 1.0e-1) or (x > 1.0e3):
        return "{:.5e}".format(x)
    else:
        return "{:.5f}".format(x)

def git_short_hash() -> str:
    """
    Retrieves and returns the short Git hash of the current revision if the current process is the root process.

    Returns:
        The short Git hash as a string, or an empty string if an error occurs.
    """
    if (parallel.is_root()):
        try:
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                                       shell=False,
                                       stdout=subprocess.PIPE)
            hash = process.communicate()[0].decode('ascii').strip()
            return "Revision "+str(hash)
        except Exception as e:
            pass
