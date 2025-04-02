import subprocess

from sparkle.src.env.parallel import parallel
from numpy import float64
from typing import Union

###############################################
### A set of functions to format printings

### Specific colors for printings
wrn_clr = '\033[93m'
err_clr = '\033[91m'
end_clr = '\033[0m'
bld_clr = '\033[1m'

### New line
def new_line():
    if (parallel.is_root()):
        print("")

### Header
def header():
    if (parallel.is_root()):
        print("#################################")

### Liner
def liner(text):
    if (parallel.is_root()):
        new_line()
        print("### "+text)

### Liner with no newline
def liner_simple(text: str):
    if (parallel.is_root()):
        print("### "+text)

### Spacer
def spacer(text: str) -> None:
    if (parallel.is_root()):
        print("# "+text)

### Sparkle disclaimer
def disclaimer():
    if (parallel.is_root()):
        header()
        liner_simple(bold("Sparkle, an optimization library"))
        liner_simple(git_short_hash())
        header()

### Print with warning color
def warn_print(text):
    if (parallel.is_root()):
        return wrn_clr + text + end_clr

### Print with error color
def err_print(text):
    if (parallel.is_root()):
        return err_clr + text + end_clr

### Print with bold text
def bold(text):
    if (parallel.is_root()):
        return bld_clr + text + end_clr

### Format float for output
def fmt_float(x: Union[float, float64]) -> str:
    if (x < 1.0e-1) or (x > 1.0e3):
        return "{:.5e}".format(x)
    else:
        return "{:.5f}".format(x)

### Print git revision
def git_short_hash() -> str:
    if (parallel.is_root()):
        try:
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                                       shell=False,
                                       stdout=subprocess.PIPE)
            hash = process.communicate()[0].decode('ascii').strip()
            return "Revision "+str(hash)
        except Exception as e:
            pass
