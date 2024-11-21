# Generic imports
import os
import subprocess

# Custom imports
from sparkle.src.env.parallel import parallel

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
def liner():
    if (parallel.is_root()):
        new_line()
        print("###", end=" ")

### Liner with no newline
def liner_simple():
    if (parallel.is_root()):
        print("###", end=" ")

### Spacer
def spacer():
    if (parallel.is_root()):
        print("#", end=" ")

### Sparkle disclaimer
def disclaimer():
    if (parallel.is_root()):
        header()
        liner_simple()
        bold("Sparkle, an optimization library")
        liner_simple()
        git_short_hash()
        header()

### Print with warning color
def warn_print(text):
    if (parallel.is_root()):
        print(wrn_clr + text + end_clr)

### Print with error color
def err_print(text):
    if (parallel.is_root()):
        print(err_clr + text + end_clr)

### Print with bold text
def bold(text):
    if (parallel.is_root()):
        print(bld_clr + text + end_clr)

### Print git revision
def git_short_hash() -> str:
    if (parallel.is_root()):
        try:
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                                       shell=False,
                                       stdout=subprocess.PIPE)
            hash = process.communicate()[0].decode('ascii').strip()
            print("Revision "+str(hash))
        except Exception as e:
            pass
