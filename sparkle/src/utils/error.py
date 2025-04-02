import sys

from sparkle.src.env.parallel import parallel
from sparkle.src.utils.prints import err_print, liner, new_line, spacer, warn_print

###############################################
### A set of functions to print errors and warnings

### Error
def error(module, function, text, call_exit=True):
    liner(err_print("Sparkle error"))
    spacer("Module "+str(module)+", function "+str(function))
    spacer(text)

    if (call_exit):
        parallel.finalize()
        sys.exit(1)

### Warning
def warning(module, function, text):
    liner(warn_print("Sparkle warning"))
    spacer("Module "+str(module)+", function "+str(function))
    spacer(text)
    new_line()

