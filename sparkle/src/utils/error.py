# Custom imports
from sparkle.src.utils.prints import liner, spacer, new_line

###############################################
### A set of functions to print errors and warnings

### Error
def error(module, function, text):
    liner()
    errr("Sparkle error")
    spacer()
    print("Module "+str(module)+", function "+str(function))
    spacer()
    print(text)
    exit(1)

### Warning
def warning(module, function, text):
    liner()
    warn("Sparkle warning")
    spacer()
    print("Module "+str(module)+", function "+str(function))
    spacer()
    print(text)
    new_line()

