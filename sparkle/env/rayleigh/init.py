# Generic imports
import time

# Custom imports
from rayleigh import *

#######################################
# Generate initial data
#######################################

plot_freq = 10000
control = 0.0
s = rayleigh(0, ".")
s.reset(0)

start_time = time.time()
c = s.cost([control]*10)
s.dump("init_field.dat")
s.render([[control]*10])
end_time = time.time()
print("# Loop time = {:f}".format(end_time - start_time))
