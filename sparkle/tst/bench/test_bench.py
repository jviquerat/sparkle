

from sparkle.src.bench.bench import combine_parameters
from sparkle.src.utils.json import JsonParser


###############################################
def test_bench():

    parser    = JsonParser()
    filename = "sparkle/tst/bench/pex.json"
    pms       = parser.read(filename)
    methods    = pms.methods
    dimensions = pms.dimensions

    keys   = ["method", "dimension"]
    values = [ methods,  dimensions]

    combinations = combine_parameters(keys, values)
    assert len(combinations) == 12
