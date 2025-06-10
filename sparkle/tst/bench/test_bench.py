from types import SimpleNamespace

from sparkle.src.bench.bench import (combine_parameters,
                                     combination_to_name,
                                     get_sweep_parameters)
from sparkle.src.utils.json import JsonParser


###############################################
def test_bench():

    parser     = JsonParser()
    filename   = "sparkle/tst/bench/pex.json"
    pms        = parser.read(filename)
    methods    = pms.method
    dimensions = pms.dimension

    keys   = ["method", "dimension"]
    values = [ methods,  dimensions]

    combinations = combine_parameters(keys, values)
    assert len(combinations) == 6

    dict0 = {"method": "random", "dimension": 2}
    dict1 = {"method": "random", "dimension": 5}
    assert combinations[0] == dict0
    assert combinations[1] == dict1

    assert combination_to_name(combinations[0]) == "2_random"

    nsp = SimpleNamespace()
    nsp.a = ["1", "2", "3"]
    nsp.b = ["11", "12", "13"]
    k, v = get_sweep_parameters(nsp)

    assert k == ["a", "b"]
    assert v == [["1", "2", "3"], ["11", "12", "13"]]
