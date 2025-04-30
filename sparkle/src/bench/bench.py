import itertools
from types import SimpleNamespace
from collections import defaultdict
from typing import Any, List, Dict, Tuple

from sparkle.src.utils.prints import spacer


def get_sweep_parameters(sweep: SimpleNamespace) -> Tuple[Dict]:
    """
    A function to retrieve the list of keys and values within the
    sweep keyword of the json benchmark file

    Args:
        sweep: a Simplenamespace containing the sweep parameters

    Returns:
        keys: a dict of keys
        values: a dict of values
    """
    keys = []
    values = []
    for k, v in vars(sweep).items():
        keys.append(k)
        values.append(v)

    return keys, values

def combine_parameters(keys: List[str], values: List[Any]) -> List[Dict]:
    """
    A function to combine parameters based on a list of keys and a list of values

    Args:
        keys: list of keys
        values: list of values

    Returns:
        combinations: a list of dicts, each containing a combination of
                      the different values

    """
    spacer("Parameter keys: "+str(keys))
    spacer("Parameter values: "+str(values))

    # Generate combinations as list of tuples
    comb_tuples = list(itertools.product(*values))

    # Convert list of tuples to list of dicts
    combinations = []
    for k in comb_tuples:
        comb_dict = defaultdict(list)
        for l in range(len(k)):
            comb_dict[keys[l]] = k[l]
        combinations.append(dict(comb_dict))
    spacer("Nb of combinations: "+str(len(combinations)))

    return combinations

def combination_to_name(cmb: Dict) -> str:
    """
    A function that takes a combination of parameters and returns
    a string name from it for storage or plotting purpose
    """

    print(cmb)
    name = ""
    for k, v in cmb.items():
        name += f"{v} "

    return name
