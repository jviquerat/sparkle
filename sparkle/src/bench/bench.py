import itertools
from collections import defaultdict
from typing import Any, List

from sparkle.src.utils.prints import spacer


def combine_parameters(keys: List[str], values: List[Any]) -> List[dict]:
    """
    A function to combine parameters based on a list of keys and a list of values

    Args:
        keys: list of keys
        values: list of values

    Returns:
        combinations: a list of dicts, each containing a combination of
                      the different values

    """
    # Printings
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
