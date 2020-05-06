from gym.spaces import Box
from gym.spaces import Discrete

# TODO: add support for other types of spaces
# Tuple, Dict, MultiBinary, MultiDiscrete
def is_discrete(space):
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError()

# TODO: add support for other types of discrete spaces
# Tuple, Dict, MultiBinary, MultiDiscrete
def cardinality(space):
    if not is_discrete(space):
        raise ValueError("Cardinality of a non-discrete space is infinite")

    if isinstance(space, Discrete):
        return space.n
    else:
        raise NotImplementedError()

# TODO: add support for other types of discrete spaces
# Tuple, Dict, MultiBinary, MultiDiscrete
def enumerate(space):
    if not is_discrete(space):
        raise ValueError("Cardinality of a non-discrete space is infinite")

    if isinstance(space, Discrete):
        return range(space.n)
