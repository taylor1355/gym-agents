from gym import spaces

# TODO: add support for other types of spaces
# Tuple, Dict, MultiBinary, MultiDiscrete
def is_discrete(space):
    if isinstance(space, spaces.Discrete):
        return True
    elif isinstance(space, spaces.Box):
        return False
    else:
        raise NotImplementedError()

# TODO: add support for other types of discrete spaces
# Tuple, Dict, MultiBinary, MultiDiscrete
def cardinality(space):
    if not is_discrete(space):
        raise ValueError("Cardinality of a non-discrete space is infinite")

    if isinstance(space, spaces.Discrete):
        return space.n
    else:
        raise NotImplementedError()
