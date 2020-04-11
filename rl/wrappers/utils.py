def implements_wrapper(environment, wrapper_class):
    env = environment
    while True:
        if isinstance(env, wrapper_class):
            return True
        try:
            env = env.env
        except AttributeError:
            break
    return False

def apply_wrapper(environment, wrapper_class, *args):
    if not implements_wrapper(environment, wrapper_class):
        return wrapper_class(environment, *args)
