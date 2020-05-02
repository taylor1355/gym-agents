from gym.core import Wrapper

class BaseWrapper(Wrapper):
    def implements_wrapper(self, wrapper_class):
        env = self
        while True:
            if isinstance(env, wrapper_class):
                return True
            try:
                env = env.env
            except AttributeError:
                break
        return False

    def apply_wrapper(self, wrapper_class, *args):
        if not self.implements_wrapper(wrapper_class):
            self.env = wrapper_class(self.env, *args)
