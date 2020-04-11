class RandomAgent:
    def __init__(self, environment):
        self.action_space = environment.action_space

    def act(self, observation):
        return self.action_space.sample()
