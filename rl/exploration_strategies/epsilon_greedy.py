import numpy as np

class EpsilonGreedy:
    def __init__(self, action_space, epsilon):
        self.action_space = action_space
        self.epsilon = epsilon

    def choose_action(self):
        return self.action_space.sample() if np.random.random() < self.epsilon else None
