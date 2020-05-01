import gym
import numpy as np
import rl

from dataclasses import dataclass
from rl import exploration

@dataclass
class EpsilonGreedy:
    action_space: gym.spaces.Space
    epsilon: float

    def action_probability(self, action, greedy_action):
        if not rl.spaces.is_discrete(self.action_space):
            raise ValueError('Need discrete action space for action probability to be well defined')

        prob = self.epsilon * (1 / rl.spaces.cardinality(self.action_space))
        if action == greedy_action:
            prob += 1 - self.epsilon
        return prob

    def choose_action(self, greedy_action):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return greedy_action
