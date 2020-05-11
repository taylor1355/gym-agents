import numpy as np

from rl import spaces
from rl.wrappers import SnapshotWrapper

class GreedyAgent:
    def __init__(self, environment, samples=100):
        environment.apply_wrapper(SnapshotWrapper)
        self.environment = environment
        self.samples = samples

    def act(self, observation):
        actions = []
        if spaces.is_discrete(self.environment.action_space):
            actions = spaces.enumerate(self.environment.action_space)
        else:
            actions = (self.environment.action_space.sample() for i in range(self.samples))

        return self.find_greedy(actions)

    def find_greedy(self, actions):
        greedy_action, max_reward = None, None
        original_state = self.environment.save_snapshot()
        for action in actions:
            _, reward, _, _ = self.environment.step(action)
            if max_reward is None or max_reward < reward:
                greedy_action, max_reward = action, reward
            self.environment.load_snapshot(original_state)
        return greedy_action
