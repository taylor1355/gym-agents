import numpy as np

from rl import spaces
from rl.agents import BaseAgent
from rl.agents.tabular import ActionValuesTable

# Double q-learning uses differemt action-value tables to estimate
# the maximum value action and the value of that action. This decorrelation
# eliminates the maximization bias present in q-learning. This means that
# state-action pairs without many visits will not be overestimated.
class DoubleQLearningAgent(BaseAgent):
    def __init__(self, environment, exploration_strategy, discount, step_size):
        super().__init__(environment, exploration_strategy, discount)
        # TODO: make sure environment has a discrete action and observation space

        self.action_values = [
            ActionValuesTable(self.environment.action_space, step_size=step_size),
            ActionValuesTable(self.environment.action_space, step_size=step_size)
        ]

    def train_episode(self):
        prev_observation = self.environment.reset()
        done = False
        while not done:
            action = self.act(prev_observation)
            observation, reward, done, _ = self.environment.step(action)
            q_index = np.random.randint(2)
            maxq_action = self.action_values[q_index].greedy_action(observation)
            sample_return = reward + self.discount * self.action_values[1 - q_index][observation, maxq_action]
            self.action_values[q_index].record(prev_observation, action, sample_return)
            prev_observation = observation

    def act_greedily(self, observation):
        return max(spaces.enumerate(self.environment.action_space),
            key=lambda action: self.action_values[0][observation, action] +
            self.action_values[1][observation, action])
