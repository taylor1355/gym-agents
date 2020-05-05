from rl.agents import BaseAgent
from rl.agents.tabular import ActionValuesTable

# This is off policy TD(0) learning
class QLearningAgent(BaseAgent):
    def __init__(self, environment, exploration_strategy, discount, step_size):
        super().__init__(environment, exploration_strategy, discount)
        # TODO: make sure environment has a discrete action and observation space

        self.action_values = ActionValuesTable(self.environment.action_space, step_size=step_size)

    def train_episode(self):
        prev_observation = self.environment.reset()
        done = False
        while not done:
            action = self.act(prev_observation)
            observation, reward, done, _ = self.environment.step(action)
            maxq_action = self.action_values.greedy_action(observation)
            sample_return = reward + self.discount * self.action_values[observation, maxq_action]
            self.action_values.record(prev_observation, action, sample_return)
            prev_observation = observation

    def act_greedily(self, observation):
        return self.action_values.greedy_action(observation)
