from rl.agents import BaseAgent
from rl.agents.tabular import ActionValuesTable

# This is on policy TD(0) learning
class SarsaAgent(BaseAgent):
    def __init__(self, environment, exploration_strategy, discount, step_size):
        super().__init__(environment, exploration_strategy, discount)
        # TODO: make sure environment has a discrete action and observation space

        self.action_values = ActionValuesTable(self.environment.action_space, step_size=step_size)

    def train_episode(self):
        prev_observation = self.environment.reset()
        prev_action = self.act(prev_observation)
        done = False
        while not done:
            observation, reward, done, _ = self.environment.step(prev_action)
            action = self.act(observation)
            sample_return = reward + self.discount * self.action_values[observation, action]
            self.action_values.record(prev_observation, prev_action, sample_return)
            prev_observation, prev_action = observation, action

    def act_greedily(self, observation):
        return self.action_values.greedy_action(observation)
