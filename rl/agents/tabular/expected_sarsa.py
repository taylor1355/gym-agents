from rl import spaces
from rl.agents import BaseAgent
from rl.agents.tabular import ActionValuesTable

# This is, like Q-Learning, off-policy TD(0) learning. The difference
# is that the expected action-value is used in the update rather than
# the maximum. It is still off-policy because the behavior that determines
# which actions to take could be different from the behavior that generates
# a lkelihood for each action. If the two aforementioned behaviors are the
# same, then the updates to the action-values will be in the same direction
# as the expected SARSA updates, hence the name expected SARSA.
class ExpectedSarsaAgent(BaseAgent):
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
            sample_return = reward + self.discount * self.expected_value(observation)
            self.action_values.record(prev_observation, action, sample_return)
            prev_observation = observation

    def expected_value(self, state):
        greedy_action = self.action_values.greedy_action(state)
        expectation = 0
        for action in spaces.enumerate(self.environment.action_space):
            likelihood = self.exploration_strategy.action_probability(action, greedy_action)
            expectation += likelihood * self.action_values[state, action]
        return expectation

    def act_greedily(self, observation):
        return self.action_values.greedy_action(observation)
