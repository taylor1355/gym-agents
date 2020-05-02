from rl.agents.tabular import ActionValuesTable

class SarsaAgent:
    def __init__(self, environment, exploration_strategy, discount, step_size):
        self.environment = environment
        # TODO: make sure environment has a discrete action and observation space

        self.exploration_strategy = exploration_strategy
        self.discount = discount

        self.action_values = ActionValuesTable(self.environment.action_space, step_size=step_size)

    def train(self, episodes):
        for episode in range(episodes):
            print(f'Training: episode {episode}')
            prev_observation = self.environment.reset()
            prev_action = self.act(prev_observation)
            done = False
            while not done:
                observation, reward, done, _ = self.environment.step(prev_action)
                action = self.act(observation)
                sample_return = reward + self.discount * self.action_values[observation, action]
                self.action_values.record(prev_observation, prev_action, sample_return)
                prev_observation, prev_action = observation, action

    def act(self, observation):
        greedy_action = self.action_values.greedy_action(observation)
        return self.exploration_strategy.choose_action(greedy_action)
