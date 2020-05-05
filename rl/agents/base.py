class BaseAgent:
    def __init__(self, environment, exploration_strategy, discount):
        self.environment = environment
        self.exploration_strategy = exploration_strategy
        self.discount = discount
        self.trained_episodes = 0

    def train(self, episodes):
        for episode in range(episodes):
            print(f'Training: episode {self.trained_episodes}')
            self.train_episode()
            self.trained_episodes += 1

    def train_episode(self):
        raise NotImplementedError()

    def act(self, observation):
        greedy_action = self.act_greedily(observation)
        return self.exploration_strategy.choose_action(greedy_action)

    def act_greedily(self, observation):
        raise NotImplementedError()
