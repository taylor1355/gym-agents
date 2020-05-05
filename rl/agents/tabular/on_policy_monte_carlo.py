from rl.agents import BaseAgent
from rl.agents.tabular import ActionValuesTable

# This is every-visit on policy Monte Carlo control.
class OnPolicyMonteCarloAgent(BaseAgent):
    def __init__(self, environment, exploration_strategy, discount):
        super().__init__(environment, exploration_strategy, discount)
        # TODO: make sure environment has a discrete action and observation space

        self.action_values = ActionValuesTable(self.environment.action_space)

    def train_episode(self):
        states, actions, rewards = [], [], []
        observation = self.environment.reset()
        done = False
        while not done:
            states.append(observation)
            action = self.act(observation)
            actions.append(action)
            observation, reward, done, _ = self.environment.step(action)
            rewards.append(reward)

        episode_return = 0
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            episode_return = reward + self.discount * episode_return
            self.action_values.record(state, action, episode_return)

    def act_greedily(self, observation):
        return self.action_values.greedy_action(observation)
