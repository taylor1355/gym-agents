from rl.agents.tabular import ActionValuesTable

# This is off policy Monte Carlo control. The target policy is the greedy policy
# (using the current action-values table) and the behavior policy is determined by
# the exploration policy. Sample returns from following the behavior policy are
# used to calculate the action-values of the target policy using weighted importance-
# sampling

class OffPolicyMonteCarloAgent:
    # TODO: for this and OnPolicyMonteCarloAgent, resolve chicken and egg problem with exploration_strategy
    def __init__(self, environment, exploration_strategy, discount):
        self.environment = environment
        # TODO: make sure environment has a discrete action and observation space

        self.exploration_strategy = exploration_strategy
        self.discount = discount

        self.action_values = ActionValuesTable(self.environment.action_space)

    def train(self, episodes):
        for episode in range(episodes):
            print(f'Training: episode {episode}')
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
            importance_sampling_ratio = 1
            for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
                episode_return = reward + self.discount * episode_return
                self.action_values.record(state, action, episode_return, weight=importance_sampling_ratio)

                greedy_action = self.action_values.greedy_action(state)
                if action != greedy_action:
                    # Since the target policy is greedy, the importance sampling ratio
                    # will be 0 if the greedy action was not chosen by the behavior policy,
                    # resulting in the returns from earlier in the episode having 0 weight
                    break
                importance_sampling_ratio = importance_sampling_ratio * (1 / self.exploration_strategy.action_probability(action, greedy_action))

    def act(self, observation):
        greedy_action = self.action_values.greedy_action(observation)
        return self.exploration_strategy.choose_action(greedy_action)
