import gym

import rl
from rl import agents
from rl import exploration_strategies

if __name__ == '__main__':
    # environment = gym.make('CartPole-v0')
    environment = gym.make('MountainCar-v0')
    explore_strategy = exploration_strategies.EpsilonGreedy(environment.action_space, epsilon=0.01)
    agent = agents.tabular.OnPolicyMonteCarloAgent(environment, explore_strategy, 0.99)
    agent.train(1000)

    runner = rl.Runner(environment, agent)
    runner.run(100, 10000)
