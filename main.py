import gym
import rl

from rl import agents
from rl import exploration
from rl import wrappers

if __name__ == '__main__':
    #environment = gym.make('Roulette-v0')
    environment = gym.make('Taxi-v3')
    # environment = gym.make('CartPole-v0')
    #environment = gym.make('MountainCar-v0')

    environment = wrappers.BaseWrapper(environment)

    explore_strategy = exploration.EpsilonGreedy(environment.action_space, epsilon=0.01)
    agent = agents.tabular.OffPolicyMonteCarloAgent(environment, explore_strategy, 0.99)
    # agent = agents.tabular.SarsaAgent(environment, explore_strategy, 0.99, 0.05)
    # agent = agents.tabular.QLearningAgent(environment, explore_strategy, 0.99, 0.05)

    runner = rl.Runner(environment, agent)
    runner.train(1000)
    # runner.run(10, 10000)
