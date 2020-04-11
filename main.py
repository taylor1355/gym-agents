import gym

import rl
from rl import agents

if __name__ == '__main__':
    #environment = gym.make('CartPole-v0')
    environment = gym.make('MountainCarContinuous-v0')
    #agent = agents.RandomAgent(environment)
    agent = agents.GreedyAgent(environment)
    runner = rl.Runner(environment, agent)

    runner.run(100, 1000)
    #params = agents.tabular_q_learning.TabularQLearningParameters(learning_rate=1e-3)
    #agent = agents.TabularQLearningAgent(environment, params)
