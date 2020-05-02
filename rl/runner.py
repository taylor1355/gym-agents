import copy
import matplotlib.pyplot as plt
import numpy as np

from gym.wrappers import RecordEpisodeStatistics

class Runner:
    def __init__(self, environment, agent):
        environment.apply_wrapper(RecordEpisodeStatistics)
        self.environment = environment
        self.agent = agent

    def train(self, episodes, graph_interval=5):
        returns = []
        episodes_completed = 0

        figure = plt.figure()
        axis = figure.add_subplot()
        line, = axis.plot(returns)
        plt.ion()
        plt.show()

        while episodes_completed < episodes:
            episodes_to_complete = min(graph_interval, episodes - episodes_completed)
            self.agent.train(episodes_to_complete)
            episodes_completed += episodes_to_complete

            returns.append(np.mean(self.environment.return_queue))
            self.environment.return_queue.clear()

            line.set_ydata(returns)
            line.set_xdata(range(0, episodes_completed, graph_interval))
            axis.set_xlim([0, episodes_completed])
            axis.set_ylim([min(returns), max(returns)])
            plt.pause(0.05)

    def run(self, episodes, max_steps):
        for episode in range(episodes):
            print(f'Episode {episode}')
            observation = self.environment.reset()
            for step in range(max_steps):
                self.environment.render()
                action = self.agent.act(observation)
                observation, reward, done, info = self.environment.step(action)
                if done:
                    self.environment.close()
                    break
