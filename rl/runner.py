class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def run(self, episodes, max_steps):
        for episode in range(episodes):
            print(f'Episode {episode}')
            observation = self.environment.reset()
            for step in range(max_steps):
                self.environment.render()
                action = self.agent.act(observation)
                print(action)
                observation, reward, done, info = self.environment.step(action)
                if done:
                    self.environment.close()
                    break
