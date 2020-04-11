from rl import wrappers
from rl.wrappers import SnapshotWrapper

class GreedyAgent:
    def __init__(self, environment, samples=100):
        self.environment = wrappers.apply_wrapper(environment, SnapshotWrapper)
        self.samples = samples

    def act(self, observation):
        best_action, best_reward = None, None
        original_state = self.environment.save_snapshot()
        for sample in range(self.samples):
            action = self.environment.action_space.sample()
            _, reward, _, _ = self.environment.step(action)
            if best_reward is None or best_reward < reward:
                best_action, best_reward = action, reward
            self.environment.load_snapshot(original_state)
        return best_action
