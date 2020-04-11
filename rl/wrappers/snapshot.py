from gym.core import Wrapper

import pickle

# TODO implement special cases for Atari / Mujoco environmenta
class SnapshotWrapper(Wrapper):
    def save_snapshot(self):
        self.close() # close popup windows since we can't pickle them
        return pickle.dumps(self.env)

    def load_snapshot(self, snapshot):
        self.close() # close popup windows since we can't load into them
        self.env = pickle.loads(snapshot)
