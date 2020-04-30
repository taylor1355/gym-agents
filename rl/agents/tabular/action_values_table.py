import numpy as np

# TODO: configurable default policy and update policy
class ActionValuesTable:
    def __init__(self):
        self.action_values = {}

    def record(self, state, action, value_sample):
        state_str = state.tostring()
        if state_str not in self.action_values:
            self.action_values[state_str] = {action: (value_sample, 1)}
        elif action not in self.action_values[state_str]:
            self.action_values[state_str][action] = (value_sample, 1)
        else:
            value, visits = self.action_values[state_str][action]
            updated_value = np.average([value, value_sample], weights=[visits, 1])
            self.action_values[state_str][action] = updated_value, visits + 1

    def best_action(self, state):
        state_str = state.tostring()
        if state_str not in self.action_values:
            return None

        actions = self.action_values[state_str]
        return max(actions, key=lambda key: actions[key][0])

    def __getitem__(self, key):
        state, action = key
        state_str = state.tostring()
        if state_str not in self.action_values or action not in self.action_values[state_str]:
            return 0
        return self.action_values[state_str][action]
