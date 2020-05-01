import numpy as np

from collections import namedtuple

WeightedValue = namedtuple('WeightedValue', ['value', 'weight'])

# TODO: configurable default policy and update policy
class ActionValuesTable:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_values = {}

    def record(self, state, action, value_sample, weight=1):
        state_str = state.tostring()
        if state_str not in self.action_values:
            self.action_values[state_str] = {action: WeightedValue(value_sample, weight)}
        elif action not in self.action_values[state_str]:
            self.action_values[state_str][action] = WeightedValue(value_sample, weight)
        else:
            value, weight_sum = self.action_values[state_str][action]
            updated_value = np.average([value, value_sample], weights=[weight_sum, weight])
            self.action_values[state_str][action] = WeightedValue(updated_value, weight_sum + weight)

    def greedy_action(self, state):
        state_str = state.tostring()
        if state_str not in self.action_values:
            return self.action_space.sample()

        actions = self.action_values[state_str]
        return max(actions, key=lambda action: actions[action].value)

    def __getitem__(self, key):
        state, action = key
        state_str = state.tostring()
        if state_str not in self.action_values or action not in self.action_values[state_str]:
            return 0
        return self.action_values[state_str][action]
