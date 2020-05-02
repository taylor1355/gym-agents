import numpy as np

from collections import namedtuple

WeightedValue = namedtuple('WeightedValue', ['value', 'weight'])

def get_hashable(x):
    if hasattr(x, 'tostring'):
        return x.tostring()
    return x

# TODO: configurable default policy and update policy
class ActionValuesTable:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_values = {}

    def record(self, state, action, value_sample, weight=1):
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values:
            self.action_values[hashable_state] = {action: WeightedValue(value_sample, weight)}
        elif action not in self.action_values[hashable_state]:
            self.action_values[hashable_state][action] = WeightedValue(value_sample, weight)
        else:
            value, weight_sum = self.action_values[hashable_state][action]
            updated_value = np.average([value, value_sample], weights=[weight_sum, weight])
            self.action_values[hashable_state][action] = WeightedValue(updated_value, weight_sum + weight)

    def greedy_action(self, state):
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values:
            return self.action_space.sample()

        actions = self.action_values[hashable_state]
        return max(actions, key=lambda action: actions[action].value)

    def __getitem__(self, key):
        state, action = key
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values or action not in self.action_values[hashable_state]:
            return 0
        return self.action_values[hashable_state][action]
