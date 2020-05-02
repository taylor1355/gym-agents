import numpy as np

from collections import namedtuple

WeightedValue = namedtuple('WeightedValue', ['value', 'weight'])

def get_hashable(x):
    if hasattr(x, 'tostring'):
        return x.tostring()
    return x

# TODO: configurable default policy and update policy
class ActionValuesTable:
    def __init__(self, action_space, step_size=None):
        self.action_space = action_space
        self.step_size = step_size

        self.action_values = {}

    def record(self, state, action, value_sample, weight=1):
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values:
            self.action_values[hashable_state] = {}
        if action not in self.action_values[hashable_state]:
            self.action_values[hashable_state][action] = WeightedValue(value_sample, weight)

        value, weight_sum = self.action_values[hashable_state][action]
        step_size = weight / (weight + weight_sum) if self.step_size is None else self.step_size

        updated_value = value + step_size * (value_sample - value)
        self.action_values[hashable_state][action] = WeightedValue(updated_value, weight_sum + weight)

    def greedy_action(self, state):
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values:
            return self.action_space.sample()

        actions = self.action_values[hashable_state]
        return max(actions, key=lambda action: actions[action].value)

    def getWeightedValue(self, key):
        state, action = key
        hashable_state = get_hashable(state)
        if hashable_state not in self.action_values or action not in self.action_values[hashable_state]:
            return WeightedValue(0, 0)  
        return self.action_values[hashable_state][action]

    def __getitem__(self, key):
        return self.getWeightedValue(key).value
