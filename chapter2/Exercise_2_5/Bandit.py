import random

import numpy as np


class Bandit():
    def __init__(self, num_of_actions):
        self.k = num_of_actions
        self.action_values = [np.random.normal(0, 1, 1) for i in range(self.k)]
        self.Q = np.full(self.k, 0, dtype=float)
        self.N = np.full(self.k, 0)

    def get_reward(self, i):
        return self.action_values[i]

    def explore(self):
        raise NotImplementedError("Not Implemented")

    def exploit(self):
        raise NotImplementedError("Not implemented")

    def is_optimal(self, action):
        return float(action == np.argmax(self.action_values))

    def random_walk(self):
        increment = [np.random.normal(0, 0.01, 1) for i in range(self.k)]
        self.action_values = [self.action_values[i] + increment[i] for i in range(self.k)]


class SampleAverageBandit(Bandit):

    def __init__(self, num_of_actions):
        super(SampleAverageBandit, self).__init__(num_of_actions)

    def explore(self):
        a = random.randint(0, self.k - 1)
        # print('explore selected action ' + str(a + 1))
        reward = self.get_reward(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])
        return a, reward

    def exploit(self):
        a = np.argmax(self.Q)
        # print('exploit action ' + str(a))
        reward = self.get_reward(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])
        return a, reward


class WeightedAverageBandit(Bandit):

    def __init__(self, num_of_actions, step_size):
        super(WeightedAverageBandit, self).__init__(num_of_actions)
        self.step_size = step_size

    def explore(self):
        a = random.randint(0, self.k - 1)
        # print('explore selected action ' + str(a + 1))
        reward = self.get_reward(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + self.step_size * (reward - self.Q[a])
        return a, reward

    def exploit(self):
        a = np.argmax(self.Q)
        # print('exploit action ' + str(a))
        reward = self.get_reward(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + self.step_size * (reward - self.Q[a])
        return a, reward


def get_instance(bandit_mode, k, alpha=0.1):
    if bandit_mode == 'sample':
        return SampleAverageBandit(k)
    elif bandit_mode == 'weighted':
        return WeightedAverageBandit(k, step_size=alpha)
