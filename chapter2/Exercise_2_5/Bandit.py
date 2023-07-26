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


class OptimisticGreedyBandit(Bandit):

    def __init__(self, num_of_actions, step_size, initial=5):
        super(OptimisticGreedyBandit, self).__init__(num_of_actions)
        self.step_size = step_size
        self.Q = np.full(self.k, initial, dtype=float)

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


class UpperConfidenceBandit(Bandit):

    def __init__(self, num_of_actions, confidence):
        super(UpperConfidenceBandit, self).__init__(num_of_actions)

        self.confidence = confidence
        self.t = 1

    def exploit(self):
        ucb = self.Q + self.confidence * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        a = np.argmax(ucb)
        # print('exploit action ' + str(a))
        reward = self.get_reward(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])
        self.t += 1
        return a, reward


class GradientBandit(Bandit):

    def __init__(self, num_of_actions, step_size):
        super(GradientBandit, self).__init__(num_of_actions)
        self.step_size = step_size

        self.H = np.full(self.k, fill_value=0.)
        self.average_reward = 0

    def exploit(self):
        sum_preferences = np.sum(np.exp(self.H))
        probs = [(np.exp(self.H[i]) / sum_preferences) for i in range(self.k)]

        actions = [i for i in range(self.k)]

        a = np.random.choice(actions, p=probs)
        reward = self.get_reward(a)

        self.H[0:a] = self.H[0:a] - self.step_size * (reward - self.average_reward) * probs[0:a]
        self.H[a] = self.H[a] + self.step_size * (reward - self.average_reward) * (1 - probs[a])
        self.H[a + 1:] = self.H[a + 1:] - self.step_size * (reward - self.average_reward) * probs[a + 1:]
        self.average_reward += self.step_size * (reward - self.average_reward)

        return a, reward


def get_instance(bandit_mode, k, alpha=0.1, confidence=2, init_g=5):
    if bandit_mode == 'sample':
        return SampleAverageBandit(k)
    elif bandit_mode == 'weighted':
        return WeightedAverageBandit(k, step_size=alpha)
    elif bandit_mode == 'ucb':
        return UpperConfidenceBandit(k, confidence=confidence)
    elif bandit_mode == 'optimistic':
        return OptimisticGreedyBandit(k, alpha, init_g)
    elif bandit_mode == 'gradient':
        return GradientBandit(k, step_size=alpha)
