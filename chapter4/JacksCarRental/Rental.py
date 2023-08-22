import math

import numpy as np
import matplotlib.pyplot as plt


class CarRental:

    def __init__(self):

        self.max_cars = 20
        self.tranport_cost = 2
        self.rent_reward = 10
        self.max_movable = 5

        self.lambda_rent_A = 3
        self.lambda_rent_B = 4

        self.lambda_return_A = 3
        self.lambda_return_B = 2

        self.max_poisson = 11

        self.V = np.full((self.max_cars + 1, self.max_cars + 1), 0.0)
        self.policies = np.full((self.max_cars + 1, self.max_cars + 1), 0, dtype=int)
        self.actions = [i for i in range(-self.max_movable, self.max_movable + 1)]
        self.poisson_cache = dict()

    def poisson_probability(self, lam, n):

        key = n * 10 + lam
        if key not in self.poisson_cache:
            self.poisson_cache[key] = ((np.power(lam, n) * np.exp(-lam)) / math.factorial(n))
        return self.poisson_cache[key]

    def expected_value(self, s, a, gamma=0.9):

        total_reward = 0.0

        num_cars_A_Initial = max(min(s[0] - a, self.max_cars), 0)
        num_cars_B_Initial = max(min(s[1] + a, self.max_cars), 0)

        total_reward -= (self.tranport_cost) * np.abs(a)
        for rent_request_A in range(self.max_poisson):
            for rent_request_B in range(self.max_poisson):
                # Check here
                rent_prob = self.poisson_probability(self.lambda_rent_A, rent_request_A) * self.poisson_probability(
                    self.lambda_rent_B, rent_request_B)

                rent_cars_A = min(rent_request_A, num_cars_A_Initial)
                rent_cars_B = min(rent_request_B, num_cars_B_Initial)

                reward = (rent_cars_A + rent_cars_B) * self.rent_reward

                num_cars_A = num_cars_A_Initial - rent_cars_A
                num_cars_B = num_cars_B_Initial - rent_cars_B

                for return_A in range(self.max_poisson):
                    for return_B in range(self.max_poisson):
                        return_prob = self.poisson_probability(self.lambda_return_A,
                                                               return_A) * self.poisson_probability(
                            self.lambda_return_B, return_B)

                        num_cars_at_A = min(return_A + num_cars_A, self.max_cars)
                        num_cars_at_B = min(return_B + num_cars_B, self.max_cars)

                        total_prob = (rent_prob * return_prob)

                        total_reward += (total_prob * (reward + gamma * self.V[num_cars_at_A, num_cars_at_B]))

        return total_reward

    def policy_evaluation(self, gamma=0.9, theta=1e-2):
        print('policy evaluation')
        while True:
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    v = self.V[i, j]
                    a = self.policies[i, j]

                    tot_val = self.expected_value([i, j], a, gamma)

                    self.V[i, j] = tot_val
                    delta = max(delta, (np.abs(v - self.V[i, j])))
            print('delta ' + str(delta))
            if (delta < theta):
                break

    def policy_improvement(self):

        policy_stable = True
        print('policy improvement')
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):

                old_action = self.policies[i, j]
                actions_values = []
                for a in self.actions:
                    if (0 <= a <= i) or (-j <= a <= 0):
                        tot_val = self.expected_value([i, j], a)
                        actions_values.append(tot_val)
                    else:
                        actions_values.append(-np.inf)

                self.policies[i, j] = self.actions[np.argmax(actions_values)]
                if old_action != self.policies[i, j]:
                    policy_stable = False
        print('policy stability ' + str(policy_stable))
        return policy_stable

    def save_policy(self, count):
        plt.figure()
        plt.title("Policy" + str(count))
        plt.ylabel("Cars at Location 1")
        plt.xlabel("Cars at Location 2")

        plt.imshow(self.policies, origin='lower',
                   vmin=-5, vmax=+5)
        plt.colorbar()
        plt.savefig('plots/policy' + str(count) + '.jpg')
        plt.close()

    def save_value(self, count):
        plt.figure()
        plt.title("Policy" + str(count))
        plt.ylabel("Cars at Location 1")
        plt.xlabel("Cars at Location 2")

        plt.imshow(self.V, origin='lower')
        plt.colorbar()
        plt.savefig('plots/value' + str(count) + '.jpg')
        plt.close()

    def train(self):

        count = 0
        while True:

            self.policy_evaluation(gamma=0.9, theta=0.01)
            stable = self.policy_improvement()
            self.save_policy(count)
            self.save_value(count)
            if stable:
                print('stable breaking')
                break

            count += 1

            print('count ' + str(count))


rental = CarRental()
rental.train()
