import numpy as np
import matplotlib.pyplot as plt


class Gambler:
    def __init__(self, head_prob):
        self.head_prob = head_prob

    def expected_value(self, s, a):

        head_val = s + a
        tail_val = s - a

        tot_reward = self.head_prob * V[head_val] + (1 - self.head_prob) * V[tail_val]
        return tot_reward

    def value_iteration(self, S, V, pi, theta=1e-9):
        count = 0
        plt.figure()
        plt.title("Value Function")
        plt.ylabel("Value Estimate")
        plt.xlabel("Capital")
        while True:
            delta = 0
            for s in S:

                v_old = V[s]

                action_vals = []

                for a in range(1, min(s, 100 - s) + 1):
                    val = self.expected_value(s, a)
                    action_vals.append(val)
                V[s] = max(action_vals)

                delta = max(delta, (np.abs(v_old - V[s])))
            plt.plot(V, label='sweep ' + str(count))
            if delta < theta:
                break
            count += 1

        for s in S:

            action_vals = []
            for a in range(1, min(s, 100 - s) + 1):
                val = self.expected_value(s, a)
                action_vals.append(val)

            pi[s] = np.argmax(np.round(action_vals, 5)) + 1
        plt.legend(loc='upper left', ncols=2)
        plt.savefig('plots/value_function' + str(self.head_prob) + '.jpg')
        return V, pi


phs = [0.4, 0.25, 0.55]

for ph in phs:
    S = np.arange(1, 100)
    V = np.zeros(101)
    V[100] = 1
    pi = np.zeros(101)
    gambler = Gambler(ph)
    vals, policy = gambler.value_iteration(S, V, pi)

    plt.figure()
    plt.title("Final Policy")
    plt.ylabel("Stake")
    plt.xlabel("Capital")
    plt.bar(range(101), policy)
    plt.savefig('plots/policy' + str(ph) + '.jpg')

    plt.figure()
    plt.title("Final Value")
    plt.ylabel("Stake")
    plt.xlabel("Capital")
    plt.plot(range(101), vals)
    plt.savefig('plots/value_final' + str(ph) + '.jpg')


