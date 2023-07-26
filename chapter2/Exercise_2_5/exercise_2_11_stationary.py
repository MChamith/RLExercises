import random
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bandit import get_instance


def perform_task(k, epsilon, N_STEPS, N_RUNS, mode, alpha=0.1, stationary=True, confidence=2, init_g=0):
    Optimal_percentage = np.full((N_RUNS, N_STEPS), 0, dtype=float)
    Reward = np.full((N_RUNS, N_STEPS), 0, dtype=float)

    for run_i in tqdm(range(1, N_RUNS + 1)):
        bandit = get_instance(mode, k, alpha, confidence, init_g)
        for step in range(1, N_STEPS + 1):

            prob = random.uniform(0, 1)

            if mode == 'sample' or mode == 'weighted':
                if prob < epsilon:
                    step_action, reward = bandit.explore()
                else:
                    step_action, reward = bandit.exploit()
            else:
                step_action, reward = bandit.exploit()

            optimal = bandit.is_optimal(step_action)

            Optimal_percentage[run_i - 1][step - 1] = optimal
            Reward[run_i - 1][step - 1] = reward
            if not stationary:
                bandit.random_walk()

    return Optimal_percentage, Reward


hyper_parameters = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
epsilons = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1/4]
alphas = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2]
confidences = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
Q_zeros = [ 1 / 4, 1 / 2, 1, 2, 4]


gradient_rewards = []
epsilon_rewards = []
ucb_rewards = []
optimistic_rewards = []

steps = 10000
runs = 2000
for alpha in alphas:
    percent_optimal_gradient, reward_gradient = perform_task(10,0.1,steps,runs,'gradient',alpha)

    avg_reward_gradient = np.mean(reward_gradient)
    gradient_rewards.append(avg_reward_gradient)

for epsilon in epsilons:
    percent_optimal, reward = perform_task(10, epsilon, steps, runs, 'sample')
    avg_reward = np.mean(reward)
    epsilon_rewards.append(avg_reward)

for param in confidences:
    percent_optimal, reward = perform_task(10, 0.1, steps, runs, 'ucb', confidence=param)
    avg_reward = np.mean(reward)
    ucb_rewards.append(avg_reward)

for param in Q_zeros:
    percent_optimal, reward = perform_task(10, 0.1, steps, runs, 'optimistic', init_g=param)
    avg_reward = np.mean(reward)
    optimistic_rewards.append(avg_reward)



plt.figure(1)

plt.xscale('log', base=2)
# X_tick = [Fraction(item).limit_denominator() for item in alphas]
# plt.xticks(np.unique(alphas), X_tick)
plt.plot(alphas, gradient_rewards, label='gradient')

# plt.xscale('log', base=2)
# X_tick = [Fraction(item).limit_denominator() for item in epsilons]
# plt.xticks(np.unique(epsilons), X_tick)
plt.plot(epsilons,epsilon_rewards, label='ε-greedy' )
# plt.xscale('log', base=2)
# X_tick = [Fraction(item).limit_denominator() for item in confidences]
# plt.xticks(np.unique(confidences), X_tick)
plt.plot(confidences, ucb_rewards, label='UCB')
# plt.xscale('log', base=2)
# X_tick = [Fraction(item).limit_denominator() for item in Q_zeros]
# plt.xticks(np.unique(Q_zeros), X_tick)
plt.plot(Q_zeros, optimistic_rewards, label='optimistic greedy')

plt.legend()
plt.xlabel("hyperparameter")
plt.ylabel("Average reward")

plt.show()
