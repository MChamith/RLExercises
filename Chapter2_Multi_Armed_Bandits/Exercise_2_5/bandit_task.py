import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bandit import get_instance


def perform_task(k, epsilon, N_STEPS, N_RUNS, mode, alpha=0.1):
    Optimal_percentage = np.full((N_RUNS, N_STEPS), 0, dtype=float)
    Reward = np.full((N_RUNS, N_STEPS), 0, dtype=float)

    for run_i in tqdm(range(1, N_RUNS + 1)):
        bandit = get_instance(mode, k, alpha)
        for step in range(1, N_STEPS + 1):

            prob = random.uniform(0, 1)

            if prob < epsilon:
                step_action, reward = bandit.explore()
            else:
                step_action, reward = bandit.exploit()

            optimal = bandit.is_optimal(step_action)

            Optimal_percentage[run_i - 1][step - 1] = optimal
            Reward[run_i - 1][step - 1] = reward
            bandit.random_walk()

    return Optimal_percentage, Reward


# percent_optimal_sample, reward_sample = perform_task(10, 0.1, 10000, 2000, 'sample')
# avg_optimal_sample = np.mean(percent_optimal_sample, axis=0)
# avg_reward_sample = np.mean(reward_sample, axis=0)

percent_optimal_weighted, reward_weighted = perform_task(10, 0.1, 10000, 2000, 'weighted', 0.1)
avg_optimal_weighted = np.mean(percent_optimal_weighted, axis=0)
avg_reward_weighted = np.mean(reward_weighted, axis=0)

plt.figure(1)
plt.plot(avg_optimal_sample, label='ε = 0.1')
plt.plot(avg_optimal_weighted, label='ε = 0.1 α = 0.1')
plt.legend()
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")

plt.figure(2)
plt.plot(avg_reward_sample, label='ε = 0.1')
plt.plot(avg_reward_weighted, label='ε = 0.1 α = 0.1')
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Average reward")

plt.show()
