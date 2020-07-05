import random
import time
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 根据公式更新Q值表
def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

# epsilon 贪婪策略
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: q[(state,x)])


alpha = 0.4
gamma = 0.999
epsilon = 0.017

# 所有轮次的回报
episode_rewards = []

env = gym.make('Taxi-v3')

print("observations:", env.observation_space)
print("actions:", env.action_space)

q = {}
# for s in range(env.observation_space.n):
#     for a in range(env.action_space.n):
#         q[(s, a)] = 0.0
q = np.zeros((env.observation_space.n,env.action_space.n))

for i in range(2000):
    r = 0
    prev_state = env.reset()
    taxirow, taxicol, passloc, destidx = env.unwrapped.decode(prev_state)
    #print("出租车位置: ",taxirow, taxicol)
    #print("乘客位置:", passloc)
    #print("乘客目的地：", destidx)

    while True:
        # env.render()
        # 根据贪婪策略， 选择一个行为
        action = epsilon_greedy_policy(prev_state, epsilon)
        # 执行行为，得到下一个状态，以及奖励
        nextstate, reward, done, _ = env.step(action)
        # 更新下一个状态来Q值表
        update_q_table(prev_state, action, reward, nextstate, alpha, gamma)
        # 更新总奖励值
        r += reward
        # 转换到下一个状态
        prev_state = nextstate
        # 判断是否结束
        if done:
            break
        #time.sleep(1)
    episode_rewards.append(r)
    print("total reward: ", r)

print('Q:\n', q)
env.close()

plt.plot(episode_rewards)
plt.show()

