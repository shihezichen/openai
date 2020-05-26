import gym
import numpy as np


# build the environment
gEnv = gym.make("FrozenLake-v0")

print("show env:")
gEnv.render()

# show the enviroment size: 4*4=16
print("env space size: ", gEnv.observation_space.n)

# show the action space size: 上下左右
print("action space size: ", gEnv.action_space.n)

print("env: ", dir(gEnv))


def value_iteration(env, gamma=1.0):
    env = env.unwrapped
    # initial the V(s) to zero
    value_table = np.zeros(env.observation_space.n)
    # iterate 100000 times
    no_of_iterations = 10 ** 5
    threshold = 1e-20
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        # 遍历所有的状态
        for state in range(env.observation_space.n):
            Q_value = []
            # 遍历所有的动作
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    action_reward = trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                    next_states_rewards.append(action_reward)
                state_reward = np.sum(next_states_rewards)
                Q_value.append(state_reward)
            value_table[state] = max(Q_value)
        delta = np.sum(np.fabs(updated_value_table - value_table))
        if delta <= threshold:
            print("converged at iteration # %d." % (i + 1))
            break
    return value_table


def extract_policy(env, value_table, gamma=1.0):
    env = env.unwrapped
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                action_reward = trans_prob * (reward_prob + gamma * value_table[next_state])
                Q_table[action] += action_reward
        policy[state] = np.argmax(Q_table)
    return policy


optimal_value_function = value_iteration(env=gEnv, gamma=1.0)
optimal_policy = extract_policy(env=gEnv, value_table=optimal_value_function, gamma=1.0)

print(optimal_policy)
