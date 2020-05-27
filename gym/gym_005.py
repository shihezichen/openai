import gym
import numpy as np

# build the environment
gEnv = gym.make("FrozenLake-v0")
gEnv.reset()
print("show env:")
gEnv.render()

# show the environment size: 4*4=16
print("env space size: ", gEnv.observation_space.n)
# show the action space size: 上下左右
print("action space size: ", gEnv.action_space.n)


# 值迭代过程:  本质是状态V值的迭代刷新过程
#   1. 先初始化一个为0的 V值表 V(s)
#   2. 根据V(s) 和 从env获取的所有action得到的rewards, 计算出 Q(s,a)
#   3. 通过 maxQ(s,s) 刷新 V(s)
#   4. 不断重复以上过程2, 3过程, 直到V(s)刷新前后差别很小时, 退出迭代过程
def value_iteration(env, gamma=1.0):
    env = env.unwrapped
    # 初始化 V(s) 为全0
    value_table = np.zeros(env.observation_space.n)
    # 最高迭代10^5次
    no_of_iterations = 10 ** 5
    # 迭代退出条件: V(s)刷新前后的差值<=1^(-20)
    threshold = 1e-5

    policy_table = np.zeros(env.observation_space.n)
    # 开始迭代
    for i in range(no_of_iterations):
        # 保存原有的 V(s), 以便于最后求V(s)更新前后的差值
        pre_value_table = np.copy(value_table)
        # 遍历所有的状态, 以便于更新每个状态对应的V值
        for state in range(env.observation_space.n):
            # 由于某个状态的V值本质是最大的Q(s,a),因此用此Q值表保存所有的动作a对应的Q值
            # 由于一个动作对应的下一个状态可能更会有多,需要求和得到本状态下, 执行本动作后, 进入所有可能的下一个状态时获得的奖励总和
            Q_table = np.zeros(env.action_space.n)
            # 遍历所有的动作, 求出动作action对应Q值, 添加到Q值表中
            for action in range(env.action_space.n):
                # 针对本状态执行action后所有的可能下一个状态,都计算其 转移概率, 奖励
                for next_sr in env.P[state][action]:
                    # 转移概率trans_prob, 奖励 reward
                    trans_prob, next_state, reward, _ = next_sr
                    # 按照公式求奖励:
                    #     Q(s,a) = 转移概率 * ( 本此动作的奖励 + gamma * 下一个状态的V值)
                    action_reward = trans_prob * (reward + gamma * value_table[next_state])
                    # 把本次状态转移的奖励添加到总奖励中
                    Q_table[action] += action_reward
            # 在本状态的所有动作中挑出最大的奖励, 作为本状态新的V值
            value_table[state] = np.max(Q_table)
            # 从Q值表中选出最大的那个动作(np.argmax获得序号,且序号为action), 作为本状态的新的Policy
            policy_table[state] = np.argmax(Q_table)

        # 求两次迭代之间的V值表差值
        delta = np.sum(np.fabs(value_table - pre_value_table))
        # 若达到迭代退出条件, 则退出
        if delta <= threshold:
            print("converged at iteration #%d." % (i + 1))
            break
    # 返回刷新后的V值表
    return value_table, policy_table


v_table, p_table = value_iteration(env=gEnv, gamma=1.0)

print("value_table: \n", v_table)
print("policy_table: \n", p_table)


