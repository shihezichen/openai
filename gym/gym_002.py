

import gym
env = gym.make("BipedalWalker-v3")

for episode in range(100):
    env.reset()
    env.render()
    for i in range(10000):
        env.render()
        # random sample from the 'action space'
        action = env.action_space.sample()
        observation, rewoard, done, info = env.step( action )
        if done:
            print("{} timesteps taken for the Episode.".format(i+1))
            break
env.close()
