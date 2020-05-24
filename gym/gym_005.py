import gym
import numpy as np


# build the environment
env = gym.make("FrozenLake-v0")

print( "show env:")
env.render()

# show the enviroment size: 4*4=16
print( "env space size: ", env.observation_space.n)

# show the action space size: 上下左右
print( "action space size: ", env.action_space.n)

# initial the V(s) to zero
value_table = np.zeros( env.observation_space.n )
# iterate 100000 times
no_of_iterations = 10**5

for i in range( no_of_iterations ):
    # 
    updated_value_table = np.copy( value_table )