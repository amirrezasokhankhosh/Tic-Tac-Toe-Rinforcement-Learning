
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from RLGlue.rl_glue import RLGlue
from numpy.lib.function_base import add_newdoc_ufunc
from environment import BaseEnvironment
from environment import TicTacToeEnvironment
from agent import Agent
from RLGlue import BaseAgent
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import os
import shutil
from plot_script import plot_result
import random


agent_config = {
    "optimizer_config" : {
        "num_states" : 3 ** 9,
        "num_hidden_layer" : 1,
        "num_hidden_units" : 3 ** 6,
        "step_size" : 3e-5,
        "beta_m": 0.9,
        "beta_v": 0.999,
        "epsilon": 1e-8
    },
    "network_config" : {
        "num_states" : 3 ** 9,
        "num_hidden_layer" : 1,
        "num_hidden_units" : 3 ** 6,
        'gamma': 0.99
    },
    "num_states" : 3 ** 9,
    "num_hidden_layer" : 1,
    "num_hidden_units" : 3 ** 6,
    'replay_buffer_size': 32,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 1000.0,
    'seed': 0
}


avg_rewards = np.zeros(20)

agent = Agent()
agent.agent_init(agent_config)

env = TicTacToeEnvironment()
env.env_init()
for j in range(len(avg_rewards)):
    print("Progress : " + str(j * 5) + "%.")
    rewards = np.zeros(15)
    for i in range(len(rewards)):
        obs = env.env_start()
        action = agent.agent_start(obs)
        reward_obs_term = env.env_step(action)
        while reward_obs_term[2] != True:
            action = agent.agent_step(reward_obs_term[0], reward_obs_term[1])
            reward_obs_term = env.env_step(action)
        agent.agent_end(reward_obs_term[0])
        rewards[i] = agent.sum_rewards
    avg_rewards[j] = np.average(rewards)

x_axis = np.zeros(20)
for i in range(len(x_axis)):
    x_axis[i] = i + 1
plt.plot(x_axis, avg_rewards)
plt.xlabel("Runs (Each run = 30 episodes)")
plt.ylabel("Average Reward")
plt.show()
