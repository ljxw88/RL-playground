import gym
import torch
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from IPython import display as ipythondisplay
import random

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater than this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0', render_mode='rgb_array')
# env.seed(1)     # reproducible, general Policy gradient has high variance
env.action_space.seed(42)
np.random.seed(1)
torch.manual_seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(1000):
    observation, info = env.reset()
    env.render()

    while True:
        action = RL.choose_action(observation) # policy network
        observation_, reward, done, _, info = env.step(action) # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train

            # if i_episode == 30:
            #     plt.plot(vt)  # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()

            break

        observation = observation_
env.close()