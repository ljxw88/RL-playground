import gym
import torch.nn as nn
import torch
import copy
import random
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Bernoulli

cuda = torch.device('cuda:0')

class Net(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        dropout_rate = 0.25
        hidden = 10
        self.fc1 = nn.Linear(in_feature, hidden)
        self.fc2 = nn.Linear(hidden, out_feature)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = torch.tanh(x)
        return x

class PolicyGradient(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.policy_net = self._build_net()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def _build_net(self):
        model = Net(self.n_features, self.n_actions)
        return model

    def choose_action(self, observation):
        state = torch.from_numpy(observation[np.newaxis, :]).float()
        probs = self.policy_net(state)
        prob_weights = F.softmax(probs, dim=1).detach().cpu().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # gradient descent 
        self.optimizer.zero_grad()

        criteria = nn.NLLLoss()
        observation_array = np.array(self.ep_obs)
        state = torch.FloatTensor(observation_array)
        action = torch.tensor([self.ep_as], dtype = torch.long).T.squeeze(1)
        for i in range(len(self.ep_as)):
            probs = self.policy_net(state[i])
            loss = 1.* criteria(probs, action[i]) * discounted_ep_rs_norm[i]
            loss.backward()

        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # print('calculate discount rewards...')
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    