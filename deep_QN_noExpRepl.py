
# import ipympl
import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import glob, os


e_greedy = 0.3
reward_decay = 0.99
episodes = 3000
max_steps = 200
learning_rate = 0.001
#
# class Policy(nn.Module):
#     def __init__(self, action_n, state_n):
#         super(Policy, self).__init__()
#         self.fc1 = nn.Linear(state_n, 200)
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.out = nn.Linear(200, action_n)
#         self.out.weight.data.normal_(0, 0.1)   # initialization
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value
class Policy(nn.Module):
    def __init__(self, action_n, state_n):
        super(Policy, self).__init__()
        self.state_space = state_n
        self.action_space = action_n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)



class DQN:

    def __init__(self, env, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy, episodes=episodes, max_steps=max_steps ):
        # Parameters
        self.env = env
        self.action_n = len(env.action_space)
        self.state_n = env.observation_space['low'].shape[0]
        self.epsilon = e_greedy
        self.gamma = reward_decay

        self.episodes = episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.final_positions = []
        self.rewards= []

        # Initialize Policy
        self.policy = self.create_model()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    def create_model(self):
        return Policy(self.action_n, self.state_n)

    def update(self):
        state = self.env.reset()
        for episode in range(self.episodes):
            episode_loss = 0
            episode_reward = 0
            state = self.env.reset()

            if (episode+1)%5==0:
                self.viz()
            for s in range(self.max_steps):
                # Get first action value function
                Q = self.policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))


                # Choose self.epsilon-greedy action
                if np.random.rand(1) < self.epsilon:
                    action = np.random.randint(0,3)
                else:
                    _, action = torch.max(Q, -1)
                    action = action.item()

                # Step forward and receive next state and reward
                state_1, reward, done, _ = self.env.step(action)

                # Find max Q for t+1 state
                Q1 = self.policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
                maxQ1, _ = torch.max(Q1, -1)

                # Create target Q value for training the self.policy
                Q_target = Q.clone()
                Q_target = Variable(Q_target.data)
                Q_target[action] = reward + torch.mul(maxQ1.detach(), self.gamma)

                # Calculate loss
                loss = self.loss_fn(Q, Q_target)

                # Update self.policy
                self.policy.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record history
                episode_loss += loss.item()
                episode_reward += reward



                if done:

                    if state_1[0] >= 0.5:
                        # On successful epsisodes, adjust the following parameters

                        # Adjust self.epsilon
                        self.epsilon *= .99

                        # Adjust learning rate
                        self.scheduler.step()

                    # Record history

                    break

                else:
                    state = state_1
            self.final_positions.append(state_1[0])
            self.rewards.append(episode_reward)
            print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, s, episode_reward))


    def viz(self, save=True):
        fig, ax = plt.subplots(2,sharex=True, figsize=(10, 7))
        ax[0].plot(np.arange(len(self.rewards)) + 1, self.rewards)
        ax[0].set(xlabel="Episodes",ylabel="Reward")
        ax[0].set_title('Average Reward vs Episodes')

        ax[1].plot(np.arange(len(self.final_positions)) + 1, self.final_positions)
        ax[1].set(xlabel="Episodes",ylabel="Final Pos")
        ax[1].set_title('Final Pos vs Episodes')
        plt.savefig('rewards_qn_noER.jpg')





from game import MountainCarEnv
env = MountainCarEnv()
dqn_agent = DQN(env, learning_rate, reward_decay, e_greedy, episodes, max_steps)

dqn_agent.update()
