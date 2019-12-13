import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
learning_rate = 0.01
reward_decay = 0.9
batch_size = 32
e_greedy = 0.999
target_update_freq = 100
memory_buffer_size = 2000
episodes = 400
max_steps=200

class Policy(nn.Module):
    def __init__(self, action_n, state_n):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_n, 200)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(200, action_n)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, env, learning_rate=learning_rate, reward_decay=reward_decay, batch_size=batch_size, e_greedy=e_greedy, target_update_freq=target_update_freq, memory_buffer_size = memory_buffer_size, episodes=episodes, max_steps=max_steps):
        self.env = env

        self.action_n = len(env.action_space)
        self.state_n = env.observation_space['low'].shape[0]

        self.eval_policy, self.target_policy = self.create_model()
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.batch_size=batch_size
        self.target_update_freq = target_update_freq   # target update frequency
        self.memory_buffer_size = memory_buffer_size
        self.episodes = episodes
        self.max_steps = max_steps



        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((memory_buffer_size, self.state_n * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_policy.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.rewards = []
        self.final_positions = []

    def create_model(self):
        return Policy(action_n=self.action_n, state_n=self.state_n), Policy(action_n=self.action_n, state_n=self.state_n)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            actions_value = self.eval_policy.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
            # action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)
        else:  # random
            action = np.random.randint(0, self.action_n)
            # action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.episodes
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.eval_policy.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.episodes, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_n])
        b_a = torch.LongTensor(b_memory[:, self.state_n:self.state_n+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_n+1:self.state_n+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_n:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_policy(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_policy(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_policy.state_dict(), 'model/DQN/eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))
        torch.save(self.target_policy.state_dict(), 'model/DQN/target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))

    def load_model(self, model_name):
        self.eval_policy.load_state_dict(torch.load(model_name))

    def newReward(self, obsesrvation, obsesrvation_):
        return abs(obsesrvation_[0] - (-0.5))

    def update(self):
        records = []
        for episode in range(self.episodes):
            # initial
            observation = self.env.reset()

            iter_cnt, total_reward = 0, 0
            true_reward = 0

            if (episode+1)%5==0:
                self.viz()
            # self.epsilon*=0.99
            while True:
                iter_cnt += 1

                # fresh env
                # env.render()

                # self choose action based on observation
                action = self.choose_action(observation)
                # self take action and get next observation and reward
                observation_, reward, done, _ = self.env.step(action)
                true_reward+=reward

                reward = self.newReward(observation, observation_)
                # self learn from this transition
                self.store_transition(observation, action, reward, observation_)
                if self.memory_counter > self.episodes:
                    self.learn()

                # accumulate reward
                total_reward += reward
                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done or iter_cnt>=self.max_steps:
                    total_reward = round(total_reward, 2)
                    records.append((iter_cnt, total_reward))
                    print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                    print("true reward", true_reward)
                    self.final_positions.append(observation[0])
                    self.rewards.append(true_reward)
                    break

    def viz(self, save=True):
        fig, ax = plt.subplots(2,sharex=True, figsize=(10, 7))
        ax[0].plot(np.arange(len(self.rewards)) + 1, self.rewards)
        ax[0].set(xlabel="Episodes",ylabel="Reward")
        ax[0].set_title('Average Reward vs Episodes')

        ax[1].plot(np.arange(len(self.final_positions)) + 1, self.final_positions)
        ax[1].set(xlabel="Episodes",ylabel="Final Pos")
        ax[1].set_title('Final Pos vs Episodes')
        plt.savefig('rewards_qn_ER.jpg')



from game import MountainCarEnv
env = MountainCarEnv()
dqn_agent = DQN(env, learning_rate, reward_decay, batch_size, e_greedy, target_update_freq, memory_buffer_size, episodes, max_steps)

dqn_agent.update()
dqn_agent.viz()
