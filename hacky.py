import gym
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json

import pickle

import matplotlib.pyplot as plt
from collections import deque

from game import MountainCarEnv

class DQN:

    def __init__(self, env, gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=.005, tau=0.125, max_buffer=10000, episodes = 500, max_episode_len = 1500):
        self.env = env
        self.memory = deque(maxlen=max_buffer)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau


        self.episodes  = episodes
        self.max_episode_len = max_episode_len

        self.model = self.create_model()
        self.target_model = self.create_model()

        self.rewards = []
        self.avg_rewards = []
        self.iterations = []
        self.final_positions = []

        # try:
        #     with open("deep_qn/rewards.pkl", "wb") as f:
        #         self.rewards = pickle.load(f)
        #     with open("deep_qn/iterations.pkl", "wb") as f:
        #         self.iterations =  pickle.load(f)
        #     with open("deep_qn/final_positions.pkl", "wb") as f:
        #         self.final_positions =  pickle.load(f)
        #     print("loaded logs")
        # except:
        #     pass

    def create_model(self):


        model   = Sequential()
        state_shape  = self.env.observation_space['low'].shape
        model.add(Dense(64, input_dim=state_shape[0], activation="relu"))
        # model.add(Dense(48, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(len(self.env.action_space)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        try:
            json_file = open('model-hack.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model-hack.h5")
            print("Loaded model from disk")
            loaded_model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate))
            return loaded_model
        except Exception as e:
            print(e)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(np.arange(len(self.env.action_space)))
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory = []
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, samples):
        batch_size = 1
        # if len(self.memory) < batch_size:
        #     return

        # samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
        model_json = self.model.to_json()
        with open("model-hack.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model-hack.h5")

        with open("deep_qn_hack/rewards.pkl", "wb") as f:
            pickle.dump(self.rewards, f)
        with open("deep_qn_hack/iterations.pkl", "wb") as f:
            pickle.dump(self.iterations, f)
        with open("deep_qn_hack/final_positions.pkl", "wb") as f:
            pickle.dump(self.final_positions, f)
        print("Saved model to disk")

    def train(self):

        steps = []
        for i in range(self.episodes):

            tot_reward = 0
            cur_state = self.env.reset().reshape(1,2)
            pos = False
            for step in range(self.max_episode_len):
                action = self.act(cur_state)
                new_state, reward, done, _ = self.env.step(action)

                # reward = reward if not done else -20
                new_state = new_state.reshape(1,2)
                # self.remember(cur_state, action, reward, new_state, done)

                self.replay([[cur_state, action, reward, new_state, done]])       # internally iterates default (prediction) model
                self.target_train() # iterates target model

                cur_state = new_state
                tot_reward += reward
                if (step==200 or done) and not pos:
                    self.final_positions.append(new_state[0])
                    pos = True
                if done:
                    break

            self.rewards.append(tot_reward)
            self.iterations.append(step)



            avg_reward = np.mean(self.rewards)
            self.avg_rewards.append(avg_reward)
            # self.rewards = []
            print('Episode {} Average Reward: {}'.format(i+1, avg_reward))
            if (i+1) % 5 == 0:
                self.save_model("success-my-2-hack.model")
                self.viz()

    def viz(self, save=True):
            fig, ax = plt.subplots(3,sharex=True, figsize=(10, 10))
            ax[0].plot(np.arange(len(self.rewards)) + 1, self.rewards)
            ax[0].set(xlabel="Episodes",ylabel="Reward")
            ax[0].set_title('Reward vs Episodes')

            ax[1].plot(np.arange(len(self.iterations)) + 1, self.iterations)
            ax[1].set(xlabel="Episodes",ylabel="Iterations")
            ax[1].set_title('Iterations till goal/termination ')

            ax[2].plot(np.arange(len(self.rewards)) + 1, np.array(self.final_positions)[:, 0])
            ax[2].set(xlabel="Episodes",ylabel="Final Pos")
            ax[2].set_title('Final Position vs Episodes')
            plt.savefig('rewards_dqn-2-hack.jpg')

env = MountainCarEnv()


# updateTargetNetwork = 1000
dqn_agent = DQN(env=env)
dqn_agent.train()
# dqn_agent.viz()
