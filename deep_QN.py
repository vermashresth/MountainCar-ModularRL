import gym
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

from game import MountainCarEnv

class DQN:

    def __init__(self, env, gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=.01, tau=0.125, max_buffer=2000, episodes = 1000, max_episode_len = 200):
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

    def create_model(self):


        model   = Sequential()
        state_shape  = self.env.observation_space['low'].shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(self.env.action_space)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        try:
            model = load_model("success-my.model")
            print("success")
        # except:
        #     pass
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
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

    def train(self):

        steps = []
        for i in range(self.episodes):

            tot_reward = 0
            cur_state = self.env.reset().reshape(1,2)

            for step in range(self.max_episode_len):
                action = self.act(cur_state)
                new_state, reward, done, _ = self.env.step(action)

                # reward = reward if not done else -20
                new_state = new_state.reshape(1,2)
                self.remember(cur_state, action, reward, new_state, done)

                self.replay()       # internally iterates default (prediction) model
                self.target_train() # iterates target model

                cur_state = new_state
                tot_reward += reward
                if done:
                    break

            self.rewards.append(tot_reward)
            self.iterations.append(step)


            avg_reward = np.mean(self.rewards)
            self.avg_rewards.append(avg_reward)
            self.rewards = []
            print('Episode {} Average Reward: {}'.format(i+1, avg_reward))
            if (i+1) % 10 == 0:
                self.save_model("success-my.model")

    def viz(self, save=True):
            fig, ax = plt.subplots(3,sharex=True, figsize=(10, 10))
            ax[0].plot(100*(np.arange(len(self.rewards)) + 1), self.rewards)
            ax[0].set(xlabel="Episodes",ylabel="Reward")
            ax[0].set_title('Average Reward vs Episodes')

            ax[1].plot(100*(np.arange(len(self.iterations)) + 1), self.iterations)
            ax[1].set(xlabel="Episodes",ylabel="Iterations")
            ax[1].set_title('Iterations till goal/termination ')

            ax[2].plot(100*(np.arange(len(self.rewards)) + 1), self.final_positions)
            ax[2].set(xlabel="Episodes",ylabel="Final Pos")
            ax[2].set_title('Final Position vs Episodes')
            plt.savefig('rewards_dqn.jpg')

env = MountainCarEnv()


# updateTargetNetwork = 1000
dqn_agent = DQN(env=env)
dqn_agent.train()
dqn_agent.viz()
