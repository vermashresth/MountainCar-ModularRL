import numpy as np
import gym
import matplotlib.pyplot as plt
from game import MountainCarEnv


# Define Q-learning function
class QLearning:

    def __init__(self, env, learning, discount, epsilon, min_eps, episodes):
        self.env = env
        self.learning = learning
        self.epsilon = epsilon
        self.discount = discount
        self.min_eps = min_eps
        self.episodes = episodes


        self.Q_table = self.create_model()

        # Initialize metric lists
        self.rewards = []
        self.avg_rewards = []
        self.iterations = []
        self.final_positions = []

        # Calculate episodic reduction in epsilon
        self.reduction = (self.epsilon - self.min_eps)/self.episodes

    def create_model(self):
        pos_range = self.env.max_position-env.min_position+1
        vel_range = self.env.max_vel*2+1

        pos_discrete_states = int(pos_range*10)
        vel_discrete_states = int(vel_range*100)

        # discrete_states = ((env.high-eng.low)*np.array([10, 100])+1).astype(int)

        Q_table = np.random.uniform(low = -1, high = 1,
                                    size = (pos_discrete_states, vel_discrete_states,
                                    len(self.env.action_space)))
        return Q_table

    def act(self, state):
        # Determine next action - epsilon greedy strategy
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q_table[state[0], state[1]])
        else:
            action = np.random.randint(0, len(self.env.action_space))
        return action

    def train(self):
        # Run Q learning algorithm
        for i in range(self.episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0,0
            state = self.env.reset()

            # Discretize state
            discrete_state = ((state - self.env.observation_space['low'])*np.array([10, 100])).astype(int).flatten()

            iteration = 0
            while done != True:

                action = self.act(discrete_state)

                # Get next state and reward
                new_state, reward, done, _ = self.env.step(action)

                # Discretize state2
                new_discrete_state = ((new_state - self.env.observation_space['low'])*np.array([10, 100])).astype(int).flatten()

                # Update at terminal states
                if done and new_state[0] >= 0.5:
                    self.Q_table[discrete_state[0], discrete_state[1], action] = reward

                # Adjust Q value for current state
                else:
                    delta = self.learning*(reward +
                                     self.discount*np.max(self.Q_table[new_discrete_state[0],
                                                       new_discrete_state[1]]) -
                                     self.Q_table[discrete_state[0], discrete_state[1],action])
                    self.Q_table[discrete_state[0], discrete_state[1],action] += delta

                # Update variables
                tot_reward += reward
                discrete_state = new_discrete_state

                iteration+=1
                if iteration>=self.env.max_episodes:
                    break
            self.final_positions.append(new_state[0])

            # Decay epsilon
            if self.epsilon > self.min_eps:
                self.epsilon -= self.reduction

            # Track rewards
            self.rewards.append(tot_reward)
            self.iterations.append(iteration)

            if (i+1) % 100 == 0:
                avg_reward = np.mean(self.rewards)
                self.avg_rewards.append(avg_reward)
                self.rewards = []
                print('Episode {} Average Reward: {}'.format(i+1, avg_reward))

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
        plt.savefig('rewards_qn.jpg')

# Import and initialize Mountain Car Environment
env = MountainCarEnv()
# Run Q-learning algorithm
model = QLearning(env, 0.2, 0.9, 0.8, 0, 20000)
model.train()
model.viz()
