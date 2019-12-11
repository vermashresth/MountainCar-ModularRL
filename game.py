
import math
import numpy as np


class MountainCarEnv():

    def __init__(self, seed=None):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_vel = 0.07
        self.target_position = 0.5


        self.low = np.array([self.min_position, -self.max_vel])
        self.high = np.array([self.max_position, self.max_vel])
        #
        # vel_range = self.max_vel*2+1
        # pos_range = self.max_position-self.min_position+1
        #
        # vel_discrete_obs = np.round(vel_range*100, 0).astype(int)
        # pos_discrete_obs = np.round()

        self.action_space = [-1, 0, 1]
        self.observation_space = {'low':[self.low], 'high':[self.high]}

        if seed is not None:
            np.random.seed(seed)


    def step(self, action):
        done = False
        position, velocity = self.state

        action = self.action_space[action]
        
        velocity += action*0.001 - math.cos(3*position)*0.0025
        velocity = np.clip(velocity, -self.max_vel, self.max_vel)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if (position==self.min_position and velocity<0):
            velocity = 0

        if (position >= self.target_position):
            done = True

        reward = -1.0

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}


    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)
