import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform

PERSIST_MODEL = './data/model'
PERSIST_JSON = './data/data.json'

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(action_size)
        self.model.load_weights(PERSIST_MODEL)

    def get_action(self, state):
        q_value = self.model(state)
        return np.argmax(q_value[0])

state_size = 1
def for_keras(state):
    return np.reshape(state, [1, state_size])

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    EPISODES = 10
    for current_episode in range(EPISODES):
        done = False
        score = 0

        state, info = env.reset()
        state = for_keras(state)

        while not done:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = for_keras(next_state)

            score += reward
            state = next_state

        print("episode: {:3d} | score: {:.3f} ".format(current_episode, score))
