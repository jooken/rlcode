from environment import Env, PERSIST_MODEL
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense

class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 0.01
        self.model = DeepSARSA(self.action_size)
        self.model.load_weights(PERSIST_MODEL)
    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return np.argmax(q_values[0])

if __name__ == '__main__':
    env = Env(render_speed=0.05)
    state_size = len(env.get_state())
    action_size = env.action_size
    agent = DeepSARSAgent(state_size, action_size)

    def for_keras(state):
        return np.reshape(state, [1, state_size]).astype(dtype=float)

    scores, episodes = [], []

    EPISODES = 10
    for current_episode in range(EPISODES):
        score = 0
        done = False

        state = for_keras(env.reset())

        while not done:
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = for_keras(next_state)
            
            state = next_state
            score += reward

            if done:
                print("episode: {:3d} | score: {:3d}".format(current_episode, score))
