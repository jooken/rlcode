from environment import Env, PERSIST_MODEL
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class Reinforce(tf.keras.Model):
    def __init__(self, action_size):
        super(Reinforce, self).__init__()
        self.fc1 = Dense(36, activation='relu')
        self.fc2 = Dense(36, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy

class ReinforceAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = Reinforce(self.action_size)
        self.model.load_weights(PERSIST_MODEL)

    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

if __name__ == '__main__':
    env = Env(render_speed=0.1)
    state_size = len(env.get_state())
    action_size = env.action_size
    agent = ReinforceAgent(state_size, action_size)

    def for_keras(state):
        return np.reshape(state, [1, state_size]).astype(dtype=float)
    
    EPISODES = 10
    for current_episode in range(EPISODES):
        score = 0
        done = False

        state = for_keras(env.reset())

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = for_keras(next_state)

            score += reward

            state = next_state

        print("episode: {:3d} | score: {:3d}".format(current_episode, score))

