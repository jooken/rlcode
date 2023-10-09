import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import pylab
import sys
import os
import json

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

        self.render = False

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        self.memory = deque(maxlen=2000)

        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        self.update_target_model()

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path, save_format='tf')

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))
        
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


state_size = 1
def for_keras(state):
    return np.reshape(state, [1, state_size])

def load_data(agent):
    scores = []
    score_avg = 0
    if os.path.exists(PERSIST_JSON):
        data = None
        with open(PERSIST_JSON, 'rt') as f:
            data = json.load(f)
        if data:
            agent.load_weights(PERSIST_MODEL)
            if 'scores' in data:
                scores = data['scores']
            if 'score_avg' in data:
                score_avg = data['score_avg']
            if 'memory' in data:
                agent.memory = data['memory']
            if 'epsilon' in data:
                agent.epsilon = data['epsilon']
            if 'epsilon_decay' in data:
                agent.epsilon_decay = data['epsilon_decay']
    return [*range(len(scores))], scores, score_avg

def be_serializable(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    elif isinstance(a, np.int64):
        return int(a)
    return a

def save_data(agent, scores, score_avg):
    agent.save_weights(PERSIST_MODEL)

    # memory = []
    # for sample in agent.memory:
        # print(type(sample)) #==> <class 'tuple'>
        # 0: numpy.ndarray ==> it can't be serialized
        # 1: int
        # 2: float
        # 3: numpy.ndarray ==> it can't be serialized
        # 4: bool
        # memory.append([_.tolist() if isinstance(_, np.ndarray) else _ for _ in sample])
    
    memory = [ [be_serializable(_) for _ in sample] for sample in agent.memory]
    # print(memory)
    # print(json.dumps(memory, indent=2))
    data = {
        'scores': scores,
        'score_avg': score_avg,
        'memory': memory,
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay
        }
    with open(PERSIST_JSON, 'wt') as f:
        f.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('state_size: {0}, action_size: {1}'.format(state_size, action_size))
    
    state = env.reset()

    agent = DQNAgent(state_size, action_size)

    episodes, scores, score_avg = load_data(agent)

    EPISODES = 100
    for current_episode in range(EPISODES):
        current_episode = len(episodes)
        done = False
        score = 0

        state, info = env.reset()
        state = for_keras(state)
        
        while not done:
            if agent.render:
                env.render()
            
            action = agent.get_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = for_keras(next_state)

            score += reward
            reward = 0.1 if not done or score == 500 else -1

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            
            state = next_state
        
        agent.update_target_model()

        score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
        print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(
                      current_episode, score_avg, len(agent.memory), agent.epsilon))
        scores.append(score_avg)
        episodes.append(current_episode)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("average score")
        pylab.savefig("./save_graph/graph.png")

        if score_avg > 400:
            # agent.model.save_weights("./save_model/model", save_format="tf")
            break

    save_data(agent, scores, score_avg)
    env.close()
