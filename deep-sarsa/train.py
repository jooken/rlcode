import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import pylab
from environment import Env

class DeepSARSA(tf.keras.Model):
    '''
        `DeepSARSA`는 큐함수[Q(s,a)]를 근사하는 신경망
    '''
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        '''
            `input`으로 `state` 벡터가 들어오면 출력으로 모든액션의 큐값을 리턴한다
            [참고] instance 그 자체를 호출하면 call function을 호출한다!
        '''
        # print('input =>', x)
        x = self.fc1(x)
        # print('layer1 =>', x)
        x = self.fc2(x)
        # print('layer2 =>', x)
        q = self.fc_out(x)
        # print('output =>', q)
        return q

class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def load_weights(self, path):
        # tf.keras.models.load_model(path)
        self.model.load_weights(path)

    def save_weights(self, path):
        # tf.keras.models.save_model(self.model, path)
        self.model.save_weights(path, save_format='tf')

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def train_model(self, s_0, a_0, r_1, s_1, a_1, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            predict = self.model(s_0)[0]
            one_hot_action = tf.one_hot([a_0], self.action_size)
            # print('model_params =>', model_params)
            predict = tf.reduce_sum(one_hot_action * predict, axis=1)
            # print('model_params =>', model_params)
            q_1 = self.model(s_1)[0][a_1]
            target = r_1 + (1 - done) * self.discount_factor * q_1

            loss = tf.reduce_mean(tf.square(target - predict))
            # print('model_params =>', model_params)

        # print('model_params =>', model_params)
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

if __name__ == '__main__':
    env = Env(render_speed=0.501)
    state_size = len(env.get_state())
    action_size = env.action_size
    agent = DeepSARSAgent(state_size, action_size)

    scores = []
    episodes = []

    def for_keras(state):
        return np.reshape(state, [1, state_size]).astype(dtype=float)

    episodes, scores = env.load_data(agent)

    EPISODES = 500
    for _ in range(EPISODES):
        current_episode = len(episodes)
        score = 0
        done = False

        state = for_keras(env.reset())

        while not done:
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = for_keras(next_state)
            next_action = agent.get_action(next_state)

            agent.train_model(state, action, reward, next_state, next_action, done)

            score += reward
            state = next_state

        print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      current_episode, score, agent.epsilon))
        scores.append(score)
        episodes.append(current_episode)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("score")
        pylab.savefig("./save_graph/graph.png")

    env.save_data(agent, scores)
