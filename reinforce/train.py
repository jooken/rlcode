from environment import Env
import numpy as np
import pylab
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = Reinforce(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path, save_format='tf')

    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = np.float32(self._discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)

    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


if __name__ == '__main__':
    env = Env(render_speed=0.01)
    state_size = len(env.get_state())
    action_size = env.action_size
    agent = ReinforceAgent(state_size, action_size)

    def for_keras(state):
        return np.reshape(state, [1, state_size]).astype(dtype=float)
    
    episodes, scores = env.load_data(agent)

    EPISODES = 300
    for _ in range(EPISODES):
        current_episode = len(episodes)
        score = 0
        done = False

        state = for_keras(env.reset())

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = for_keras(next_state)

            agent.append_sample(state, action, reward)
            score += reward

            state = next_state

        entropy = agent.train_model()
        print("episode: {:3d} | score: {:3d} | entropy: {:.3f}".format(
                      current_episode, score, entropy))
        scores.append(score)
        episodes.append(current_episode)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("score")
        pylab.savefig("./save_graph/graph.png")
    
    env.save_data(agent, scores)
