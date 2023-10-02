from environment import Env, GraphicDisplay
import numpy as np
import random
from collections import defaultdict

class TemporalDiffAgent:
    @staticmethod
    def arg_max(next_state_value):
        max_index_list = []
        max_value = next_state_value[0]
        for index, value in enumerate(next_state_value):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.value_table = defaultdict(float)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.env.possible_actions)
        else:
            # take action according to the q function table
            next_state_value = self._possible_next_states_value(state)
            action = self.arg_max(next_state_value)
        return int(action)

    def from_json(self, json):
        if 'value_table' in json:
            value_table = json['value_table']
            for state in value_table:
                self.set_value(eval(state), value_table[state])

    def to_json(self):
        value_table = {}
        for state in self.value_table:
            value_table[str(state)] = self.value_table[state]
        return {'value_table': value_table}

    def get_value(self, state):
        return self.value_table[state]

    def set_value(self, state, value):
        self.value_table[state] = value

    def _possible_next_states_value(self, state):
        next_states_value = [0.0]*len(self.env.possible_actions)
        for index,action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(action)
            next_states_value[index] = self.get_value(next_state)
        return next_states_value

    def update(self, state, next_state, reward):
        '''
          V(S_t) <- V(S_t) + alpha * (G_t - V(S_t))
          V(S_t) <- V(S_t) + alpha * (R + gamma * V(S_t+1) - V(S_t))
        '''
        value = self.get_value(state)
        next_value = self.get_value(next_state)
        value = value + self.learning_rate * (reward + self.discount_factor * next_value - value)
        # print('v({0}) <- v({1}) + alpha * (R({2}) + gamma * next_v({3}))'.format(state, value, reward, next_value))
        self.set_value(state, value)

if __name__ == '__main__':
    env = Env()
    agent = TemporalDiffAgent(env)
    grid_world = GraphicDisplay(agent)

    grid_world.load_value_function()

    for episode in range(10):
        state = grid_world.reset()
        action = agent.get_action(state)

        while True:
            next_state, reward, done = grid_world.step(action)

            agent.update(state, next_state, reward)

            state = next_state
            action = agent.get_action(state)

            if done:
                print("episode : ", episode)
                agent.update(state, state, reward)
                break

        print(agent.to_json())
    grid_world.store_value_function()
