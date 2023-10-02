import numpy as np
import random
from collections import defaultdict
from environment import Env, GraphicDisplay


# Monte Carlo Agent which learns every episodes from the sample
class MCAgent:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = []
        self.value_table = defaultdict(float)

    def print_value_table(self):
        print("value_table : ", self.value_table)

    # append sample to memory(state, reward, done)
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # for every episode, agent updates q function of visited states
    def update(self):
        G_t = 0
        visit_state = []
        for step in reversed(self.samples):
            state, reward, done = step
            if state not in visit_state:
                visit_state.append(state)
                G_t = self.discount_factor * (reward + G_t)
                value = self.value_table[state]
                print('{0} => R:{1}, v_mu:{2}'.format(state, reward, value))
                self.value_table[state] = (value +
                                           self.learning_rate * (G_t - value))

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
        # print('get_value state = {0}({1})'.format(type(state), state))
        return self.value_table[state]
    
    def set_value(self, state, value):
        self.value_table[state] = value

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.env.possible_actions)
        else:
            # take action according to the q function table
            next_state = self.possible_next_states_value(state)
            action = self.arg_max(next_state)
        return int(action)

    # compute arg_max if multiple candidates exit, pick one randomly
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # get the possible next states
    def possible_next_states_value(self, state):
        next_states_value = [0.0]*len(self.env.possible_actions)
        for index,action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(action)
            next_states_value[index] = self.get_value(next_state)
        return next_states_value


# main loop
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(env)
    grid_world = GraphicDisplay(agent)

    grid_world.load_value_function()

    # print(agent.to_json())

    for episode in range(200):
        state = grid_world.reset()
        action = agent.get_action(state)

        while True:
            grid_world.render()

            # forward to next state. reward is number and done is boolean
            next_state, reward, done = grid_world.step(action)
            agent.save_sample(next_state, reward, done)

            # get next action
            action = agent.get_action(next_state)

            # at the end of each episode, update the q function table
            if done:
                print("episode : ", episode)
                agent.update()
                agent.samples.clear()
                # agent.print_value_table()
                break
    
    # print(agent.to_json())
    grid_world.store_value_function()
