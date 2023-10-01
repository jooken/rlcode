from environment import Env, GraphicDisplay
import numpy as np

class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.discount_factor = 0.9
        self.reset()

    def reset(self):
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

    def from_json(self, json):
        if 'value_table' in json:
            self.value_table = json['value_table']

    def to_json(self):
        return {'value_table': self.value_table}

    def get_value(self, state):
        row, col = state[0], state[1]
        return self.value_table[row][col]

    def set_value(self, state, value):
        row, col = state[0], state[1]
        self.value_table[row][col] = value

    def get_action(self, state):
        if self.env.is_final_state(state):
            return []

        value_list = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = reward + self.discount_factor*next_value
            value_list.append(value)

        max_idx_list = np.argwhere(value_list == np.amax(value_list))
        action_list = max_idx_list.flatten().tolist()
        return action_list


    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            row, col = state[0], state[1]
            if self.env.is_final_state(state):
                next_value_table[row][col] = 0.0
                continue
            
            value_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor*next_value
                # print('v({0}) = R({1}) + γ*v({2}) | s({3}), next_s({4})'.format(value, reward, next_value, state, next_state))
                value_list.append(value)

            next_value_table[row][col] = max(value_list)

        self.value_table = next_value_table

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
