from environment import Env, GraphicDisplay
import copy
import numpy as np

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.discount_factor = 0.9
        self.reset()

    def reset(self):
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

        for row, col in [ state for state in self.env.get_all_states() if self.env.is_final_state(state)]:
            self.policy_table[row][col] = []

    def policy_evaluation(self):
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            #print('x={0},y={1}'.format(state[0], state[1]))
            value = 0.0
            st_row, st_col = state[0], state[1]

            # print('state={0}================================'.format(state))

            if self.env.is_final_state(state):
                # print('Final State: ({0},{1})'.format(state[0],state[1]))
                next_value_table[st_row][st_col] = value
                continue
            
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (
                    self.get_policy(state)[action] * (
                        reward + self.discount_factor * next_value
                    )
                )
                # print('r={3},rxt={4},v={5}|s=({0},{1}),a={2}'.format(st_row, st_col, action, reward, next_value, value))

            next_value_table[st_row][st_col] = value

        self.value_table = next_value_table

    def get_value(self, state):
        row, col = state[0], state[1]
        return self.value_table[row][col]

    def get_policy(self, state):
        row, col = state[0], state[1]
        return self.policy_table[row][col]

    def policy_improvement(self):
        next_policy = copy.deepcopy(self.policy_table)
        for state in self.env.get_all_states():
            if self.env.is_final_state(state):
                continue

            st_row, st_col = state[0], state[1]
            value_list = []
            result = [0.0]*4
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)
            
            max_idx_list = np.argwhere(value_list == np.amax(value_list))
            max_idx_list = max_idx_list.flatten().tolist()
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob

            next_policy[st_row][st_col] = result
        self.policy_table = next_policy

if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
