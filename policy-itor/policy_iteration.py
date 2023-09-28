from environment import Env, GraphicDisplay

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def policy_evaluation(self):
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            #print('x={0},y={1}'.format(state[0], state[1]))
            value = 0.0
            st_row, st_col = state[0], state[1]

            # print('state={0}'.format(state))

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
                # print('v={3}|s=({0},{1}),a={2}'.format(st_row, st_col, action, value))

            next_value_table[st_row][st_col] = value

        self.value_table = next_value_table

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]

    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
