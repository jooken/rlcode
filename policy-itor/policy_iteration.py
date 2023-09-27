from environment import Env, GraphicDisplay

class PolicyIteration:
    def __init__(self, env):
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

    def policy_evaluation(self):
        pass



if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
