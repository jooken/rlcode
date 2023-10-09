import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import time
import os
import json

UNIT = 100
HEIGHT = 5
WIDTH = 6

PERSIST_MODEL = './data/model'
PERSIST_JSON = './data/data.json'

class Env(tk.Tk):
    def __init__(self, render_speed=0.01):
        super(Env, self).__init__()
        self.render_speed = render_speed
        self.action_sapce = ['s','l','r','u','d']
        self.action_size = len(self.action_sapce)
        self.title('DeepSARSA')
        self.geometry('{2}x{3}+{0}+{1}'.format(2600,400, WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self._load_images()
        self.canvas = self._build_canvas()
        
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.reset_reward()

    def load_data(self, agent):
        scores = []
        if os.path.exists(PERSIST_JSON):
            data = None
            with open(PERSIST_JSON, 'rt') as f:
                data = json.load(f)
            if data:
                agent.load_weights(PERSIST_MODEL)
                if 'scores' in data:
                    scores = data['scores']
                if 'epsilon' in data:
                    agent.epsilon = data['epsilon']
                if 'epsilon_decay' in data:
                    agent.epsilon_decay = data['epsilon_decay']

        return [*range(len(scores))], scores

    def save_data(self, agent, scores):
        agent.save_weights(PERSIST_MODEL)
        data = {
            'scores': scores, 
            'epsilon': agent.epsilon,
            'epsilon_decay': agent.epsilon_decay
            }
        with open(PERSIST_JSON, 'wt') as f:
            f.write(json.dumps(data, indent=4))

    def reset(self):
        # self.counter = 0
        self._move_to_origin()
        # self.reset_reward()
        self.update()
        time.sleep(0.5)
        return self.get_state()

    def step(self, action):
        self.counter += 1

        if self.counter % 2 == 1:
            self.rewards = self._move_rewards()

        if not self._move_rectangle(action):
            return self.get_state(), -1, False

        self._render()

        row, col = self._find_cell_of_rectangle()
        reward, done = self._check_which_reward(row, col)

        return self.get_state(), reward, done

    def reset_reward(self):
        for reward in self.rewards:
            self.canvas.delete(reward['figure'])
        self.rewards.clear()
        self.goal.clear()

        self.set_reward(row=1, col=0)
        self.set_reward(row=2, col=1)
        self.set_reward(row=3, col=2)
        self.set_reward(row=4, col=5, reward=1)

    def set_reward(self, row, col, reward=-1):
        obj = {}

        obj['row'] = row
        obj['col'] = col
        obj['reward'] = reward

        figure = None
        if reward > 0:
            figure = self._draw_shape(self.canvas, row, col, image=self.shapes[2])
        elif reward < 0:
            figure = self._draw_shape(self.canvas, row, col, image=self.shapes[1])
            obj['direction'] = -1

        obj['figure'] = figure
        obj['coords'] = self.canvas.coords(figure)
        self.rewards.append(obj)

    def get_state(self):
        agent_row, agent_col = self._find_cell_of_rectangle()

        states = list()
        for reward in self.rewards:
            obj_row = reward['row']
            obj_col = reward['col']
            states.append(obj_row - agent_row)
            states.append(obj_col - agent_col)
            states.append(reward['reward'])
            if 'direction' in reward:
                states.append(reward['direction'])

        return states

    def _load_images(self):
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def _build_canvas(self):
        w, h = WIDTH*UNIT, HEIGHT*UNIT

        canvas = tk.Canvas(self, bg='white', height=h, width=w)

        for x in range(0, w, UNIT):
            x0, y0, x1, y1 = x, 0, x, h
            canvas.create_line(x0, y0, x1, y1)

        for y in range(0, h, UNIT):
            x0, y0, x1, y1 = 0, y, w, y
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = self._draw_shape(canvas, row=0, col=0, image=self.shapes[0])

        canvas.pack()

        return canvas

    def _check_which_reward(self, row, col):
        rewards = 0
        done = False
        for obj in self.rewards:
            obj_row, obj_col = obj['row'], obj['col']
            if row == obj_row and col == obj_col:
                rewards += obj['reward']
                if 'direction' not in obj:
                    done = True
                break

        return rewards, done

    def _draw_shape(self, canvas, row, col, image):
        x, y = col*UNIT+UNIT/2, row*UNIT+UNIT/2
        return canvas.create_image(x, y, image=image)

    def _render(self):
        self.update()
        time.sleep(self.render_speed)

    def _move_rewards(self):
        new_rewards = []
        for reward in self.rewards:
            if 'direction' not in reward:
                new_rewards.append(reward)
                continue
            reward['coords'] = coords = self._move_const(reward)
            reward['row'], reward['col'] = self._find_cell_by_coords(coords)
            new_rewards.append(reward)
        return new_rewards

    def _move_to_origin(self):
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, -x +UNIT/2, -y +UNIT/2)

    def _move_rectangle(self, action):
        x0, y0 = self.canvas.coords(self.rectangle)

        dx, dy = 0, 0
        if action == 0: dx = -UNIT   # Left
        elif action == 1: dx = UNIT  # Right
        elif action == 2: dy = -UNIT # Up
        elif action == 3: dy = UNIT  # Down

        x1, y1 = x0 + dx, y0 + dy
        if x1 < 0 or x1 > WIDTH*UNIT:
            return False
        if y1 < 0 or y1 > HEIGHT*UNIT:
            return False

        self.canvas.move(self.rectangle, dx, dy)
        self.canvas.tag_raise(self.rectangle)
        return True

    def _move_const(self, reward):
        figure = reward['figure']
        dir = reward['direction']

        row, col = self._find_cell_by_coords(self.canvas.coords(figure))
        
        if col == 0:
            reward['direction'] = dir = 1
        elif col == (WIDTH-1):
            reward['direction'] = dir = -1

        dx = UNIT * dir

        self.canvas.move(figure, dx, 0)
        return self.canvas.coords(figure)

    def _find_cell_of_rectangle(self):
        coords = self.canvas.coords(self.rectangle)
        return self._find_cell_by_coords(coords)

    def _find_cell_by_coords(self, coords):
        x, y = coords
        row, col = (y-UNIT/2)/UNIT, (x-UNIT/2)/UNIT
        return int(row), int(col)
