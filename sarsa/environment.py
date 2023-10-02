import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import os
import json

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 필셀 수
HEIGHT = 5  # 그리드 월드 세로
WIDTH = 6  # 그리드 월드 가로

TARGET_LOCATIONS=[(2,3)]
OBSTACLE_LOCATIONS=[(1,1),(1,3),(2,1),(2,2),(1,4),(3,3)]

PERSIST_JSON = './data/data.json'

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['l', 'r', 'u', 'd']
        self.n_actions = len(self.action_space)
        self.title('SARSA')
        self.geometry('{2}x{3}+{0}+{1}'.format(2600,400,WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def load_data(self, agent):
        if os.path.exists(PERSIST_JSON):
            data = None
            with open(PERSIST_JSON, 'rt') as f:
                data = json.load(f)
            if data:
                episode = 0
                if 'episode' in data:
                    episode = data['episode']
                agent.from_json(data)
                return episode
        return 0

    def save_data(self, agent, episode):
        data = agent.to_json()
        if data:
            data['episode'] = episode
            with open(PERSIST_JSON, 'wt') as f:
                f.write(json.dumps(data, indent=4))
                return

    def _build_canvas(self):
        w, h = WIDTH*UNIT, HEIGHT*UNIT

        canvas = tk.Canvas(self, bg='white', height=h, width=w)

        for pos_x in range(0, w, UNIT):
            x0, y0, x1, y1 = pos_x, 0, pos_x, h
            canvas.create_line(x0, y0, x1, y1)

        for pos_y in range(0, h, UNIT):
            x0, y0, x1, y1 = 0, pos_y, w, pos_y
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = self._draw_shape(canvas, row=0, col=0, image=self.shapes[0])
        self.triangles = []
        for row, col in OBSTACLE_LOCATIONS:
            self.triangles.append(self._draw_shape(canvas, row, col, image=self.shapes[1]))
        self.circles = []
        for row, col in TARGET_LOCATIONS:
            self.circles.append(self._draw_shape(canvas, row, col, image=self.shapes[2]))

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def print_value_all(self, q_table):
        for text in self.texts:
            self.canvas.delete(text)
        self.texts.clear()

        for row in range(HEIGHT):
            for col in range(WIDTH):
                for index, action in enumerate(self.action_space):
                    state = (row, col)
                    q = q_table[state][index]
                    self._text_value(row, col, index, round(q, 3))

    def reset(self):
        self.update()
        time.sleep(0.05)
        self._move_to_origin()
        return self._find_state_of_rectangle()

    def step(self, action):
        state = self._find_state_of_rectangle()
        self._move_rectangle(action)
        next_state = self._find_state_of_rectangle()

        if state == next_state:
            return next_state, -100, False

        for circle in self.circles:
            if self._is_state_meet_shape(next_state, circle):
                return next_state, 100, True

        for triangle in self.triangles:
            if self._is_state_meet_shape(next_state, triangle):
                return next_state, -100, True

        return next_state, 0, False

    def render(self):
        time.sleep(0.01)
        self.update()

    def _text_value(self, row, col, dir, content, font='Helvetica', size=9, style='normal', anchor='nw'):
        offset = 0
        if dir == 0:   # Left
            offset, origin_x, origin_y = 0,3,50
        elif dir == 1: # Right
            offset, origin_x, origin_y =  -1,97,50
        elif dir == 2: # Up
            offset, origin_x, origin_y = -0.5,50,3
        elif dir == 3: # Down
            offset, origin_x, origin_y = -0.5,50,87

        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        fill = 'black' if content <= 0.0 else 'blue'
        font = (font, str(size), style)
        text = self.canvas.create_text(x+offset*len(str(content))*6.5, y, fill=fill, text=content, font=font, anchor=anchor)
        return self.texts.append(text)

    def _draw_shape(self, canvas, row, col, image):
        '''
            _draw_shape(): 지정한 이미지(image)를 지정 셀에 출력
        '''
        x, y = col*UNIT+UNIT/2, row*UNIT+UNIT/2
        shape = canvas.create_image(x, y, image=image)
        return shape

    def _move_to_origin(self):
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, -x +UNIT/2, -y +UNIT/2)

    def _move_rectangle(self, action):
        self.render()

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

    def _find_state_of_rectangle(self):
        coords = self.canvas.coords(self.rectangle)
        state = self._find_state_by_coords(coords)
        return state

    def _find_state_by_coords(self, coords):
        x, y = coords
        row, col = (y-UNIT/2)/UNIT, (x-UNIT/2)/UNIT
        return int(row), int(col)

    def _is_state_meet_shape(self, state, shape):
        coords = self.canvas.coords(shape)
        found_state = self._find_state_by_coords(coords)
        return (state == found_state)