import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import json
import os

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage

UNIT = 100  # Pixel Count of a cell
WIDTH = 6 # Horizontal Cell Count on Grid World
HEIGHT = 5 # Vertical Cell Count on Grid World
POSSIBLE_ACTIONS = [0, 1, 2, 3] #Left, Right, Up, Down

TARGET_LOCATIONS=[(2,3)]
OBSTACLE_LOCATIONS=[(1,1),(1,3),(2,1),(2,2),(1,4),(3,3)]

PERSIST_JSON = './data/data.json'

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('temporal difference')
        self.geometry('{2}x{3}+{0}+{1}'.format(2600,400, WIDTH * UNIT, HEIGHT * UNIT))
        self.texts = []
        self.env = agent.env
        self.agent = agent
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()

    def load_value_function(self):
        if os.path.exists(PERSIST_JSON):
            data = None
            with open(PERSIST_JSON, 'rt') as f:
                data = json.load(f)
            if data:
                self.agent.from_json(data)

    def store_value_function(self):
        data = self.agent.to_json()
        if data:
            with open(PERSIST_JSON, 'wt') as f:
                f.write(json.dumps(data, indent=4))
                return

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def _build_canvas(self):
        w, h = WIDTH*UNIT, HEIGHT*UNIT

        canvas = tk.Canvas(self, bg='white', width=w, height=h)

        for pos_x in range(0, w, UNIT):
            x0, y0, x1, y1 = pos_x, 0, pos_x, h
            canvas.create_line(x0, y0, x1, y1)

        for pos_y in range(0, h, UNIT):
            x0, y0, x1, y1 = 0, pos_y, w, pos_y
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = self._draw_shape(canvas, row=0, col=0, image=self.shapes[0])
        self.circles = []
        self.triangles = []
        for row, col in OBSTACLE_LOCATIONS:
            self.triangles.append(self._draw_shape(canvas, row, col, image=self.shapes[1]))
        for row, col in TARGET_LOCATIONS:
            self.circles.append(self._draw_shape(canvas, row, col, image=self.shapes[2]))

        # pack all
        canvas.pack()
        return canvas

    def reset(self):
        self.update()
        time.sleep(0.5)
        self._move_to_origin()
        return self._find_state_of_rectangle()

    def step(self, action):
        self._move_rectangle(action)
        next_state = self._find_state_of_rectangle()

        reward = 0
        done = False

        # print('check circle - {0}'.format(len(self.circles)))
        for circle in self.circles:
            if self._is_state_meet_shape(next_state, circle):
                reward = 100
                done = True
                # print('meet circle')

        # print('check triangles - {0}'.format(len(self.triangles)))
        if not done:
            for triangle in self.triangles:
                if self._is_state_meet_shape(next_state, triangle):
                    reward = -100
                    done = True
                    # print('meet triangle')

        return next_state, reward, done

    def _is_state_meet_shape(self, state, shape):
        coords = self.canvas.coords(shape)
        found_state = self._find_state_by_coords(coords)
        # print('{0} == {1}=>{2}'.format(state, coords, found_state))
        return (state == found_state)

    def render(self):
        time.sleep(0.03)
        self.canvas.tag_raise(self.rectangle)
        self.update()

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
        if action == 0: dx = -UNIT
        elif action == 1: dx = UNIT
        elif action == 2: dy = -UNIT
        elif action == 3: dy = UNIT

        x1, y1 = x0 + dx, y0 + dy
        if x1 < 0 or x1 > WIDTH*UNIT:
            return False
        if y1 < 0 or y1 > HEIGHT*UNIT:
            return False

        self.canvas.move(self.rectangle, dx, dy)
        return True

    def _find_state_of_rectangle(self):
        coords = self.canvas.coords(self.rectangle)
        state = self._find_state_by_coords(coords)
        self.env.current_state = state
        # print('current_state = {0}({1})'.format(type(state), state))
        return state

    def _find_state_by_coords(self, coords):
        x, y = coords
        row, col = (y-UNIT/2)/UNIT, (x-UNIT/2)/UNIT
        return int(row), int(col)

class Env:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.possible_actions = POSSIBLE_ACTIONS
        self.current_state = (0,0)

    def state_after_action(self, action):
        row, col = self.current_state
        if action == 0: # Left
            col = col - 1 if col > 0 else col
        elif action == 1: # Right
            col = col + 1 if col < WIDTH-1 else col
        elif action == 2: # Up
            row = row - 1 if row > 0 else row
        elif action == 3: # Down
            row = row + 1 if row < HEIGHT-1 else row
        return row, col
