import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import time
import json
import os
import random
import logging

UNIT = 100  # Pixel Count of a cell
WIDTH = 6 # Horizontal Cell Count on Grid World
HEIGHT = 5 # Vertical Cell Count on Grid World
POSSIBLE_ACTIONS = [0, 1, 2, 3] #Left, Right, Up, Down
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Vector

TARGET_LOCATIONS=[(2,3)]
OBSTACLE_LOCATIONS=[(1,1),(1,3),(2,1),(2,2),(1,4),(3,3)]

PERSIST_JSON = './data/data.json'

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Value Iteration')
        self.geometry('{2}x{3}+{0}+{1}'.format(2600,400,WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = agent.env
        self.agent = agent
        self.iteration_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        
        for row, col in TARGET_LOCATIONS:
            self._text_reward(row, col, "R : 1.0")
        for row, col in OBSTACLE_LOCATIONS:
            self._text_reward(row, col, "R : -1.0")
        
        if os.path.exists(PERSIST_JSON):
            self._load_from_persist_json()

    def load_images(self):
        up = PhotoImage(Image.open("../img/up.png").resize((13,13)))
        down = PhotoImage(Image.open("../img/down.png").resize((13,13)))
        left = PhotoImage(Image.open("../img/left.png").resize((13,13)))
        right = PhotoImage(Image.open("../img/right.png").resize((13,13)))
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65,65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65,65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65,65)))
        return (up, down, left, right), (rectangle, triangle, circle)

    def _build_canvas(self):
        w, h = WIDTH*UNIT, HEIGHT*UNIT

        canvas = tk.Canvas(self, bg='white', width=w, height=h)

        calculate_button = tk.Button(self, text="Calculate", command=self.calculate_value)
        calculate_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.11, h+14, window=calculate_button)

        print_policy_button = tk.Button(self, text="Print Policy", command=self.print_policy)
        print_policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.30, h+14, window=print_policy_button)

        move_by_policy_button = tk.Button(self, text="Move", command=self.move_by_policy)
        move_by_policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.49, h+14, window=move_by_policy_button)

        reset_button = tk.Button(self, text="Reset", command=self.reset)
        reset_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.68, h+14, window=reset_button)

        store_button = tk.Button(self, text="Store", command=self.store_policy)
        store_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.87, h+14, window=store_button)

        for pos_x in range(0, w, UNIT):
            x0, y0, x1, y1 = pos_x, 0, pos_x, h
            canvas.create_line(x0, y0, x1, y1)

        for pos_y in range(0, h, UNIT):
            x0, y0, x1, y1 = 0, pos_y, w, pos_y
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = self._draw_shape(canvas, row=0, col=0, image=self.shapes[0])
        for row, col in OBSTACLE_LOCATIONS:
            self._draw_shape(canvas, row, col, image=self.shapes[1])
        for row, col in TARGET_LOCATIONS:
            self._draw_shape(canvas, row, col, image=self.shapes[2])
        
        canvas.pack()
        return canvas

    def calculate_value(self):
        self.iteration_count += 1
        self._clear_values()
        self.agent.value_iteration()
        self._print_values(self.agent.value_table)

    def print_policy(self):
        self.improvement_count += 1
        self._draw_arrows_of_all_states()

    def move_by_policy(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1
            self._move_to_origin()
            row, col = self._find_rectangle_matrix_pos()
            while len(self.agent.get_action([row,col])) != 0:
                action = random.sample(self.agent.get_action([row,col]), 1)[0]
                self.after(100, self._move_rectangle(action))
                row, col = self._find_rectangle_matrix_pos()

            self.is_moving = 0

    def reset(self):
        if self.is_moving == 0:
            self.iteration_count = 0
            self.improvement_count = 0
            self._clear_values()
            self._clear_arrows()
            self.agent.reset()
            self._move_to_origin()

    def store_policy(self):
        self._store_to_persist_json()

    def _load_from_persist_json(self):
        '''
            _load_from_persist_json(): 저장된 json으로 재구성
        '''
        data = None
        with open(PERSIST_JSON, 'rt') as f:
            data = json.load(f)
        if data:
            if 'iteration_count' in data:
                self.iteration_count = data['iteration_count']
            if 'improvement_count' in data:
                self.improvement_count = data['improvement_count']          
            self.agent.from_json(data)
            self._print_values(self.agent.value_table)
            self._draw_arrows_of_all_states()

    def _store_to_persist_json(self):
        '''
            _store_to_persist_json(): json 파일로 agent 훈련 데이터 저장
        '''
        data = self.agent.to_json()
        if data:
            data['iteration_count'] = self.iteration_count
            data['improvement_count'] = self.improvement_count
            with open(PERSIST_JSON, 'wt') as f:
                f.write(json.dumps(data, indent=4))
                return
        logging.info('no persistence!!')

    def _draw_shape(self, canvas, row, col, image):
        '''
            _draw_shape(): 지정한 이미지(image)를 지정 셀에 출력
        '''
        x, y = col*UNIT+UNIT/2, row*UNIT+UNIT/2
        shape = canvas.create_image(x, y, image=image)
        return shape

    def _text_reward(self, row, col, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        '''
            _text_reward() : 지정한 상태(=셀)로의 보상(Reward) 출력
        '''
        origin_x, origin_y = 5, 5
        x, y = origin_x + (UNIT * col), origin_y + (UNIT *row)
        font = (font, str(size), style)
        self.canvas.create_text(x, y, fill='black', text=content, font=font, anchor=anchor)

    def _clear_values(self):
        for text in self.texts:
            self.canvas.delete(text)

    def _print_values(self, value_table):
        '''
            _print_values() : 모든 상태에 대한 가치함수를 출력
        '''
        for col in range(WIDTH):
            for row in range(HEIGHT):
                self._text_value(row, col, round(value_table[row][col], 2))

    def _text_value(self, row, col, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        '''
            _text_value() : 지정한 상태(=셀)의 가치함수(value function)를 출력
        '''
        origin_x, origin_y = 70, 85
        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill='black', text=content, font=font, anchor=anchor)
        return self.texts.append(text)

    def _clear_arrows(self):
        for arrow in self.arrows:
            self.canvas.delete(arrow)

    def _draw_arrows_of_all_states(self):
        self._clear_arrows()
        for state in self.env.get_all_states():
            actions = self.agent.get_action(state)
            self._draw_arrows(state, actions)

    def _draw_arrows(self, state, actions):
        row, col = state[0], state[1]
        for action in actions:
            self._draw_one_arrow(row, col, action)

    def _draw_one_arrow(self, row, col, action):
        if self.env.is_final_state([row, col]): return
        
        if action == 0: # Left
            x, y = col * UNIT + 10, row * UNIT + 50
            arrow = self.canvas.create_image(x, y, image=self.left)
            self.arrows.append(arrow)
        elif action == 1: # Right
            x, y = col * UNIT + 90, row * UNIT + 50
            arrow = self.canvas.create_image(x, y, image=self.right)
            self.arrows.append(arrow)
        elif action == 2: # Up
            x, y = col * UNIT + 50, row * UNIT + 10
            arrow = self.canvas.create_image(x, y, image=self.up)
            self.arrows.append(arrow)
        elif action == 3: # Down
            x, y = col * UNIT + 50, row * UNIT + 90
            arrow = self.canvas.create_image(x, y, image=self.down)
            self.arrows.append(arrow)

    def _move_to_origin(self):
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, -x +UNIT/2, -y +UNIT/2)

    def _move_rectangle(self, action):
        self._render()
        dx, dy = 0, 0
        if action == 0: dx = -UNIT
        elif action == 1: dx = UNIT
        elif action == 2: dy = -UNIT
        elif action == 3: dy = UNIT
        self.canvas.move(self.rectangle, dx, dy)

    def _render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()

    def _find_rectangle_matrix_pos(self):
        x, y = self.canvas.coords(self.rectangle)
        # print('x={0},y={1}'.format(x,y))
        row, col = (y-UNIT/2)/UNIT, (x-UNIT/2)/UNIT
        # print('row={0},col={1}'.format(row,col))
        return int(row), int(col)

class Env:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward = self._init_reward()
        self.all_state = self._init_state()

    def _init_reward(self):
        reward = [[0]*WIDTH for _ in range(HEIGHT)]
        for row, col in TARGET_LOCATIONS:
            reward[row][col] = 1.0
        for row, col in OBSTACLE_LOCATIONS:
            reward[row][col] = -1.0
        return reward

    def _init_state(self):
        states = []
        for col in range(WIDTH):
            for row in range(HEIGHT):
                state = [row, col]
                states.append(state)
        return states

    def get_all_states(self):
        # 상태는 Grid World의 좌표!!
        return self.all_state
    
    def is_final_state(self, state):
        current = tuple(state)
        for alt in TARGET_LOCATIONS:
            if current == alt:
                return True
        return False

    def state_after_action(self, state, action):
        row, col = state[0], state[1]
        if action == 0: # Left
            col = col - 1 if col > 0 else col
        elif action == 1: # Right
            col = col + 1 if col < WIDTH-1 else col
        elif action == 2: # Up
            row = row - 1 if row > 0 else row
        elif action == 3: # Down
            row = row + 1 if row < HEIGHT-1 else row
        return [row, col]

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        next_row, next_col = next_state
        # print('state={0},action={1} -> next_state={2}'.format(state, action, next_state))
        reward = self.reward[next_row][next_col]
        return reward
