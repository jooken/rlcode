import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import time
import json
import os

UNIT = 100  # Pixel Count of a cell
WIDTH = 6 # Horizontal Cell Count on Grid World
HEIGHT = 5 # Vertical Cell Count on Grid World
POSSIBLE_ACTIONS = [0, 1, 2, 3] #Left, Right, Up, Down
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Vector

TARGET_LOCATIONS=[(2,3)]
OBSTACLE_LOCATIONS=[(1,1),(1,3),(2,1),(2,2),(1,4),(3,3)]

POLICY_JSON = './data/policy.json'

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        self.geometry('{2}x{3}+{0}+{1}'.format(2600,400,WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = agent.env
        self.agent = agent
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        
        for row, col in TARGET_LOCATIONS:
            self.text_reward(row, col, "R : 1.0")
        for row, col in OBSTACLE_LOCATIONS:
            self.text_reward(row, col, "R : -1.0")
        
        if os.path.exists(POLICY_JSON):
            self._load_policy_json()

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

        evaluate_button = tk.Button(self, text="Evaluate", command=self.evaluate_policy)
        evaluate_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.11, h+14, window=evaluate_button)

        improve_button = tk.Button(self, text="Improve", command=self.improve_policy)
        improve_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(w*0.30, h+14, window=improve_button)

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

    def _draw_shape(self, canvas, row, col, image):
        x, y = col*UNIT+UNIT/2, row*UNIT+UNIT/2
        shape = canvas.create_image(x, y, image=image)
        return shape

    def text_reward(self, row, col, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 5, 5
        x, y = origin_x + (UNIT * col), origin_y + (UNIT *row)
        font = (font, str(size), style)
        self.canvas.create_text(x, y, fill='black', text=content, font=font, anchor=anchor)

    def evaluate_policy(self):
        self.evaluation_count += 1
        for text in self.texts:
            self.canvas.delete(text)
        self.agent.policy_evaluation() #<-------An agent evaluate value of current state
        self.print_value_table(self.agent.value_table) #------ value function

    def improve_policy(self):
        self.improvement_count += 1
        for arrow in self.arrows:
            self.canvas.delete(arrow)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)

    def move_by_policy(self):
        # print('improvement_count={0}, is_moving={1}'.format(self.improvement_count, self.is_moving))
        if self.improvement_count != 0 and self.is_moving == 0:
            self.is_moving = 1

            self._move_to_origin()

            # self.update()
            row, col = self._find_rectangle_matrix_pos()
            while len(self.agent.policy_table[row][col]) != 0:
                self.after(100, self._move_rectangle(self.agent.get_action([row,col])))
                row, col = self._find_rectangle_matrix_pos()

            self.is_moving = 0

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

    def reset(self):
        if self.is_moving == 0:
            self.evaluation_count = 0
            self.improvement_count = 0
            self._move_to_origin()
            for text in self.texts:
                self.canvas.delete(text)
            for arrow in self.arrows:
                self.canvas.delete(arrow)
            self.agent.reset()

    def store_policy(self):
        policy_dict, value_dict = {}, {}
        for state in self.env.get_all_states():
            policy_dict[str(tuple(state))] = self.agent.get_policy(state)
            value_dict[str(tuple(state))] = self.agent.get_value(state)
        
        with open(POLICY_JSON, 'wt') as f:
            f.write(json.dumps({
                'improvement_count': self.improvement_count,
                'policy_table': policy_dict, 
                'evaluation_count': self.evaluation_count,
                'value_table': value_dict
                }, indent=4))

    def _load_policy_json(self):
        with open(POLICY_JSON, 'rt') as f:
            dump = json.load(f)
            if 'improvement_count' in dump and 'policy_table' in dump:
                self.improvement_count = dump['improvement_count']
                policy_table = dump['policy_table']
                for state in policy_table:
                    self.agent.set_policy(list(eval(state)), policy_table[state])
                self.draw_from_policy(self.agent.policy_table)
            if 'evaluation_count' in dump and 'value_table' in dump:
                self.evaluation_count = dump['evaluation_count']
                value_table = dump['value_table']
                for state in value_table:
                    self.agent.set_value(list(eval(state)), value_table[state])
                self.print_value_table(self.agent.value_table)


    def print_value_table(self, value_table):
        for col in range(WIDTH):
            for row in range(HEIGHT):
                self.text_value(row, col, round(value_table[row][col], 2))

    def text_value(self, row, col, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 70, 85
        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill='black', text=content, font=font, anchor=anchor)
        return self.texts.append(text)
    
    def draw_from_policy(self, policy_table):
        for col in range(WIDTH):
            for row in range(HEIGHT):
                self._draw_on_arrow(row, col, policy_table[row][col])

    def _draw_on_arrow(self, row, col, policy):
        if self.env.is_final_state([row, col]): return

        if policy[0] > 0: # Left
            x, y = col * UNIT + 10, row * UNIT + 50
            arrow = self.canvas.create_image(x, y, image=self.left)
            self.arrows.append(arrow)

        if policy[1] > 0: # Right
            x, y = col * UNIT + 90, row * UNIT + 50
            arrow = self.canvas.create_image(x, y, image=self.right)
            self.arrows.append(arrow)

        if policy[2] > 0: # Up
            x, y = col * UNIT + 50, row * UNIT + 10
            arrow = self.canvas.create_image(x, y, image=self.up)
            self.arrows.append(arrow)

        if policy[3] > 0: # Down
            x, y = col * UNIT + 50, row * UNIT + 90
            arrow = self.canvas.create_image(x, y, image=self.down)
            self.arrows.append(arrow)

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
